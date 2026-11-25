"""CPU continuation from the official MLP and a compact additive residual head."""
import json
from functools import lru_cache
from pathlib import Path

from baseline_model import Model
import numpy as np


@lru_cache(maxsize=8)
def fields(paths):
    return tuple(tuple(item[1] for item in json.loads(path)) for path in paths)


def vector(inputs, schema):
    def walk(path):
        node = inputs
        for key in path:
            node = node[key]
        return node
    return [walk(path) for path in schema]


def dataset(rows, model):
    schema = fields(tuple(model.meta["paths"]))
    labels = model.meta["classes"]
    x, y = [], []
    for row in rows:
        x.append(vector(row["inputs"], schema))
        y.append(labels.index(row["answer"]) if labels else float(row["answer"]))
    x, y = np.asarray(x, dtype=float), np.asarray(y)
    if not len(y) or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Empty or nonfinite public projection")
    return x, y


def signals(key, x, head, fit):
    if key == "A":
        s = np.sign(x) * np.log1p(abs(x))
        grid = s.reshape(len(x), 16, 8)
        relative = grid - np.median(grid, axis=1, keepdims=True)
        return np.column_stack((s, relative.reshape(len(x), -1)))
    if key == "B":
        return np.sign(x) * np.log1p(abs(x))
    # The official model flattens dictionary keys in sorted order: sensors then settings.
    sensors, settings = x[:, :21], x[:, 21:]
    if fit:
        centers, inv, counts = np.unique(np.round(settings[:, [0, 2]]), axis=0,
                                          return_inverse=True, return_counts=True)
        keep = np.argsort(counts)[-8:]
        head["regimes"] = centers[keep]
        head["means"] = np.array([sensors[inv == k].mean(0) for k in keep])
        head["stds"] = np.array([sensors[inv == k].std(0) for k in keep])
    dist = np.sum((settings[:, None, [0, 2]] - head["regimes"][None, :, :]) ** 2, axis=2)
    group = dist.argmin(axis=1)
    adjusted = (sensors - head["means"][group]) / np.maximum(head["stds"][group], 1e-4)
    return np.column_stack((np.clip(adjusted, -7, 7), settings))


def basis(key, x, head, seed=None):
    raw = signals(key, x, head, seed is not None)
    if seed is not None:
        head["center"] = np.median(raw, axis=0)
        q1, q3 = np.percentile(raw, [25, 75], axis=0)
        head["scale"] = np.maximum(q3 - q1, np.maximum(.05 * raw.std(0), 1e-6))
        rng = np.random.default_rng(seed)
        head["proj"] = rng.normal(0, 1.6 / np.sqrt(raw.shape[1]), (raw.shape[1], 96))
        head["phase"] = rng.uniform(-np.pi, np.pi, 96)
    z = np.clip((raw - head["center"]) / head["scale"], -6, 6)
    parts = [np.ones(len(z)), z, np.cos(z @ head["proj"] + head["phase"])]
    if key == "A":
        # Normalized response shape reduces sensitivity to overall array gain.
        shape = z / np.maximum(np.linalg.norm(z, axis=1, keepdims=True), 1e-6)
        if seed is not None:
            anchors = np.random.default_rng(seed).choice(len(shape), min(160, len(shape)), replace=False)
            head["anchors"] = shape[anchors]
        if "anchors" in head:
            cosine = np.clip(shape @ head["anchors"].T, -1, 1)
            parts.extend((np.exp(3 * (cosine - 1)), np.exp(12 * (cosine - 1))))
    return np.column_stack(parts)


def base_output(model, x):
    result = model.predict(x)
    return result if result.ndim == 2 else result[:, None]


def adjusted(model, head, key, x):
    values = base_output(model, x)
    if head:
        values += basis(key, x, head) @ head["coef"]
    return values


def metric(key, y, scores):
    if key != "A":
        return float(np.sqrt(np.mean((np.maximum(scores[:, 0], 0) - y) ** 2)))
    guess = scores.argmax(1)
    present = np.union1d(y, guess)
    return float(np.mean([2 * np.sum((guess == i) & (y == i)) /
                          (np.sum(guess == i) + np.sum(y == i)) for i in present]))


def load_head(path):
    if path is None:
        return {}
    with np.load(path, allow_pickle=False) as f:
        return {k: f[k] for k in f.files}


def experiments(key, parent, old_head, train, valid, index):
    x, y = train
    xv, yv = valid
    before = metric(key, yv, adjusted(parent, old_head, key, xv))
    # Continue every original Adam moment and both original scalers.
    parent.meta["learning_rate"] = {"A": .001, "B": .0008, "C": .0006}[key]
    rate_factor = (1., .45, .15, 1.5, .7)[(index - 1) % 5]
    parent.meta["learning_rate"] *= rate_factor
    parent.fit(x, y, 3 if key == "A" else 1)
    raw = base_output(parent, x)
    raw_valid = base_output(parent, xv)
    scores = {"parent": before, "adam": metric(key, yv, raw_valid)}
    heads = {"adam": {}}
    # Keep the previous residual basis and update its coefficients in place when selected.
    # Once a basis is deployed it remains frozen even if the next public projection changes.
    head = dict(old_head)
    design = basis(key, x, head, seed=None if head else 2026 + index)
    design_val = basis(key, xv, head)
    targets = np.eye(raw.shape[1])[y] if key == "A" else y[:, None]
    residual = targets - raw
    prev = head.get("coef", np.zeros((design.shape[1], raw.shape[1])))
    scores["continued"] = metric(key, yv, raw_valid + design_val @ prev)
    heads["continued"] = old_head
    # Bearing lifetimes are strongly right skewed. Compare ordinary and robust updates.
    for name, shrink, penalty in (("small", .35, .01), ("full", .85, .001), ("robust", .90, .004)):
        penalty *= (1., .3, 3., .08, 8.)[(index - 1) % 5]
        if name == "small":
            shrink /= 1 + (index - 1) % 4
        weights = np.ones(len(y))
        if name == "robust" and key == "B":
            weights = 1 / np.maximum(1, (1 + y) / (1 + np.median(y))) ** .75
        if name == "robust" and key == "A":
            weights = 1 / np.sqrt(np.bincount(y, minlength=raw.shape[1])[y])
        weights /= weights.mean()
        wt = design.T * weights
        gram = wt @ design
        gram.flat[::len(gram) + 1] += max(1e-6, penalty * len(y))
        update = np.linalg.solve(gram, wt @ (residual - design @ prev))
        coef = prev + shrink * update
        scores[name] = metric(key, yv, raw_valid + design_val @ coef)
        heads[name] = {**head, "coef": coef}
    return scores, heads


def drift(x, xv, model):
    # Absolute shift in mean, measured in training standard deviations. Diagnostic only.
    delta = abs(xv.mean(0) - x.mean(0)) / np.maximum(x.std(0), 1e-6)
    top = np.argsort(delta)[-3:][::-1]
    return {"/".join(map(str, fields(tuple(model.meta["paths"]))[i])): round(float(delta[i]), 3)
            for i in top}


def output(model, head, key, inputs):
    x = np.asarray([vector(inputs, fields(tuple(model.meta["paths"])))], dtype=float)
    values = adjusted(model, head, key, x)[0]
    if not model.meta["classes"]:
        return {"value": max(0., float(values[0]))}
    # Emit bounded scores preserving the residual classifier's ordering.
    p = np.exp(values - values.max())
    p /= p.sum()
    if np.sum(p == p.max()) > 1:
        i = int(p.argmax())
        p[i] = np.nextafter(p[i], 1.)
    return [{"label": label, "score": float(v)} for label, v in zip(model.meta["classes"], p)]
