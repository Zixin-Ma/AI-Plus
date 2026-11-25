"""Frozen C candidate confirmation, disjoint engines, unseen settings and scalar latency."""
import json
import time
from copy import deepcopy

import numpy as np

from bench_engine import C, WORK, Model, adjusted, collect, experiments, metrics
from engine import fit, predict


def confirm(model, train, val, split):
    x, y = train[:2]
    xv, yv, units, domain = val
    start = time.process_time()
    full = deepcopy(model)
    scores, heads = experiments("C", full, {}, (x, y), (xv, yv), 1)
    choice = min(heads, key=scores.get)
    def v1(z):
        return np.maximum(0, adjusted(full, heads[choice], "C", z)[:, 0])
    pv1 = v1(xv)
    c1 = time.process_time() - start
    start = time.process_time()
    model.meta["learning_rate"] = .0006
    idx = np.random.default_rng(2027).choice(len(y), min(len(y), 1536), replace=False)
    model.fit(x[idx], y[idx], 1)
    head = fit(model, x, y, 1)
    def v2(z):
        return predict(model, head, z)
    pv2 = v2(xv)
    c2 = time.process_time() - start
    result = {"v1": {**metrics(yv, pv1, domain), "cpu": c1},
              "v2": {**metrics(yv, pv2, domain), "cpu": c2}}
    # Paired cluster bootstrap, resampling complete operating histories.
    labels = np.char.add(domain, np.strings.mod("-%04d", units))
    _, inv = np.unique(labels, return_inverse=True)
    n = np.bincount(inv)
    se1 = np.bincount(inv, weights=(pv1-yv)**2)
    se2 = np.bincount(inv, weights=(pv2-yv)**2)
    rng = np.random.default_rng(422)
    picks = rng.integers(len(n), size=(1000, len(n)))
    delta = np.sqrt(se2[picks].sum(1)/n[picks].sum(1)) - np.sqrt(se1[picks].sum(1)/n[picks].sum(1))
    result["engine_bootstrap_delta_rmse_95"] = np.percentile(delta, [2.5, 97.5]).tolist()
    result["engines"] = int(len(n))
    if split == "public":
        bench = xv[rng.choice(len(xv), 2048, replace=False)]
        for name, fn in (("v1", v1), ("v2", v2)):
            batch = fn(bench)
            times = []
            for _ in range(5):
                start = time.process_time()
                row = np.array([fn(z[None])[0] for z in bench])
                times.append((time.process_time()-start)*1e6/len(bench))
            result[name]["microseconds_per_row"] = float(np.median(times))
            result[name]["max_row_batch_difference"] = float(np.max(abs(batch-row)))
        result["weight_bytes"] = sum(a.nbytes for a in head.values())
    print(split, result, flush=True)
    return result


def main():
    tr = collect(C/"public/train.jsonl")
    valid = collect(C/"public/validation.jsonl")
    official = Model.load(C/"checkpoint.npz")
    subsets = [(f"prior_fold_{i}", tr[2]%5 == i) for i in range(3)]
    subsets += [(f"reserved_fold_{i}", tr[2]%5 == i) for i in (3,4)]
    subsets += [(f"unseen_altitude_{alt}", np.round(tr[0][:,21]) == alt) for alt in (0,25)]
    result = {}
    for tag, hold in subsets:
        sub = tuple(a[~hold] for a in tr)
        audit = tuple(a[hold] for a in tr)
        m = Model("regression", official.meta["paths"], [], *sub[:2], hidden=32, seed=42,
                  learning_rate=.002, batch_size=256)
        m.fit(*sub[:2], 15)
        result[tag] = confirm(m, sub, audit, tag)
    result["public"] = confirm(official, tr, valid, "public")
    path = WORK/"v2/engine/confirm.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
