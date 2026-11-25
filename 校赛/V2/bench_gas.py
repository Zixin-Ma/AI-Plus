"""Reproduce gas-array model ablations and blocked, public-data-only evaluations.

Run with OPENBLAS_NUM_THREADS=1 and the server's ai-plus Python.
"""
import copy
import json
from pathlib import Path
import sys
from time import process_time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))
from industrial import Model, adjusted, dataset, experiments, metric
import gas


def projection(split, model):
    path = ROOT / "work/A/public" / (split + ".jsonl")
    with path.open() as stream:
        rows = [json.loads(line) for line in stream]
    rows.sort(key=lambda row: (row["meta"]["source_batch"], row["meta"]["source_index"]))
    return dataset(rows, model)


def dev_mask(y, fraction=0.60):
    result = np.zeros(len(y), dtype=bool)
    for k in np.unique(y):
        positions = np.flatnonzero(y == k)
        result[positions[:int(len(positions) * fraction)]] = True
    return result


def blocked(y, count=3):
    blocks = np.zeros(len(y), dtype=int)
    for k in np.unique(y):
        for i, indices in enumerate(np.array_split(np.flatnonzero(y == k), count)):
            blocks[indices] = i
    return blocks


def quality(y, scores):
    n_classes = scores.shape[1]
    labels = scores.argmax(1)
    matrix = np.bincount(n_classes*y + labels, minlength=n_classes*n_classes).reshape(n_classes, -1)
    f1 = 2 * matrix.diagonal() / np.maximum(1, matrix.sum(0) + matrix.sum(1))
    return dict(n=len(y), macro_f1=float(f1.mean()), min_class_f1=float(f1.min()),
                class_f1=f1.tolist(), support=matrix.sum(1).tolist(), confusion=matrix.tolist())


def v1(model, train, dev, rounds):
    head = {}
    for i in range(1, rounds + 1):
        candidate = copy.deepcopy(model)
        scores, heads = experiments("A", candidate, head, train, dev, i)
        best = max(heads, key=lambda name: scores[name])
        if scores[best] > scores["parent"] + 1e-8:
            model, head = candidate, heads[best]
    return model, head


def compare(initial, train, dev, test, ten_rounds):
    x, y = train
    xv, yv = test
    methods = {"official_start": (lambda v: initial.predict(v), 0.0)}
    for rounds in ([1, 10] if ten_rounds else [1]):
        start = process_time()
        deployed, head = v1(initial, train, dev, rounds)
        methods["v1_" + str(rounds)] = (
            lambda v, m=deployed, h=head: adjusted(m, h, "A", v), process_time()-start)
    start = process_time()
    trained = copy.deepcopy(initial)
    trained.meta["learning_rate"] = 0.001
    trained.fit(x, y, 1)
    continuation_cost = process_time()-start
    methods["continued_adam"] = (trained.predict, continuation_cost)
    start = process_time()
    head = gas.fit(trained, x, y, 1)
    cost = continuation_cost + process_time()-start
    def subset(v, kind):
        base, absolute, relative = gas.components(trained, head, v)
        return base + (absolute if kind == "absolute" else relative)
    methods.update(absolute_lda=(lambda v: subset(v, "absolute"), cost),
                   relative_qda=(lambda v: subset(v, "relative"), cost),
                   combined=(lambda v: gas.predict(trained, head, v), cost))
    summaries = {}
    mask = dev_mask(yv)
    blocks = blocked(yv)
    for name, (predict, seconds) in methods.items():
        start = process_time()
        sc = predict(xv)
        elapsed = process_time() - start
        single = []
        if ten_rounds:
            for i in range(3):
                start = process_time()
                chunk = np.vstack([predict(xv[ix:ix+1]) for ix in range(min(len(xv), 240))])
                single.append((process_time()-start) / len(chunk))
                np.testing.assert_array_equal(sc[:len(chunk)].argmax(1), chunk.argmax(1))
        summaries[name] = dict(all=quality(yv, sc), early=quality(yv[mask], sc[mask]),
                               late=quality(yv[~mask], sc[~mask]),
                               blocks=[quality(yv[blocks==i], sc[blocks==i]) for i in range(3)],
                               train_cpu_seconds=seconds, batch_predict_cpu_seconds=elapsed,
                               single_row_cpu_seconds=np.median(single).item() if single else None)
    return summaries


def main():
    start = Model.load(ROOT / "work/A/checkpoint.npz")
    x, y = projection("train", start)
    xv, yv = projection("validation", start)
    dev = dev_mask(yv)
    data = {"classes": start.meta["classes"],
            "split": "first 60% by source_index within each batch-2 gas is development, last 40% audit",
            "official": compare(start, (x,y), (xv[dev],yv[dev]), (xv,yv), True),
            "train_block_holdout": []}
    # Rebuild the model on retained rows: the packaged checkpoint has already
    # seen batch 1, so using it here would contaminate internal holdouts.
    blocks = blocked(y)
    for i in range(3):
        available = np.flatnonzero(blocks != i)
        dev = dev_mask(y[available], 0.8)
        fit_ix, check_ix = available[dev], available[~dev]
        model = Model("classification", start.meta["paths"], start.meta["classes"], x[fit_ix], y[fit_ix],
                      seed=start.meta["seed"], batch_size=start.meta["batch_size"], hidden=start.meta["hidden"])
        model.fit(x[fit_ix], y[fit_ix], 20)
        result = compare(model, (x[fit_ix], y[fit_ix]), (x[check_ix], y[check_ix]), (x[blocks==i], y[blocks==i]), False)
        data["train_block_holdout"].append({k: v["all"] for k,v in result.items()})
    folder = ROOT / "work/v2/gas"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "comparison.json").write_text(json.dumps(data, indent=2, allow_nan=False))
    for name, values in data["official"].items():
        print(name, *(round(values[s]["macro_f1"],4) for s in ["early","late","all"]),
              "worst_class", round(values["all"]["min_class_f1"],4),
              "worst_block", round(min(s["macro_f1"] for s in values["blocks"]),4),
              "train_s", round(values["train_cpu_seconds"],4),
              "predict_us",round(values["single_row_cpu_seconds"]*1e6,1))
    for i, split in enumerate(data["train_block_holdout"]):
        print("internal",i,{k:round(v["macro_f1"],3) for k,v in split.items()})


if __name__ == "__main__":
    main()
