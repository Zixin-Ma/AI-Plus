"""Locked cross-day transfer experiment on Wörner et al. (Zenodo 15681119).

Use the authors' extracted_features.csv only. Supports the full ZIP from
Supp_Datasets or a hash-checked CSV under work/v2/external. Nothing here
is imported by the competition image.
"""
import csv
import hashlib
import io
import json
from pathlib import Path
import sys
from time import process_time
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))
from baseline_model import Model
from gas import _discriminant

SHA256 = "9933dca26ad0837c25eecca188d9a0499d848bb08e4c27574538e9e8fc69a38d"
FOLDER = ROOT / "work/v2/external"


def read_dataset():
    path = FOLDER / "extracted_features.csv"
    if path.is_file():
        raw = path.read_bytes()
    else:
        archive = ROOT.parent / "Supp_Datasets/A/01_Long_Term_Drift_2025/Dataset.zip"
        with zipfile.ZipFile(archive) as data:
            raw = data.read("extracted_features.csv")
    if hashlib.sha256(raw).hexdigest() != SHA256:
        raise ValueError("Unrecognized external file revision")
    rows = list(csv.DictReader(io.StringIO(raw.decode()), delimiter=" "))
    days = np.array([row["Date[yyyy-mm-dd]"] for row in rows])
    classes, y = np.unique([row["Class"].split()[0] for row in rows], return_inverse=True)
    columns = ["mean_R{}[Ohm]", "rel_diff_R{}", "sample_startslope_R{}", "rec_lvl_R{}"]
    array = np.array([[[float(row[k.format(i)]) for k in columns] for i in range(1, 63)]
                      for row in rows])
    if not np.all(np.isfinite(array)):
        raise ValueError("Nonfinite observation")
    return days, classes, y, array


def f1(y, scores):
    n = scores.shape[1]
    table = np.bincount(n*y+scores.argmax(1), minlength=n*n).reshape(n,n)
    values = 2*np.diag(table) / np.maximum(1,table.sum(1)+table.sum(0))
    return {"n":len(y), "macro_f1":float(values.mean()), "class_f1":values.tolist(),
            "support":table.sum(1).tolist()}


def bounded(scores):
    return np.clip(scores - scores.max(1, keepdims=True), -10, 0)


def main():
    days, classes, y, array = read_dataset()
    unique_days = np.unique(days)
    fit = np.isin(days, unique_days[:12])
    tests = [np.isin(days, block) for block in np.array_split(unique_days[12:], 3)]
    # Use instrument-native features. The competition has 16 sensors and
    # 8 response summaries, while these instruments have 62 and 4.
    features = array.copy()
    features[:, :, 0] = np.log(np.maximum(features[:, :, 0], 1e-6))
    absolute = features.reshape(len(y), -1)
    shape = (features-features.mean(1,keepdims=True)).reshape(len(y),-1)
    model_input = array.reshape(len(y), -1)
    model = Model("classification", [str(i) for i in range(model_input.shape[1])],
                  list(classes), model_input[fit],y[fit])
    model.fit(model_input[fit], y[fit], 20)
    model.meta["learning_rate"] = 0.001
    model.fit(model_input[fit], y[fit], 1)
    base = np.log(np.maximum(model.predict(model_input), 1e-12))
    start = process_time()
    a = _discriminant(absolute[fit], y[fit], len(classes), 48, 0, .005)
    b = _discriminant(shape[fit], y[fit], len(classes), 24, .8, .005)
    cpu = process_time()-start
    norm = np.clip((absolute-a["center"])/a["scale"], -8, 8)
    pooled = bounded((norm@a["weights"]+a["bias"])*0.25)
    norm = np.clip((shape-b["center"])/b["scale"], -8, 8)
    dist = (norm@b["projection"])[:,None,:] - b["means"]
    shape_score = bounded(-.5*(np.einsum("ncd,cde,nce->nc", dist,b["precision"],dist)+b["logdet"]))
    result = dict(source="https://zenodo.org/records/15681119", file_sha256=SHA256, n=len(y),
                  classes=classes.tolist(), training_days=unique_days[:12].tolist(), train_n=int(fit.sum()),
                  test_day_blocks=[np.unique(days[t]).tolist() for t in tests],
                  head_fit_cpu_seconds=cpu, scores={})
    for name, scores in {"MLP":base, "MLP_absolute":base+pooled,
                         "MLP_relative":base+shape_score, "MLP_combined":base+pooled+shape_score}.items():
        results = [f1(y[m], scores[m]) for m in tests]
        result["scores"][name] = dict(all=f1(y[~fit],scores[~fit]), blocks=results)
        print(name, result["scores"][name]["all"]["macro_f1"], [r["macro_f1"] for r in results])
    FOLDER.mkdir(parents=True, exist_ok=True)
    (FOLDER / "external.json").write_text(json.dumps(result,allow_nan=False,indent=2))


if __name__ == "__main__":
    main()
