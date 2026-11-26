"""Round-trip the exact official and continued Adam state through both ZIP encodings."""
import json
from pathlib import Path
from statistics import median
from tempfile import TemporaryDirectory
from time import process_time

import numpy as np

import baseline_model
from contest_submission import projection

HERE = Path(__file__).resolve().parent
WORK = HERE.parent / "work/v3"


def check(path, model):
    restored = baseline_model.Model.load(path)
    assert model.meta == restored.meta
    for x, y in [(model.params, restored.params), (model.first, restored.first), (model.second, restored.second)]:
        for name in x:
            np.testing.assert_array_equal(x[name], y[name])
    np.testing.assert_array_equal(model.mean, restored.mean)
    np.testing.assert_array_equal(model.scale, restored.scale)


def main():
    measurements = []
    with TemporaryDirectory(dir=WORK) as directory:
        for key in "ABC":
            data = HERE.parents[1] / "校赛/work" / key
            model = baseline_model.Model.load(data / "checkpoint.npz")
            with (data / "public/train.jsonl").open() as stream:
                rows = [json.loads(next(stream)) for _ in range(256)]
            x, y, _ = projection(rows, model, key)
            for phase in ("starter", "continued"):
                if phase == "continued":
                    model.fit(x, y, 1)
                for compress in (False, True):
                    baseline_model.CHECKPOINT_DEFLATE = compress
                    saves, loads = [], []
                    for i in range(7):
                        path = Path(directory) / f"{key}-{phase}-{compress}-{i}.npz"
                        t = process_time()
                        model.save(path)
                        saves.append(process_time()-t)
                        t = process_time()
                        baseline_model.Model.load(path)
                        loads.append(process_time()-t)
                    check(path, model)
                    measurements.append(dict(task=key, phase=phase, compressed=compress, bytes=path.stat().st_size,
                                             write_ms=median(saves)*1000, read_ms=median(loads)*1000, exact=True))
    (WORK / "checkpoint-io.json").write_text(json.dumps(measurements, indent=2))


if __name__ == "__main__":
    main()
