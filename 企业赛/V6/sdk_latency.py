"""Time one JSON-line request at a time through the released image's real SDK."""
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import tempfile
from time import perf_counter_ns

import numpy as np

HERE = Path(__file__).resolve().parent
WORK = HERE.parent / "work"
PUBLIC = HERE.parents[1] / "校赛/work"
ASSETS = WORK / "v6/serving-assets"


def export_image(version, root, archive=None):
    root.mkdir(mode=0o755)
    with (archive or WORK / version / "delivery/harness.tar").open("rb") as image:
        with subprocess.Popen([PUBLIC / "tools/crane", "export", "-", "-"], stdin=image,
                              stdout=subprocess.PIPE) as exporter:
            subprocess.run(["tar", "-xf", "-", "-C", root], stdin=exporter.stdout, check=True)
            exporter.stdout.close()
            if exporter.wait():
                raise RuntimeError(f"Image export failed for {version}")
    (root / "model").mkdir(exist_ok=True)


def warm_and_measure(root, source, key, lines):
    # Create a fresh process for each checkpoint/runtime pair. The SDK resolves /model.
    target = root / "model"
    for name in ("checkpoint.npz", "head.npz", "reference.npz"):
        origin = ASSETS / source / key / name
        if origin.is_file():
            shutil.copyfile(origin, target / name)
        elif (target / name).is_file():
            (target / name).unlink()
    command = [shutil.which("chroot"), "--userspec=65534:65534", str(root), "/usr/local/bin/python3",
               "-B", "/app/harness.py", "predict"]
    env = dict(PATH="/usr/local/bin:/usr/bin:/bin", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    start = perf_counter_ns()
    with subprocess.Popen(command, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE) as proc:
        def one(line):
            proc.stdin.write(line)
            proc.stdin.flush()
            raw = proc.stdout.readline()
            if not raw:
                raise RuntimeError(f"SDK ended early with {proc.poll()}")
            return raw
        one(lines[0])
        cold_ms = (perf_counter_ns() - start) / 1e6
        for line in lines[:32]:
            one(line)
        times, scores = [], []
        for line in lines:
            tic = perf_counter_ns()
            raw = one(line)
            times.append((perf_counter_ns() - tic) / 1000)
            pred = json.loads(raw)["outputs"]
            scores.append([p["score"] for p in pred] if key == "A" else [pred["value"]])
        proc.stdin.close()
        if proc.wait(timeout=8):
            raise RuntimeError("SDK exited with an error")
    p50, p95, p99 = np.quantile(times, [.5, .95, .99]).tolist()
    return dict(cold_process_ms=cold_ms, p50_us=p50, p95_us=p95, p99_us=p99, n=len(times)),np.array(scores)


def main():
    cpus = sorted(os.sched_getaffinity(0))
    os.sched_setaffinity(0, {cpus[7 % len(cpus)]})
    results = []
    with tempfile.TemporaryDirectory(dir=WORK / "v6") as tmp:
        roots = {v:Path(tmp) / v for v in ("v5", "v6")}
        for version, root in roots.items():
            export_image(version, root)
        for key in "ABC":
            task = "classification" if key == "A" else "regression"
            with (PUBLIC / key / "public/validation.jsonl").open() as fh:
                lines = [json.dumps({"task":task, "inputs":json.loads(line)["inputs"]},
                                    separators=(",", ":")).encode()+b"\n" for line in fh]
            for i in range(5):
                chosen = random.Random(20261127+i).sample(lines,min(600,len(lines)))
                cases = [("v5","v5"),("v5","v6"),("v6","v6")]
                responses = {}
                for source, version in (reversed(cases) if i%2 else cases):
                    measured, scores = warm_and_measure(roots[version],source,key,chosen)
                    results.append(dict(task=key, source=source, runtime=version, repeat=i, **measured))
                    responses[source,version] = scores
                old, new = responses["v5","v5"], responses["v5","v6"]
                np.testing.assert_allclose(old,new,rtol=1e-9,atol=1e-8)
                if key=="A":
                    np.testing.assert_array_equal(old.argmax(axis=1),new.argmax(axis=1))
    path=WORK/"v6/sdk-latency.json"
    path.write_text(json.dumps(dict(cpu=sorted(os.sched_getaffinity(0)),rows=results),indent=2))


if __name__=="__main__":
    main()
