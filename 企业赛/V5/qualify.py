"""Freeze, run and summarize version-paired public evaluations on the server."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
OUT = ROOT.parent / "work/v5"
PUBLIC = ROOT.parents[1] / "校赛/work"
RUNTIME = ("baseline_model", "industrial", "contest_submission", "policy", "guard",
           "gas", "prototype", "bearing", "engine", "serving", "benchmark")


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def manifest():
    code = {n + ".py": sha(ROOT / (n + ".py")) for n in RUNTIME}
    inputs = {f"{k}/{name}": sha(PUBLIC / k / name) for k in "ABC"
              for name in ("checkpoint.npz", "public/train.jsonl", "public/validation.jsonl")}
    return dict(code=code, public_inputs=inputs, protocol=sha(OUT / "protocol.json"))


def path_for(case, version, key, split):
    return OUT / "replay" / f"{case}-{version}-{key}-{split}.json"


def freeze():
    protocol = json.loads((OUT / "protocol.json").read_text())
    if sha(ROOT / "benchmark.py") != protocol["evaluator_sha256"]:
        raise ValueError("Evaluator changed after the schedules were registered")
    lock = OUT / "freeze.json"
    if lock.exists():
        if json.loads(lock.read_text()) != manifest():
            raise ValueError("Existing freeze differs from current runtime or data")
        return
    if any(p.stem.rsplit("-",1)[-1] in ("11","12","13") for p in (OUT / "replay").glob("*.json")):
        raise ValueError("Confirmation must begin after freezing")
    lock.write_text(json.dumps(manifest(), indent=2))


def run_one(spec):
    key, version, split, case = spec
    measured = case.endswith("-measured")
    cmd = [sys.executable, "-B", str(ROOT / "benchmark.py"), key, version, str(split),
           "--setting", case.split("-")[0], "--review", "measured" if measured else "controlled", "--profile"]
    logs = OUT / "paired-logs"
    logs.mkdir(exist_ok=True)
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", BASELINE_CPU_THREADS="1")
    with (logs / f"{case}-{version}-{key}-{split}.log").open("w") as stream:
        subprocess.run(cmd, stdout=stream, stderr=subprocess.STDOUT, env=env, check=True)


def matrix(part):
    if part == "confirm":
        specs = [(case, n) for case in ["drift", "fresh", "official", "fresh-measured","trace"] for n in (11,12,13)]
    else:
        specs = [(case,n) for case in ("drift","official","trace") for n in range(11)]
        specs.extend((case,n) for case in ("fresh","fresh-measured") for n in (3,4,5,6,7,8,9,10))
    return [(k,v,n,c) for k in "ABC" for v in ("v4","v5") for c,n in specs]


def pool(rows):
    rounds = [z for r in rows for z in r["rounds"]]
    return dict(**{k:statistics.mean(r[k] for r in rows) for k in ("mean","worst","forget","delay")},
                lowest=min(r["worst"] for r in rows), calls=sum(z["calls"] for z in rounds),
                prompt_bytes=sum(z["prompt_bytes"] for z in rounds),
                checkpoint_bytes=sum(z["checkpoint_bytes"] for z in rounds), cpu=sum(z["cpu"] for z in rounds),
                streams=len(rows))


def summarize():
    result = {}
    for part in ("dev", "confirm"):
        for k,v,n,c in matrix(part):
            path = path_for(c,v,k,n)
            record = json.loads(path.read_text())
            if len(record["rounds"]) != 10 or any(row["audit"] <= 0 or row["selection"] <= 0 for row in record["rounds"]):
                raise ValueError(f"Incomplete stream {path}")
            bucket = result.setdefault(part + "/" + c, {})
            bucket.setdefault(v, []).append((k,n,record))
    final = {}
    for case,versions in result.items():
        final[case] = {v:{"all":pool([r for _,_,r in values]),
                          **{k:pool([r for task,_,r in values if task==k]) for k in "ABC"}}
                       for v,values in versions.items()}
    (OUT / "paired-summary.json").write_text(json.dumps(final,indent=2))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=["freeze","dev","confirm","summary"])
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--version",choices=["v4","v5"])
    p.add_argument("--tasks",choices=list("ABC"),nargs="+",default=list("ABC"))
    a = p.parse_args()
    if a.action == "freeze":
        freeze()
    elif a.action == "summary":
        summarize()
    else:
        if a.action == "confirm" and json.loads((OUT/"freeze.json").read_text()) != manifest():
            raise ValueError("Frozen manifest mismatch")
        with ThreadPoolExecutor(max_workers=a.workers) as ex:
            list(ex.map(run_one, [spec for spec in matrix(a.action) if spec[0] in a.tasks and (a.version is None or spec[1] == a.version)]))


if __name__ == "__main__":
    main()
