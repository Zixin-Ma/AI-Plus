"""Replay the pre-registered reserved schedules against the live PyTRiO controller."""
from concurrent.futures import ThreadPoolExecutor
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import qualify

HERE = Path(__file__).resolve().parent
CLOUD = qualify.OUT / "cloud"


def replay(case):
    task, version, split = case
    name = f"schedule-{version}-{task}-{split}"
    results = CLOUD / "streams"
    trace, receipts = results / (name + ".json"), results / ("receipts-" + name + ".json")
    if trace.is_file() and receipts.is_file() and len(json.loads(trace.read_text())["rounds"]) == 10 and json.loads(receipts.read_text()):
        return
    args = [sys.executable, "-B", str(HERE / "live_replay.py"), task, version, "schedule", str(split)]
    with (CLOUD / f"{name}.log").open("w") as log:
        subprocess.run(args, stdout=log, stderr=subprocess.STDOUT, check=True,
                       env=dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    CLOUD.mkdir(parents=True, exist_ok=True)
    published = (HERE / "cloud_protocol.json").read_bytes()
    registered = CLOUD / "confirmation-design.json"
    if registered.exists() and registered.read_bytes() != published:
        raise ValueError("Public cloud study design differs from the preregistration")
    if not registered.exists():
        registered.write_bytes(published)
    design = json.loads(published)
    assert design["registered_before_model_freeze"] and design["splits"] == [101, 104, 107]
    assert qualify.manifest() == json.loads((qualify.OUT / "freeze.json").read_text())
    cases = [(task, version, split) for split in design["splits"] for task in design["tasks"]
             for version in design["versions"]]
    with ThreadPoolExecutor(args.workers) as pool:
        list(pool.map(replay, cases))


if __name__ == "__main__":
    main()
