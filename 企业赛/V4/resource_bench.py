"""Three alternating paired runs per task pinned to one available CPU."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("task", choices=list("ABC"))
    args = parser.parse_args()
    available = sorted(os.sched_getaffinity(0))
    os.sched_setaffinity(0, {available["ABC".index(args.task) % len(available)]})
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", BASELINE_CPU_THREADS="1")
    for repeat in range(3):
        versions = ("v3", "v4") if repeat % 2 == 0 else ("v4", "v3")
        for version in versions:
            out = HERE.parent / "work/v4/resources" / str(repeat)
            subprocess.run([sys.executable, "-B", str(HERE / "benchmark.py"), args.task, version, "0",
                            "--output", str(out), "--profile"], env=env, check=True)


if __name__ == "__main__":
    main()
