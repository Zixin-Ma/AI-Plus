"""Paired controller-only ablations on development schedules."""
import argparse
from pathlib import Path

import benchmark
import contest_submission as harness


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("route", choices=["unlimited", "class04", "class09", "class16", "always_plan"])
    parser.add_argument("split", type=int, choices=[0, 1, 2])
    parser.add_argument("--setting", choices=["official", "drift"], default="drift")
    args = parser.parse_args()
    limits = {"unlimited": None, "class04": .04, "class09": .09, "class16": .16}
    if args.route == "always_plan":
        harness.REUSE_ENGINE_PLAN = False
        task = "C"
    else:
        harness.CLASS_DROP_LIMIT = limits[args.route]
        task = "A"
    benchmark.OUT = Path(__file__).resolve().parent.parent / "work/v2/controller" / args.route
    benchmark.run(task, "v2", args.split, args.setting, profile=True)


if __name__ == "__main__":
    main()
