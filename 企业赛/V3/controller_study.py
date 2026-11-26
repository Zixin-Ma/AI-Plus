"""Causal policy ablations with the V2 task experts kept fixed by default."""
import argparse
import importlib.util
from pathlib import Path
import sys

import benchmark
import contest_submission as harness

HERE = Path(__file__).resolve().parent
ROUTES = ("control", "anchor", "recall", "both", "exact", "switch05", "switch1", "switch2", "order", "cross", "block25", "block75")


def use_v2_experts():
    for name in ("prototype", "gas", "bearing", "engine"):
        spec = importlib.util.spec_from_file_location(name, HERE.parent / "V2" / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("task", choices=list("ABC"))
    p.add_argument("route", choices=ROUTES)
    p.add_argument("split", type=int, choices=range(5))
    p.add_argument("setting", choices=["drift", "official", "fresh"])
    p.add_argument("--review", choices=["measured", "controlled"], default="controlled")
    p.add_argument("--current", action="store_true")
    a = p.parse_args()
    if not a.current:
        use_v2_experts()
    harness.ANCHOR_TASKS = {a.task} if a.route in ("anchor","both","order","cross","block25","block75") else set()
    harness.CONTEXT_TASKS = {a.task} if a.route in ("recall","exact","both") else set()
    harness.guard.BEARING_PANEL = a.route if a.route in ("order","cross") else "amplitude"
    harness.BLOCK_WEIGHT = {"block25":.25,"block75":.75}.get(a.route,0.)
    harness.SWITCH_SE = {"switch05":.5, "switch1":1., "switch2":2.}.get(a.route,0.)
    harness.CONTEXT_RADIUS = .001 if a.route == "exact" else .18
    panel = "integrated" if a.current else "v2-experts"
    benchmark.OUT = HERE.parent / "work/v3/controller" / panel / a.route
    benchmark.run(a.task, "v3", a.split, a.setting, True, a.review)


if __name__ == "__main__":
    main()
