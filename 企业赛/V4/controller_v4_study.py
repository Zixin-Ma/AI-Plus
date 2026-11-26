"""Hold the engine expert fixed to isolate stage-risk selection and fallback endpoints."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import importlib.util
import os
from pathlib import Path
import subprocess
import sys

import benchmark
import contest_submission as harness

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "work/v4/controller"
ROUTES = ("original","wide","static","predicted","observed","offset","up","coherent15","coherent40","taper15","taper30")


def apply(route, version):
    if version == "v3":
        spec = importlib.util.spec_from_file_location("engine", HERE.parent / "V3/engine.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules["engine"] = module
        spec.loader.exec_module(module)
    harness.ANCHOR_TASKS = {"B"}
    harness.ENGINE_PANEL = None
    harness.OFFSET["C"] = 0.
    harness.WEIGHTS["C"] = (1., .85) if route == "original" else (1., .85, .60, .30)
    if route in ("static","predicted","observed","offset"):
        harness.ANCHOR_TASKS.add("C")
    if route in ("predicted","observed"):
        harness.ENGINE_PANEL = "observed_life" if route == "observed" else "predicted_life"
    harness.ENGINE_OFFSET_MODE = "symmetric"
    if route in ("up","coherent15","coherent40","taper15","taper30"):
        harness.WEIGHTS["C"] = (1., .85)
        harness.OFFSET["C"] = .40 if route == "coherent40" else .30 if route == "taper30" else .15
        harness.ENGINE_OFFSET_MODE = "up" if route == "up" else "taper" if route.startswith("taper") else "coherent"
    if route == "offset":
        harness.OFFSET["C"] = .15


def main():
    p = argparse.ArgumentParser()
    p.add_argument("version", choices=["v3","v4"])
    p.add_argument("route", choices=[*ROUTES,"screen"])
    p.add_argument("split", nargs="?", type=int, choices=range(8))
    p.add_argument("setting", nargs="?", choices=["drift","official","fresh"])
    p.add_argument("--routes", nargs="+", choices=ROUTES, default=list(ROUTES))
    p.add_argument("--workers", type=int, default=7)
    a = p.parse_args()
    if a.route == "screen":
        jobs = [(r,i,s) for r in a.routes for i,s in [(0,"drift"),(3,"drift"),(5,"drift"),(7,"drift"),
                (3,"fresh"),(5,"fresh"),(7,"fresh"),(1,"official"),(5,"official"),(7,"official")]]
        def run(job):
            route, split, setting = job
            dest = OUT / a.version / route
            dest.mkdir(parents=True,exist_ok=True)
            with (dest / f"{setting}-{split}.log").open("w") as f:
                env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
                subprocess.run([sys.executable,"-B",__file__,a.version,route,str(split),setting],
                               env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
        with ThreadPoolExecutor(max_workers=a.workers) as ex:
            list(ex.map(run,jobs))
    else:
        if a.split is None or a.setting is None:
            p.error("Specify a development split and setting")
        apply(a.route,a.version)
        benchmark.OUT = OUT / a.version / a.route
        benchmark.run("C","v4",a.split,a.setting,profile=True)


if __name__ == "__main__":
    main()
