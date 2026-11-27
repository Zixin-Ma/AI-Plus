"""Paired selector experiments on previously opened public validation schedules."""
import argparse
import importlib
import os
import sys
import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import benchmark
import contest_submission as harness

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent / "work/v6/selector"
ROUTES = {"base": None, "confidence": ("confidence", .6), "sequence": ("sequence", .6),
          "conservative": ("confidence", 1.5), "life":None, "residual":None, "dense":None, "near_static":None, "dense_guard":None}

def fit_one(key, route, mode, split, expert, label):
    if split >= 14:
        raise ValueError("Reserved confirmation order")
    if expert == "v5":
        # Task heads can be developed independently while testing the selector.
        from importlib.util import spec_from_file_location, module_from_spec
        for name in ("prototype", "gas", "bearing", "engine"):
            spec = spec_from_file_location(name, HERE.parent/"V5"/(name+".py"))
            module = module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
    harness.REPLAY_SAMPLING = {key:route} if route in ("life","residual") else {}
    harness.EXPOSURE = {key: ROUTES[route]} if ROUTES[route] else {}
    if route in ("dense","dense_guard"):
        # Keep original weights first to preserve the incumbent tie-breaking order.
        harness.WEIGHTS[key] = tuple(dict.fromkeys((*harness.WEIGHTS[key], .95,.70,.50,.25,.05)))
    if route == "near_static":
        harness.WEIGHTS[key] = tuple(dict.fromkeys((*harness.WEIGHTS[key], .30, .05)))
    if route == "dense_guard":
        harness.EXPOSURE = {key: ("confidence", .6)}
    benchmark.OUT = ROOT/label/route
    benchmark.run(key,"v6",split,mode)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--expert",choices=("v5","v6"),default="v5")
    p.add_argument("--label", help="Output cohort for the evaluated expert revision")
    p.add_argument("--tasks",default="ABC")
    p.add_argument("--routes",nargs="+",default=list(ROUTES),choices=ROUTES)
    p.add_argument("--seeds",nargs="+",type=int,default=[0,8,12])
    p.add_argument("--modes",nargs="+",default=["drift","fresh","trace","schedule"])
    p.add_argument("--workers",type=int,default=12)
    p.add_argument("--one",nargs=4)
    args = p.parse_args()
    label=args.label or args.expert
    if args.one:
        k,r,m,n = args.one
        fit_one(k,r,m,int(n),args.expert,label)
    else:
        def run(spec):
            k,r,m,n = spec
            folder = ROOT/label/r
            folder.mkdir(parents=True,exist_ok=True)
            with (folder/f"{m}-{k}-{n}.log").open("w") as log:
                subprocess.run([sys.executable,"-B",__file__,"--expert",args.expert,"--label",label,"--one",k,r,m,str(n)],stdout=log,stderr=log,check=True)
        with ThreadPoolExecutor(args.workers) as pool:
            list(pool.map(run,[(k,r,m,n) for k in args.tasks for r in args.routes for m in args.modes for n in args.seeds]))
