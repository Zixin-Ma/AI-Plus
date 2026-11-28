"""Paired development tests of downside-aware candidate selection; tests old orders only."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import benchmark
import contest_submission as harness

HERE=Path(__file__).resolve().parent
OUT=HERE.parent/"work/v7/controller"
ROUTES={"control":None,"penalty":("penalty",.4),"relative":("relative",.5),"lower":("se",.35),"strong":("relative",1.25)}

def one(task,expert,route,mode,seed):
    assert 0<=seed<20
    if expert=="v6":
        for n in ("prototype","gas","bearing","engine"):
            spec=importlib.util.spec_from_file_location(n,HERE.parent/"V6"/(n+".py"))
            mod=importlib.util.module_from_spec(spec)
            sys.modules[n]=mod
            spec.loader.exec_module(mod)
    harness.PAIRED_RISK={task:ROUTES[route]} if ROUTES[route] else {}
    benchmark.OUT=OUT/expert/route
    benchmark.run(task,"v7",seed,mode)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--expert",choices=["v6","v7"],default="v6")
    p.add_argument("--tasks",default="ABC")
    p.add_argument("--routes",nargs="+",default=list(ROUTES),choices=ROUTES)
    p.add_argument("--modes",nargs="+",default=["drift","schedule","transfer","fresh"])
    p.add_argument("--seeds",nargs="+",type=int,default=[0,8,16])
    p.add_argument("--single",nargs=4)
    p.add_argument("--workers",type=int,default=12)
    a=p.parse_args()
    if a.single:
        k,route,mode,seed=a.single
        one(k,a.expert,route,mode,int(seed));return
    def run(spec):
        k,r,m,n=spec
        root=OUT/a.expert/r
        root.mkdir(exist_ok=True,parents=True)
        with (root/f"{m}-{k}-{n}.log").open("w") as f:
            subprocess.run([sys.executable,"-B",__file__,"--expert",a.expert,"--single",k,r,m,str(n)],stdout=f,stderr=f,check=True)
    with ThreadPoolExecutor(a.workers) as pool:
        list(pool.map(run,[(k,r,m,n) for k in a.tasks for r in a.routes for m in a.modes for n in a.seeds]))
if __name__=="__main__":
    main()
