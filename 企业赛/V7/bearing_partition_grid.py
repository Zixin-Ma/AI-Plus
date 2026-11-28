"""Evaluate every distinct combination of the three public bearing audit windows."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys

import benchmark
import qualify
import importlib.util
# Reproduce the original three-window family for all 27 combinations, including
# seed numbers that are now used for V7's different sliding-window test.
spec=importlib.util.spec_from_file_location("legacy_windows",qualify.ROOT.parent/"V6/benchmark.py")
legacy_windows=importlib.util.module_from_spec(spec)
spec.loader.exec_module(legacy_windows)
benchmark.partitions=legacy_windows.partitions

OUT=qualify.OUT/"bearing-grid"

def register():
    valid=benchmark.WORK/"B/public/validation.jsonl"
    rows=[json.loads(line) for line in valid.read_text().splitlines()]
    devices=sorted({r["meta"]["bearing"] for r in rows})
    assert len(devices)==3
    def fingerprint(split):
        audit=benchmark.partitions(rows,"B",split)[2]
        return hashlib.sha256(audit.tobytes()).hexdigest()
    seen={}
    for split in range(20,5000):
        seen.setdefault(fingerprint(split),split)
        if len(seen)==3**len(devices):
            break
    assert len(seen)==27
    known={fingerprint(i) for i in range(20)}
    grid=[dict(split=i,audit_mask=h,seen_before=h in known) for h,i in seen.items()]
    reg=dict(devices=devices,freeze=qualify.sha(qualify.OUT/"freeze.json"),design="All 27 masks, equal weight",grid=grid)
    target=OUT/"design.json"
    if target.exists():
        if json.loads(target.read_text())!=reg:
            raise ValueError("Existing design mismatch")
    else:
        if any(OUT.glob("schedule-*.json")):
            raise ValueError("Grid must be registered before scoring")
        target.write_text(json.dumps(reg,indent=2))
    return grid

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--single",nargs=2)
    p.add_argument("--workers",type=int,default=8)
    a=p.parse_args()
    OUT.mkdir(exist_ok=True,parents=True)
    if a.single:
        version,i=a.single
        benchmark.OUT=OUT
        benchmark.run("B",version,int(i),"schedule")
        return
    assert qualify.manifest()==json.loads((qualify.OUT/"freeze.json").read_text())
    grid=register()
    def evaluate(pair):
        v,i=pair
        subprocess.run([sys.executable,"-B",__file__,"--single",v,str(i)],stdout=subprocess.DEVNULL,check=True,
                       env=dict(os.environ,OPENBLAS_NUM_THREADS="1",OMP_NUM_THREADS="1"))
    with ThreadPoolExecutor(a.workers) as pool:
        list(pool.map(evaluate,[(v,r["split"]) for r in grid for v in ("v6","v7")]))
    tables={}
    for group,subset in [("all",grid),("outside_v6_initial_twenty",[r for r in grid if not r["seen_before"]])]:
        pairs=[]
        for r in subset:
            both=[json.loads((OUT/f"schedule-{v}-B-{r['split']}.json").read_text())for v in ("v6","v7")]
            assert all(l["static"]==m["static"] and l["audit"]==m["audit"] and l["selection"]==m["selection"] for l,m in zip(both[0]["rounds"],both[1]["rounds"],strict=True))
            pairs.append(both)
        table={"masks":len(subset)}
        for key in ("stage_floor","stage_mean","worst","forget","delay"):
            xy=[[r[j][key] for r in pairs] for j in (0,1)]
            table[key]=dict(v6=statistics.mean(xy[0]),v7=statistics.mean(xy[1]),
                            increased=sum(a<b-1e-9 for a,b in zip(*xy)),decreased=sum(a>b+1e-9 for a,b in zip(*xy)),
                            positive=[sum(x>0 for x in col) for col in xy])
        table["stage_means"]=[[statistics.mean(r[j]["stage_means"][k] for r in pairs) for k in range(3)]for j in (0,1)]
        tables[group]=table
    (OUT/"summary.json").write_text(json.dumps(tables,indent=2))

if __name__=="__main__":
    main()
