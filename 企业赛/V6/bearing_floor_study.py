"""Cross-device support gating for a frozen B head. Development-only ablation."""
import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np

import bearing
import benchmark
import opening_study

HERE=Path(__file__).resolve().parent
OUT=HERE.parent/"work/v6/bearing-floor"
FIT=bearing.fit_candidates
PREDICT=bearing.predict

def features(model,h,x):
    log=np.log(np.maximum(x[:,h["curve_bands"]],1e-12))
    energy=(log[:,:2]+log[:,2:])*.5
    # Scale, shape and cross-axis mismatch of measured vibration plus frozen backbone life.
    return np.column_stack((energy.mean(1),energy[:,0]-energy[:,1],
                            (log[:,2:]-log[:,:2]).mean(1),
                            np.diff(log[:,:2],axis=1).ravel()-np.diff(log[:,2:],axis=1).ravel(),
                            model.predict(x)))

def nearest(z,centers):
    # Reuse one vector per block of rows when calibrating source coverage.
    sq=np.zeros((len(z),len(centers)))
    for i in range(z.shape[1]):
        sq+=(z[:,i,None]-centers[:,i])**2
    return np.sqrt(np.min(sq,axis=1)/z.shape[1])

def region(m,h,x,y,route):
    tracks=bearing._training_runs(y)
    if not tracks or "curve_bands" not in h:return {}
    z=features(m,h,x)
    center=np.median(z,axis=0)
    scale=np.maximum(1e-7,np.quantile(z,.75,axis=0)-np.quantile(z,.25,axis=0))
    z=np.clip((z-center)/scale,-8.,8.)
    banks=[np.stack([np.median(part,axis=0) for part in np.array_split(z[t],5)]) for t in tracks]
    # Calibrate without query data. Each source trajectory contributes equal mass.
    quantiles=[]
    for j,tr in enumerate(tracks):
        ids=np.linspace(tr.start,tr.stop-1,min(128,tr.stop-tr.start),dtype=int)
        quantiles.append(np.quantile(nearest(z[ids],np.concatenate(banks[:j]+banks[j+1:])),
                                     [.25,.75] if route=="broad" else [.5,.9]))
    cut=np.median(quantiles,axis=0)
    return dict(cover_center=center,cover_scale=scale,cover_bank=np.concatenate(banks),cover_cut=cut)

def install(route):
    if route=="base":return
    def fit(m,x,y,i):
        heads=FIT(m,x,y,i)
        model=next(iter(heads.values()))
        extra=region(m,model,x,y,route)
        return {key:{**h,**extra}for key,h in heads.items()}
    def predict(m,h,x):
        pred=PREDICT(m,h,x)
        if "cover_bank" not in h:return pred
        z=np.clip((features(m,h,x)-h["cover_center"])/h["cover_scale"],-8.,8.)
        dist=nearest(z,h["cover_bank"])
        lo,hi=h["cover_cut"]
        alpha=np.clip((dist-lo)/max(1e-6,hi-lo),0.,1.)
        return pred+alpha*(m.predict(x)-pred)
    bearing.fit_candidates=fit
    bearing.predict=predict

def run(route,mode,split):
    if split not in range(14):raise ValueError("development only")
    install(route)
    dest=OUT/route
    dest.mkdir(exist_ok=True,parents=True)
    if mode=="opening":
        opening_study.OUT=dest
        opening_study.run("B","v6")
    else:
        benchmark.OUT=dest
        benchmark.run("B","v6",split,mode)

def summarize():
    metrics=("stage_mean","stage_floor","worst","forget","delay")
    result={"reference":"enterprise V6 frozen bearing","routes":{}}
    for route in ("base","tail","broad"):
        r={}
        for mode in ("schedule","drift"):
            paths=sorted((OUT/route).glob(f"{mode}-v6-B-*.json"))
            splits=sorted(int(f.stem.split("-")[-1]) for f in paths)
            if not splits or min(splits)<0 or max(splits)>13:
                raise ValueError("Invalid development cohort")
            records=[json.loads((OUT/route/f"{mode}-v6-B-{i}.json").read_text()) for i in splits]
            frozen=[json.loads((OUT/"base"/f"{mode}-v6-B-{i}.json").read_text()) for i in splits]
            gain=[q["stage_floor"]-ref["stage_floor"] for q,ref in zip(records,frozen,strict=True)]
            r[mode]={"orders":splits,"metrics":{k:float(np.mean([v[k] for v in records])) for k in metrics},
                     "floor_up":sum(d>1e-10 for d in gain),"floor_down":sum(d < -1e-10 for d in gain)}
        r["opening"]=json.loads((OUT/route/"v6-B.json").read_text())
        result["routes"][route]=r
    (OUT/"summary.json").write_text(json.dumps(result,indent=2))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--one",nargs=3)
    parser.add_argument("--summarize",action="store_true")
    parser.add_argument("--seeds",type=int,nargs="+",default=[0,5,8,10])
    parser.add_argument("--modes",nargs="+",default=["schedule","drift","opening"])
    parser.add_argument("--routes",nargs="+",default=["base","tail","broad"])
    parser.add_argument("--workers",type=int,default=9)
    a=parser.parse_args()
    if a.summarize:
        summarize()
        return
    if a.one:
        run(a.one[0],a.one[1],int(a.one[2]))
        return
    OUT.mkdir(parents=True,exist_ok=True)
    def job(spec):
        route,mode,i=spec
        with (OUT/f"{route}-{mode}-{i}.log").open("w") as log:
            subprocess.run([sys.executable,"-B",__file__,"--one",route,mode,str(i)],stdout=log,stderr=log,check=True)
    with ThreadPoolExecutor(a.workers) as pool:
        list(pool.map(job,[(r,m,i) for r in a.routes for m in a.modes for i in ([0] if m=="opening" else a.seeds)]))

if __name__=="__main__":
    main()
