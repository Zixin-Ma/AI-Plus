"""Measure equivalent mix fast paths on official checkpoints with historical expert heads."""
import importlib.util
import json
from pathlib import Path
from statistics import median
from time import process_time

import numpy as np
import contest_submission as current
import guard

HERE=Path(__file__).resolve().parent
WORK=HERE.parent/"work/v8"

def main():
    spec=importlib.util.spec_from_file_location("prior_loop",HERE.parent/"V7/contest_submission.py")
    prior=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(prior)
    rows=[]
    for key in "ABC":
        public=HERE.parents[1]/"校赛/work"/key
        model=current.Model.load(public/"checkpoint.npz")
        data=[json.loads(s) for s in (public/"public/validation.jsonl").open()]
        x,_,_=current.projection(data,model,key)
        directory=WORK.parent/"v7/serving-assets/v7"/key
        parent,head,_=current.artifact(directory)
        for weight in (0.,.35,.65,.85,1.):
            mixed=guard.freeze(guard.strip_mix(head),model,weight)
            for size in (1,min(512,len(x))):
                xb=x[:size]
                reference=None if key=="A" else current.values(key,model,{},xb)
                old=lambda:prior.values(key,parent,mixed,xb)
                new=lambda:current.values(key,parent,mixed,xb,reference if key!="A" else None)
                error=float(np.max(np.abs(old()-new())))
                assert error < 1.e-10,(key,weight,size,error)
                jobs=(old,new) 
                records=[[],[]]
                for i in range(8):
                    for j in ((0,1) if i%2 else (1,0)):
                        tic=process_time()
                        for _ in range(8): jobs[j]()
                        records[j].append((process_time()-tic)/8)
                rows.append(dict(task=key,weight=weight,rows=size,max_error=error,
                                 original_us=median(records[0])*1e6,fast_us=median(records[1])*1e6))
    (WORK/"reference-prediction-study.json").write_text(json.dumps(rows,indent=2))
    print(len(rows),"pairs, max error",max(r["max_error"] for r in rows))
if __name__=="__main__":
    main()
