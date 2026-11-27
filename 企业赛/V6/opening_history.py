"""Repeated initial deployments with an entire public domain held outside selection.

The official full-size starting pool and within-stage action requests are replayed.
All data, eligibility checks and deployments are computed locally; neither hidden
stage gains nor the published choices of artifacts enter the selection rule.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile

import numpy as np
import benchmark
from opening_study import domain_ids

HERE=Path(__file__).resolve().parent
OUT=HERE.parent/"work/v6/opening-history"
# Public V5 formal run 80900572e91d7d33d6637088915dc5e6, first unchanged training release.
MOVES={"A":("refit","continue"),"B":("refit","continue","continue","refit"),"C":("refit",)*3}

def run(task,version):
    if version=="v5":
        sys.path.insert(0,str(HERE.parent/"V5"))
    import contest_submission as harness
    src=benchmark.WORK/task
    def rows(name):
        with (src/"public"/(name+".jsonl")).open() as fh:
            return [json.loads(line) for line in fh]
    train,valid=rows("train"),rows("validation")
    group_ids=domain_ids(valid,task)
    for row in train+valid:
        row.pop("meta",None)
    initial=harness.Model.load(src/"checkpoint.npz")
    x,y,_=harness.projection(valid,initial,task)
    reference=harness.values(task,initial,{},x)
    results=[]
    OUT.mkdir(exist_ok=True,parents=True)
    for g in np.unique(group_ids):
        sel=np.flatnonzero(group_ids!=g)
        test=group_ids==g
        static=harness.metric(task,y[test],reference[test])
        with tempfile.TemporaryDirectory(dir=OUT) as td:
            root=Path(td)
            os.environ["CONTEST_WORK"]=td
            parent=root/"initial"
            parent.mkdir()
            shutil.copyfile(src/"checkpoint.npz",parent/"checkpoint.npz")
            artifacts=[parent]
            history=[]
            metrics=[]
            for i,action in enumerate(MOVES[task],1):
                rnd=benchmark.PublicRound()
                rnd.root,rnd.dataset_key,rnd.agent,rnd.round_index=root,task,harness,i
                rnd.review_mode="measured"
                rnd.parent,rnd.records,rnd.call_ids=benchmark.Parent(parent),history,[]
                rnd.prompt_bytes=0
                rnd.plan=f"{action}|half|replay the first recorded release"
                rnd.artifacts=[{"artifact_id":f.name} for f in artifacts]
                rnd.selection=x[sel],y[sel]
                rnd.data=benchmark.SimpleNamespace(training_rows=lambda:iter(train),validation_rows=lambda:(valid[j] for j in sel))
                harness.train(rnd)
                rnd.record["id"]=f"{task}-{i}"
                history.append(rnd.record)
                artifacts.append(rnd.candidate)
                parent=root/rnd.record["selected_artifact_id"]
                model,head,_=harness.artifact(parent)
                pred=harness.values(task,model,head,x[test])
                metric=harness.metric(task,y[test],pred)
                outcome=dict(round=i,metric=metric,gain=harness.gain(task,static,metric)/max(1e-9,static),
                             deployed=parent.name)
                if task=="C":
                    err=pred[:,0]-y[test]
                    outcome["nasa_per_row"]=float(np.expm1(np.maximum(err/10.,-err/13.)).mean())
                metrics.append(outcome)
        results.append(dict(domain=int(g)+1,train=len(train),selection=len(sel),audit=int(test.sum()),static=static,
                            mean_gain=float(np.mean([r["gain"] for r in metrics])),rounds=metrics))
    (OUT/f"{version}-{task}.json").write_text(json.dumps(results,indent=2))

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("task",choices=list("ABC"))
    parser.add_argument("version",choices=("v5","v6"))
    args=parser.parse_args()
    run(args.task,args.version)
