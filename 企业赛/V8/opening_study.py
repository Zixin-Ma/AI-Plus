"""First deployment on an entire public domain excluded from model selection."""
import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import benchmark

HERE = Path(__file__).resolve().parent
OUT = HERE.parent/"work/v8/opening"

def domain_ids(rows,key):
    if key == "A":
        groups = np.empty(len(rows),dtype=int)
        labels = np.array([r["answer"] for r in rows])
        for label in np.unique(labels):
            ids = np.flatnonzero(labels == label)
            ids = sorted(ids,key=lambda j:rows[j]["meta"]["source_index"])
            for group,block in enumerate(np.array_split(ids,8)):
                groups[block] = group
        return groups
    if key == "B":
        return np.array([int(r["meta"]["bearing"].split("_")[0][-1])-1 for r in rows])
    return np.array([int(r["meta"]["source_subset"][-1])-1 for r in rows])

def run(key,version):
    if version != "v8":
        sys.path.insert(0,str(HERE.parent/version.upper()))
    import contest_submission as agent
    src = benchmark.WORK/key
    def read(split):
        with (src/"public"/(split+".jsonl")).open() as stream:
            return [json.loads(line) for line in stream]
    train, valid = read("train"), read("validation")
    groups = domain_ids(valid,key)
    for row in train+valid:
        row.pop("meta",None)
    static = agent.Model.load(src/"checkpoint.npz")
    x,y,_ = agent.projection(valid,static,key)
    ref = agent.values(key,static,{},x)
    result = []
    OUT.mkdir(exist_ok=True,parents=True)
    for group in np.unique(groups):
        test = groups == group
        selected = np.flatnonzero(~test)
        with tempfile.TemporaryDirectory(dir=OUT) as tmp:
            rnd=benchmark.PublicRound()
            rnd.agent,rnd.root,rnd.dataset_key,rnd.round_index=agent,Path(tmp),key,1
            rnd.review_mode="controlled"
            rnd.parent,rnd.records,rnd.call_ids=benchmark.Parent(src),[],[]
            rnd.artifacts=[]
            rnd.prompt_bytes=0
            rnd.plan="refit|half|fit the first visible release"
            rnd.selection=x[selected],y[selected]
            rnd.data=benchmark.SimpleNamespace(training_rows=lambda:iter(train),validation_rows=lambda:(valid[i] for i in selected))
            os.environ["CONTEST_WORK"]=tmp
            agent.train(rnd)
            source=rnd.candidate if rnd.record["configuration"]["deployed"]=="candidate" else src
            model,head,_=agent.artifact(source)
            pred=agent.values(key,model,head,x[test])
            metric=agent.metric(key,y[test],pred)
            initial=agent.metric(key,y[test],ref[test])
            item=dict(domain=int(group)+1,rows=int(test.sum()),selection=len(selected),static=initial,metric=metric,
                      gain=agent.gain(key,initial,metric)/max(initial,1e-9),deployment=rnd.record["configuration"]["deployed"])
            if key=="C":
                e=pred[:,0]-y[test]
                item["nasa"]=float(np.expm1(np.maximum(e/10.,-e/13.)).mean())
            result.append(item)
    (OUT/f"{version}-{key}.json").write_text(json.dumps(result,indent=2))
if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("task",choices=list("ABC"))
    p.add_argument("version",choices=["v5","v7","v8"])
    a=p.parse_args()
    run(a.task,a.version)
