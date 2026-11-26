"""Paired latency on immutable V3/V4 artifacts. Measures JSON input, inference and JSON output."""
import argparse
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import tempfile
from time import perf_counter_ns
from types import SimpleNamespace

import numpy as np
import benchmark

HERE = Path(__file__).resolve().parent
WORK = HERE.parent / "work/v4"
ASSETS = WORK / "serving-assets"
RESULT = WORK / "serving"


def runtime(version):
    if version == "v3":
        sys.path.insert(0,str(HERE.parent/"V3"))
    import contest_submission
    return contest_submission


def rows(key, split):
    with (benchmark.WORK/key/"public"/(split+".jsonl")).open() as f:
        return [json.loads(line) for line in f]


def make_artifact(key, version):
    mod = runtime(version)
    data = {s:rows(key,s) for s in ("train","validation")}
    for group in data.values():
        for item in group:
            item.pop("meta",None)
    dest = ASSETS/version/key
    dest.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(dir=WORK) as tmp:
        r = benchmark.PublicRound()
        r.agent,r.root,r.dataset_key,r.round_index = mod,Path(tmp),key,1
        r.parent,r.records,r.call_ids,r.prompt_bytes = benchmark.Parent(benchmark.WORK/key),[],[],0
        r.artifacts=[]
        r.plan="refit|half|compare the expert"
        r.selection=mod.projection(data["validation"],mod.Model.load(r.parent.path("checkpoint.npz")),key)[:2]
        r.data=SimpleNamespace(training_rows=lambda:iter(data["train"]),validation_rows=lambda:iter(data["validation"]))
        os.environ["CONTEST_WORK"]=tmp
        mod.train(r)
        choice=r.record["configuration"]["deployed"]
        src=r.candidate if choice=="candidate" else r.parent.folder
        for name in ("checkpoint.npz","head.npz","reference.npz"):
            if (src/name).is_file():
                shutil.copyfile(src/name,dest/name)
            elif (dest/name).exists():
                (dest/name).unlink()
        (dest/"record.json").write_text(json.dumps({"choice":choice,"family":r.record["configuration"]["head_family"],
                                                  "weight":r.record["configuration"]["expert_weight"]}))


def standard_one(h, key, folder, sample):
    model,head,read=h.artifact(folder)
    x=np.asarray([read(sample.inputs)],dtype=float)
    v=h.values(key,model,head,x)[0]
    if key!="A":
        return {"value":float(v[0])}
    p=np.exp(v-v.max()) if head else v.copy()
    p/=p.sum()
    i=int(p.argmax())
    if np.count_nonzero(p==p[i])>1:
        p[i]=np.nextafter(p[i],1.)
    return [{"label":label,"score":float(z)} for label,z in zip(model.meta["classes"],p)]


def measure(key, source, version, repeat):
    h=runtime(version)
    folder=ASSETS/source/key
    data=rows(key,"validation")
    task="classification" if key=="A" else "regression"
    chosen=np.random.default_rng(91413+repeat).choice(len(data),min(600,len(data)),replace=False)
    lines=[json.dumps(dict(task=task,dataset_key=key,inputs=data[i]["inputs"]),separators=(",",":")) for i in chosen]
    RESULT.mkdir(exist_ok=True)
    modes=["legacy"] if version=="v3" else ["reference","dispatch","compiled"]
    modes=modes[repeat%len(modes):]+modes[:repeat%len(modes)]
    result={}
    for mode in modes:
        h.artifact.cache_clear()
        if version=="v4":
            h.prepared_artifact.cache_clear()
        start=perf_counter_ns()
        if mode=="dispatch":
            import serving
            model,head,read=h.artifact(folder)
            bound=serving.prepare(key,model,head,use_expert_hook=False)
            predict=lambda s:bound(read(s.inputs))
        elif mode=="reference":
            predict=lambda s:standard_one(h,key,folder,s)
        else:
            predict=lambda s:h.predict(s,folder)
        def call(line):
            return json.dumps(predict(SimpleNamespace(**json.loads(line))),allow_nan=False)
        call(lines[0])
        cold=(perf_counter_ns()-start)/1000
        for line in lines[:32]:
            call(line)
        durations=[]
        for line in lines:
            tic=perf_counter_ns()
            call(line)
            durations.append((perf_counter_ns()-tic)/1000)
        result[mode]=dict(cold_us=cold,p50_us=float(np.median(durations)),p95_us=float(np.quantile(durations,.95)),
                          p99_us=float(np.quantile(durations,.99)),mean_us=statistics.mean(durations),n=len(durations))
    if version=="v4" and repeat==0:
        model,head,_=h.artifact(folder)
        x,_,_=h.projection(data,model,key)
        score=h.values(key,model,head,x)
        if key=="A":
            expected=np.exp(score-score.max(1,keepdims=True)) if head else score.copy()
            expected/=expected.sum(1,keepdims=True)
        else:
            expected=score.reshape(-1)
        predictions=[]
        for row in data:
            out=h.predict(SimpleNamespace(task=task,dataset_key=key,inputs=row["inputs"]),folder)
            predictions.append([p["score"] for p in out] if key=="A" else out["value"])
        actual=np.asarray(predictions)
        error=float(np.max(np.abs(actual-expected)))
        if not np.allclose(actual,expected,rtol=1e-9,atol=1e-8):
            raise ValueError(f"Reference discrepancy {key} {source}: {error}")
        if key=="A" and not np.array_equal(actual.argmax(1),expected.argmax(1)):
            raise ValueError("Classification choice changed")
        result["equivalence"]={"rows":len(data),"max_abs":error}
    (RESULT/f"{key}-{source}-{version}-{repeat}.json").write_text(json.dumps(result,indent=2))


def suite(key):
    cpus=sorted(os.sched_getaffinity(0))
    os.sched_setaffinity(0,{cpus[("ABC".index(key)+4)%len(cpus)]})
    def run(*args):
        subprocess.run([sys.executable,"-B",__file__,*map(str,args)],check=True)
    for version in ("v3","v4"):
        run("prepare",key,version)
    for repeat in range(5):
        pairs=[("v3","v3"),("v3","v4"),("v4","v4")]
        if repeat%2:
            pairs.reverse()
        for src,version in pairs:
            run("measure",key,src,version,repeat)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("action",choices=["suite","prepare","measure"])
    parser.add_argument("task",choices=list("ABC"))
    parser.add_argument("source",nargs="?",choices=["v3","v4"])
    parser.add_argument("version",nargs="?",choices=["v3","v4"])
    parser.add_argument("repeat",nargs="?",type=int)
    args=parser.parse_args()
    if args.action=="suite":
        suite(args.task)
    elif args.action=="prepare":
        make_artifact(args.task,args.source)
    else:
        measure(args.task,args.source,args.version,args.repeat)
