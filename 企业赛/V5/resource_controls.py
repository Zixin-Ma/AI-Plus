"""Isolate regime partitioning and context/anchor serialization on V4 public artifacts."""
import importlib.util
import json
import os
import shutil
from pathlib import Path
from statistics import median
from tempfile import TemporaryDirectory
from time import process_time_ns

import numpy as np
import contest_submission as harness
import guard

HERE=Path(__file__).resolve().parent
WORK=HERE.parent/'work'
PUBLIC=HERE.parents[1]/'校赛/work'
spec=importlib.util.spec_from_file_location('old_guard', HERE.parent/'V4/guard.py')
old_guard=importlib.util.module_from_spec(spec)
spec.loader.exec_module(old_guard)

def paired(left,right,repeat):
    runs=[[],[]]
    for i in range(repeat):
        for k in ((0,1) if i%2==0 else (1,0)):
            start=process_time_ns()
            (left,right)[k]()
            runs[k].append((process_time_ns()-start)/1e6)
    return list(map(median,runs))

def main():
    cpus=sorted(os.sched_getaffinity(0))
    os.sched_setaffinity(0,{cpus[-1]})
    model=harness.Model.load(PUBLIC/'C/checkpoint.npz')
    def read(name):
        with (PUBLIC/'C/public'/name).open() as f:
            return harness.projection((json.loads(line) for line in f),model,'C')[:2]
    x,_=read('train.jsonl')
    y,labels=read('validation.jsonl')
    def partition(module,train=x):
        return module.strata('C',model,train,y,labels)
    cases={}
    renamed={}
    for name,train in [('full',x),('decimated',x[::4]),('short',x[:512])]:
        old=partition(old_guard,train)
        new=partition(guard,train)
        canonical=np.array([s.replace("settings--0/","settings-0/").replace("/-0","/0") for s in old])
        renamed[name]=int(np.count_nonzero(old!=new))
        cases[name]=int(np.count_nonzero(canonical!=new))
        assert cases[name]==0
    ms=paired(lambda:partition(old_guard),lambda:partition(guard),13)
    record=json.loads((WORK/'v4/platform/trial-report.json').read_text())['report']['workspace_report']
    encoded=[0,0]
    maxdiff=0.
    for ds in record['datasets']:
        for row in ds['rounds']:
            original=row['configuration']['context_profile']
            packed=np.rint(np.array(original)*1000).astype(int).tolist()
            for i,data in enumerate((original,packed)):
                encoded[i]+=len(json.dumps(data,separators=(',',':')).encode())
            if original:
                maxdiff=max(maxdiff,float(np.max(np.abs(np.asarray(packed)/1000-original))))
                cur=np.array(original)
                old={'configuration':{'context_profile':original}}
                new={'configuration':{'context_milli':packed}}
                assert harness.prior_contexts([old],cur,.18)[0][:2]==harness.prior_contexts([new],cur,.18)[0][:2]
    refs={}
    with TemporaryDirectory(dir=WORK/'v5') as temp:
        dest=Path(temp)/'reference.npz'
        for key in 'ABC':
            src=WORK/'v4/serving-assets/v4'/key/'reference.npz'
            with np.load(src,allow_pickle=False) as arrays:
                weights=dict(arrays)
            times=paired(lambda:np.savez_compressed(dest,**weights),lambda:shutil.copyfile(src,dest),31)
            with np.load(dest,allow_pickle=False) as written:
                assert all(np.array_equal(written[k],v) for k,v in weights.items())
            refs[key]=times
    result=dict(group_label_mismatches=cases,labels_renamed=renamed,group_cpu_ms=ms,context_bytes=encoded,context_max_error=maxdiff,reference_cpu_ms=refs)
    (WORK/'v5/resource-controls.json').write_text(json.dumps(result,indent=2))
    print(result)
if __name__=='__main__':
    main()
