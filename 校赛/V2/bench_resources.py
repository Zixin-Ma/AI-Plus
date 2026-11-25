"""Paired first-round process CPU with replayed answers and equal public host measurement."""
import json
import os
from pathlib import Path
from statistics import median
import tempfile
import time
from types import SimpleNamespace

from replay import WORK, load_v1, v2
from agent_replay import Parent
from industrial import adjusted, metric, load_head
from contest_harness.data import iter_jsonl


def run(key, version, folder, validation):
    base=WORK/key
    if version=="v1":
        original=json.loads((WORK/"qwen3"/key/"round-01/record.json").read_text())
        texts=[original["response"]]
    else:
        row=json.loads((WORK/"v2/agent"/f"{key}-fixed.json").read_text())[0]
        texts=[row["planner_response"],row["decision_response"]]
    def measure(folder, **spec):
        base_model=v2.Model.load(folder/"checkpoint.npz")
        name="residual.npz" if version=="v1" else "head.npz"
        aux=load_head(folder/name if (folder/name).exists() else None)
        x,y,_=validation
        scores=(adjusted(base_model,aux,key,x) if version=="v1" else v2.values(key,base_model,aux,x))
        count=sum(f.stat().st_size for f in folder.iterdir())
        rnd.upload_bytes=count
        return {"artifact_id":"candidate","summary":{"metrics":{v2.METRIC[key]:metric(key,y,scores)}}}
    def call(messages,**opts):
        idx=len(rnd.call_ids)
        rnd.call_ids.append(f"replay-{idx}")
        return SimpleNamespace(text=texts[idx])
    rnd=SimpleNamespace(model_name=v2.MODEL, round_index=1,dataset_key=key,parent=Parent(base),artifacts=[],
        data=SimpleNamespace(training_rows=lambda:iter_jsonl(base/"public/train.jsonl"),
                             validation_rows=lambda:iter_jsonl(base/"public/validation.jsonl")),
        history=lambda:[],call_ids=[],max_output_tokens=160,submit_candidate=measure,
        model=call,finish=lambda **record:None)
    os.environ["CONTEST_WORK"]=str(folder)
    harness=load_v1() if version=="v1" else v2
    t=time.process_time()
    harness.train(rnd)
    return time.process_time()-t, rnd.upload_bytes


def main():
    results={}
    for key in "ABC":
        base=WORK/key
        model=v2.Model.load(base/"checkpoint.npz")
        val=v2.projection(iter_jsonl(base/"public/validation.jsonl"),model,key,True)
        runs={"v1":[],"v2":[]}
        for version in ("v1","v2","v2","v1","v1","v2"):
            with tempfile.TemporaryDirectory(dir=WORK/"v2") as tmp:
                runs[version].append(run(key,version,Path(tmp),val))
        results[key]={v:{"cpu_seconds":median(i[0] for i in rows),"uploaded_bytes":rows[0][1],
                         "cpu_trials":[i[0] for i in rows]} for v,rows in runs.items()}
        print(key,results[key],flush=True)
    (WORK/"v2/paired-resources.json").write_text(json.dumps(results,indent=2))


if __name__=="__main__":
    main()
