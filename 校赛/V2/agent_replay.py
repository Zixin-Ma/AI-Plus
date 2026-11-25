"""Exercise the production two-call train entry with real PyTRIO and only public rows."""
import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import sys
import time
import tomllib
from types import SimpleNamespace

import numpy as np
import requests
from replay import WORK, schedules
import contest_submission as agent
from contest_harness.data import iter_jsonl


def write_json(path, obj):
    temporary = path.with_suffix(".part")
    temporary.write_text(json.dumps(obj, ensure_ascii=False, indent=2))
    temporary.replace(path)


class Parent:
    def __init__(self, path):
        self.folder = path
        self.artifact_id = path.name

    def path(self, name):
        return self.folder / name

    def declared_paths(self):
        return [p.name for p in self.folder.glob("*.npz")]


class PublicRound:
    model_name = agent.MODEL
    max_output_tokens = 56

    def __init__(self, key, index, parent, data, records, available, workspace, token):
        self.dataset_key, self.round_index = key, index
        self.parent, self.data = Parent(parent), data
        self.records, self.paths, self.folder = records, available, workspace
        self.artifacts = [{"artifact_id": name} for name in available]
        self.call_ids, self.responses = [], []
        self.token = token
        self.model_wall = 0.

    def history(self):
        return self.records

    def model(self, messages, **options):
        if len(self.responses) >= 2 or len(messages[0]["content"].encode()) > 2400:
            raise ValueError("Development call budget exceeded")
        payload = dict(model=self.model_name, messages=messages, **options)
        digest = sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()
        cache = self.folder / f"receipt-{self.round_index}-{len(self.responses)}-{digest}.json"
        begin=time.monotonic()
        # Exact prompt/options, round, dataset and schedule must match before
        # reusing an earlier development receipt. The production SDK always calls live.
        if not cache.exists():
            previous = WORK/"v2/agent_artifacts-multiline"/self.dataset_key/self.folder.name/cache.name
            if previous.is_file():
                shutil.copyfile(previous,cache)
        if cache.exists():
            body=json.loads(cache.read_text())
        else:
            response = requests.post("https://pytrio.com/api/openai/v1/chat/completions",
                headers={"Authorization":"Bearer "+self.token}, json=payload, timeout=(15,90), allow_redirects=False)
            response.raise_for_status()
            raw=response.json()
            body=dict(id=raw["id"],text=raw["choices"][0]["message"]["content"],usage=raw["usage"],
                      input_bytes=len(messages[0]["content"].encode()))
            write_json(cache,body)
        self.model_wall+=time.monotonic()-begin
        self.call_ids.append(body["id"])
        self.responses.append(body)
        return SimpleNamespace(text=body["text"])

    def submit_candidate(self, folder, **spec):
        self.candidate = self.folder / f"round-{self.round_index:02}"
        if self.candidate.exists():
            # Same deterministic round may be resumed after a network error.
            for file in spec["checkpoint_paths"]:
                if (self.candidate/file).read_bytes() != (folder/file).read_bytes():
                    raise ValueError("Resumed artifact differs")
        else:
            shutil.copytree(folder,self.candidate)
        # Independent public measurement through the saved artifact, like host validation.
        model, head, _ = agent.artifact(self.candidate)
        x,y,g = self.validation
        score=agent.measures(self.dataset_key,y,agent.values(self.dataset_key,model,head,x),g)
        return dict(artifact_id=self.candidate.name, summary={"metrics":{agent.METRIC[self.dataset_key]:score["primary"]}})

    def restore_artifact(self, name, dest):
        return shutil.copytree(self.paths[name],dest)

    def finish(self, **record):
        if record["model_call_ids"] != self.call_ids or len(self.responses) != 2:
            raise AssertionError("Missing real planner or reviewer receipt")
        self.record=record


def run(key, mode):
    base = WORK/key
    starter = agent.Model.load(base/"checkpoint.npz")
    public=list(iter_jsonl(base/"public/validation.jsonl"))
    xv,yv,groups=agent.projection(public,starter,key,True)
    out=WORK/"v2/agent"
    out.mkdir(exist_ok=True,parents=True)
    output=out/f"{key}-{mode}.json"
    results=json.loads(output.read_text()) if output.exists() else []
    records=[row["record"] for row in results]
    folder=WORK/"v2/agent_artifacts"/key/mode
    folder.mkdir(parents=True,exist_ok=True)
    paths={base.name:base, **{p.name:p for p in folder.glob("round-*") if p.is_dir()}}
    current=results[-1]["record"]["selected_artifact_id"] if results else base.name
    token=tomllib.loads((Path.home()/".pytrio/config.toml").read_text())["api_key"]
    for index,(sel,audit,stage) in enumerate(schedules(groups,mode),1):
        if index<=len(results) or (mode=="fixed" and index>1):continue
        log=folder/f"step-{index:02}"
        log.mkdir(exist_ok=True)
        os.environ["CONTEST_WORK"]=str(log)
        validrows=[public[j] for j in np.flatnonzero(sel)]
        data=SimpleNamespace(training_rows=lambda:iter_jsonl(base/"public/train.jsonl"),
                             validation_rows=lambda:iter(validrows))
        rnd=PublicRound(key,index,paths[current],data,records,paths,folder,token)
        # Receipts are scoped per round and stage. Store deployable candidate folders together.
        rnd.validation=(xv[sel],yv[sel],groups[sel])
        t=time.process_time()
        agent.train(rnd)
        cpu=time.process_time()-t
        rec=rnd.record
        current=rec["selected_artifact_id"]
        paths[rnd.candidate.name]=rnd.candidate
        model,head,_=agent.artifact(paths[current])
        def assess(base_model,aux,mask):
            return agent.measures(key,yv[mask],agent.values(key,base_model,aux,xv[mask]),groups[mask])
        aud=assess(model,head,audit)
        st=assess(model,head,stage)
        ref=assess(starter,{},stage)
        cfg=rec["configuration"]
        rec["id"]=f"{key}-{mode}-{index}"
        usage={k:sum(int(c["usage"][k]) for c in rnd.responses) for k in ("prompt_tokens","completion_tokens")}
        row=dict(round=index,plan=cfg["plan"],requested=cfg["requested"],deployed=cfg["deployed"],
                 primary=aud["primary"],worst=aud["worst"],stage_relative_gain=agent.gain(key,ref["primary"],st["primary"])/max(ref["primary"],1e-12),
                 cpu_s=cpu,model_wall_s=rnd.model_wall,usage=usage,prompt_bytes=sum(r["input_bytes"] for r in rnd.responses),
                 planner_response=rnd.responses[0]["text"],decision_response=rnd.responses[1]["text"],record=rec)
        records.append(rec)
        results.append(row)
        write_json(output,results)
        print(key,mode,index,cfg["plan"]["tool"],cfg["deployed"],round(aud["primary"],5),usage,flush=True)
        for temp in log.glob("candidate-*"):
            shutil.rmtree(temp)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("task",choices=list("ABC"))
    args=parser.parse_args()
    for mode in ("fixed","rotating"):
        run(args.task,mode)
