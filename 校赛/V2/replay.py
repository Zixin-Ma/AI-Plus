"""Paired ten-round public replay and optional three-call PyTRIO smoke check."""
import argparse
import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.extend([str(ROOT), str(ROOT / "app")])
import contest_submission as v2
import control
from industrial import adjusted
from baseline_model import Model
from prepare import WORK
from contest_harness.data import iter_jsonl


def load_v1():
    spec = importlib.util.spec_from_file_location("v1_harness", ROOT / "app/contest_submission.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def schedules(groups, mode):
    if mode == "fixed":
        return [(np.ones(len(groups), bool), np.ones(len(groups), bool), np.ones(len(groups), bool))] * 10
    # Hold every fifth observation within each public group strictly outside selection
    # for this replay. Rotate the remaining chronological quarters to simulate releases.
    audit = np.zeros(len(groups), bool)
    phase = np.empty(len(groups), int)
    for g in np.unique(groups):
        idx = np.flatnonzero(groups == g)
        audit[idx[::5]] = True
        for p, chunk in enumerate(np.array_split(idx, 4)):
            phase[chunk] = p
    return [(~audit & (phase == p), audit, audit & (phase == p)) for p in (0,0,1,1,2,2,3,3,0,1)]


def consult(msg):
    import requests
    import tomllib
    key = tomllib.loads((Path.home()/".pytrio/config.toml").read_text())["api_key"]
    response = requests.post("https://pytrio.com/api/openai/v1/chat/completions",
        headers={"Authorization":"Bearer "+key},json=dict(model=v2.MODEL, messages=msg,
        max_tokens=20, temperature=0, enable_thinking=False), timeout=(15,90), allow_redirects=False)
    response.raise_for_status()
    body = response.json()
    return body["choices"][0]["message"]["content"], body["usage"]


def run(key, modes, live, out):
    start = WORK / key
    base = Model.load(start / "checkpoint.npz")
    train = v2.projection(iter_jsonl(start/"public/train.jsonl"),base,key)
    valid = v2.projection(iter_jsonl(start/"public/validation.jsonl"),base,key,True)
    xv,yv,groups = valid
    v1 = load_v1()
    for mode in modes:
        for version in (("v2",) if live else ("v1","v2")):
            result=out/f"{version}-{key}-{mode}.json"
            if live and result.exists():
                raise FileExistsError(f"Live receipt already exists: {result}")
            model, old, history = copy.deepcopy(base), {}, []
            for index,(selection,audit,stage) in enumerate(schedules(groups, mode),1):
                if live and index > 1: break
                public = tuple(arr[selection] for arr in valid)
                next_model=copy.deepcopy(model)
                begin = time.process_time()
                text, usage = "", {}
                if version == "v1":
                    with tempfile.TemporaryDirectory(dir=out) as tmp:
                        name,scores,conf = v1.prepare(key,index,next_model,old,train[:2],public[:2],Path(tmp))
                        headpath = Path(tmp)/"residual.npz"
                        head = v2.load_head(headpath if headpath.exists() else None)
                    msg = v1.prompt(key,index,conf,scores[name],history)
                    take = v1.improvement(key,scores["parent"],scores[name]) > 1e-8
                else:
                    head,conf = control.experiment(key,index,next_model,old,train,public)
                    name=conf["method"]
                    with tempfile.TemporaryDirectory(dir=out) as tmp:
                        next_model.save(Path(tmp)/"checkpoint.npz")
                        if head:
                            np.savez_compressed(Path(tmp)/"head.npz", **head)
                    score=conf["trials"][name]
                    msg=control.message(key,index,conf,score,history)
                    if live:
                        text,usage=consult(msg)
                    take=control.choose(key,conf,score,text if live else "DEPLOY replay")
                cpu=time.process_time()-begin
                if take:
                    model,old=next_model,head
                begin=time.process_time()
                pred=(np.maximum(adjusted(model,old,key,xv[audit]),0) if version=="v1" and key!="A"
                      else adjusted(model,old,key,xv[audit]) if version=="v1" else v2.values(key,model,old,xv[audit]))
                met=v2.measures(key,yv[audit],pred,groups[audit])
                stage_in_audit = stage[audit]
                stage_score = v2.measures(key, yv[stage], pred[stage_in_audit], groups[stage])
                start_score = v2.measures(key, yv[stage], v2.values(key,base,{},xv[stage]),groups[stage])
                relative = v2.gain(key,start_score["primary"],stage_score["primary"]) / max(1e-12,start_score["primary"])
                row=dict(stage=stage_score, stage_static=start_score, relative_stage_gain=relative,id=f"{version}-{key}-{index}", round=index, method=name, accepted=take,
                         reason=text or ("deployed" if take else "kept parent"), config=conf, audit=met,
                         cpu_s=cpu, audit_cpu_s=time.process_time()-begin,
                         prompt_bytes=len(msg[0]["content"].encode()), usage=usage)
                history.append(row)
            result.write_text(json.dumps(history,indent=2,ensure_ascii=False))
            print(result.name, history[-1]["audit"], sum(r["cpu_s"] for r in history),flush=True)


if __name__=="__main__":
    arg=argparse.ArgumentParser()
    arg.add_argument("--live",action="store_true")
    arg.add_argument("--keys",default="ABC")
    arg.add_argument("--modes",nargs="+",choices=["fixed","rotating"],default=["fixed","rotating"])
    args=arg.parse_args()
    output=WORK/"v2"/("live" if args.live else "replay")
    output.mkdir(parents=True,exist_ok=True)
    for key in dict.fromkeys(args.keys):
        run(key,["fixed"] if args.live else args.modes,args.live,output)
