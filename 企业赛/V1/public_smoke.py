"""One paid, receipt-cached Qwen planning/review pair per task on its official public starter."""
import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import tempfile
import tomllib
from types import SimpleNamespace

import requests
import contest_submission as agent
from benchmark import HERE, WORK, Parent, PublicRound


def run(key):
    out=HERE.parent/"work"/"live"/key
    out.mkdir(exist_ok=True,parents=True)
    keyfile=Path.home()/".pytrio/config.toml"
    def invoke(messages, **options):
        payload=dict(model=agent.MODEL,messages=messages,**options)
        digest=sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()
        path=out/f"call-{len(r.call_ids)+1}.json"
        if path.exists() and json.loads(path.read_text())["digest"] != digest and len(r.call_ids)==1:
            path=out/"review-v2.json"
        if path.exists():
            receipt=json.loads(path.read_text())
            if receipt["digest"]!=digest:
                raise ValueError("Existing paid call has different inputs")
        else:
            token=tomllib.loads(keyfile.read_text())["api_key"]
            response=requests.post("https://pytrio.com/api/openai/v1/chat/completions",
                headers={"Authorization":"Bearer "+token},json=payload,timeout=(15,90),allow_redirects=False)
            response.raise_for_status()
            body=response.json()
            receipt=dict(id=body["id"],text=body["choices"][0]["message"]["content"],
                         usage=body["usage"],digest=digest)
            tmp=path.with_suffix(".tmp")
            tmp.write_text(json.dumps(receipt,ensure_ascii=False,indent=2)); tmp.replace(path)
        r.call_ids.append(receipt["id"])
        return SimpleNamespace(text=receipt["text"])
    def rows(name):
        with (WORK/key/"public"/(name+".jsonl")).open() as stream:
            for line in stream:
                row=json.loads(line)
                yield {"inputs":row["inputs"],"answer":row["answer"]}
    r=PublicRound()
    r.agent,r.dataset_key,r.round_index=agent,key,1
    r.parent,r.records,r.call_ids=Parent(WORK/key),[],[]
    r.artifacts=[]
    r.data=SimpleNamespace(training_rows=lambda:rows("train"),validation_rows=lambda:rows("validation"))
    r.selection=agent.projection(rows("validation"),agent.Model.load(r.parent.path("checkpoint.npz")),key)[:2]
    r.model=invoke
    with tempfile.TemporaryDirectory(dir=out) as t:
        os.environ["CONTEST_WORK"]=t
        r.root=Path(t)
        agent.train(r)
    assert len(set(r.call_ids))==2
    (out/"round.json").write_text(json.dumps(r.record,ensure_ascii=False,indent=2))
    print(key,r.record["configuration"]["plan"],r.record["configuration"]["deployed"])


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("task",choices=list("ABC"));run(p.parse_args().task)
