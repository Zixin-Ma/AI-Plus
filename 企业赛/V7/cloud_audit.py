"""Run Qwen on released metrics only and cache identifiable responses by exact request."""
import hashlib
import json
import time
import fcntl
import os
from pathlib import Path
import tomllib
import requests

OUT=Path(__file__).resolve().parent.parent/"work/v7/cloud"

def invoke(messages,**options):
    cfg={"model":"Qwen/Qwen3.8-27B","messages":messages,**options}
    digest=hashlib.sha256(json.dumps(cfg,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
    receipts=OUT/"receipts"
    receipts.mkdir(parents=True,exist_ok=True)
    path=receipts/(digest+".json")
    with (receipts/(digest+".lock")).open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if path.exists():
            return json.loads(path.read_text())
        token=tomllib.loads((Path.home()/".pytrio/config.toml").read_text())["api_key"]
        for attempt in range(6):
            try:
                response=requests.post("https://pytrio.com/api/openai/v1/chat/completions",
                    headers={"Authorization":"Bearer "+token},json=cfg,timeout=(12,100),allow_redirects=False)
            except requests.ConnectionError as error:
                # Name lookup and connection establishment fail before the request is sent.
                safe=isinstance(error,requests.ConnectTimeout) or "NameResolutionError" in str(error)
                if not safe or attempt==5:
                    raise
            else:
                if response.status_code!=429 or attempt==5:
                    break
            time.sleep(min(18,3+3*attempt))
        response.raise_for_status()
        body=response.json()
        item=dict(request_sha256=digest,id=body["id"],model=body.get("model",cfg["model"]),
                  reply=body["choices"][0]["message"]["content"],usage=body["usage"])
        tmp=path.with_suffix(".tmp")
        tmp.write_text(json.dumps(item,ensure_ascii=False,indent=2))
        os.replace(tmp,path)
        return item
