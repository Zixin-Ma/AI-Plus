"""Ten-round server replay with actual, cached PyTRiO controller replies."""
import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace
import benchmark
from cloud_audit import OUT,invoke

def run(task,version,setting,split):
    stream=OUT/"streams"
    stream.mkdir(parents=True,exist_ok=True)
    benchmark.OUT=stream
    receipts=[]
    def model(rnd,messages,**opts):
        r=invoke(messages,**opts)
        rnd.prompt_bytes+=sum(len(m["content"].encode()) for m in messages)
        rnd.call_ids.append(r["id"])
        receipts.append(dict(round=rnd.round_index,kind="review" if "Choose one of" in messages[0]["content"] else "plan",
                             sha256=r["request_sha256"],tokens=r["usage"].get("total_tokens",0)))
        return SimpleNamespace(text=r["reply"])
    benchmark.PublicRound.model=model
    benchmark.run(task,version,split,setting)
    (stream/f"receipts-{setting}-{version}-{task}-{split}.json").write_text(json.dumps(receipts,indent=2))

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("task",choices=list("ABC"))
    ap.add_argument("version",choices=("v5","v7","v8"))
    ap.add_argument("--split",type=int,choices=(201,203,207),default=201)
    args=ap.parse_args()
    if args.split!=201 and args.task!="B":
        ap.error("Supplementary controller diagnostics are defined for bearings")
    import qualify
    if qualify.manifest()!=json.loads((qualify.OUT/"freeze.json").read_text()):
        raise ValueError("Check the frozen manifest before running the live controller")
    run(args.task,args.version,"anchored",args.split)
