"""Ten-round server replay with actual, cached PyTRiO controller replies."""
import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace
import benchmark
from cloud_audit import OUT,invoke

def run(task,version,setting,split,ablation=None):
    stream=OUT/"streams"
    if ablation:
        if version!="v6":
            raise ValueError("The prompt ablation holds V6 industrial heads and training code fixed")
        sys.path.insert(0,str(Path(__file__).resolve().parent.parent/"V6"))
        import policy
        from prompt_study import compact_plan, explicit_review
        policy.final_request=explicit_review
        if ablation=="compact":
            policy.plan_request=compact_plan
        stream=OUT/"ablations"/ablation
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
    ap.add_argument("version",choices=("v6","v7"))
    ap.add_argument("setting",choices=("official","schedule"))
    ap.add_argument("split",type=int,choices=(*range(20),101,103,104,107,108))
    ap.add_argument("--ablation",choices=["review","compact"])
    a=ap.parse_args()
    if a.split in (103,108) and (a.task!="C" or a.setting!="schedule"):
        ap.error("Splits 103/108 are a targeted engine cost diagnostic")
    if a.split>=101:
        import qualify
        if a.setting!="schedule" or a.ablation or qualify.manifest()!=json.loads((qualify.OUT/"freeze.json").read_text()):
            raise ValueError("Reserved cloud schedules require frozen code and public inputs")
    run(a.task,a.version,a.setting,a.split,a.ablation)
