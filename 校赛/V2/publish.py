"""Publish the decision traces and public metrics to the team's SwanLab project."""
import json
from pathlib import Path
import sys

import swanlab

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "work/v2"


def main():
    settings = swanlab.Settings(terminal={"proxy_type":"none"}, probe={
        "hardware":False, "runtime":False, "requirements":False,
        "conda":False, "git":False, "swanlab":False, "monitor":False})
    receipts=WORK/"swan-runs.json"
    done=json.loads(receipts.read_text()) if receipts.exists() else {}
    for condition in ("agent-naive","agent"):
        for task in "ABC":
            name=f"v2-{condition}-{task}"
            if name in done:
                continue
            fixed=json.loads((WORK/condition/f"{task}-fixed.json").read_text())
            rotation=json.loads((WORK/condition/f"{task}-rotating.json").read_text())
            if len(fixed)!=1 or len(rotation)!=10:
                raise RuntimeError("Incomplete public records")
            run=swanlab.init(project="ai-plus-campus-rsi",public=True,name=name,
                mode="online",on_online_error="raise",log_dir=str(ROOT/"work/swanlog"),settings=settings,
                config={"dataset":task,"planner":condition,"llm":"Qwen/Qwen3.8-27B",
                        "split":"organizer public only","schedule":"one fixed, ten rotating"})
            for i,row in enumerate(fixed+rotation):
                cfg=row["record"]["configuration"]
                logs={"audit/primary":row["primary"], "audit/worst":row["worst"],
                      "audit/stage_relative_gain":row["stage_relative_gain"],
                      "usage/input":row["usage"]["prompt_tokens"], "usage/output":row["usage"]["completion_tokens"],
                      "action/plan":swanlab.Text(json.dumps(cfg["plan"],ensure_ascii=False)),
                      "action/diagnosis":swanlab.Text(json.dumps(cfg["diagnosis"],ensure_ascii=False)),
                      "action/public_scores":swanlab.Text(json.dumps(cfg["scores"],ensure_ascii=False)),
                      "action/decision":swanlab.Text(row["decision_response"]),
                      "action/deployed":swanlab.Text(cfg["deployed"]),
                      "action/hypothesis":swanlab.Text(row["record"]["hypothesis"]),
                      "training/adam_steps":cfg["steps"], "training/learning_rate":cfg["learning_rate"]}
                run.log(logs,step=i+1)
            url=getattr(run,"url",None)
            run.finish()
            done[name]=str(url or run.id)
            receipts.write_text(json.dumps(done,indent=2))
            print(name,done[name],flush=True)


if __name__=="__main__":
    main()
