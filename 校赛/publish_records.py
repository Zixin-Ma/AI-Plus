"""Publish completed public development records to SwanLab. Do not run in the contest image."""
import argparse
import json
from pathlib import Path

import swanlab
from prepare import WORK

PROJECT = "ai-plus-campus-rsi"


def publish(experiment):
    settings = swanlab.Settings(terminal={"proxy_type":"none"}, probe={
        "hardware":False, "runtime":False, "requirements":False, "conda":False, "git":False,
        "swanlab":False, "monitor":False})
    for key in "ABC":
        records = [json.loads(f.read_text()) for f in sorted((WORK / experiment / key).glob("*/record.json"))]
        if not records:
            continue
        run = swanlab.init(project=PROJECT, public=True, name=f"{experiment}-{key}",
                           mode="online", on_online_error="raise", settings=settings,
                           log_dir=str(WORK/"swanlog"), config={"split":"organizer public validation",
                           "dataset":key, "rounds":len(records), "driver":records[0]["mode"]})
        for r in records:
            cfg = r["config"]
            logs = {f"trial/{k}":v for k,v in cfg["public_ablation"].items()}
            logs.update({"validation/selected":r["selected_score"],"decision/accept":int(r["accepted"]),
                         "training/steps":cfg["steps"],"resource/seconds":r["elapsed_s"],
                         "resource/prompt_tokens":r["usage"].get("prompt_tokens",0),
                         "resource/output_tokens":r["usage"].get("completion_tokens",0),
                         "audit/record":swanlab.Text(json.dumps(r,ensure_ascii=False))})
            run.log(logs,step=r["round"])
        run.finish()
        print(key,len(records),flush=True)


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("experiment")
    publish(parser.parse_args().experiment)
