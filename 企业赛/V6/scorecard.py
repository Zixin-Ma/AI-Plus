"""Produce a paired scorecard and preserve the official V5 reference used for design."""
import json
from pathlib import Path
import statistics as st

import qualify

OUT = qualify.OUT


def read(path):
    return json.loads(path.read_text())


def write(name, data):
    (OUT/name).write_text(json.dumps(data, ensure_ascii=False, indent=2))


def historical():
    older = OUT.parent/"v5"
    official = read(older/"formal/formal-report.json")
    trials = read(older/"platform/comparison.json")
    results = {}
    for ver in ("V4","V5"):
        trial = trials[ver]
        results[ver] = {"run":trial["run"]["id"], "tasks":{v["key"]:v["metrics"] for v in trial["datasets"]}}
    summary = read(older/"paired-summary.json")
    obj = official["objective_result"]
    result = dict(run=official["run"]["id"], url="https://rsi-index.115.zone/api/v1/runs/"+official["run"]["id"]+"/report",
                  score=obj["objective_score"], components=obj["objective_components"],
                  tasks=obj["datasets"], final_metrics={r["key"]:r["metrics"] for r in official["report"]["workspace_report"]["datasets"]},
                  trials=results, local_v5_reference={k:v for k,v in summary.items() if k.startswith("confirm/")})
    write("official-reference.json",result)


def paired(left,right):
    delta=[y-x for x,y in zip(left,right,strict=True)]
    return dict(v5=st.mean(left),v6=st.mean(right),delta=st.mean(delta),
                increased=sum(d>1e-9 for d in delta),decreased=sum(d< -1e-9 for d in delta),
                tied=sum(abs(d)<=1e-9 for d in delta),delta_range=[min(delta),max(delta)])


def main():
    historical()
    verified = read(OUT/"audit.json")["data"]
    results = {}
    for case in ("schedule","trace","official","drift","fresh","fresh-measured"):
        for key in "ABC":
            audit=[a for a in verified if a["mode"]==case and a["task"]==key and a["part"]=="confirm"]
            runs=[[read(qualify.path_for(case,version,key,a["split"])) for a in audit] for version in ("v5","v6")]
            data={m:paired(*[[r[m] for r in rows] for rows in runs]) for m in ("mean","worst","stage_mean","stage_floor","forget","delay")}
            if key=="C":
                data["nasa"]=paired(*zip(*(a["nasa"] for a in audit),strict=True))
            data["positive_floor"]=[sum(r["stage_floor"]>1e-9 for r in rows) for rows in runs]
            data["stages"]=[paired(*[[r["stage_means"][idx] for r in rows] for rows in runs])
                            for idx in range(len(runs[0][0]["stage_means"]))]
            old_weak, new_weak=[],[]
            weak_ids=[]
            for left,right in zip(*runs,strict=True):
                stage=min(range(len(left["stage_means"])),key=left["stage_means"].__getitem__)
                old_weak.append(left["stage_means"][stage])
                new_weak.append(right["stage_means"][stage])
                weak_ids.append(stage+1)
            data["v5_weakest_stage_followup"]=paired(old_weak,new_weak)
            data["v5_weakest_stage_ids"]=weak_ids
            data["floor_range"]=[[min(r["stage_floor"] for r in rows),max(r["stage_floor"] for r in rows)] for rows in runs]
            results[case+"/"+key]=data
    write("scorecard.json",results)


if __name__=="__main__":
    main()
