"""Compare archived official score reports and the two public trial receipts."""
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "work"


def records(path):
    found = []
    for line in path.read_text().splitlines():
        if line.startswith('{"') and '"scores":' in line:
            found.append(json.loads(line))
    if len(found) != 30:
        raise ValueError("Expected 30 expanded official V2 configurations")
    return found


def main():
    configs = records(WORK / "platform/official-v2-full.txt")
    result = {"objective_v1":15.054075, "objective_v2":16.769565,
              "displayed_total_v1":58.1, "displayed_total_v2":57.8, "datasets":{}}
    for key, group in zip("ABC", (configs[:10], configs[10:20], configs[20:])):
        parent = [r["scores"]["parent"] for r in group]
        result["datasets"][key] = {
            "training_rows":[r["train_rows"] for r in group],
            "validation_rows":[r["validation_rows"] for r in group],
            "collapsed_group_rounds":[i+1 for i,r in enumerate(parent) if r.get("worst_group")=="all"],
            "weakest_equals_overall_rounds":[i+1 for i,r in enumerate(parent)
                                             if abs(r["worst"]-r["primary"]) < 1e-9],
            "actions":[dict(round=i+1,tool=r["method"],deployed=r["deployed"])
                       for i,r in enumerate(group)],
        }
    trials = {}
    for ver, file in [("v1",WORK/"platform/trial-report.txt"),
                      ("v2",WORK/"v2/platform-trial-report.txt")]:
        text = file.read_text()
        cpu = [float(v.replace(",","")) for v in re.findall(r"CPU 用时\n([\d,.]+) 秒",text)]
        if len(cpu)!=3:
            raise ValueError("Missing measured CPU")
        tokens = int(re.search(r"合计 Token\n([\d,]+)",text).group(1).replace(",",""))
        trials[ver]={"cpu_s":dict(zip("ABC",cpu)), "total_cpu_s":sum(cpu),"tokens":tokens}
    result["trials"]=trials
    result["cpu_change_percent"]=100*(trials["v2"]["total_cpu_s"]/trials["v1"]["total_cpu_s"]-1)
    result["token_change_percent"]=100*(trials["v2"]["tokens"]/trials["v1"]["tokens"]-1)
    path=WORK/"v2/platform-audit.json"
    path.write_text(json.dumps(result,ensure_ascii=False,indent=2))
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
