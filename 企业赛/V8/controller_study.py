"""Compare two cold-start planner formats on previously published platform diagnoses."""
import json
from importlib.util import spec_from_file_location, module_from_spec
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import policy
from cloud_audit import invoke, OUT

def cold_prompt(key, index, diagnosis):
    evidence = {"shift_SD": diagnosis["drift_train_sd"],
                "primary": diagnosis["parent"]["primary"], "weak": diagnosis["parent"]["worst"]}
    message = (f"{key} round {index}, {policy.METRIC[key]}. Cold start. Choose refit (fit expert) "
               "or adam (update backbone). Prefer refit unless shift favors adam. "
               "One line: action|half or full|reason in 2-5 words. "
               + policy._pack(evidence))
    return [{"role": "user", "content": message}]

def main():
    work = OUT.parent.parent
    spec = spec_from_file_location("v7_policy", Path(__file__).resolve().parent.parent / "V7/policy.py")
    prior = module_from_spec(spec)
    spec.loader.exec_module(prior)
    jobs = []
    for version, path in (("v5",work/"v5/formal/formal-report.json"),("v7",work/"v7/platform-trial/report.json")):
        tasks = json.loads(path.read_text())["report"]["workspace_report"]["datasets"]
        for task in tasks:
            for style in ("existing","short"):
                jobs.append((version,task["key"],task["rounds"][0]["configuration"]["diagnosis"],style))
    def run(case):
        v,k,d,style = case
        msg = prior.plan_request(k,1,d,False) if style=="existing" else cold_prompt(k,1,d)
        response=invoke(msg,temperature=0,enable_thinking=False,max_tokens=56)
        decision=policy.read_plan(response["reply"],False)
        return dict(source=v,task=k,style=style,decision=decision,id=response["id"],usage=response["usage"],
                    receipt=response["request_sha256"])
    with ThreadPoolExecutor(3) as pool:
        rows=list(pool.map(run,jobs))
    (OUT/"cold-plan-comparison.json").write_text(json.dumps(rows,indent=2))
    for r in rows:
        print(r["source"],r["task"],r["style"],r["decision"]["tool"],r["decision"]["fallback"],r["usage"]["total_tokens"])

if __name__=="__main__":
    main()
