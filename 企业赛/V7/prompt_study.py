"""Matched replay of historical public planning and review inputs on real PyTRiO."""
import argparse
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import policy
from cloud_audit import OUT,invoke

def compact_plan(key,index,d,has_head):
    h=d["history"]; parent=d["parent"]
    evidence={"driftSD":d["drift_train_sd"],"parent":parent,"wins/tries":h["positive_primary_over_attempts"],"last":h["last"]}
    options="refit,adam,continue" if has_head else "refit,adam"
    first="" if has_head else " Cold start: favor refit."
    text=(f"{key} r{index}, {policy.METRIC[key]}. refit=train task head; adam=backbone; continue=reuse head.{first} "
          f"Allowed {options}. Output exactly one line: action|half or full|reason (1-5 words). No other text. "
          +policy._pack(evidence))
    return [{"role":"user","content":text}]

def compact_review(key,plan,scores):
    p=scores["parent"];sign=1 if key=="A" else -1
    rows=[]
    for n,d in scores.items():
        if n=="parent":continue
        rows.append([n,int(d["allowed"]),policy._rounded(sign*100*(d["primary"]-p["primary"])/max(1e-9,p["primary"])),policy._rounded(sign*(d["worst"]-p["worst"])),policy._rounded(100*(d["floor"]-p["floor"]))])
    names=", ".join(["parent"]+[r[0] for r in rows])
    return [{"role":"user","content":f"{key}. Choose one of {names}. Rows [model,allowed,meanGain%,weakGain,floorGainPt]. Parent baseline. Prefer an allowed gain. One line: choice|reason (1-5 words). "+policy._pack(rows)}]

def admissible_review(key,plan,scores):
    p=scores["parent"]
    sign=1 if key=="A" else -1
    opts=["parent"]
    rows=[]
    for name,d in scores.items():
        if name=="parent" or not d["allowed"]:
            continue
        opts.append(name)
        rows.append([name,policy._rounded(sign*100*(d["primary"]-p["primary"])/max(1e-9,p["primary"])),
                     policy._rounded(sign*(d["worst"]-p["worst"])),policy._rounded(100*(d["floor"]-p["floor"]))])
    content=f"{key}. Choose one of {', '.join(opts)}. "
    if rows:
        content+="Compared with parent [model,meanGain%,weakGain,floorGainPt]: "+policy._pack(rows)+". Prefer measured gain. "
    else:
        content+="All proposed alternatives failed the measured deployment guard. "
    return [{"role":"user","content":content+"Return choice|reason (1-5 words)."}]

def explicit_review(key,plan,scores):
    parent=scores["parent"]
    direction=1 if key=="A" else -1
    options=["parent"]
    rows=[["parent",0,0,0]]
    for name,entry in scores.items():
        if name!="parent" and entry["allowed"]:
            options.append(name)
            rows.append([name,policy._rounded(100*direction*(entry["primary"]-parent["primary"])/max(1e-9,parent["primary"])),
                         policy._rounded(direction*(entry["worst"]-parent["worst"])),policy._rounded(100*(entry["floor"]-parent["floor"]))])
    if len(options)==1:
        text=f"{key}. Choose one of parent. Parent is approved and available. Return exactly: parent|retain validated model"
    else:
        text=(f"{key}. Choose one of {', '.join(options)}. All listed models passed. "
              f"Rows [option,meanGain%,weakGain,floorGainPt], higher favors promotion: {policy._pack(rows)}. "
              "Reply option|short reason.")
    return [{"role":"user","content":text}]

def main():
    from importlib.util import module_from_spec, spec_from_file_location
    spec=spec_from_file_location("v6policy",Path(__file__).resolve().parent.parent/"V6/policy.py")
    old=module_from_spec(spec);spec.loader.exec_module(old)
    d=json.loads((OUT.parent.parent/"v5/formal/formal-report.json").read_text())["report"]["workspace_report"]["datasets"]
    ap=argparse.ArgumentParser()
    ap.add_argument("--check",action="store_true")
    args=ap.parse_args()
    jobs=[]
    for task in d:
        k=task["key"]
        for idx in ((2,4,5,6,8,9) if args.check else (1,3,7,10)):
            cfg=task["rounds"][idx-1]["configuration"]
            for style in (("v6","explicit") if args.check else ("v6","compact","admissible","explicit")):
                for stage in ("plan","review"):
                    if stage=="plan" and (args.check or style in ("admissible","explicit")):
                        continue
                    jobs.append((k,idx,style,stage,cfg))
    def execute(task):
        k,i,style,stage,cfg=task
        if stage=="plan":
            fn=old.plan_request if style=="v6" else compact_plan
            msgs=fn(k,i,cfg["diagnosis"],i>1)
        else:
            fn={"v6":old.final_request,"compact":compact_review,"admissible":admissible_review,"explicit":explicit_review}[style]
            msgs=fn(k,cfg["plan"],cfg["scores"])
        receipt=invoke(msgs,temperature=0,enable_thinking=False,max_tokens=56 if stage=="plan" else 32)
        reply=receipt["reply"]
        if stage=="plan":
            parsed=old.read_plan(reply,i>1)
            valid=not parsed["fallback"]; decision=parsed["tool"];allowed=valid
        else:
            decision,reason=old.read_final(reply)
            valid=not reason.startswith("Invalid review")
            allowed=valid and decision in cfg["scores"] and (decision=="parent" or cfg["scores"][decision]["allowed"])
        return dict(task=k,round=i,style=style,stage=stage,valid=valid,allowed=allowed,decision=decision,
                    receipt=receipt["request_sha256"],usage=receipt["usage"])
    with ThreadPoolExecutor(3) as pool:
        results=list(pool.map(execute,jobs))
    (OUT/("prompt-check.json" if args.check else "prompt-study.json")).write_text(json.dumps(results,indent=2))

if __name__=="__main__":
    main()
