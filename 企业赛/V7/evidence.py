"""Verify the release signature and summarize frozen, hardware, and cloud comparisons."""
import json
import statistics as st
from pathlib import Path
import qualify

OUT=qualify.OUT
def read(path):
    return json.loads(path.read_text())
def main():
    assert qualify.manifest()==read(OUT/"freeze.json")
    audit=read(OUT/"audit.json")
    assert audit["pairs"]==294 and audit["observations"]==5880
    deps=read(OUT/"platform-reference.json")
    mask_checks={}
    for k in "ABC":
        def signature(i):
            data=read(qualify.path_for("schedule","v6",k,i))["rounds"]
            return tuple(sorted(set((d["stage"],d["audit_ids"])for d in data)))
        dev={signature(i) for i in (0,5,8,13,16,19)}
        confirmation={signature(i) for i in range(101,109)}
        assert len(confirmation)==8 and not (dev & confirmation)
        mask_checks[k]={"distinct_confirmation":len(confirmation),"matches_six_dev":len(dev&confirmation)}
    folds={}
    for k in "ABC":
        folds[k]={}
        for v in ("v6","v7"):
            ds=read(OUT/"opening-history"/f"{v}-{k}.json")
            folds[k][v]=dict(floor=min(z["mean_gain"]for z in ds),mean=st.mean(z["mean_gain"]for z in ds),
                              groups=len(ds),rounds=len(ds[0]["rounds"]))
    cloud=OUT/"cloud"
    true_runs={}
    for m,i in (("schedule",17),("official",13)):
        for k in "ABC":
            s={}
            for v in ("v6","v7"):
                d=read(cloud/"streams"/f"{m}-{v}-{k}-{i}.json")
                receipts=read(cloud/"streams"/f"receipts-{m}-{v}-{k}-{i}.json")
                assert len(d["rounds"])==10
                s[v]={field:d[field]for field in ("stage_floor","stage_mean","forget","delay")}
                s[v].update(calls=len(receipts),tokens=sum(q["tokens"]for q in receipts))
                if k=="C":s[v]["nasa"]=st.mean(r["nasa_mean"]for r in d["rounds"])
            true_runs[m+"/"+k]=s
    reserved={}
    for k in "ABC":
        versions={}
        for v in ("v6","v7"):
            cases=[]
            for i in (101,104,107):
                file=cloud/"streams"/f"schedule-{v}-{k}-{i}.json"
                trace=read(file)
                matched=read(qualify.path_for("schedule",v,k,i))
                assert len(trace["rounds"])==10
                assert all(all(a[key]==b[key] for key in ("stage","train_ids","selection_ids","audit_ids","static")) for a,b in zip(trace["rounds"],matched["rounds"],strict=True))
                saved=read(file.with_name("receipts-"+file.name))
                cases.append(dict(split=i,**{key:trace[key] for key in ("stage_floor","stage_mean","forget","delay")},tokens=sum(d["tokens"] for d in saved),calls=len(saved)))
            versions[v]={key:st.mean(d[key] for d in cases) for key in ("stage_floor","stage_mean","forget","delay","calls","tokens")}
            versions[v]["parts"]=cases
        reserved[k]=versions
    targeted={}
    for i in (103,108):
        targeted[i]={}
        for controller in ("fixed","qwen"):
            for v in ("v6","v7"):
                fixed=read(qualify.path_for("schedule",v,"C",i))["rounds"]
                rounds=fixed if controller=="fixed" else read(cloud/"streams"/f"schedule-{v}-C-{i}.json")["rounds"]
                assert all(r["audit_ids"]==t["audit_ids"] for r,t in zip(rounds,fixed,strict=True))
                targeted[i][controller+"_"+v]=dict(nasa=st.mean(r["nasa_mean"] for r in rounds),
                    fd004=st.mean(r["nasa_mean"] for r in rounds if r["stage"]==3))
    prompts={}
    for file in ("prompt-study.json","prompt-check.json"):
        rows=read(cloud/file); group={}
        for style in dict.fromkeys(d["style"] for d in rows):
            review=[d for d in rows if d["style"]==style and d["stage"]=="review"]
            group[style]=dict(n=len(review),valid=sum(d["valid"]for d in review),allowed=sum(d["allowed"]for d in review),tokens=sum(d["usage"]["total_tokens"]for d in review))
        prompts[file]=group
    receipts=[read(f) for f in (cloud/"receipts").glob("*.json")]
    result=dict(mask_checks=mask_checks,opening=folds,cloud_runs=true_runs,cloud_reserved=reserved,targeted_nasa=targeted,prompts=prompts,
                unique_requests=len(receipts),tokens=sum(d["usage"]["total_tokens"]for d in receipts),
                prompt_tokens=sum(d["usage"]["prompt_tokens"]for d in receipts),completion_tokens=sum(d["usage"]["completion_tokens"]for d in receipts),
                platform_checked=deps["checked"])
    (OUT/"evidence.json").write_text(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=="__main__":main()
