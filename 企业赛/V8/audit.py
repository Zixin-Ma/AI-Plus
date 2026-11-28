"""Independently check the release pairing, static-relative scores, and per-stage means."""
import json
import math
import statistics as st
from collections import defaultdict
import qualify

def audit(d,k):
    peaks={}
    visits=defaultdict(list)
    sequence=[]
    gains=[]
    worst_drop=0.
    for i,r in enumerate(d["rounds"],1):
        assert r["round"]==i and r["audit"]>0 and r["selection"]>0
        ref=r["static"]
        pred=r["metrics"]
        assert len(pred)==len(ref)
        all_gain=[(x-y if k!="A" else y-x)/max(x,1e-9) for x,y in zip(ref,pred)]
        for j in peaks:
            worst_drop=max(worst_drop,peaks[j]-all_gain[j])
        stage=r["stage"]
        peaks.setdefault(stage,all_gain[stage])
        for j in peaks:
            peaks[j]=max(peaks[j],all_gain[j])
        assert all(math.isclose(v,w,abs_tol=1e-12) for v,w in zip(all_gain,r["gains"],strict=True))
        sequence.append(stage)
        gains.append(all_gain[stage])
        visits[stage].append(all_gain[stage])
    means=[st.mean(visits[i]) for i in sorted(visits)]
    delays=[]
    for i in range(1,len(sequence)):
        if sequence[i]!=sequence[i-1]:
            j=i
            while j<len(sequence) and sequence[j]==sequence[i] and gains[j]<0:
                j+=1
            delays.append(j-i)
    verify={"stage_mean":st.mean(means),"stage_floor":min(means),"worst":min(gains),"mean":st.mean(gains),"forget":worst_drop,"delay":st.mean(delays)}
    assert all(math.isclose(x,d[k],abs_tol=1e-12) for k,x in verify.items())
    assert len(sequence)==10
    return means

def main():
    pairs=[]
    for part in ("dev","confirm"):
        for key,version,i,mode in qualify.matrix(part):
            if version!="v5":
                continue
            a,b,c=[json.loads(qualify.path_for(mode,v,key,i).read_text()) for v in ("v5","v7","v8")]
            for left,baseline,right in zip(a["rounds"],b["rounds"],c["rounds"],strict=True):
                assert all(left[f]==right[f]==baseline[f] for f in ("stage","train","selection","audit","static","train_ids","selection_ids","audit_ids"))
            stages=[audit(res,key) for res in [a,b,c]]
            item=dict(split=i,part=part,task=key,mode=mode,stage_means=stages)
            if key=="C":
                item["nasa"]=[st.mean(r["nasa_mean"] for r in res["rounds"]) for res in (a,b,c)]
            pairs.append(item)
    result=dict(pairs=len(pairs),observations=30*len(pairs),data=pairs)
    (qualify.OUT/"audit.json").write_text(json.dumps(result,indent=2))
    print(result["pairs"],result["observations"])
if __name__=="__main__":
    main()
