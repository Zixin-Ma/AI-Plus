"""Independently audit paired V4 qualification replays from their per-round audit metrics."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
from tempfile import NamedTemporaryFile

WORK=Path(__file__).resolve().parent.parent/"work/v4"
VERSIONS=("v3","v4")
METRICS=("mean","worst","forget","delay")
COUNTS=("train","selection","audit")
RTOL=1e-12
ATOL=1e-12
DEVELOPMENT_ORDERS=(
    (0,0,1,1,2,2,3,3,0,1), (2,2,0,0,3,3,1,1,2,0),
    (3,3,2,2,1,1,0,0,3,2), (1,3,0,2,1,0,3,2,2,1),
    (3,0,3,1,1,2,0,2,1,3), (0,2,3,1,0,3,1,2,2,0),
    (2,1,3,0,1,3,2,0,0,1), (3,3,1,2,0,2,1,3,0,2))


def check(condition, message):
    if not condition:
        raise ValueError(message)


def finite(x):
    return type(x) in (int,float) and math.isfinite(x)


def compare(a,b,field):
    check(finite(a) and finite(b),f"{field}: non-finite or missing number {a!r}, {b!r}")
    check(math.isclose(a,b,rel_tol=RTOL,abs_tol=ATOL),f"{field}: calculated {a!r}, stored {b!r}")
    return abs(a-b)


def cases(scope):
    if scope=="sample":
        return [("dev",case,task,0) for case in ("drift","official") for task in "ABC"]
    matrix=[]
    for case in ("drift","official","fresh","fresh-measured"):
        dev=range(8) if case in ("drift","official") else range(3,8)
        for task in "ABC":
            matrix.extend(("dev",case,task,n) for n in dev)
            matrix.extend(("confirm",case,task,n) for n in (8,9,10))
    return matrix


def recovery(ordered_stages, current_gains):
    lengths=[]
    for i in range(1,len(ordered_stages)):
        if ordered_stages[i]==ordered_stages[i-1]:
            continue
        end=i
        while end<len(ordered_stages) and ordered_stages[end]==ordered_stages[i] and current_gains[end]<0.:
            end+=1
        lengths.append(end-i)
    return lengths


def stream_audit(payload,task,order):
    rows=payload["rounds"]
    check(len(rows)==10,"expected ten rounds")
    ref=rows[0]["static"]
    check(len(ref)==4 and all(finite(x) and x>=0 for x in ref),"invalid four-stage static reference")
    peaks={}
    current=[]; drawdowns=[]
    largest_delta=0.
    for index,(stage,row) in enumerate(zip(order,rows),1):
        check(row["round"]==index and row["stage"]==stage,f"round {index}: wrong order or stage")
        check(all(type(row[c]) is int and row[c]>0 for c in COUNTS),f"round {index}: invalid sample counts")
        check(row["static"]==ref,f"round {index}: static reference changed")
        measured=row["metrics"]
        check(len(measured)==len(row["gains"])==4 and all(finite(x) and x>=0. for x in measured),
              f"round {index}: invalid four-stage metrics")
        if task=="A":
            check(all(0.<=x<=1. for x in [*measured,*ref]),f"round {index}: F1 outside [0,1]")
        direction=1. if task=="A" else -1.
        gain=[direction*(value-start)/max(start,1e-9) for start,value in zip(ref,measured)]
        for j,(g,saved) in enumerate(zip(gain,row["gains"])):
            largest_delta=max(largest_delta,compare(g,saved,f"round {index} stage {j} gain"))
        # Only stages already seen before this round can contribute a drawdown.
        drop=max([0.,*(high-gain[j] for j,high in peaks.items())])
        largest_delta=max(largest_delta,compare(drop,row["forget"],f"round {index} historical drawdown"),
                          compare(gain[stage],row["stage_gain"],f"round {index} active stage gain"))
        current.append(gain[stage]);drawdowns.append(drop)
        peaks.setdefault(stage,gain[stage])
        for previous in peaks:
            peaks[previous]=max(peaks[previous],gain[previous])
    delays=recovery(order,current)
    check(delays==payload["recovery_delays"],"stored stage-switch delay sequence differs")
    result={"mean":statistics.fmean(current),"worst":min(current),"forget":max(drawdowns),"delay":statistics.fmean(delays)}
    for key in METRICS:
        largest_delta=max(largest_delta,compare(result[key],payload[key],key))
    return result,largest_delta


def bearing_compare(left,right,field,counts):
    if isinstance(left,dict):
        check(isinstance(right,dict) and left.keys()==right.keys(),f"{field}: bearing score schema changed")
        for name in left:
            bearing_compare(left[name],right[name],f"{field}/{name}",counts)
    elif isinstance(left,list):
        check(isinstance(right,list) and len(left)==len(right),f"{field}: bearing vector length changed")
        for i,(a,b) in enumerate(zip(left,right)):
            bearing_compare(a,b,f"{field}/{i}",counts)
    elif finite(left):
        diff=compare(left,right,field)
        counts["numeric_comparisons"]+=1
        counts["nonidentical_values"]+=int(diff!=0.)
        counts["max_abs_difference"]=max(counts["max_abs_difference"],diff)
    else:
        check(type(left)==type(right) and left==right,f"{field}: bearing choice differs: {left!r}, {right!r}")


def paired_rounds(left,right,task,bearing):
    for index,(a,b) in enumerate(zip(left["rounds"],right["rounds"]),1):
        check(a["static"]==b["static"],f"round {index}: starting metrics differ between versions")
        for name in ("round","stage",*COUNTS):
            check(a[name]==b[name],f"round {index}: unmatched {name}: {a[name]}, {b[name]}")
        if task=="B":
            for name in ("deployed","head_family","expert_weight","replanned","scores","metrics","gains",
                         "stage_gain","forget"):
                bearing_compare(a[name],b[name],f"round {index}/{name}",bearing)
            bearing["paired_rounds"]+=1


def verify(scope):
    protocol=json.loads((WORK/"protocol.json").read_text())
    orders=[*DEVELOPMENT_ORDERS,*map(tuple,protocol["new_orders"])]
    check(len(orders)==11 and all(len(o)==10 for o in orders),"invalid stage schedule")
    pairs=cases(scope)
    root=WORK/"replay"
    streams={}
    bearing={"paired_rounds":0,"numeric_comparisons":0,"nonidentical_values":0,"max_abs_difference":0.}
    counts={"dev":0,"confirm":0}
    hashes=hashlib.sha256()
    max_delta=0.
    for phase,case,task,split in pairs:
        pair=[]
        for version in VERSIONS:
            path=root/f"{case}-{version}-{task}-{split}.json"
            try:
                data=path.read_bytes()
                record=json.loads(data)
                scores,delta=stream_audit(record,task,orders[split])
            except (ValueError,KeyError,TypeError,IndexError) as exc:
                raise ValueError(f"{path.name}: {exc}") from exc
            hashes.update(path.name.encode()+b"\0"+hashlib.sha256(data).digest())
            pair.append(record)
            streams[path.name]=scores
            max_delta=max(max_delta,delta)
        try:
            paired_rounds(*pair,task,bearing)
        except (ValueError,KeyError,TypeError) as exc:
            raise ValueError(f"{case}/{task}/{split}: {exc}") from exc
        counts[phase]+=20
    if scope=="full":
        observed={p.name for p in root.glob("*.json")}
        check(observed==set(streams),f"unexpected or missing files: {sorted(observed.symmetric_difference(streams.keys()))}")
        check(len(streams)==228 and sum(counts.values())==2280 and bearing["paired_rounds"]==380,
              "qualification coverage mismatch")
    return dict(status="passed",scope=scope,files=len(streams),recorded_rounds=sum(counts.values()),
        counts_by_phase=counts,version_pairs=len(pairs),paired_rounds=10*len(pairs),
        sample_fields=list(COUNTS),sample_field_comparisons=10*len(pairs)*len(COUNTS),
        bearing=bearing,max_metric_residual=max_delta,atol=ATOL,rtol=RTOL,
        input_sha256=hashes.hexdigest(),per_stream=streams)


def self_test():
    # A repeated unseen stage, a late negative gain and a truncated recovery.
    order=(0,0,1,1,2,2,3,3,0,1)
    rows=[];peaks={};current=[];drawdowns=[]
    for i,stage in enumerate(order):
        g=[(.12 if j==0 else -.08)+.01*((i+2*j)%4) for j in range(4)]
        row=dict(round=i+1,stage=stage,metrics=[.5*(1+v) for v in g],static=[.5]*4,gains=g,
                 stage_gain=g[stage],forget=max([0.]+[peaks[j]-g[j] for j in peaks]),train=5,selection=2,audit=3)
        peaks[stage]=max(peaks.get(stage,-1.),g[stage])
        for j in peaks: peaks[j]=max(peaks[j],g[j])
        rows.append(row);current.append(g[stage]);drawdowns.append(row["forget"])
    data=dict(rounds=rows,mean=statistics.fmean(current),worst=min(current),forget=max(drawdowns),
              delay=1.4,recovery_delays=[2,2,2,0,1])
    stream_audit(data,"A",order)
    rows[-1]["gains"][0]+=.1
    try: stream_audit(data,"A",order)
    except ValueError: pass
    else: raise AssertionError("a mutated phase gain was accepted")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample",action="store_true",help="audit twelve public development flows without writing the final result")
    args=parser.parse_args()
    self_test()
    result=verify("sample" if args.sample else "full")
    if not args.sample:
        with NamedTemporaryFile("w",encoding="utf8",dir=WORK,prefix=".verification-",suffix=".tmp",delete=False) as out:
            tmp=Path(out.name)
            try:
                json.dump(result,out,ensure_ascii=False,indent=2,allow_nan=False)
                out.write("\n")
            except BaseException:
                tmp.unlink(missing_ok=True)
                raise
        tmp.replace(WORK/"verification.json")
    print(json.dumps({k:v for k,v in result.items() if k!="per_stream"},ensure_ascii=False))


if __name__=="__main__":
    main()
