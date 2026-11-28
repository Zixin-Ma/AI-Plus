"""Compare saved controller trials, industrial replay, and model usage receipts."""
from collections import Counter, defaultdict
import json
from pathlib import Path
from statistics import mean, median

HERE = Path(__file__).resolve().parent
WORK = HERE.parent / "work/v7"
METRICS = ("stage_mean", "stage_floor", "forget", "delay")
DECISIONS = ("deployed", "head_family", "expert_weight", "replanned")
SPLITS = ("train_ids", "selection_ids", "audit_ids")
TOL = 1e-9


def read(path):
    return json.loads(path.read_text())


def index(folder, version):
    # Search exactly the published development seeds. Confirmation seeds are not inputs.
    paths = {}
    for task in "ABC":
        for seed in range(20):
            for mode in ("schedule", "trace", "fresh", "drift", "transfer", "official"):
                f = folder / f"{mode}-{version}-{task}-{seed}.json"
                if f.is_file():
                    paths[(task, mode, seed)] = f
    return paths


def sample_counts(keys):
    return dict(sorted(Counter(task + "/" + mode for task, mode, _ in keys).items()))


def equivalent(left, right, required):
    aligned = sorted(left.keys() & right.keys())
    differences, missing, top_diff = Counter(), Counter(), Counter()
    round_examples, score_max = {}, defaultdict(float)
    times = [[], []]
    def deviation(x, y):
        if isinstance(x, dict) and isinstance(y,dict):
            if x.keys() != y.keys():
                return None
            vals = [deviation(x[k],y[k]) for k in x]
        elif isinstance(x,list) and isinstance(y,list):
            if len(x) != len(y):
                return None
            vals = [deviation(a,b) for a,b in zip(x,y)]
        elif isinstance(x,(int,float)) and isinstance(y,(int,float)):
            return abs(x-y)
        else:
            return 0.0 if x==y else None
        return None if any(v is None for v in vals) else max(vals,default=0.)
    for k in aligned:
        a,b=read(left[k]),read(right[k])
        assert len(a["rounds"]) == len(b["rounds"]) == 10, k
        for field in (a.keys() | b.keys()) - {"rounds"}:
            if field not in a or field not in b or a[field] != b[field]:
                top_diff[field] += 1
        for x,y in zip(a["rounds"],b["rounds"]):
            for j,z in enumerate((x,y)):
                times[j].append(z["cpu"])
            for field in x.keys() | y.keys():
                if field not in x or field not in y:
                    missing[field]+=1
                elif x[field]!=y[field]:
                    differences[field]+=1
                    round_examples.setdefault(field, [*k,x["round"]])
            for field in ("scores","metrics","static","gains","stage_gain","forget"):
                d=deviation(x[field],y[field])
                if d is None:
                    raise ValueError((k,x["round"],field))
                score_max[field]=max(score_max[field],d)
            assert all(x[key]==y[key] for key in required), (k,x["round"])
    return dict(streams=len(aligned),rounds=sum(map(len,times))//2,groups=sample_counts(aligned),
                round_field_changes=dict(differences),missing_round_fields=dict(missing),
                stream_field_changes=dict(top_diff),max_numeric_delta=dict(score_max),
                median_round_cpu_seconds=[median(arr) for arr in times],first_changed=round_examples)


def compare(control, experiment, keys):
    keys=sorted(keys)
    result={}
    for metric in METRICS:
        a=[control[k][metric] for k in keys]
        b=[experiment[k][metric] for k in keys]
        delta=[y-x for x,y in zip(a,b)]
        result[metric]=dict(base=mean(a),new=mean(b),delta=mean(delta),
                            up=sum(x>TOL for x in delta),down=sum(x< -TOL for x in delta),
                            tie=sum(abs(x)<=TOL for x in delta))
    changed=Counter()
    differs=0
    recorded_changes=Counter()
    for k in keys:
        for x,y in zip(control[k]["rounds"],experiment[k]["rounds"]):
            assert len(control[k]["rounds"])==len(experiment[k]["rounds"])==10
            assert all(x.get(a)==y.get(a) for a in SPLITS), k
            for a in DECISIONS:
                changed[a]+=int(x[a]!=y[a])
            differs+=int(any(x[a]!=y[a] for a in DECISIONS))
            for metric in ("metrics","nasa_mean","calls"):
                recorded_changes[metric]+=int(x[metric]!=y[metric])
    return dict(pairs=len(keys), metrics=result, changed_round_fields=dict(changed),
                changed_decision_rounds=differs, recorded_changes=dict(recorded_changes), groups=sample_counts(keys))


def variants(folder):
    dirs=sorted(d for d in folder.iterdir() if d.is_dir())
    tables={d.name:{k:read(f) for k,f in index(d,"v7").items()} for d in dirs}
    ctrl=tables.pop("control")
    result={}
    for task in "ABC":
        candidates={r:set(d) & set(ctrl) for r,d in tables.items() if any(k[0]==task for k in d)}
        if not candidates:
            continue
        common=set.intersection(*candidates.values())
        result[task]={}
        for route,keys in candidates.items():
            selected={k for k in keys if k[0]==task}
            both=selected & common
            data=tables[route]
            result[task][route]={"available":compare(ctrl,data,selected),
                                 "common":compare(ctrl,data,both),
                                 "by_protocol":{m:compare(ctrl,data,[k for k in selected if k[1]==m])
                                                for m in sorted({k[1] for k in selected})}}
    return result



def transfer_effect(tasks):
    """Compare each ranking rule on exactly the same protocol and data split across experts."""
    out = {}
    for task,rule in tasks:
        runs = {}
        for expert in ("v6","v7"):
            for branch in ("control",rule):
                entries = index(WORK/"controller"/expert/branch, "v7")
                runs[(expert,branch)] = {k:read(f) for k,f in entries.items() if k[0]==task}
        keys = sorted(set.intersection(*(set(dataset) for dataset in runs.values())))
        for k in keys:
            for x,y in zip(runs[("v6","control")][k]["rounds"], runs[("v7","control")][k]["rounds"]):
                assert all(x[id_] == y[id_] for id_ in SPLITS), k
        out[task] = {expert:compare(runs[(expert,"control")], runs[(expert,rule)],keys)
                        for expert in ("v6","v7")}
    return out

def cloud():
    base=WORK/"cloud"
    routes={"original":base/"streams","review":base/"ablations/review",
            "compact":base/"ablations/compact"}
    conditions=defaultdict(dict)
    all_ids=set()
    receipts_by_route={}
    for route,folder in routes.items():
        ids=set()
        for task in "ABC":
            suffix=f"schedule-v6-{task}-17.json"
            stream=read(folder/suffix)
            receipt=read(folder/("receipts-"+suffix))
            tokens=defaultdict(Counter)
            replies={}
            for row in receipt:
                sha=row["sha256"]
                full=read(base/"receipts"/(sha+".json"))
                assert full["request_sha256"]==sha and full["model"]=="Qwen/Qwen3.8-27B"
                usage=full["usage"]
                assert usage["total_tokens"]==row["tokens"]==usage["prompt_tokens"]+usage["completion_tokens"]
                ids.add(sha);all_ids.add(sha)
                meter=tokens[row["kind"]]
                meter["calls"]+=1
                for name in ("prompt_tokens","completion_tokens","total_tokens"):
                    meter[name]+=usage[name]
                reply=full["reply"].strip().split("|")
                replies[(row["round"],row["kind"])]=reply[:2] if row["kind"]=="plan" else reply[:1]
            assert len(receipt)==sum(r["calls"] for r in stream["rounds"])
            conditions[task][route]={"tokens":dict(tokens),"summary":{k:stream[k] for k in METRICS},
                                     "decisions":{k:[r[k] for r in stream["rounds"]] for k in DECISIONS},
                                     "requests":{f"{n}-{role}":value for (n,role),value in replies.items()}}
        receipts_by_route[route]=len(ids)
    fixed={}
    for file in ("prompt-study.json","prompt-check.json"):
        groups=defaultdict(Counter)
        for row in read(base/file):
            verified=read(base/"receipts"/(row["receipt"]+".json"))
            assert verified["request_sha256"]==row["receipt"]
            assert verified["model"]=="Qwen/Qwen3.8-27B" and verified["usage"]==row["usage"]
            g=groups[row["stage"]+"/"+row["style"]]
            g["samples"]+=1
            g["valid"]+=row["valid"]
            g["allowed"]+=row["allowed"]
            g["tokens"]+=row["usage"]["total_tokens"]
        fixed[file]=groups
    totals={}
    for route in routes:
        by_kind={}
        for kind in ("plan","review"):
            acc=Counter()
            for task in "ABC":
                acc.update(conditions[task][route]["tokens"][kind])
            by_kind[kind]=dict(acc)
        by_kind["total_tokens"]=sum(v["total_tokens"] for v in by_kind.values())
        totals[route]=by_kind
    return dict(conditions=conditions,totals=totals,unique_sha_all=len(all_ids),unique_sha_by_condition=receipts_by_route,
                fixed_prompts=fixed)


def main():
    control=index(WORK/"controller/v6/control","v7")
    original=index(WORK/"replay","v6")
    old_archive=index(HERE.parent/"work/v6/replay","v6")
    required=("deployed","head_family","expert_weight","replanned","calls","checkpoint_bytes","prompt_bytes")
    out={"fast_path":equivalent(original,control,required+SPLITS),
         "archive":equivalent(old_archive,control,required),
         "v6_experts":variants(WORK/"controller/v6"),
         "v7_experts":variants(WORK/"controller/v7"),
         "common_expert_effect":transfer_effect([("A","relative"),("B","penalty")]),
         "cloud":cloud()}
    (WORK/"controller-audit.json").write_text(json.dumps(out,indent=2)+"\n")
    print("streams",out["fast_path"]["streams"],"rounds",out["fast_path"]["rounds"],
          "unique_qwen_request_hashes",out["cloud"]["unique_sha_all"])


if __name__=="__main__":
    main()
