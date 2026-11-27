"""Paired, ten-round public replay with growing training releases and isolated audit rows."""
import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time
from types import SimpleNamespace

import numpy as np

HERE = Path(__file__).resolve().parent
SCHOOL = HERE.parents[1] / "校赛"
WORK = SCHOOL / "work"
OUT = HERE.parent / "work" / "v6" / "replay"


# Public row counts and action names from V4 trial 235fea6ebd8481811331281798df1821.
TRACE_SCHEDULE = {'A': [(445, 1244, 'refit'),
       (445, 1244, 'continue'),
       (952, 1371, 'refit'),
       (952, 1371, 'refit'),
       (1003, 1384, 'refit'),
       (1065, 1400, 'continue'),
       (1801, 1584, 'refit'),
       (2957, 1873, 'refit'),
       (3050, 1897, 'refit'),
       (3200, 1935, 'refit')],
 'B': [(7032, 1679, 'refit'),
       (7032, 1679, 'continue'),
       (7032, 1679, 'continue'),
       (7032, 1679, 'refit'),
       (7048, 1683, 'continue'),
       (7048, 1683, 'continue'),
       (7048, 1683, 'refit'),
       (7156, 1710, 'continue'),
       (7156, 1710, 'continue'),
       (7156, 1710, 'refit')],
 'C': [(113089, 23049, 'refit'),
       (113089, 23049, 'refit'),
       (113089, 23049, 'refit'),
       (114141, 23313, 'refit'),
       (114141, 23313, 'refit'),
       (114141, 23313, 'refit'),
       (116746, 23965, 'refit'),
       (116746, 23965, 'refit'),
       (117909, 24256, 'refit'),
       (117909, 24256, 'refit')]}


def partitions(rows, key, split):
    groups = {}
    for i, row in enumerate(rows):
        m = row["meta"]
        g = row["answer"] if key == "A" else m["bearing"] if key == "B" else (m["source_subset"], m["unit"])
        groups.setdefault(g, []).append(i)
    phase = np.zeros(len(rows), dtype=int)
    audit = np.zeros(len(rows), dtype=bool)
    release = np.zeros(len(rows))
    for g, ids in groups.items():
        ids = np.asarray(ids)
        if key != "A":
            ids = ids[np.argsort([rows[i]["meta"]["snapshot" if key == "B" else "cycle"] for i in ids])]
        for n, block in enumerate(np.array_split(ids, 4)):
            phase[block] = n
            # Blocked halves for gas and bearing. Hold out whole engines for C.
            if split >= 8:
                import hashlib
                seed = f"enterprise4-{split}-{g}"
                code = int.from_bytes(hashlib.sha256(seed.encode()).digest()[:4], "big")
                if key == "C":
                    audit[block] = code % 4 == 0
                else:
                    audit[np.array_split(block, 3)[(code+n) % 3]] = True
            elif split >= 5:
                if key == "C":
                    audit[block] = (g[1] % 5 == split - 5)
                else:
                    # Rotate contiguous within-stage blocks across classes/devices and phases.
                    import hashlib
                    domain = int.from_bytes(hashlib.blake2b(str(g).encode(), digest_size=2).digest(), "big")
                    pick = (domain + n + split - 5) % 4
                    audit[np.array_split(block, 4)[pick]] = True
            elif key == "C":
                audit[block] = (g[1] % 3 == 1) if split==2 else (g[1] % 4 == split-2) if split>=3 else (g[1] % 2 == split)
            else:
                audit[np.array_split(block, 3)[1] if split==2 else np.array_split(block,4)[split-2] if split>=3 else np.array_split(block, 2)[split]] = True
        release[ids] = (np.arange(len(ids)) + .5) / len(ids)
    return release, phase, audit


class Parent:
    def __init__(self, folder):
        self.folder = folder
        self.artifact_id = folder.name

    def path(self, name):
        return self.folder / name

    def declared_paths(self):
        return [p.name for p in self.folder.glob("*.npz")]


class PublicRound:
    model_name = "Qwen/Qwen3.8-27B"
    max_output_tokens = 56

    def history(self):
        return self.records

    def model(self, messages, **opts):
        # Fixed planner and reviewer for a paired CPU comparison of the two harnesses.
        # Only the production gateway supplies real model decisions.
        review = "Choose one of" in messages[0]["content"]
        text = "candidate|use the measured candidate" if review else self.plan
        if review and getattr(self, "review_mode", "") == "measured":
            cfg = self.candidate_config
            panel = cfg["scores"]
            allowed = [n for n in panel if n != "parent" and self.agent.eligible(self.dataset_key, panel["parent"], panel[n])]
            if allowed:
                sign = 1 if self.dataset_key == "A" else -1
                best = max(allowed, key=lambda n: sign*panel[n]["primary"])
            else:
                best = "parent"
            text = best + "|best permitted measured primary"
        self.prompt_bytes += sum(len(m["content"].encode()) for m in messages)
        self.call_ids.append("offline-" + str(len(self.call_ids)+1))
        return SimpleNamespace(text=text)

    def submit_candidate(self, folder, **spec):
        self.candidate_config = spec["configuration"]
        self.candidate = self.root/f"round-{self.round_index}"
        shutil.copytree(folder,self.candidate)
        model,head,_ = self.agent.artifact(self.candidate)
        x,y = self.selection
        score = self.agent.metric(self.dataset_key,y,self.agent.values(self.dataset_key,model,head,x))
        return {"artifact_id":self.candidate.name,"summary":{"metrics":{self.agent.METRIC[self.dataset_key]:score}}}

    def restore_artifact(self, name, dest):
        return shutil.copytree(self.root/name,dest)

    def finish(self, **record):
        self.record = record


def run(key, version, split, setting="drift", profile=False, review="controlled"):
    if version == "v5":
        sys.path.insert(0,str(HERE.parent/"V5"))
    import contest_submission as agent
    base = WORK/key
    def read(name):
        with (base/"public"/(name+".jsonl")).open() as f:
            return [json.loads(line) for line in f]
    train, valid = read("train"), read("validation")
    release,_,_ = partitions(train,key,split)
    _,phase,audit = partitions(valid,key,split)
    official = agent.Model.load(base/"checkpoint.npz")
    xt,yt,_ = agent.projection(train,official,key)
    xv,yv,_ = agent.projection(valid,official,key)
    # Rebuild the starting model on the first release: later training pairs are unseen.
    if setting in ("trace","schedule"):
        # Apply only the released row-count ratios and action schedule from the V4
        # trial to public data. The hidden samples and hidden metrics are never used.
        observed = TRACE_SCHEDULE[key]
        train_sizes = np.array([row[0] for row in observed])
        valid_sizes = np.array([row[1] for row in observed])
        train_rates = train_sizes / train_sizes[-1]
        valid_rates = valid_sizes / valid_sizes[-1]
        valid_release = partitions(valid,key,split)[0]
    else:
        train_rates = np.linspace(.55,1.,10)
    first = release <= train_rates[0]
    m = official.meta
    static = agent.Model(m["task"], m["paths"], m["classes"],xt[first],yt[first],
                         hidden=m["hidden"], seed=m["seed"], batch_size=m["batch_size"])
    if setting == "official":
        static = official
        release[:] = 0.
    else:
        static.fit(xt[first],yt[first],15)
    selection_domains = [row["answer"] if key=="A" else row["meta"]["bearing"] if key=="B"
                         else (row["meta"]["source_subset"], row["meta"]["unit"]) for row in valid]
    if setting == "schedule":
        # Stages follow the observed V5 round sequence. Public-only domains are proxies.
        if key == "A":
            import hashlib
            for label in set(selection_domains):
                ids = [i for i,v in enumerate(selection_domains) if v == label]
                ids.sort(key=lambda i: valid[i]["meta"]["source_index"])
                for group, block in enumerate(np.array_split(ids, 8)):
                    phase[block] = group
                    audit[block] = False
                    if len(block):
                        fold = (group+split)%3 if split < 14 else int.from_bytes(hashlib.blake2b(f"{split}:{label}:{group}".encode(), digest_size=4).digest(), "big") % 3
                        audit[np.array_split(block, 3)[fold]] = True
        elif key == "B":
            phase = np.array([int(v.split("_")[0][-1])-1 for v in selection_domains])
        else:
            phase = np.array([int(v[0][-1])-1 for v in selection_domains])
    # Platform deployments sometimes omit device metadata.
    for r in train+valid:
        r.pop("meta",None)
    OUT.mkdir(exist_ok=True,parents=True)
    records, report, seen, peaks = [], [], set(), {}
    phases = [0,0,1,1,2,2,3,3,0,1] if split==0 else [2,2,0,0,3,3,1,1,2,0]
    if split == 2:
        phases = [3,3,2,2,1,1,0,0,3,2]
    if split==3:
        phases=[1,3,0,2,1,0,3,2,2,1]
    elif split==4:
        phases=[3,0,3,1,1,2,0,2,1,3]
    if split >= 14:
        phases = np.random.default_rng(6020+split).permutation([0,0,0,1,1,1,2,2,3,3]).tolist()
    elif split >= 5:
        phases = {5:[0,2,3,1,0,3,1,2,2,0], 6:[2,1,3,0,1,3,2,0,0,1],
                  7:[3,3,1,2,0,2,1,3,0,2],
                  8:[2,0,2,3,1,1,0,3,2,1],9:[1,3,3,0,2,1,0,2,3,0],
                  10:[0,3,1,2,2,0,1,3,1,0],
                  11:[2,2,0,3,2,1,0,0,3,1],12:[3,1,1,0,2,1,3,0,2,2],
                  13:[1,0,3,3,2,1,0,2,0,1]}[split]
    if setting == "schedule":
        phases = {"A":[0,0,1,1,2,3,4,5,6,7], "B":[0,0,0,0,1,1,1,2,2,2],
                  "C":[0,0,0,1,1,1,2,2,3,3]}[key]
    masks = [(phase==p)&audit for p in range(max(phases)+1)]
    def metrics(model,head):
        pred = agent.values(key,model,head,xv[audit])
        return [agent.metric(key,yv[mask],pred[mask[audit]]) for mask in masks]
    ref = metrics(static,{})
    with tempfile.TemporaryDirectory(dir=OUT) as t:
        root = Path(t)
        os.environ["CONTEST_WORK"]=t
        parent=root/"initial"; parent.mkdir(); static.save(parent/"checkpoint.npz")
        paths=[parent]
        for i,stage in enumerate(phases,1):
            tr=np.flatnonzero(release <= train_rates[i-1])
            sel=np.flatnonzero((phase==stage)&~audit)
            if setting in ("trace","schedule"):
                sel = np.flatnonzero((valid_release <= valid_rates[i-1]) & ~audit)
            if setting == "fresh":
                # Each visit receives a different selection block inside each domain.
                # The audit side remains completely isolated in this replay.
                visit = phases[:i-1].count(stage)
                visits = phases.count(stage)
                buckets = {}
                for j in sel:
                    buckets.setdefault(selection_domains[j], []).append(j)
                parts = [np.array_split(ids, visits)[visit] for ids in buckets.values()]
                sel = np.asarray(sorted(index for part in parts for index in part), dtype=int)
            r=PublicRound()
            r.review_mode = review
            r.agent,r.root,r.dataset_key,r.round_index=agent,root,key,i
            r.parent,r.records,r.call_ids=Parent(parent),records,[]
            r.artifacts=[{"artifact_id":p.name} for p in paths]
            r.selection=(xv[sel],yv[sel])
            r.data=SimpleNamespace(training_rows=lambda:(train[j] for j in tr),
                                   validation_rows=lambda:(valid[j] for j in sel))
            # Include backbone-only actions for bearings, matching the prevalent V2 choice.
            tool="adam" if key=="B" and i not in (1,5,9) else "refit"
            if setting in ("trace","schedule"):
                tool=observed[i-1][2]
            r.plan=f"{tool}|half|test changed training release"
            r.prompt_bytes = 0
            tick = time.process_time()
            agent.train(r)
            elapsed = time.process_time()-tick
            paths.append(r.candidate)
            r.record["id"]=f"{key}-{i}"
            records.append(r.record)
            parent=root/r.record["selected_artifact_id"]
            model,head,_=agent.artifact(parent)
            pred=metrics(model,head)
            gain=[agent.gain(key,a,b)/max(a,1e-9) for a,b in zip(ref,pred)]
            if key == "C":
                error = agent.values(key,model,head,xv[masks[stage]])[:,0]-yv[masks[stage]]
                nasa = float(np.mean(np.expm1(np.maximum(error/10.,-error/13.))))
            drops=[max(0.,peaks[p]-gain[p]) for p in seen]
            seen.add(stage)
            for p in seen:
                peaks[p]=max(peaks.get(p,-np.inf),gain[p])
            cfg = r.record["configuration"]
            outcome = {a:{k:v for k,v in values.items() if k in ("primary","worst","floor","memory_loss","release_movement","allowed")}
                       for a,values in cfg["scores"].items()}
            report.append(dict(round=i,stage=stage,scores=outcome,expert_weight=cfg.get("expert_weight"),head_family=r.record["configuration"].get("head_family"),replanned=r.record["configuration"].get("replanned"),cpu=elapsed,calls=len(r.call_ids),prompt_bytes=r.prompt_bytes,checkpoint_bytes=sum(p.stat().st_size for p in r.candidate.glob("*.npz")),static=ref,metrics=pred,gains=gain,
                       nasa_mean=nasa if key=="C" else None,stage_gain=gain[stage],forget=max(drops,default=0.),train=len(tr),
                       selection=len(sel),audit=int(masks[stage].sum()),deployed=r.record["configuration"]["deployed"]))
        if profile:
            ids=np.linspace(0,len(valid)-1,min(512,len(valid)),dtype=int)
            task=m["task"]
            samples=[SimpleNamespace(inputs=valid[j]["inputs"],task=task,dataset_key=key) for j in ids]
            t0=time.process_time()
            for sample in samples:
                json.dumps(agent.predict(sample,parent),allow_nan=False)
            inference_us=1e6*(time.process_time()-t0)/len(samples)
    gains=[row["stage_gain"] for row in report]
    delays=[]
    for i,p in enumerate(phases):
        if i and phases[i-1]!=p:
            j=i
            while j<len(phases) and phases[j]==p and gains[j]<0:
                j+=1
            delays.append(j-i)
    result=dict(rounds=report,mean=float(np.mean(gains)),worst=min(gains),
                forget=max(r["forget"] for r in report),delay=float(np.mean(delays)),
                recovery_delays=delays)
    stage_means = [float(np.mean([r["stage_gain"] for r in report if r["stage"] == p])) for p in sorted(set(phases))]
    result.update(stage_means=stage_means, stage_mean=float(np.mean(stage_means)), stage_floor=min(stage_means))
    if profile:
        result["prediction_us"]=inference_us
    suffix = "" if review == "controlled" else "-"+review
    (OUT/f"{setting}{suffix}-{version}-{key}-{split}.json").write_text(json.dumps(result,indent=2))
    print(version,key,split,{k:v for k,v in result.items() if k not in ("rounds","recovery_delays")},flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("task",choices=list("ABC"))
    parser.add_argument("version",choices=["v5","v6"])
    parser.add_argument("split",type=int,choices=list(range(20)))
    parser.add_argument("--setting",choices=["drift","official","fresh","trace","schedule"],default="drift")
    parser.add_argument("--output",type=Path,default=OUT)
    parser.add_argument("--profile",action="store_true")
    parser.add_argument("--review",choices=["controlled","measured"],default="controlled")
    a=parser.parse_args()
    OUT=a.output
    run(a.task,a.version,a.split,a.setting,a.profile,a.review)
