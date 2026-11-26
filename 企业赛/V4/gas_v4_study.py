"""Gas-array drift studies. Day indices are zero based and end exclusive."""
import argparse
import csv
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import time
import zipfile
import numpy as np
import gas
import prototype
from industrial import metric
from baseline_model import Model

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
OUT=HERE.parent/"work/v4/gas"
# All routes and hyperparameters must be frozen before reading terminal test scores.
DEVELOPMENT=((6,8,12),(12,14,18),(18,20,24))
TERMINAL=(24,28,36)
HORIZONS=((28,32),(32,36))
ROUTES=("ridge","shared","shape")
SEED=20261127


def load_module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def instrument():
    path=ROOT/"Supp_Datasets/A/01_Long_Term_Drift_2025/Dataset.zip"
    with zipfile.ZipFile(path) as fp:
        data=fp.read("extracted_features.csv")
    if hashlib.sha256(data).hexdigest()!="9933dca26ad0837c25eecca188d9a0499d848bb08e4c27574538e9e8fc69a38d":
        raise ValueError("external feature version changed")
    rows=list(csv.DictReader(io.StringIO(data.decode()),delimiter=" "))
    dates,days=np.unique([r["Date[yyyy-mm-dd]"] for r in rows],return_inverse=True)
    classes,y=np.unique([r["Class"].split()[0] for r in rows],return_inverse=True)
    keys=("mean_R{}[Ohm]","rel_diff_R{}","sample_startslope_R{}","rec_lvl_R{}")
    sig=np.array([[[float(r[t.format(i)]) for t in keys] for i in range(1,63)] for r in rows])
    x=sig.reshape(len(y),-1);native=sig.copy();native[:,:,0]=np.log(np.maximum(native[:,:,0],1e-6))
    relative=(native-native.mean(1,keepdims=True)).reshape(len(y),-1)
    raw=np.sign(x)*np.log1p(abs(x));q=np.log1p(abs(sig[:,:,1])).mean(1)
    # Use gain-normalized slope, direct relative response and relative recovery.
    z=np.stack([sig[:,:,1],sig[:,:,2]/np.maximum(abs(sig[:,:,0]),1.),sig[:,:,3]-1.],axis=2)
    return x,native.reshape(len(y),-1),relative,raw,q,z,classes,y,days


def uci_views(x):
    obs=x.reshape(-1,16,8)
    response=obs[:,:,1]
    slope=obs[:,:,2:]/np.maximum(abs(obs[:,:,:1]),1.)
    return np.concatenate([response[:,:,None],slope],axis=2)


def shape_features(z,loc,scale,mode):
    u=np.arcsinh((z-loc)/scale)
    centered=u-u.mean(1,keepdims=True)
    if mode=="shape":
        rms=np.sqrt(np.mean(centered**2,axis=1,keepdims=True))
        centered/=np.maximum(rms,.25)
    return np.concatenate((centered.reshape(len(u),-1),u.mean(1)),axis=1)


def physical_fit(z,y,route):
    C=int(max(y))+1;loc=np.median(z,axis=0)
    scale=np.maximum(np.median(abs(z-loc),axis=0)*1.4826,1e-8)
    f=shape_features(z,loc,scale,route)
    target=np.eye(C)[y]
    weights=np.mean(np.bincount(y))/np.bincount(y)[y]
    mid=np.average(f,weights=weights,axis=0)
    std=np.maximum(np.sqrt(np.average((f-mid)**2,axis=0,weights=weights)),.15)
    f=np.clip((f-mid)/std,-6,6)
    if route=="shared":
        mu=np.stack([f[y==i].mean(0) for i in range(C)])
        resid=f-mu[y]
        cov=(resid.T*weights)@resid/len(y)
        W=np.linalg.solve(cov+.65*np.eye(f.shape[1]),mu.T)
        bias=-.5*(mu*W.T).sum(1)
    else:
        gram=(f.T*weights)@f/len(y)
        W=np.linalg.solve(gram+.25*np.eye(f.shape[1]),(f.T*weights)@target/len(y))
        bias=np.zeros(C)
    return {"phys_location":loc,"phys_scale":scale,"phys_mean":mid,"phys_std":std,
            "phys_weight":W,"phys_bias":bias,"phys_mode":np.array(ROUTES.index(route))}


def physical_scores(h,z):
    mode=ROUTES[int(h["phys_mode"])]
    f=shape_features(z,h["phys_location"],h["phys_scale"],mode)
    s=np.clip((f-h["phys_mean"])/h["phys_std"],-6,6)@h["phys_weight"]+h["phys_bias"]
    return s


def trial_fit(z,y,groups,route,anchors=12):
    C=int(max(y))+1; loc=np.median(z,axis=0)
    scale=np.maximum(np.median(abs(z-loc),axis=0)*1.4826,1e-8)
    f=shape_features(z,loc,scale,"ridge")
    mid=f.mean(0);std=np.maximum(f.std(0),.1);f=np.clip((f-mid)/std,-8.,8.)
    masks=[y==c for c in range(C)]
    means=np.stack([f[m].mean(0) for m in masks])
    residual=f-means[y]
    between=[]
    for c,m in enumerate(masks):
        for b in np.unique(groups[m]):
            mask=m&(groups==b)
            between.append(residual[mask].mean(0))
    J=np.stack(between);cov=J.T@J/len(J)
    if route=="invariant":
        ev,V=np.linalg.eigh(cov)
        proj=V/np.sqrt(1+ev)
    elif route=="within":
        obs=np.bincount(y,minlength=C)
        weights=len(y)/(C*obs[y])
        cov=(residual.T*weights)@residual/len(y)+cov
        ev,V=np.linalg.eigh(cov)
        proj=V/np.sqrt(.35+np.maximum(ev,0))
    else:
        proj=np.eye(f.shape[1])
    v=f@proj
    nodes=[];node_labels=[]
    for c,m in enumerate(masks):
        ids=np.flatnonzero(m); ids=ids[np.argsort(groups[ids],kind="stable")]
        for piece in np.array_split(ids,min(anchors,max(1,len(ids)//3))):
            nodes.append(v[piece].mean(0));node_labels.append(c)
    nodes=np.stack(nodes)
    width=max(float(np.mean((v-v.mean(0))**2)),1e-5)
    def kernel(a,b):
        d=np.mean(a*a,1)[:,None]+np.mean(b*b,1)-(2/a.shape[1])*(a@b.T)
        return np.exp(-.85*np.maximum(0.,d)/width)
    K=kernel(v,nodes);ww=len(y)/(C*np.bincount(y)[y])
    W=np.linalg.solve((K.T*ww)@K+.003*len(y)*kernel(nodes,nodes)+.001*np.eye(len(nodes)), (K.T*ww)@np.eye(C)[y])
    return dict(phys_location=loc,phys_scale=scale,phys_mean=mid,phys_std=std,
                drift_project=proj,drift_nodes=nodes,drift_norm=(nodes*nodes).mean(1),drift_width=np.asarray(width),drift_coef=W)


def trial_scores(h,z):
    f=shape_features(z,h["phys_location"],h["phys_scale"],"ridge")
    v=np.clip((f-h["phys_mean"])/h["phys_std"],-8,8)@h["drift_project"]
    d=(v*v).mean(1)[:,None]+h["drift_norm"]-(2/v.shape[1])*(v@h["drift_nodes"].T)
    return np.exp(-.85*np.maximum(0.,d)/h["drift_width"])@h["drift_coef"]


def native_scores(args,fit,query):
    x,features,relative,raw,q,physical,classes,y,days=args
    mask=fit|query
    m=load_module("v2_external",HERE.parent/"V2/gas_v2_study.py")
    baseline=m.native_compare(x[mask],features[mask],relative[mask],raw[mask],q[mask],
                    classes,y[mask],fit[mask],[],return_scores=True)
    b,v1,curve=(baseline[k] for k in ("mlp","v1","v2"))
    head=prototype.fit_curve(features[fit],raw[fit],q[fit],y[fit],len(classes))
    level=np.clip((q[fit]-head["amp_center"])/head["amp_scale"],-3.,3.)
    head["amplitude_bounds"]=np.stack([np.quantile(level[y[fit]==i],[.05,.95]) for i in range(len(classes))])
    delta=curve-b
    out={"tail":b+prototype.robust_score(features[query],raw[query],q[query],head),
         "independent":v1+np.clip(delta-delta.max(1,keepdims=True),-4,0)}
    for name in ROUTES:
        h=physical_fit(physical[fit],y[fit],name)
        out[name]=physical_scores(h,physical[query])
    for name in ("plain_kernel","invariant","within"):
        h=trial_fit(physical[fit],y[fit],days[fit],name)
        out[name]=trial_scores(h,physical[query])
    for name in ("tail","independent"):
        vote=out["plain_kernel"]-out["plain_kernel"].max(1,keepdims=True)
        out["amend_"+name]=out[name]+np.clip(5*vote,-3,0)
    short=trial_fit(physical[fit],y[fit],days[fit],"plain_kernel",6)
    guide=trial_scores(short,physical[query])
    vote=guide-guide.max(1,keepdims=True)
    out["compact_amend"]=out["independent"]+np.clip(5*vote,-3,0)
    margins=-np.partition(vote,-2,axis=1)[:,-2]
    out["reliable_amend"]=out["independent"]+np.clip(8*vote,-6,0)*np.clip(margins/.4,.25,1.)[:,None]
    return out


def eval_days(args,dates,keep):
    *_,y,days=args
    start,cut,end=dates
    pre=days<start;val=(days>=start)&(days<cut);train=days<cut;test=(days>=cut)&(days<end)
    cal=native_scores(args,pre,val);scores=native_scores(args,train,test)
    head_metrics={name:{"select":metric("A",y[val],cal[name]),"test":metric("A",y[test],scores[name]),
        "test_days":[metric("A",y[days==d],scores[name][days[test]==d]) for d in range(cut,end)],
        "horizons":[metric("A",y[(days>=a)&(days<b)],scores[name][(days[test]>=a)&(days[test]<b)]) for a,b in HORIZONS] if dates==TERMINAL else []}
        for name in scores}
    choices={}
    for name,options in keep.items():
        pick=max(options,key=lambda k:head_metrics[k]["select"])
        choices[name]={"family":pick,**head_metrics[pick]}
    return {"window":dates,"n":[int(pre.sum()),int(val.sum()),int(test.sum())],"heads":head_metrics,"selected":choices}


def day_study(final=False):
    args=instrument()
    options={"v3":("tail","independent")}
    for r in (*ROUTES,"plain_kernel","invariant","within"):
        options[r]=("tail","independent",r)
    options["compact_amend"]=("tail","compact_amend")
    options["reliable_amend"]=("tail","reliable_amend")
    options["amend_tail"]=("amend_tail","independent")
    options["amend_independent"]=("tail","amend_independent")
    OUT.mkdir(parents=True,exist_ok=True)
    if final:
        lock=json.loads((OUT/"protocol.json").read_text())
        options={"v3":("tail","independent"),"v4":tuple(lock["families"])}
        rows=[eval_days(args,TERMINAL,options)];name="day-confirmation"
    else:
        rows=[eval_days(args,w,options) for w in DEVELOPMENT];name="day-development"
    (OUT/(name+".json")).write_text(json.dumps(rows,indent=2,allow_nan=False))
    for r in rows:print(r["window"],[(name,round(v["select"],3),round(v["test"],3)) for name,v in r["heads"].items()])


def reference():
    return load_module("reference_A",HERE.parent/"V3/gas.py")


def install(route):
    # Training labels and within-class ranks are restricted to the current release.
    prev=reference();original_fit,original_predict=prev.fit_candidates,prev.predict
    def fits(m,x,y,i):
        heads=original_fit(m,x,y,i)
        if "independent" not in heads: return heads
        groups=np.zeros(len(y),dtype=int)
        for cls in np.unique(y):
            for j,block in enumerate(np.array_split(np.flatnonzero(y==cls),4)):
                groups[block]=j
        kind="plain_kernel" if route!="within_trial" else "within"
        h=trial_fit(uci_views(x),y,groups,kind,6 if route in ("compact_amend","reliable_amend") else 12)
        if route in ("compact_amend","reliable_amend"):
            h["reliable_vote"]=np.asarray(route=="reliable_amend")
            heads["independent"].update(h)
        elif route in ("amend_tail","amend_independent"):
            family=route[len("amend_"):]
            heads[family].update(h)
        else:
            heads["physical"]=h
        return heads
    def pred(m,h,x):
        if "drift_coef" not in h: return original_predict(m,h,x)
        evidence=trial_scores(h,uci_views(x))
        if "curve_coeff" in h:
            signal=evidence-evidence.max(1,keepdims=True)
            if h.get("reliable_vote",False):
                margin=-np.partition(signal,-2,axis=1)[:,-2]
                return original_predict(m,h,x)+np.clip(8*signal,-6,0)*np.clip(margin/.4,.25,1.)[:,None]
            return original_predict(m,h,x)+np.clip(5*signal,-3,0)
        return 10*(evidence-evidence.max(1,keepdims=True))
    gas.fit_candidates,gas.predict=fits,pred


def replay(route):
    import benchmark
    if route=="v3":
        prev=reference();gas.fit_candidates,gas.predict=prev.fit_candidates,prev.predict
    elif route!="frozen": install(route)
    benchmark.OUT=OUT/"replay"/route
    for setting in ("drift","official"):
        for i in range(8):
            benchmark.run("A","v4",i,setting,profile=True)


def dsps():
    m=load_module("external_ds",HERE.parent/"V2/gas_v2_study.py")
    def compare(x,absolute,relative,raw,q,classes,y,train,blocks,cluster=None,inner=None):
        n=len(y);pair=x.reshape(n,2,-1)
        floor=np.maximum(abs(pair[:,1]),1e-6)
        phys=np.stack((pair[:,1],pair[:,0]/floor),axis=2)
        # A group index encodes a distinct run of all six concentrations.
        reps=np.array([(i//6)%10 for i in range(n)])
        args=(x,absolute,relative,raw,q,phys,classes,y,reps)
        return evaluate_split(args,inner,train&~inner,train,~train,blocks)
    m.native_compare=compare
    result=m.dsps()
    (OUT/"ds-ps.json").write_text(json.dumps(result,indent=2))
    return result


def evaluate_split(args,pre,val,train,test,blocks):
    y=args[-2];vi=native_scores(args,pre,val);te=native_scores(args,train,test)
    return {name:{"select":metric("A",y[val],vi[name]),"test":metric("A",y[test],ts),
                  "blocks":[metric("A",y[m],ts[m[test]]) for m in blocks]} for name,ts in te.items()}


def paired_days():
    args=instrument();y,days=args[-2:]
    lock=json.loads((OUT/"protocol.json").read_text())
    families={"v3":("tail","independent"),"v4":lock["families"]}
    def measure(a,b,c):
        before=days<a;valid=(days>=a)&(days<b);fit=days<b;test=(days>=b)&(days<c)
        sel=native_scores(args,before,valid);scores=native_scores(args,fit,test)
        picked={k:max(opts,key=lambda f:metric("A",y[valid],sel[f])) for k,opts in families.items()}
        nd=np.unique(days[test]);C=len(args[-3])
        draw=np.random.default_rng(SEED).integers(len(nd),size=(4000,len(nd)))
        boot=[];measured={}
        for version,name in picked.items():
            out=scores[name];pred=out.argmax(1);yt=y[test]
            conf=np.stack([np.bincount(C*yt[days[test]==d]+pred[days[test]==d],minlength=C*C).reshape(C,C) for d in nd])
            cm=conf[draw].sum(1)
            boot.append((2*np.diagonal(cm,axis1=1,axis2=2)/np.maximum(1,cm.sum(1)+cm.sum(2))).mean(1))
            measured[version]={"family":name,"f1":metric("A",yt,out),"confusion_by_date":conf.tolist()}
        return {"dates":[a,b,c],"train":int(fit.sum()),"selection":int(valid.sum()),"test":int(test.sum()),
                "results":measured,"ci95":np.quantile(boot[1]-boot[0],[.025,.975]).tolist()}
    data={"terminal":measure(*TERMINAL),"v3_original_window":measure(8,12,36)}
    (OUT/"day-paired.json").write_text(json.dumps(data,indent=2,allow_nan=False))
    print([(k,v["ci95"],[(n,r["family"],r["f1"]) for n,r in v["results"].items()]) for k,v in data.items()])


def resource_pairs():
    import os
    import sys
    import subprocess
    os.sched_setaffinity(0,{max(os.sched_getaffinity(0))})
    for i in range(3):
        for setting in ("drift","official"):
            for version in (("v3","v4") if i%2==0 else ("v4","v3")):
                subprocess.run([sys.executable,"-B",str(HERE/"benchmark.py"),"A",version,"0","--setting",setting,
                    "--profile","--output",str(OUT/"cost"/str(i))],check=True)


def full_round(version):
    import os
    import tempfile
    from types import SimpleNamespace
    import contest_submission as harness
    from benchmark import PublicRound,Parent,WORK
    if version=="v3":
        earlier=reference();gas.fit_candidates,gas.predict=earlier.fit_candidates,earlier.predict
    data=WORK/"A"
    def samples(name):
        with (data/"public"/(name+".jsonl")).open() as fp:
            for line in fp:
                r=json.loads(line);yield {"inputs":r["inputs"],"answer":r["answer"]}
    run=PublicRound();run.agent,run.dataset_key,run.round_index=harness,"A",1
    run.parent,run.records,run.call_ids=Parent(data),[],[]
    run.artifacts=[];run.prompt_bytes=0;run.plan="refit|half|test sensor evidence"
    run.data=SimpleNamespace(training_rows=lambda:samples("train"),validation_rows=lambda:samples("validation"))
    run.selection=harness.projection(samples("validation"),harness.Model.load(run.parent.path("checkpoint.npz")),"A")[:2]
    with tempfile.TemporaryDirectory(dir=OUT) as tmp:
        os.environ["CONTEST_WORK"]=tmp;run.root=Path(tmp);harness.train(run)
        cfg=run.record["configuration"]; name=cfg["deployed"]
        info={"version":version,"family":cfg["head_family"],"weight":cfg["expert_weight"],"rows":len(run.selection[1]),
              "parent":cfg["scores"]["parent"]["primary"],"selected":cfg["scores"][name]["primary"],
              "weakest":cfg["scores"][name]["worst"],"deployed":name,
              "candidate_npz":sum(p.stat().st_size for p in run.candidate.glob("*.npz"))}
    (OUT/("full-"+version+".json")).write_text(json.dumps(info,indent=2))
    print(info)


def hot():
    import os
    os.sched_setaffinity(0,{max(os.sched_getaffinity(0))})
    base=ROOT/"校赛/work/A"
    m=Model.load(base/"checkpoint.npz")
    from industrial import dataset
    previous=reference()
    def read_rows(kind):
        with (base/"public"/(kind+".jsonl")).open() as f:
            return dataset([json.loads(line) for line in f],m)
    x,y=read_rows("train");v,_=read_rows("validation")
    same=previous.fit_candidates(m,x,y,1)
    new=gas.fit_candidates(m,x,y,1)
    extra=np.vstack((np.zeros(128),v[:4]*120,v[5:9]*.001,-v[10:14]))
    check=np.vstack((v,extra))
    sample=v[np.random.default_rng(SEED).permutation(len(v))[:256]]
    info={}
    for label,h in [(f"{ver}-{k}",hd) for ver,models in (("v3",same),("v4",new)) for k,hd in models.items()]:
        scalar=gas.prepare_inference(m,h)
        eager=gas.predict(m,h,check)
        fast=np.stack([scalar(row) for row in check])
        diff=float(np.max(abs(eager-fast)))
        np.testing.assert_allclose(fast,eager,atol=1e-9,rtol=1e-9)
        tests={"rows":len(check),"max_difference":diff,"argmax_changes":int(np.sum(eager.argmax(1)!=fast.argmax(1)))}
        times=[[],[]];before=lambda r:gas.predict(m,h,r[None])
        for repeat in range(5):
            order=(0,1) if repeat%2==0 else (1,0)
            for opt in order:
                fn=(before,scalar)[opt]
                for warmup in sample[:4]: fn(warmup)
                start=time.process_time()
                for r in sample: fn(r)
                times[opt].append(1e6*(time.process_time()-start)/len(sample))
        tests["median_eager_us"],tests["median_compiled_us"]=map(float,np.median(times,axis=1))
        info[label]=tests
    older=load_module("v2_for_format",HERE.parent/"V2/gas.py")
    for name,h in older.fit_candidates(m,x,y,1).items():
        old_score=older.predict(m,h,v);score=gas.predict(m,h,v)
        info["v2_"+name]={"plan_fallback":gas.prepare_inference(m,h) is None,
            "max_difference":float(np.max(abs(score-old_score)))}
    (OUT/"single-inference.json").write_text(json.dumps(info,indent=2))
    print(info)


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("action");a=p.parse_args()
    if a.action in ("development","terminal"): day_study(a.action=="terminal")
    elif a.action=="paired-days": paired_days()
    elif a.action=="hot": hot()
    elif a.action=="cost": resource_pairs()
    elif a.action.startswith("full-"): full_round(a.action.split("-")[1])
    elif a.action=="ds-ps": print(dsps())
    else: replay(a.action)
