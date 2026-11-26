"""Compare lifetime trust regions, novelty estimates and a compact physical prior."""
import importlib.util
import json
from pathlib import Path
import sys
import time

import numpy as np

import bearing
from industrial import Model

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
V1=HERE.parent/"V1"
OUT=HERE.parent/"work/v2/bearing"


def read_module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


old=read_module("old_bearing",V1/"bearing.py")
loader=read_module("bearing_v1_study",V1/"bearing_study.py")


def fit_support(x, y, parent, state, size=32):
    """Estimate train-only distances to local operating regions and residual envelopes."""
    order=np.argsort(parent,kind="stable")
    bins=np.array_split(order,min(12,len(order)))
    centers=np.array([parent[idx].mean() for idx in bins])
    spread=np.array([np.quantile(y[idx]-parent[idx],[.10,.90]) for idx in bins])
    dist, keep=np.unique(centers,return_index=True)
    rng=np.random.default_rng(107)
    keep_rows=rng.choice(len(y),min(2048,len(y)),replace=False)
    s=np.sign(x[keep_rows])*np.log1p(np.abs(x[keep_rows]))
    z=np.clip((s-state["center"])/state["scale"],-6,6)
    centers=z[rng.choice(len(z),min(size,len(z)),replace=False)].copy()
    for _ in range(6):
        d=np.maximum(np.sum(z*z,axis=1)[:,None]+np.sum(centers*centers,axis=1)-2*z@centers.T,0.)
        ids=np.argmin(d,axis=1)
        for j in range(len(centers)):
            seen=ids==j
            if np.any(seen): centers[j]=z[seen].mean(axis=0)
    d=np.maximum(np.sum(z*z,axis=1)[:,None]+np.sum(centers*centers,axis=1)-2*z@centers.T,0.)
    ids=np.argmin(d,axis=1)
    typical=np.empty(len(centers))
    fallback=np.quantile(d.min(axis=1),.65)
    for j in range(len(centers)):
        sample=d[ids==j,j]
        typical[j]=np.quantile(sample,.65) if len(sample)>=8 else fallback
    return dict(prediction_knots=dist, residual_band=spread[keep], operating_centers=centers,
                operating_norm=np.sum(centers*centers,axis=1),operating_radius=np.maximum(typical,.1))


def support_weight(x, state):
    s=np.sign(x)*np.log1p(np.abs(x))
    z=np.clip((s-state["center"])/state["scale"],-6,6)
    d=np.maximum(np.sum(z*z,axis=1)[:,None]+state["operating_norm"]-2*z@state["operating_centers"].T,0)
    ids=np.argmin(d,axis=1)
    event=d[np.arange(len(d)),ids]/state["operating_radius"][ids]
    return np.clip((event-.65)/1.7,0.,1.)


def bounds(base,state):
    knots=state["prediction_knots"]
    lo,hi=state["residual_band"].T
    return np.minimum(0.,np.interp(base,knots,lo)),np.maximum(0.,np.interp(base,knots,hi))




def compare(model,x,y,xt,yt,groups,return_scores=False):
    rng=np.random.default_rng(2027)
    ids=rng.choice(len(x),min(768,len(x)),replace=False)
    model.meta["learning_rate"]=.0004
    model.fit(x[ids],y[ids],1)
    state=old.fit(model,x,y,1)
    prior=old.order_predict(xt,state)
    base=model.predict(xt)
    v1=old.predict(model,state,xt)
    state.update(fit_support(x,y,model.predict(x),state))
    novelty=support_weight(xt,state)
    lo,hi=bounds(base,state)
    amp=old._amplitude(xt,state["rms_columns"])
    a,b=state["amplitude_bounds"]
    gate=1-np.clip((amp-a)/max(1e-8,b-a),0,1)
    physics=base+gate*(.5*prior+.1*state["prior"]-.6*base)
    variants={"v1":v1,"backbone":base, "pure_order":physics,
              "novelty":base+(.15+.85*novelty)*(v1-base)}
    for tag,estimate in (("rich",v1),("compact",physics)):
        for scale in (1.,2.,4.):
            variants[f"{tag}_q{scale:g}"]=base+np.clip(estimate-base,scale*lo,scale*hi)
        # More room to repair underestimation, but cap the correction to long predicted lifetimes.
        variants[tag+"_upper_open"]=base+np.maximum(estimate-base,lo)
        variants[tag+"_support_band"]=base+np.clip(estimate-base,(1+2*novelty)*lo,(1+2*novelty)*hi)
        # Protection of familiar observations is strongest at both ends of predicted life.
        variants[tag+"_novelty"]=base+(.12+.88*novelty)*(estimate-base)
    for quantile in (.50,.70,.85):
        cut=np.quantile(model.predict(x),quantile)
        smooth=np.maximum(1.,np.std(y)*.40)
        high=np.clip((base-cut)/smooth,0.,1.)
        delta=v1-base
        for protected in (.75,1.):
            name=f"tail-{quantile:g}-{protected:g}"
            variants[name]=base+delta-high*protected*np.minimum(delta,0)
    probs=np.exp(7*bearing.order_evidence(x,state["order_cols"]))
    probs/=probs.sum(1,keepdims=True)
    vt=np.exp(7*bearing.order_evidence(xt,state["order_cols"]))
    vt/=vt.sum(1,keepdims=True)
    gram=probs.T@probs
    alpha=.015*len(y)
    means=np.linalg.solve(gram+np.eye(3)*alpha,probs.T@np.sqrt(np.maximum(y,0))+alpha*np.sqrt(state["prior"]))
    new=vt@np.sort(np.maximum(0.,means)**2)
    variants["soft-prior"]=v1+.5*gate*(new-prior)
    for frac in (.35,.65):
        cut=np.quantile(model.predict(x),frac)
        high=np.clip((base-cut)/max(1.,np.std(y)*.4),0,1)
        d=variants["soft-prior"]-base
        variants[f"soft-tail-{frac:g}"]=base+d-high*np.minimum(0.,d)
    d=variants["soft-prior"]-base
    for cutq in (.50,.65,.75):
        scale=max(1.,np.std(y)*.4)
        cut=np.quantile(model.predict(x),cutq)
        urgency=np.clip((base-cut)/scale,0.,1.)
        for min_gate in (0.,.35,.65):
            protection=urgency*(1.-(1.-min_gate)*novelty)
            variants[f"support-tail-{cutq:g}-{min_gate:g}"]=base+d-protection*np.minimum(0.,d)
    for cutq in (.50,.65,.72):
        ts=bearing.tail_state(x,y,model.predict(x),state,cutq)
        fit=bearing.familiarity(xt,{**state,**ts})
        high=np.clip((base-ts["tail_cut"])/ts["tail_range"],0,1)
        for blend in (0.,.2):
            protection=high*((1-blend)*fit+blend)
            variants[f"tail-support-{cutq:g}-{blend:g}"]=base+d-protection*np.minimum(0.,d)
    ts=bearing.tail_state(x,y,model.predict(x),state,.6)
    high=np.clip((base-ts["tail_cut"])/ts["tail_range"],0,1)
    for limit in (.45,.7,1.):
        h={**state,**ts,"tail_threshold":ts["tail_threshold"]*limit}
        fit=bearing.familiarity(xt,h)
        for strength in (.5,.75,1.):
            estimate=v1+gate*(strength*(new-state["prior"])-.5*(prior-state["prior"]))
            delta=estimate-base
            variants[f"narrow-{limit:g}-{strength:g}"]=base+delta-high*fit*np.minimum(delta,0.)
    ts=bearing.tail_state(x,y,model.predict(x),state,.6)
    fit=bearing.familiarity(xt,{**state,**ts,"tail_threshold":ts["tail_threshold"]*.60})
    guard=np.clip((base-ts["tail_cut"])/ts["tail_range"],0,1)*fit
    ids=np.random.default_rng(2027).choice(len(y),min(1536,len(y)),replace=False)
    zx=old._design(x[ids],state)
    zv=old._design(xt,state)
    for width in (0,16,32,64):
        for penalty in (.008,.03):
            n=1+x.shape[1]+width
            design=zx[:,:n]
            gram=design.T@design
            gram.flat[::n+1]+=len(ids)*penalty
            coef=np.linalg.solve(gram,design.T@(y[ids]-model.predict(x[ids])))
            estimate=np.maximum(0.,base+.85*(zv[:,:n]@coef))
            raw=(1-.6*gate)*estimate+gate*(.55*new+.05*state["prior"])
            delta=raw-base
            variants[f"lowrank-{width}-{penalty:g}"]=base+delta-guard*np.minimum(delta,0.)
    deployed_state=bearing.fit(model,x,y,1)
    variants["production"]=bearing.predict(model,deployed_state,xt)
    # Public replay includes various combinations of selected rows, but the
    # head's compatibility with old checkpoints can be checked independently.
    if return_scores:
        return variants
    return {name:{str(g):float(np.sqrt(np.mean((pred[groups==g]-yt[groups==g])**2))) for g in np.unique(groups)}
            | {"all":float(np.sqrt(np.mean((pred-yt)**2)))} for name,pred in variants.items()}


def guarded_full():
    import contest_submission as cs
    import guard
    from copy import deepcopy
    m=Model.load(loader.DATA/"checkpoint.npz")
    x,y,_=loader.load("train",m)
    v,t,_=loader.load("validation",m)
    ids=np.linspace(0,len(x)-1,min(768,len(x)),dtype=int)
    probes=np.linspace(0,len(x)-1,min(512,len(x)),dtype=int)
    xx=np.vstack([v,x[ids],x[probes]])
    static=m.predict(xx)
    vals=compare(deepcopy(m),x,y,xx,None,None,True)
    groups=guard.strata("B",m,x,v,t)
    panel=guard.Panel("B",t,groups)
    static_groups=panel.score(static[:len(v),None])
    parent=dict(primary=float(np.mean((static[:len(v)]-t)**2)**.5),floor=0.)
    bins=np.searchsorted(np.unique(np.quantile(y[ids],[.25,.50,.75])),y[ids])
    def errs(v):
        return np.array([np.mean((v[bins==g]-y[ids][bins==g])**2)**.5 for g in np.unique(bins)])
    base_err=errs(static[len(v):len(v)+len(ids)])
    results={}
    for tag,pred in vals.items():
        step=cs.OFFSET["B"]*np.median(abs(y[probes]-pred[-len(probes):]))
        choices=[(False,0.,1.,0.,parent["primary"],0.)]
        for off in (0.,-step,step):
            for wt in cs.WEIGHTS["B"]:
                pr=static*(1-wt)+np.maximum(0.,pred+off)*wt
                gr=panel.score(pr[:len(v),None])
                floor=guard.relative_floor("B",gr,static_groups)
                rmse=float(np.mean((pr[:len(v)]-t)**2)**.5)
                memory=max(errs(pr[len(v):len(v)+len(ids)])-base_err)/m.meta["target_scale"]
                sc=dict(primary=rmse,floor=floor,memory_loss=memory)
                gain=(parent["primary"]-rmse)/parent["primary"]
                choices.append((cs.eligible("B",parent,sc),gain+.45*min(floor,0.),wt,off,rmse,memory))
        eligible,_,wt,off,rmse,memory=max(choices)
        results[tag]=dict(rmse=rmse if eligible else parent["primary"],weight=wt,
                          offset=off if eligible else 0.,memory=memory,eligible=bool(eligible))
    print("full",[(n,round(z["rmse"],2),z["weight"]) for n,z in results.items()])
    (OUT/"guarded.json").write_text(json.dumps(results,indent=2))


def run(external=False):
    OUT.mkdir(parents=True,exist_ok=True)
    template=Model.load(loader.DATA/"checkpoint.npz")
    if external:
        loader.WORK=OUT
        x,y,groups=loader.femto(template)
    else:
        x,y,groups=loader.load("train",template)
    folds={}
    for g in np.unique(groups):
        mask=groups==g
        model=Model("regression",template.meta["paths"],[],x[~mask],y[~mask],hidden=24)
        model.fit(x[~mask],y[~mask],15)
        folds[g]=compare(model,x[~mask],y[~mask],x[mask],y[mask],groups[mask])
    summary={}
    for name in next(iter(folds.values())):
        scores=np.array([z[name]["all"] for z in folds.values()])
        summary[name]=dict(mean=float(scores.mean()),worst=float(scores.max()))
    delta=np.array([f["production"]["all"]-f["v1"]["all"] for f in folds.values()])
    reps=np.random.default_rng(2026).choice(delta,(4000,len(delta)),replace=True).mean(1)
    output={"folds":folds,"summary":summary,"rows":len(x),
            "paired_mean_difference":float(delta.mean()),"devices_improved":int((delta<0).sum()),
            "paired_95percentiles":np.percentile(reps,[2.5,97.5]).tolist()}
    if not external:
        xv,yv,gs=loader.load("validation",template)
        pub=compare(template,x,y,xv,yv,gs)
        output["public"]=pub
        for name in summary:summary[name]["public"]=pub[name]["all"]
    (OUT/("external.json" if external else "loo.json")).write_text(json.dumps(output,indent=2))
    for name,record in summary.items():print(name,*[round(v,2) for v in record.values()])


def deployed_check():
    import tempfile
    import os
    from types import SimpleNamespace
    import benchmark
    import contest_submission as cs
    key="B"
    base=loader.DATA
    def rows(split):
        with (base/"public"/f"{split}.jsonl").open() as f:
            return [{"inputs":r["inputs"],"answer":r["answer"]} for r in map(json.loads,f)]
    training,validation=rows("train"),rows("validation")
    m=Model.load(base/"checkpoint.npz")
    x,y,_=cs.projection(training,m,key)
    xv,yv,_=cs.projection(validation,m,key)
    with tempfile.TemporaryDirectory(dir=OUT) as temp:
        r=benchmark.PublicRound()
        r.agent,r.root,r.dataset_key,r.round_index=cs,Path(temp),key,1
        r.parent,r.records,r.call_ids=benchmark.Parent(base),[],[]
        r.plan="refit|half|evaluate physical prior"; r.prompt_bytes=0
        r.artifacts=[];r.selection=(xv,yv)
        r.data=SimpleNamespace(training_rows=lambda:iter(training),validation_rows=lambda:iter(validation))
        os.environ["CONTEST_WORK"]=temp
        cs.train(r)
        current=base if r.record["selected_artifact_id"]==base.name else r.candidate
        model,head,_=cs.artifact(current)
        rated=cs.values(key,model,head,xv)
        cfg=r.record["configuration"]
        record={"selected":cfg["deployed"],"expert_weight":cfg["expert_weight"],
                "rmse":float(np.sqrt(np.mean((rated[:,0]-yv)**2))),
                "model_calls":len(r.call_ids),"upload_npz_bytes":sum(f.stat().st_size for f in r.candidate.glob("*.npz"))}
    ix=np.random.default_rng(2027).choice(len(x),min(768,len(x)),replace=False)
    m.meta["learning_rate"]=.0004
    m.fit(x[ix],y[ix],1)
    state=old.fit(m,x,y,1)
    head=bearing.fit(m,x,y,1)
    matrix,phase=bearing.feature_map(head)
    assert np.array_equal(matrix,state["projection"]) and np.array_equal(phase,state["phase"])
    # Verify single-row predictions across the whole observed range, with an intervening cache clear.
    query=np.r_[x[::15],xv[::4]]
    bat=bearing.predict(m,head,query)
    bearing._feature_map.cache_clear()
    scalar=np.array([bearing.predict(m,head,z[None])[0] for z in query])
    np.testing.assert_allclose(scalar,bat,rtol=1e-11,atol=1e-9)
    record["batch_rows_checked"]=len(query)
    record["max_single_vs_batch_error"]=float(np.max(np.abs(scalar-bat)))
    timing={}
    for name,algo in (("v1",old),("v2",bearing)):
        fits=[];inferences=[]
        for _ in range(5):
            begin=time.process_time();head=algo.fit(m,x,y,1);fits.append(time.process_time()-begin)
            begin=time.process_time()
            for row in xv:algo.predict(m,head,row[None])
            inferences.append(1e6*(time.process_time()-begin)/len(xv))
        timing[name]=dict(head_bytes=sum(v.nbytes for v in head.values()),
                         fit_seconds=float(np.median(fits)),single_us=float(np.median(inferences)))
    record["timing"]=timing
    (OUT/"deployed.json").write_text(json.dumps(record,indent=2))
    print(json.dumps(record,ensure_ascii=False))


def transition_sweep(kind,cap):
    """Override only the diagnostic of the SDK entry for a paired development ablation."""
    import benchmark
    import contest_submission as cs
    release_only=kind.startswith("release-")
    metric=kind.split("-")[-1]
    if metric not in ("global","quartile","upper") or cap<=0:
        raise ValueError("Use global, quartile, upper, or a release- variant and a positive cap.")
    def movement(key,changed,has_head,pred,parent,bins,scale):
        if key!="B" or not has_head or (release_only and not changed):
            return 0.
        diff=(pred-parent).ravel()
        if metric=="global":
            size=np.sqrt(np.mean(diff**2))
        elif metric=="upper":
            size=np.quantile(abs(diff),.80)
        else:
            size=max(abs(np.mean(diff[bins==g])) for g in np.unique(bins))
        return float(size/max(scale,1.)*(.15/cap))
    cs.release_movement=movement
    benchmark.OUT=OUT/f"transition-{kind}-{cap:g}"
    for split in (0,1,2):
        for mode in ("drift","official"):
            benchmark.run("B","v2",split,mode)


def memory_sweep(memory,offset):
    import benchmark
    import contest_submission as cs
    cs.release_movement=lambda *args:0.
    cs.MEMORY["B"]=memory
    cs.OFFSET["B"]=offset
    benchmark.OUT=OUT/f"guard-{memory}-{offset}"
    for split in (0,1,2):
        for mode in ("drift","official"):
            benchmark.run("B","v2",split,mode)


if __name__=="__main__":
    if "--transition" in sys.argv: transition_sweep(sys.argv[-2],float(sys.argv[-1]))
    elif "--memory" in sys.argv: memory_sweep(float(sys.argv[-2]),float(sys.argv[-1]))
    elif "--deploy" in sys.argv: deployed_check()
    elif "--guard" in sys.argv: guarded_full()
    else: run("--external" in sys.argv)
