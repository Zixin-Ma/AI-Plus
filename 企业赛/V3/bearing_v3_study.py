"""Development and external evidence for stage-sensitive bearing RUL heads."""
from datetime import datetime
from importlib.util import module_from_spec, spec_from_file_location
from itertools import islice
from pathlib import Path
from zipfile import ZipFile
import io
import json
import os
import tempfile
from types import SimpleNamespace

import numpy as np

import bearing
from industrial import Model, dataset

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = HERE.parent / "work/v3/bearing"
OUT.mkdir(exist_ok=True, parents=True)


def source(name, path):
    spec = spec_from_file_location(name, path)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


v2 = source("bearing_v2", HERE.parent/"V2/bearing.py")
loader = source("v1_loader", HERE.parent/"V1/bearing_study.py")


def cols(model, name):
    return np.asarray([i for i,p in enumerate(model.meta["paths"]) if json.loads(p)[-1][1] == name])


def health(x, ix):
    logs = np.log(np.maximum(x[:,ix], 1e-12))
    return np.column_stack((logs[:,:2].mean(1), logs[:,2:4].mean(1), logs[:,4:6].mean(1)))


def orders(x, ix):
    evidence = v2.order_evidence(x, ix)
    posterior = np.exp(7.*(evidence-evidence.max(1,keepdims=True)))
    return posterior / posterior.sum(1,keepdims=True)


def weighted_quantile(vals, ids, weights, fractions):
    w = np.cumsum(weights[ids])
    return np.interp(fractions*w[-1], w, vals[ids])


def build(model, x, y):
    ix=np.r_[cols(model,"band3_energy_6k_12k"),cols(model,"band1_energy_0_2k"),cols(model,"kurtosis")]
    freq=cols(model,"dominant_freq")
    physical=health(x,ix)
    weights=orders(x,freq)
    frac=np.linspace(.025,.975,13)
    y_order=np.argsort(y)
    knots=np.empty((3,3,len(frac)))
    means=np.zeros_like(knots)
    typical=[]
    # Increasing high-frequency impact energy indicates lower remaining life.
    # A smooth decreasing life curve is constrained inside each load/speed region.
    for k in range(3):
        ww=weights[:,k]+1e-6
        typical.append(weighted_quantile(y,y_order,ww,np.array([.1,.25,.5,.75,.9,.98])))
        for f in range(3):
            feature=physical[:,f]
            order=np.argsort(feature)
            knots[k,f]=weighted_quantile(feature,order,ww,frac)
            for i,center in enumerate(knots[k,f]):
                radius=max((knots[k,f,min(len(frac)-1,i+2)]-knots[k,f,max(0,i-2)])/2,.05)
                local=ww*np.exp(-.5*((feature-center)/radius)**2)
                means[k,f,i]=(local@y+8*(ww@y/ww.sum()))/(local.sum()+8)
    typical=np.asarray(typical)
    return dict(curve_ix=ix, curve_freq=freq, curve_knots=knots,curve_life=means,curve_rates=typical)


def options(model, x, state):
    feat=health(x,state["curve_ix"])
    wt=orders(x,state["curve_freq"])
    # Integrate per-speed conditional curves; alternatives probe strong/weak physical ordering.
    vals={}
    frac=np.linspace(.025,.975,13)
    for f in range(3):
        qs=np.stack([np.interp(feat[:,f],state["curve_knots"][k,f],frac) for k in range(3)],axis=1)
        cond=np.stack([np.interp(feat[:,f],state["curve_knots"][k,f],state["curve_life"][k,f]) for k in range(3)],axis=1)
        vals["observed"+str(f)]=(cond*wt).sum(1)
        # Lifetime scale estimates for long-running bearings under each regime.
        for q in (2,4,5):
            span=np.sort(state["curve_rates"][:,q]) / ([.1,.25,.5,.75,.9,.98][q])
            vals[f"rank{f}q{q}"]=((1-qs)*span*wt).sum(1)
    vals["multirank"]=(.5*vals["rank0q4"]+.35*vals["rank1q4"]+.15*vals["rank2q4"])
    # Operating-regime evidence partly identifies long/short life but it can be ambiguous.
    for pw in (1.,2.,4.):
        sharply=wt**pw
        sharply/=sharply.sum(1,keepdims=True)
        for q in (1,2,3,4):
            vals[f"regime{pw:g}q{q}"]=sharply@np.sort(state["curve_rates"][:,q])
    return vals


def fold(model,x,y,q,t,groups):
    import contest_submission as cs
    cs.continue_mlp("B",1,model,x,y,"half")
    h=v2.fit(model,x,y,1); old=v2.predict(model,h,q)
    a=build(model,x,y)
    preds=options(model,q,a)
    candidates={"v2":old,"mlp":model.predict(q)}
    for key,pred in preds.items():
        for factor in (.35,.7,1.):
            candidates[f"{key}-{factor:g}"]=(1-factor)*old+factor*pred
    return {key:{str(g):float(np.mean((v[groups==g]-t[groups==g])**2)**.5) for g in np.unique(groups)}
            | {"all":float(np.mean((v-t)**2)**.5)} for key,v in candidates.items()}


def external(m):
    path=HERE.parent/"work/v2/bearing/femto-stride10-v1.npz"
    if path.exists():
        with np.load(path,allow_pickle=False) as a:
            return tuple(a[k] for k in ("x","y","g"))
    loader.WORK=OUT
    return loader.femto(m)


def kaist(m):
    cache=HERE.parent/"work/v3/bearing/kaist-finite-hourly.npz"
    if cache.exists():
        with np.load(cache,allow_pickle=False) as a:
            return a["x"],a["y"]
    archive=next((ROOT/"Supp_Datasets/B/02_KAIST_Run_to_Failure").glob("*.zip"))
    times=[]; rows=[]
    with ZipFile(archive) as z:
        names=sorted(n for n in z.namelist() if n.endswith(".csv"))
        for n in names:
            times.append(datetime.strptime(Path(n).stem,"LogFile_%Y-%m-%d-%H-%M-%S"))
            with z.open(n) as inp:
                frame=np.loadtxt(io.BytesIO(b"".join(islice(inp,16384))),delimiter=",",usecols=(0,1))
            valid=np.isfinite(frame).all(1)
            edges=np.flatnonzero(np.diff(np.r_[False,valid,False]))
            start=next(a for a,b in edges.reshape(-1,2) if b-a>=8192)
            frame=frame[start:start+8192]
            rows.append({"inputs":loader.extract_features(frame,25600.),"answer":0.})
    xx,_=dataset(rows,m)
    y=np.array([(times[-1]-t).total_seconds()/60 for t in times])
    np.savez_compressed(cache,x=xx,y=y)
    return xx,y


def panel():
    template=Model.load(loader.DATA/"checkpoint.npz")
    x,y,g=loader.load("train",template)
    xv,yv,gv=loader.load("validation",template)
    xf,yf,gf=external(template)
    results={}
    for tag,xt,yt,gt in (("xjtu",x,y,g),("femto",xf,yf,gf)):
        folds={}
        for name in np.unique(gt):
            ix=gt!=name
            model=Model("regression",template.meta["paths"],[],xt[ix],yt[ix],hidden=24)
            model.fit(xt[ix],yt[ix],15)
            folds[str(name)]=fold(model,xt[ix],yt[ix],xt[~ix],yt[~ix],gt[~ix])
        results[tag]={key:{"mean":float(np.mean(v)),"worst":max(v)} for key in next(iter(folds.values()))
                      if (v:=[f[key]["all"] for f in folds.values()])}
        (OUT/f"{tag}-folds.json").write_text(json.dumps(folds))
    results["official"]=fold(template,x,y,xv,yv,gv)
    (OUT/"panel.json").write_text(json.dumps(results,indent=2))
    kx,ky=kaist(template)
    h=v2.fit(template,x,y,1); old=v2.predict(template,h,kx)
    opt=options(template,kx,build(template,x,y))
    from statistics import correlation
    def rank(a):
        _,idx,counts=np.unique(a,return_inverse=True,return_counts=True)
        return (counts.cumsum()-.5*counts)[idx]
    out={"n":len(ky),"target_span_minutes":float(ky.max()),"models":{}}
    for name,pred in {"v2":old,**opt}.items():
        out["models"][name]={"rmse":float(np.mean((pred-ky)**2)**.5),
                             "rank":float(correlation(rank(ky),rank(pred))) if np.unique(pred).size>1 else 0.}
    (OUT/"kaist.json").write_text(json.dumps(out,indent=2))



def route_fit(model,x,y,index,route):
    if route.startswith("impact"):
        _,branch,limit=route.split("-")
        state=route_fit(model,x,y,index,"joint-step-0.05" if branch in ("supported","strict") else "joint-step-0.025")
        hf=health(x,state["curve_ix"])[:,0]
        state["impact_bounds"]=np.quantile(hf,[.2,float(limit)])
        state["choice"]=np.asarray(route)
        return state
    if route.startswith("joint"):
        _,branch,threshold=route.split("-")
        cov=min(1.,np.mean(y<=.02*np.quantile(y,.9))/float(threshold))
        state=route_fit(model,x,y,index,"safe-twoway-0.7")
        state["transition"]=np.asarray(cov)
        state["choice"]=np.asarray(route)
        return state
    if route.startswith("mature"):
        _,mode,strength,power=route.split("-")
        coverage=min(1.,np.mean(y<=.02*np.quantile(y,.9))/.025)
        if coverage<.001: return v2.fit(model,x,y,index)
        adjusted=f"safe-{mode}-{float(strength)*coverage**float(power):g}"
        return route_fit(model,x,y,index,adjusted)
    if route=="v2":
        return v2.fit(model,x,y,index)
    state=v2.fit(model,x,y,index)
    if route.startswith("level"):
        ix=cols(model,"dominant_freq")
        weights=orders(x,ix)
        order=np.argsort(y)
        quantiles=np.asarray([.10,.25,.5,.75,.9])
        shape=np.vstack([weighted_quantile(y,order,ww,quantiles) for ww in weights.T])
        state.update(levels=np.sort(shape,axis=0),choice=np.asarray(route))
    else:
        state.update(build(model,x,y))
        state["choice"]=np.asarray(route)
    if route.startswith("safe"):
        base=model.predict(x)
        state.update(protect_cut=np.asarray(np.quantile(base,.5)),protect_width=np.asarray(max(1.,y.std()*.5)),short_life=np.sort(state["curve_rates"][:,1]))
    return state


def route_predict(model,state,x):
    if state.get("backbone_only",False): return model.predict(x)
    old=v2.predict(model,state,x)
    if "choice" not in state: return old
    route=str(state["choice"])
    if route.startswith("impact"):
        _,branch,limit=route.split("-")
        opts=options(model,x,state)["rank1q2"]
        wt=orders(x,state["order_cols"])
        low,high=state["impact_bounds"]
        damage=np.clip((health(x,state["curve_ix"])[:,0]-low)/max(high-low,1e-8),0,1)
        mature=protected("safe-twoway-0.7",model,state,x,old,opts,wt)
        initial=protected("safe-distinct-1",model,state,x,old,opts,wt)
        if branch=="always": alpha=1.
        else: alpha=float(state["transition"]>=1.)
        if branch=="full": return old+(1-damage)*((1-alpha)*initial+alpha*mature-old)
        if branch=="both": return old+(1-damage)**2*((1-alpha)*initial+alpha*mature-old)
        return (1-alpha)*initial+alpha*(old+(1-damage)*(mature-old))
    if route.startswith("joint"):
        _,branch,threshold=route.split("-")
        opts=options(model,x,state)["rank1q2"]
        wt=orders(x,state["order_cols"])
        initial=protected("safe-distinct-1",model,state,x,old,opts,wt)
        mature=protected("safe-twoway-0.7",model,state,x,old,opts,wt)
        alpha=float(state["transition"])
        if branch=="step": alpha=float(alpha>=1.)
        if branch=="smooth": alpha=alpha*alpha*(3-2*alpha)
        return (1-alpha)*initial+alpha*mature
    if route.startswith("safe"):
        return protected(route,model,state,x,old,options(model,x,state)["rank1q2"],orders(x,state["order_cols"]))
    if route.startswith("level"):
        _,q,w=route.split("-")
        if len(x)==1:
            a,b=state["order_cols"]
            weights=np.array([v2._pair_weights(float(x[0,a]),float(x[0,b]))])
        else:
            weights=orders(x,state["order_cols"])
        pred=weights @ state["levels"][:,int(q)]
    else:
        key,w=route.split("-")
        pred=options(model,x,state)[key]
    return old + float(w)*(pred-old)



def protected(route, model, state, x, old, pred, weights):
    """Restrict the healthy-life prior by mechanical order and familiar long-life predictions."""
    parts=route.split("-")
    mode,w=parts[1],float(parts[2])
    # Long-life corrections are most defensible when the longest-life load/speed order is supported.
    confidence=weights[:,2]
    base=model.predict(x)
    long=np.clip((base-state["protect_cut"])/state["protect_width"],0.,1.)
    if mode in ("increase","asymmetric","distinct","confident","directional"):
        above=np.maximum(0,pred-old)
        below=np.minimum(0,pred-old)
        low=weights @ state["short_life"]
        pi=confidence**2
        new=pi*above
        if mode!="increase": new+=pi*below*(.25 if mode=="asymmetric" else 1.)
        if mode=="distinct": new+=.4*(weights[:,0]**2)*np.minimum(0,low-old)
        if mode=="directional": new+=.5*(weights[:,0]**2)*np.minimum(0,low-old)+.5*confidence*(1-confidence)*above
        return old+w*new
    if mode=="long":
        return old+w*confidence*(pred-old)
    delta=pred-old
    if mode in ("floor","twoway","balance","soft"):
        delta=np.maximum(delta,0)*confidence + np.minimum(delta,0)*(1-long)
    else:
        delta=delta*(1-long)
    if mode in ("twoway","balance","soft"):
        low=weights @ state["short_life"]
        modifier=(1-confidence)*(1-long)
        if mode=="soft": modifier*=1-confidence
        delta+=modifier*np.minimum(0,low-old)*(1. if mode=="balance" else .5)
    return old+w*delta


def test_protected(model,x,y,q,t,groups):
    import contest_submission as cs
    cs.continue_mlp("B",1,model,x,y,"half")
    h=route_fit(model,x,y,1,"safe-floor-0.7")
    old=v2.predict(model,h,q)
    opts=options(model,q,h)
    wt=orders(q,h["order_cols"])
    candidates={"v2":old}
    for mode in ("long","floor","twoway","balance","soft","cap","increase","asymmetric","distinct","confident","directional"):
        for w in (.4,.7,1.):
            label=f"safe-{mode}-{w:g}"
            candidates[label]=protected(label,model,h,q,old,opts["rank1q2"],wt)
    return {key:{str(g):float(np.mean((v[groups==g]-t[groups==g])**2)**.5) for g in np.unique(groups)}
            | {"all":float(np.mean((v-t)**2)**.5)} for key,v in candidates.items()}


def test_impact(model,x,y,q,t,groups):
    import contest_submission as cs
    cs.continue_mlp("B",1,model,x,y,"half")
    state=route_fit(model,x,y,1,"impact-full-0.75")
    old=v2.predict(model,state,q)
    candidates={"v2":old}
    for branch in ("full","always","both"):
        for upper in (.45,.60,.75,.9):
            route=f"impact-{branch}-{upper:g}"
            state["choice"]=np.asarray(route)
            state["impact_bounds"]=np.quantile(health(x,state["curve_ix"])[:,0],[.2,upper])
            candidates[route]=route_predict(model,state,q)
    return {key:{str(g):float(np.mean((v[groups==g]-t[groups==g])**2)**.5) for g in np.unique(groups)}
            | {"all":float(np.mean((v-t)**2)**.5)} for key,v in candidates.items()}


def fixed_round(engine):
    """Compare the selected public checkpoint on exactly one full, metadata-free public release."""
    import benchmark
    import contest_submission as cs
    def read(split):
        with (loader.DATA/"public"/(split+".jsonl")).open() as f:
            return [{"inputs":r["inputs"],"answer":r["answer"]} for r in map(json.loads,f)]
    train,valid=read("train"),read("validation")
    m=Model.load(loader.DATA/"checkpoint.npz")
    xv,yv,_=cs.projection(valid,m,"B")
    original_fit,original_predict=bearing.fit,bearing.predict
    try:
        bearing.fit,bearing.predict=engine.fit,engine.predict
        with tempfile.TemporaryDirectory(dir=OUT) as tmp:
            r=benchmark.PublicRound()
            r.agent,r.root,r.dataset_key,r.round_index=cs,Path(tmp),"B",1
            r.parent,r.records,r.call_ids=benchmark.Parent(loader.DATA),[],[]
            r.plan="refit|half|evaluate RUL conditional on vibration";r.prompt_bytes=0
            r.artifacts=[];r.selection=(xv,yv)
            r.data=SimpleNamespace(training_rows=lambda:iter(train),validation_rows=lambda:iter(valid))
            os.environ["CONTEST_WORK"]=tmp
            cs.train(r)
            selected=loader.DATA if r.record["selected_artifact_id"]==loader.DATA.name else r.candidate
            fitted,head,_=cs.artifact(selected)
            pred=cs.values("B",fitted,head,xv)[:,0]
            cfg=r.record["configuration"]
            return dict(rmse=float(np.sqrt(np.mean((pred-yv)**2))),expert_weight=cfg["expert_weight"],
                        deployed=cfg["deployed"],calls=len(r.call_ids),
                        upload_bytes=sum(f.stat().st_size for f in r.candidate.glob("*.npz")))
    finally:
        bearing.fit,bearing.predict=original_fit,original_predict


def final_check():
    from time import process_time
    from statistics import correlation
    import hashlib
    import contest_submission as cs
    model=Model.load(loader.DATA/"checkpoint.npz")
    x,y,g=loader.load("train",model)
    xv,yv,gv=loader.load("validation",model)
    cs.continue_mlp("B",1,model,x,y,"half")
    kx,ky=kaist(model)
    def rank(a):
        _,idx,counts=np.unique(a,return_inverse=True,return_counts=True)
        return (counts.cumsum()-.5*counts)[idx]
    result={"source":hashlib.sha256((HERE/"bearing.py").read_bytes()).hexdigest(),
            "kaist_rows":len(ky),"kaist_span_hours":float(ky.max()/60)}
    for name,engine in (("v2",v2),("v3",bearing)):
        ft=[];pt=[]
        for _ in range(7):
            start=process_time(); h=engine.fit(model,x,y,1);ft.append(process_time()-start)
            start=process_time()
            for row in xv:engine.predict(model,h,row[None])
            pt.append(1e6*(process_time()-start)/len(xv))
        result[name]={"fit_seconds":float(np.median(ft)), "single_us":float(np.median(pt)),
                      "head_bytes":sum(z.nbytes for z in h.values())}
        pred=engine.predict(model,h,kx)
        result[name]["kaist"]={"rmse_minutes":float(np.sqrt(np.mean((ky-pred)**2))),
                               "spearman":float(correlation(rank(ky),rank(pred)))}
        p=engine.predict(model,h,xv)
        result[name]["public"]={str(a):float(np.sqrt(np.mean((p[gv==a]-yv[gv==a])**2))) for a in np.unique(gv)}
        result[name]["public"]["all"]=float(np.sqrt(np.mean((p-yv)**2)))
        result[name]["selected"]=fixed_round(engine)
    for label,mask in (("mature",np.ones(len(y),dtype=bool)),("truncated",y>55)):
        prod=bearing.fit(model,x[mask],y[mask],1)
        ref=route_fit(model,x[mask],y[mask],1,"impact-supported-0.75")
        expect=route_predict(model,ref,xv)
        bat=bearing.predict(model,prod,xv)
        one=np.array([bearing.predict(model,prod,z[None])[0] for z in xv])
        np.testing.assert_allclose(one,expect,rtol=1e-10,atol=1e-9)
        result[label]={"batch_vs_reference":float(np.max(abs(bat-expect))),"single_vs_batch":float(np.max(abs(one-bat)))}
    for tag in ("xjtu","femto"):
        folds=json.loads((OUT/"impact"/(tag+"-folds.json")).read_text())
        diffs=[v["impact-full-0.75"]["all"]-v["v2"]["all"] for v in folds.values()]
        vals=np.random.default_rng(206).choice(diffs,(4000,len(diffs))).mean(1)
        result[tag]={"mean_diff":float(np.mean(diffs)),"improved":int(sum(v<0 for v in diffs)),
                     "devices":len(diffs),"paired_device_bootstrap":np.quantile(vals,[.025,.975]).tolist()}
    (OUT/"final.json").write_text(json.dumps(result,indent=2))
    print(json.dumps(result,ensure_ascii=False,indent=2))


def stream(route, split, setting):
    import benchmark
    import contest_submission as cs
    cs.ANCHOR_TASKS.discard("B")
    cs.CONTEXT_TASKS.discard("B")
    cs.SWITCH_SE=0.
    # Patches are confined to this fresh CLI process and never enter a submitted artifact.
    bearing.fit=lambda model,x,y,index:route_fit(model,x,y,index,route)
    bearing.predict=route_predict
    if route.startswith("paired"):
        _,mode,target=route.split("_")
        bearing.fit_candidates=lambda m,x,y,k:{name:route_fit(m,x,y,k,name) for name in (mode,target)}
    benchmark.OUT=OUT/route
    benchmark.run("B","v3",split,setting)


def stream_panel(routes, settings, splits):
    import subprocess, sys
    from concurrent.futures import ThreadPoolExecutor
    def worker(item):
        route,split,setting=item
        log=OUT/f"{route}-{setting}-{split}.log"
        with log.open("w") as f:
            subprocess.run([sys.executable,"-B",str(Path(__file__).resolve()),"stream",route,str(split),setting],
                           check=True, stdout=f,stderr=f)
    tasks=[(route,split,setting) for route in routes for setting in settings for split in splits]
    with ThreadPoolExecutor(max_workers=4) as executor:
        list(executor.map(worker,tasks))

if __name__=="__main__":
    import sys
    if len(sys.argv)==1: panel()
    elif sys.argv[1]=="grid":
        import contest_submission as cs
        out={}
        for weights in ((1.,.65,.35,.15),(1.,.9,.85,.8,.75,.65,.35,.15)):
            cs.WEIGHTS["B"]=weights
            for strength in (.18,.20,.22,.25):
                cs.MEMORY["B"]=strength
                out[str(weights)+" "+str(strength)]=fixed_round(bearing)
        (OUT/"interpolation.json").write_text(json.dumps(out,indent=2))
    elif sys.argv[1]=="diagnose":
        import contest_submission as cs
        base_rule=cs.eligible
        sampled=[]
        def record(key,parent,candidate):
            ok=base_rule(key,parent,candidate)
            if key=="B" and "memory_loss" in candidate:
                sampled.append({k:candidate[k] for k in ("primary","floor","memory_loss","release_movement")} | {"allowed":bool(ok)})
            return ok
        cs.eligible=record
        fixed_round(bearing)
        (OUT/"gate-diagnostics.json").write_text(json.dumps(sampled,indent=2))
    elif sys.argv[1]=="check": final_check()
    elif sys.argv[1]=="impact":
        fold=test_impact
        OUT=OUT/"impact"; OUT.mkdir(exist_ok=True)
        panel()
    elif sys.argv[1]=="protect":
        fold=test_protected
        OUT=OUT/"safety"; OUT.mkdir(exist_ok=True)
        panel()
    elif sys.argv[1]=="stream":
        split=int(sys.argv[3])
        if split not in range(5): raise ValueError("Development studies use splits 0 to 4")
        stream(sys.argv[2],split,sys.argv[4])
    elif sys.argv[1]=="sweep": stream_panel(sys.argv[2].split(","),sys.argv[3].split(","),[int(a) for a in sys.argv[4].split(",") if 0<=int(a)<=4])
