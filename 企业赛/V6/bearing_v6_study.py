"""Cross-bearing lifetime-transfer study with training-only lifecycle supervision."""
from concurrent.futures import ThreadPoolExecutor
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import json
import os
import subprocess
import sys

import numpy as np

HERE=Path(__file__).resolve().parent
OUT=HERE.parent/"work/v6/bearing"
OUT.mkdir(parents=True,exist_ok=True)
LEGACY=HERE.parent/"V5"
def module(name,path):
    spec=spec_from_file_location(name,path)
    value=module_from_spec(spec)
    spec.loader.exec_module(value)
    return value
v5=module("frozen_bearing_v5",LEGACY/"bearing.py")

def track_intervals(y):
    # Ordered labels of a run-to-failure trace decrease between snapshots.
    # If a release is not chronological, skip cross-bearing estimation.
    delta=np.diff(y)
    if len(y)<32 or np.mean(delta>=0)>.14:
        return []
    boundaries=np.r_[0,np.flatnonzero(delta>0)+1,len(y)]
    tracks=[slice(a,b) for a,b in zip(boundaries[:-1],boundaries[1:]) if b-a>=10]
    if not 3<=len(tracks)<=80 or sum(s.stop-s.start for s in tracks)<.85*len(y):
        return []
    return tracks

def signal(x,bands):
    lo,hi=np.split(np.log(np.maximum(x[:,bands],1e-12)),2,axis=1)
    return np.column_stack((lo.mean(1),hi.mean(1),hi.mean(1)-lo.mean(1),
                            lo[:,0]-lo[:,1],hi[:,0]-hi[:,1]))

def augment(model,h,x,y,route):
    if route=="baseline":return h
    near=float(np.mean(y<=.02*np.quantile(y,.9)))
    if route in ("combined","safepair"):
        if near<.025:return augment(model,h,x,y,"ramp0.1")
        return augment(model,h,x,y,"taper")
    if route.startswith("taper"):
        if near<.025:
            if near<=0:return {"backbone_only":np.asarray(1)}
            return h
        strength=.75*(1.-np.clip((near-.04)/.025,0.,1.))
        stabilized=augment(model,h,x,y,"isotropic"+str(strength))
        return stabilized
    if route.startswith("ramp"):
        if near>=.025:return augment(model,h,x,y,"isopart0.75")
        floor=float(route[4:])
        h["prefix_fraction"]=np.array(floor+(1.-floor)*near/.025)
        return h
    if route.startswith(("conditional","minimum","bands")):
        if near>=.025:
            return augment(model,h,x,y,"isopart0.75")
        if route=="minimum" or route=="bands":
            if near>0:return h
            # Guard the first head when the start has no failure-period evidence.
            # Before applying any correction, check that at least half the visible 
            # correction magnitudes are small relative to the source target scale.
            raw=v5.predict(model,h,x);base=model.predict(x)
            d=raw-base
            h["risk_median"]=np.asarray(np.median(abs(d))/max(1.,y.std()))
            if route=="bands":
                h["risk_median"]=np.array(np.quantile(abs(d),.85)/max(1.,y.std()))
            if h["risk_median"]>.4:return {"backbone_only":np.asarray(1)}
            return h
        if near<=0.:return {"backbone_only":np.asarray(1)}
        return h
    if route.startswith(("defer","ladder","maturity")):
        # No fitted RUL correction for a right-censored release without terminal support.
        if near<(0.01 if route=="defer1" else .025 if route=="defer25" else .0025):
            return {"backbone_only":np.asarray(1)}
        if route in ("ladder","maturity") and .025<=near<.05:
            return augment(model,h,x,y,"partial0.75" if route=="ladder" else "bounded0.75")
        return h
    if route.startswith(("table","partial","isotropic","isopart")):
        if near<.025 or (near>=.05 and route.startswith(("partial","isopart"))):return h
        tracks=track_intervals(y)
        if not tracks:return h
        old=v5.predict(model,h,x); base=model.predict(x)
        lo,hi=np.split(np.log(np.maximum(x[:,h["curve_bands"]],1e-12)),2,axis=1)
        a=lo.mean(1);b=(hi-lo).mean(1)
        if route.startswith(("isotropic","isopart")):a=base
        cuts=np.quantile(a,[.20,.40,.60,.80]); knots=np.quantile(b,[.20,.40,.60,.80])
        cells=(np.searchsorted(cuts,a)*5+np.searchsorted(knots,b))
        target=y-old
        votes=[]
        for tr in tracks:
            num=np.bincount(cells[tr],weights=target[tr],minlength=25).reshape(5,5)
            den=np.bincount(cells[tr],minlength=25).reshape(5,5)
            # Adjacent rank cells borrow half weight in each axis.
            ker=np.array([.5,1.,.5])
            def smear(values):
                a=np.apply_along_axis(lambda a:np.convolve(a,ker,"same"),0,values)
                return np.apply_along_axis(lambda a:np.convolve(a,ker,"same"),1,a)
            votes.append(smear(num)/(smear(den)+8.))
        votes=np.array(votes)
        correction=np.median(votes,axis=0)*abs(np.sign(votes).mean(0))**2
        strength=float(route.removeprefix("table").removeprefix("partial").removeprefix("isotropic").removeprefix("isopart"))
        h.update(safe_cuts=cuts,safe_knots=knots,safe_bias=strength*np.clip(correction,-.2*y.std(),.2*y.std()),
                 safe_base=np.array(route.startswith(("isotropic","isopart"))))
        return h
    if route.startswith(("prefix","hybrid","envelope")):
        if route.startswith("hybrid") and near>=.025 and near<.05:
            return augment(model,h,x,y,"bounded0.75")
        # A censored release cannot calibrate the direction of large corrections.
        if near>=.025:return h
        target=v5.predict(model,h,x)-model.predict(x)
        if route.startswith("envelope"):
            # Use training-only RMS strata to estimate a conditional movement scale.
            amp=x[:,h["rms_columns"]].max(1)
            cuts=np.quantile(amp,[.25,.50,.75])
            ids=np.searchsorted(cuts,amp)
            limits=np.array([np.quantile(abs(target[ids==i]),.55) if np.any(ids==i) else np.quantile(abs(target),.55) for i in range(4)])
            h.update(prefix_cuts=cuts,prefix_local=np.maximum(1.,limits))
        weight=float(route.removeprefix("prefix").removeprefix("hybrid").removeprefix("envelope"))
        h.update(prefix_degree=np.array(weight*(1.-near/.025)),prefix_scale=np.array(max(1.,.3*np.std(y))))
        return h
    s=track_intervals(y)
    if not s or "curve_bands" not in h:return h
    data=signal(x,h["curve_bands"])
    lo,hi=np.quantile(data,[.20,.80],axis=0)
    scale=np.maximum(hi-lo,.3)
    centers=[]; spans=[]; relative=[]
    for sl in s:
        loc=np.arange(sl.start,sl.stop);mx=y[sl].max()
        if mx<=0:continue
        for block in np.array_split(loc,min(5,len(loc)//6)):
            centers.append(np.median(data[block],axis=0));spans.append(mx)
            relative.append(np.median(y[block])/mx)
    centers=np.asarray(centers)
    if not len(centers):return h
    variance=np.median([np.median(((data[sl]-np.median(data[sl],axis=0))/scale)**2,axis=0) for sl in s],axis=0)
    h.update(stage_centers=centers/scale,stage_scale=scale,stage_var=np.clip(variance,.15,1.5),
             stage_span=np.array(spans),stage_fraction=np.array(relative))
    h["stage_route"]=np.asarray(route)
    if route.startswith("consensus") or route.startswith("bounded"):
        # Estimate how the held-out trajectories disagree about a local head error.
        old=v5.predict(model,h,x)
        estimates=[]
        for sl in s:
            delta=y[sl]-old[sl]
            d=((centers[:,None,:]-data[None,sl,:])/scale)**2
            weights=np.exp(-.5*np.minimum(60.,(d/h["stage_var"]).mean(2)))
            estimates.append((weights@delta)/(weights.sum(1)+1e-8))
        vote=np.asarray(estimates)
        h["stage_error"]=np.median(vote,axis=0)
        h["stage_agree"]=abs(np.sign(vote).mean(0))
        h["stage_cap"]=np.asarray(max(1.,.20*np.std(y)))
    return h

def stage_outputs(h,x):
    f=signal(x,h["curve_bands"])/h["stage_scale"]
    # 5 source-stage centroids per available device. No target identifiers needed.
    distances=np.zeros((len(f),len(h["stage_span"])))
    for i in range(f.shape[1]):
        distances+=(f[:,i,None]-h["stage_centers"][None,:,i])**2/h["stage_var"][i]
    distances/=f.shape[1]
    w=np.exp(-np.minimum(60.,distances))
    prox=np.exp(-.5*np.min(distances,axis=1)); w/=np.maximum(w.sum(1,keepdims=True),1e-25)
    return w,prox

def predict(model,h,x):
    if "backbone_only" in h:return model.predict(x)
    old=v5.predict(model,h,x)
    if "prefix_fraction" in h:
        base=model.predict(x);return np.maximum(0.,base+h["prefix_fraction"]*(old-base))
    if "safe_bias" in h:
        base=model.predict(x)
        lo,hi=np.split(np.log(np.maximum(x[:,h["curve_bands"]],1e-12)),2,axis=1)
        a=base if h["safe_base"] else lo.mean(1)
        ix=np.searchsorted(h["safe_cuts"],a-1e-10);iy=np.searchsorted(h["safe_knots"],(hi-lo).mean(1)-1e-10)
        bias=h["safe_bias"][ix,iy]
        diff=base-old
        return np.maximum(0.,old+np.sign(diff)*np.minimum(abs(diff),np.maximum(0.,np.sign(diff)*bias)))
    if "prefix_degree" in h:
        base=model.predict(x)
        diff=old-base
        scale=h["prefix_scale"]
        if "prefix_local" in h:
            ids=np.searchsorted(h["prefix_cuts"],x[:,h["rms_columns"]].max(1))
            scale=h["prefix_local"][ids]
        small=diff/(1.+abs(diff)/scale)
        return np.maximum(0.,old+h["prefix_degree"]*(small-diff))
    if "stage_route" not in h:return old
    w,prox=stage_outputs(h,x); mode=str(h["stage_route"])
    if mode.startswith("stage"):
        life=(w@np.sqrt(h["stage_span"]*h["stage_fraction"]))**2
        return np.maximum(0.,old+float(mode[5:])*prox*(life-old))
    if mode.startswith("consensus") or mode.startswith("bounded"):
        correction=np.clip(w@(h["stage_error"]*h["stage_agree"]**2),-h["stage_cap"],h["stage_cap"])
        if mode.startswith("bounded"):
            base=model.predict(x)
            correction=np.sign(base-old)*np.minimum(abs(base-old),np.maximum(0,correction*np.sign(base-old)))
        return np.maximum(0.,old+float(mode.removeprefix("consensus").removeprefix("bounded"))*prox*correction)
    raise ValueError(mode)

def build(model,x,y,i,route):
    return augment(model,v5.fit(model,x,y,i),x,y,route)

def replay(route,k,setting):
    if not 0<=k<=13:raise ValueError("Study uses released development orders 0–13")
    runner=module("development_bench",HERE/"benchmark.py")
    # Head ablations share the same V5 controller and V6 replay protocol.
    sys.path.insert(0,str(LEGACY))
    import contest_submission
    impl=module("tested_bearing",HERE/"bearing.py")
    if route!="frozen":
        if hasattr(impl,"fit_candidates"):del impl.fit_candidates
        impl.fit=lambda m,x,y,i:build(m,x,y,i,route)
        impl.predict=predict
        if route in ("safepair","initialpair"):
            def mature_options(m,x,y,i):
                h=v5.fit(m,x,y,i); new=augment(m,dict(h),x,y,"combined")
                if (route=="safepair" or i==1) and np.mean(y<=.02*np.quantile(y,.9))>=.05 and "safe_bias" in new:
                    return {"bearing":h,"consensus":new}
                return {"bearing":new}
            impl.fit_candidates=mature_options
        if route.startswith("optional"):
            def options(m,x,y,i):
                h=v5.fit(m,x,y,i)
                near=np.mean(y<=.02*np.quantile(y,.9))
                if near<.0025 and route=="optionalDefer":
                    return {"bearing":{"backbone_only":np.asarray(1)}}
                calibrated=augment(m,dict(h),x,y,"isopart0.75")
                if "safe_bias" not in calibrated:return {"bearing":h}
                return {"bearing":h,"consensus":calibrated}
            impl.fit_candidates=options
    sys.modules["bearing"]=impl
    runner.OUT=OUT/route
    runner.run("B","v6",k,setting)

def fixture():
    return module("bearing_fixtures",HERE.parent/"V3/bearing_v3_study.py")

def folds(routes):
    from industrial import Model
    import contest_submission as cs
    helper=fixture(); start=Model.load(helper.loader.DATA/"checkpoint.npz")
    result={}
    def evaluate(model,x,y,q,t):
        base=v5.fit(model,x,y,1); record={}
        for mode in routes:
            if mode=="frozen":
                current=module("production",HERE/"bearing.py"); head=current.fit(model,x,y,1);values=current.predict(model,head,q)
            else:
                head=augment(model,dict(base),x,y,mode);values=predict(model,head,q)
            delta=t-values
            record[mode]={"rmse":float(np.sqrt(np.mean(delta**2))),"mae":float(np.mean(abs(delta)))}
        return record
    for name,(x,y,g) in (("XJTU",helper.loader.load("train",start)),("FEMTO",helper.external(start))):
        rows={}
        for device in np.unique(g):
            t=g!=device
            m=Model("regression",start.meta["paths"],[],x[t],y[t],hidden=24)
            m.fit(x[t],y[t],15);cs.continue_mlp("B",1,m,x[t],y[t],"half")
            rows[str(device)]=evaluate(m,x[t],y[t],x[~t],y[~t])
        result[name]={"folds":rows,"summary":{}}
        for mode in routes:
            a=np.array([record[mode]["rmse"] for record in rows.values()])
            b=np.array([record["baseline"]["rmse"] for record in rows.values()])
            boot=np.random.default_rng(606).choice(a-b,(4000,len(b))).mean(1)
            result[name]["summary"][mode]={"mean":float(a.mean()),"worst":float(a.max()),"wins":int(np.sum(a<b-1e-8)),"paired_ci":np.quantile(boot,[.025,.975]).tolist()}
    x,y,_=helper.loader.load("train",start);q,t,_=helper.loader.load("validation",start)
    cs.continue_mlp("B",1,start,x,y,"half");result["public"]=evaluate(start,x,y,q,t)
    extra=module("bearing_ext",HERE.parent/"V4/bearing_v4_study.py");extra.OUT=OUT
    q,t=extra.kaist_samples(start,helper);result["KAIST"]=evaluate(start,x,y,q,t)
    (OUT/("folds-"+"-".join(routes)+".json")).write_text(json.dumps(result,indent=2))

def sweep(routes,keys,settings):
    def job(entry):
        route,k,s=entry
        if (OUT/route/f"{s}-v6-B-{k}.json").exists():return
        with (OUT/f"{route}-{s}-{k}.log").open("w") as log:
            subprocess.run([sys.executable,"-B",str(__file__),"one",route,str(k),s],stdout=log,stderr=subprocess.STDOUT,check=True)
    with ThreadPoolExecutor(max_workers=int(os.environ.get("BEARING_WORKERS","6"))) as pool:
        list(pool.map(job,[(r,k,s) for k in keys for s in settings for r in routes]))

def verify():
    """Compare raw and prepared inference, complete first release and isolated single-core timings."""
    import hashlib, time
    sys.path.insert(0,str(LEGACY))
    from industrial import Model
    import contest_submission as cs
    helper=fixture();helper.OUT=OUT
    model=Model.load(helper.loader.DATA/"checkpoint.npz")
    x,y,g=helper.loader.load("train",model);q,t,_=helper.loader.load("validation",model)
    fem,life,device=helper.external(model)
    cs.continue_mlp("B",1,model,x,y,"half")
    current=module("frozen_bearing",HERE/"bearing.py")
    results={"sha256":hashlib.sha256((HERE/"bearing.py").read_bytes()).hexdigest(),"conditions":{}}
    for name,xx,yy,query in (("complete",x,y,q),("censored",x[y>100],y[y>100],q),
                             ("FEMTO",fem,life,fem),("permuted",x[::-1],y[::-1],q)):
        old=v5.fit(model,xx,yy,1);new=current.fit(model,xx,yy,1)
        proto=augment(model,dict(old),xx,yy,"combined")
        expected=predict(model,proto,query)
        dedicated=current.predict(model,new,query)
        fn=current.prepare_inference(model,new)
        single=np.array([fn(z) for z in query])
        previous=current.prepare_inference(model,old)
        legacy=v5.prepare_inference(model,old)
        np.testing.assert_allclose(dedicated,expected,rtol=1e-11,atol=1e-8)
        np.testing.assert_allclose(single,dedicated,rtol=1e-11,atol=1e-8)
        np.testing.assert_allclose([previous(z) for z in query],[legacy(z) for z in query],rtol=1e-11,atol=1e-8)
        result={"max_prepared_error":float(max(abs(single-dedicated))),"max_study_error":float(max(abs(expected-dedicated))),
                "prefix_fraction":float(new.get("prefix_fraction",1.)),"residual_table":"safe_bias" in new,
                "public_n":len(query),"timings":{}}
        samples=[query[j] for j in np.arange(1024)%len(query)]
        costs={tag:{"fit":[],"row":[]} for tag in ("v5","v6")}
        for trial in range(7):
            methods=[("v5",v5,legacy,old),("v6",current,fn,new)]
            for tag,algo,inference,head in (methods if trial%2==0 else methods[::-1]):
                start=time.process_time();algo.fit(model,xx,yy,1);costs[tag]["fit"].append(time.process_time()-start)
                start=time.process_time();[inference(z) for z in samples]
                costs[tag]["row"].append((time.process_time()-start)*1e6/len(samples))
                result["timings"][tag]={"fit_ms":1000*np.median(costs[tag]["fit"]),
                    "one_us":np.median(costs[tag]["row"]),"head_bytes":sum(v.nbytes for v in head.values())}
        results["conditions"][name]=result
    # The legacy helper imports the current module; install each optional candidate provider in turn.
    orig=getattr(helper.bearing,"fit_candidates",None)
    results["complete"]={}
    try:
        for tag,engine in (("v5",v5),("v6",current)):
            if hasattr(engine,"fit_candidates"):helper.bearing.fit_candidates=engine.fit_candidates
            elif hasattr(helper.bearing,"fit_candidates"):delattr(helper.bearing,"fit_candidates")
            results["complete"][tag]=helper.fixed_round(engine)
    finally:
        if orig is not None:helper.bearing.fit_candidates=orig
    (OUT/"verified.json").write_text(json.dumps(results,indent=2))

def summarize():
    """Compare the available matched orders for each experiment; verify the final full panel."""
    measures=("mean","worst","stage_mean","stage_floor","forget","delay")
    summary={}
    for folder in sorted(OUT.iterdir()):
        if not folder.is_dir() or folder.name=="baseline":continue
        rows={}
        for setting in ("schedule","drift","trace","official","fresh"):
            matched=[]
            for k in range(14):
                name=f"{setting}-v6-B-{k}.json"
                paths=[OUT/"baseline"/name,folder/name]
                if not all(f.is_file() for f in paths):continue
                matched.append((k,[json.loads(f.read_text()) for f in paths]))
            if not matched:continue
            compare={"splits":[k for k,_ in matched],"paired":{}}
            for i,tag in enumerate(("v5","new")):
                vals=[pair[i] for _,pair in matched]
                compare["paired"][tag]={m:float(np.mean([v[m] for v in vals])) for m in measures}
                compare["paired"][tag]["lowest"]=min(v["worst"] for v in vals)
            for m in ("stage_mean","stage_floor"):
                diffs=np.array([pair[1][m]-pair[0][m] for _,pair in matched])
                bounds=np.random.default_rng(606).choice(diffs,(4000,len(diffs))).mean(1)
                compare[m+"_difference"]={"wins":int(sum(diffs>1e-10)),"ties":int(sum(abs(diffs)<=1e-10)),
                                          "order_paired_interval":np.quantile(bounds,[.025,.975]).tolist()}
            rows[setting]=compare
        if rows:summary[folder.name]=rows
    if any(len(summary["frozen"][k]["splits"])!=14 for k in ("schedule","drift","trace","official","fresh")):
        raise ValueError("70 complete frozen ten-round streams required")
    (OUT/"summary.json").write_text(json.dumps(summary,indent=2))

if __name__=="__main__":
    action,*args=sys.argv[1:]
    if action=="summarize":summarize()
    elif action=="verify":verify()
    elif action=="one":replay(args[0],int(args[1]),args[2])
    elif action=="sweep":sweep(args[0].split(","),list(map(int,args[1].split(","))),args[2].split(","))
    elif action=="folds":folds(args[0].split(","))
