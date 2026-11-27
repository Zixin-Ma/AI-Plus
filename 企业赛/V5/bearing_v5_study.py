"""Bearing V5 experiment: released end-of-life evidence and local excess-error shrinkage."""
from pathlib import Path
from importlib.util import module_from_spec, spec_from_file_location
import json
import os
import sys
import subprocess
from concurrent.futures import ThreadPoolExecutor
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
OUT=HERE.parent/"work/v5/bearing"
OUT.mkdir(exist_ok=True,parents=True)


def module(name,path):
    spec=spec_from_file_location(name,path)
    mod=module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


reference=module("reference_bearing",HERE.parent/"V4/bearing.py")


def evidence(y,x,head,rule):
    cutoff=.02*np.quantile(y,.9)
    count=int(np.sum(y<=cutoff))
    terminal=x[y==0.,:][:,head["curve_bands"]]
    landmarks=len(np.unique(terminal,axis=0))
    if rule=="endpoints":
        return landmarks>=3 and count>=12 and count/len(y)>=.025
    if rule=="wilson":
        # One-sided Wilson bound estimates how precisely the reference coverage is observed.
        z=1.64; p=count/len(y); n=len(y)
        upper=(p+z*z/(2*n)+z*np.sqrt(p*(1-p)/n+z*z/(4*n*n)))/(1+z*z/n)
        return landmarks>=3 and upper>=.05
    raise ValueError(rule)


def features(x,h):
    bands=np.log(np.maximum(x[:,h["curve_bands"]],1e-12))
    return np.stack((bands[:,:2].mean(1),bands[:,2:].mean(1)),axis=1)


def cells(f,knots,d):
    a=np.searchsorted(knots[0],f[:,0]); b=np.searchsorted(knots[1],f[:,1])
    return (a*4+b)*2+(d>=0)


def fit(model,x,y,index,route):
    h=reference.fit(model,x,y,index)
    if route=="baseline": return h
    parts=route.split("+")
    if parts[0] in ("endpoints","wilson") and not h["curve_mature"] and evidence(y,x,h,parts[0]):
        h["curve_mature"]=np.asarray(True)
        h["curve_base_cut"]=np.asarray(np.median(model.predict(x)))
        h["curve_base_width"]=np.asarray(max(1.,y.std()*.5))
    mode=parts[-1]
    if mode.startswith("trust"):
        # Train the safety correction on visible source labels. For sparse cells use the unchanged head.
        old=reference.predict(model,h,x);base=model.predict(x)
        d=old-base
        f=features(x,h)
        knots=np.quantile(f,[.25,.50,.75],axis=0).T
        ids=cells(f,knots,d)
        ss=np.bincount(ids,weights=d*d,minlength=32)
        num=np.bincount(ids,weights=d*(y-base),minlength=32)
        reg=24.*np.mean(d*d)+1e-9
        raw=np.clip((num+reg)/(ss+reg),0.,1.)
        h.update(trust_knots=knots,trust_level=1.-float(mode[5:])*(1.-raw))
    elif mode.startswith("proxy"):
        # Score each possible direction against a locally supported residual range of the old backbone.
        base=model.predict(x)
        target=y-base
        f=features(x,h)
        knots=np.quantile(f,[.20,.5,.80],axis=0).T
        ids=cells(f,knots,np.ones(len(x)))//2
        global_band=np.quantile(target,[.15,.85])
        bounds=[]
        for group in range(16):
            values=target[ids==group]
            weight=len(values)/(len(values)+30)
            empirical=np.quantile(values,[.15,.85]) if len(values) else global_band
            bounds.append(empirical*weight+global_band*(1-weight))
        h.update(proxy_knots=knots,proxy_bounds=np.asarray(bounds),proxy_strength=np.asarray(float(mode[5:])))
    return h


def predict(model,h,x):
    v=reference.predict(model,h,x)
    if "trust_level" in h:
        b=model.predict(x); delta=v-b
        return b+delta*h["trust_level"][cells(features(x,h),h["trust_knots"],delta)]
    if "proxy_bounds" in h:
        b=model.predict(x); delta=v-b
        ix=cells(features(x,h),h["proxy_knots"],np.ones(len(x)))//2
        lo,hi=h["proxy_bounds"][ix].T
        clamped=np.clip(delta,np.minimum(0,lo),np.maximum(0,hi))
        return v+h["proxy_strength"]*(clamped-delta)
    return v


def replay(route,split,setting):
    if split not in range(11): raise ValueError("Development splits 0–10 only")
    bench=module("v5_benchmark",HERE/"benchmark.py")
    sys.path.insert(0,str(HERE.parent/"V4"))
    import contest_submission
    expert=module("testing_bearing",HERE/"bearing.py")
    if route!="frozen":
        expert.fit=lambda m,x,y,i:fit(m,x,y,i,route)
        expert.predict=predict
    sys.modules["bearing"]=expert
    bench.OUT=OUT/route
    bench.run("B","v5",split,setting)


def sweep(routes,splits,settings):
    def work(job):
        route,k,policy=job
        path=OUT/route/f"{policy}-v5-B-{k}.json"
        if path.exists(): return
        with (OUT/f"{route}-{policy}-{k}.log").open("w") as log:
            subprocess.run([sys.executable,"-B",str(Path(__file__).resolve()),"one",route,str(k),policy],
                           check=True,stdout=log,stderr=subprocess.STDOUT)
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(work,[(route,k,s) for route in routes for k in splits for s in settings]))


def land():
    return module("fixtures",HERE.parent/"V3/bearing_v3_study.py")


def folds(routes):
    from industrial import Model
    import contest_submission as cs
    f=land(); f.OUT=OUT
    start=Model.load(f.loader.DATA/"checkpoint.npz")
    output={}
    def evaluate(m,x,y,q,z):
        preds={}
        for route in routes:
            h=fit(m,x,y,1,route)
            v=predict(m,h,q)
            preds[route]={"rmse":float(np.sqrt(np.mean((z-v)**2))),"mae":float(np.mean(abs(z-v))),
                          "mature":bool(h["curve_mature"]),"near_end":float(np.mean(y<=.02*np.quantile(y,.9)))}
        return preds
    for name,(x,y,g) in (("XJTU",f.loader.load("train",start)),("FEMTO",f.external(start))):
        devices={}
        for dev in np.unique(g):
            mask=g!=dev
            m=Model("regression",start.meta["paths"],[],x[mask],y[mask],hidden=24)
            m.fit(x[mask],y[mask],15)
            cs.continue_mlp("B",1,m,x[mask],y[mask],"half")
            devices[str(dev)]=evaluate(m,x[mask],y[mask],x[~mask],y[~mask])
        output[name]={"folds":devices,"summary":{}}
        for route in routes:
            scores=np.array([v[route]["rmse"] for v in devices.values()])
            baseline=np.array([v["baseline"]["rmse"] for v in devices.values()])
            diff=scores-baseline
            boot=np.random.default_rng(505).choice(diff,(4000,len(diff))).mean(1)
            output[name]["summary"][route]={"mean":float(scores.mean()),"worst":float(scores.max()),
                "wins":int(sum(diff< -1e-8)),"difference_interval":np.quantile(boot,[.025,.975]).tolist()}
    x,y,_=f.loader.load("train",start); q,z,_=f.loader.load("validation",start)
    cs.continue_mlp("B",1,start,x,y,"half")
    output["public"]=evaluate(start,x,y,q,z)
    ext=module("external_loader_v4",HERE.parent/"V4/bearing_v4_study.py")
    ext.OUT=OUT
    xx,yy=ext.kaist_samples(start,f);output["kaist"]=evaluate(start,x,y,xx,yy)
    (OUT/"folds.json").write_text(json.dumps(output,indent=2))



def confirm():
    """Recompute the selected production head, including old-head compatibility and branch boundaries."""
    import time
    import hashlib
    sys.path.insert(0,str(HERE.parent/"V4"))
    import contest_submission as cs
    from industrial import Model
    current=module("v5_frozen",HERE/"bearing.py")
    helpers=land()
    helpers.OUT=OUT
    template=Model.load(helpers.loader.DATA/"checkpoint.npz")
    x,y,g=helpers.loader.load("train",template)
    xt,yt,_=helpers.loader.load("validation",template)
    xf,yf,gf=helpers.external(template)
    cs.continue_mlp("B",1,template,x,y,"half")
    output={"sha256":hashlib.sha256((HERE/"bearing.py").read_bytes()).hexdigest(),
            "complete":{name:helpers.fixed_round(model) for name,model in (("v4",reference),("v5",current))},
            "boundaries":{},"predictions":{}}
    rng=np.random.default_rng(1435)
    chosen=x[rng.choice(len(x),600,replace=False)]
    for label,rows,nnear,distinct,expected in (("censored",600,17,0,False),("duplicate_endpoints",600,17,1,False),
                 ("two_endpoints",600,17,2,False),("third_unique",600,17,3,True),("too_few_late",300,11,3,False),
                 ("below_fraction",600,14,3,False),("mature_without_zero",600,30,0,True)):
        xx=chosen[:rows].copy(); life=np.full(rows,100.); life[:nnear]=1.
        if distinct:
            life[:3]=0.
            xx[:3]=xx[np.arange(3)%distinct]
        h=current.fit(template,xx,life,1)
        assert bool(h["curve_mature"])==expected
        output["boundaries"][label]=expected
    for label,xx,yy,q in (("XJTU",x,y,xt),("truncated",x[y>55],y[y>55],xt),
                           ("FEMTO",xf,yf,xf),("FEMTO_hold_one",xf[gf!=gf[0]],yf[gf!=gf[0]],xf[gf==gf[0]])):
        h4=reference.fit(template,xx,yy,1)
        h5=current.fit(template,xx,yy,1)
        expect_h=fit(template,xx,yy,1,"endpoints")
        assert set(h5)==set(expect_h) and all(np.array_equal(v,expect_h[k]) for k,v in h5.items())
        new=current.prepare_inference(template,h5); original=reference.prepare_inference(template,h4)
        qp=q[np.random.default_rng(2026).permutation(len(q))]
        old=np.array([original(v) for v in qp])
        pred=np.array([new(v) for v in qp]);batch=current.predict(template,h5,qp)
        np.testing.assert_allclose(pred,batch,atol=2e-9,rtol=1e-11)
        compatible=current.prepare_inference(template,h4)
        np.testing.assert_allclose(old,[compatible(z) for z in qp],atol=2e-9,rtol=1e-11)
        prof={}
        for tag,algo,fn,h in (("v4",reference,original,h4),("v5",current,new,h5)):
            fits=[]; predictions=[]
            for trial in range(7):
                start=time.process_time_ns();algo.fit(template,xx,yy,1);fits.append((time.process_time_ns()-start)/1e6)
                start=time.process_time_ns()
                for a in qp:fn(a)
                predictions.append((time.process_time_ns()-start)/1e3/len(qp))
            prof[tag]={"fit_ms":float(np.median(fits)),"single_us":float(np.median(predictions)),
                       "head_bytes":sum(v.nbytes for v in h.values()),"mature":bool(h["curve_mature"])}
        output["predictions"][label]={"old_head_compatible":True,"max_delta":float(abs(old-pred).max()),
                                      "batch_error":float(abs(batch-pred).max()),"profile":prof}
    output["streams"]={}
    for setting in ("drift","official","fresh","trace"):
        metrics=[]; choices=0; max_diff=0.
        for k in range(11):
            a,b=(json.loads((OUT/route/f"{setting}-v5-B-{k}.json").read_text()) for route in ("baseline","frozen"))
            values=["mean","worst","forget","delay"]
            metrics.append([b[z] for z in values])
            for aa,bb in zip(a["rounds"],b["rounds"],strict=True):
                choices+=int((aa["deployed"],aa["expert_weight"],aa["head_family"])==
                             (bb["deployed"],bb["expert_weight"],bb["head_family"]))
                max_diff=max(max_diff,max(abs(np.asarray(aa["metrics"])-bb["metrics"])))
        output["streams"][setting]={"equal_choices":choices,"max_rmse_delta":max_diff,
                                    "mean_metrics":dict(zip(values,np.mean(metrics,axis=0)))}
    (OUT/"frozen.json").write_text(json.dumps(output,indent=2))


if __name__=="__main__":
    action=sys.argv[1]
    if action=="confirm":confirm()
    elif action=="one": replay(sys.argv[2],int(sys.argv[3]),sys.argv[4])
    elif action=="sweep":
        sweep(sys.argv[2].split(","),[int(k) for k in sys.argv[3].split(",")],sys.argv[4].split(","))
    elif action=="folds":folds(sys.argv[2].split(","))
