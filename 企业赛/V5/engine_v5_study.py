"""Independent engine-head tests on public development schedules and whole units."""
import argparse
import concurrent.futures
import subprocess
import importlib.util
import json
from pathlib import Path
from time import process_time
import sys
import numpy as np

import benchmark
import engine
import contest_submission as harness

HERE=Path(__file__).resolve().parent
OUT=HERE.parent/"work/v5/engine"


def legacy(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    mod=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


v4=legacy("reference_engine",HERE.parent/"V4/engine.py")
diagnostics=legacy("diagnostics",HERE.parent/"V3/engine_v3_study.py")


def gradient(e,route):
    if route=="base":
        return e
    if route=="huber":
        return .35*e+.65*np.clip(e,-1.25,1.25)
    if route.startswith("tilt"):
        strength=float(route[4:])/100
        # Hotter than raw PHM risk; normalize upper and lower tails separately
        # to keep rare long-lived units from moving the whole operating regime.
        w=np.exp(np.clip(np.where(e>0,e/.85,-e/1.11),0,3.2))
        pos=e>0
        for use in (pos,~pos):
            if np.any(use):
                w[use]/=w[use].mean()
        return e*(1+strength*(w-1))
    if route.startswith("cvar"):
        strength=float(route[4:])/100
        scaled=np.where(e>0,e,-e/1.3)
        extreme=scaled>=np.quantile(scaled,.85)
        return e*(1+strength*(extreme.astype(np.float32)/max(.02,extreme.mean())-1))
    if route.startswith("gdr"):
        # Larger signed tail errors exert a bounded additional tail pressure.
        strength=float(route[3:])/100
        q=np.minimum(np.where(e>0,e,-e/1.3),3.)
        return e*(1+strength*(q-np.mean(q)))
    raise ValueError(route)

def neural(model, x, y, limit=24576, epochs=20, width=48, ordinal_weight=.7, objective="base", seed=145, keep_logits=False):
    rng = np.random.default_rng(seed)
    if len(x) > limit:
        ids = rng.choice(len(x), limit, replace=False)
        x, y = x[ids], y[ids]
    centers, counts = np.unique(np.round(x[:, [-3, -1]]), axis=0, return_counts=True)
    centers = centers[np.argsort(-counts)[:8]]
    groups = v4._regimes(x, centers)
    means = np.stack([x[groups == k, :-3].mean(0) for k in range(len(centers))])
    scales = np.stack([x[groups == k, :-3].std(0) for k in range(len(centers))])
    inv_scale = 1 / np.maximum(scales, 1e-4)
    sensors = np.clip((x[:, :-3] - means[groups]) * inv_scale[groups], -6, 6)
    base = model.predict(x).reshape(-1)
    design = np.column_stack((sensors, (base - 100) / 60, np.eye(len(centers))[groups])).astype(np.float32)
    target = ((y - base) / 55).astype(np.float32)
    thresholds = np.quantile(y, [.20, .45, .70])
    near_threshold = (y[:, None] < thresholds).astype(np.float32)

    weights = rng.normal(0, .22, (design.shape[1], width)).astype(np.float32)
    bias = np.zeros(width, dtype=np.float32)
    # Column zero predicts the standardised residual; other columns supervise
    # rank at three training-only lifetime thresholds and are discarded at serving.
    output = np.zeros((width, 4), dtype=np.float32)
    output[:, 0] = rng.normal(0, .03, width)
    intercept = np.zeros(4, dtype=np.float32)
    params = [weights, bias, output, intercept]
    first = [np.zeros_like(a) for a in params]
    second = [np.zeros_like(a) for a in params]
    step = 0
    for epoch in range(epochs):
        order = rng.permutation(len(y))
        rate = .003 / (1 + epoch / 10)
        for idx in np.array_split(order, max(1, len(y) // 256)):
            batch = design[idx]
            hidden = np.tanh(batch @ weights + bias)
            delta = hidden @ output + intercept
            error = delta[:, 0] - target[idx]
            if objective.startswith(("deploy","joint")):
                # Account for the real inference-time one-sided residual limiter.
                strength=(int(objective[6:]) if objective.startswith("deploy") else int(objective[5:7]))/100
                raw=delta[:,0]
                scale=np.where(raw>0,.50,1.)
                clipped=np.clip(raw,-65/55,65/55)
                slope=scale*np.where(abs(raw)<65/55,1.,.08)
                error=(1-strength)*error+strength*(scale*clipped-target[idx])*slope
                if objective.startswith("joint"):
                    error=gradient(error,"tilt"+objective[7:])
            else:
                error=gradient(error,objective)
            delta[:, 0] = error * (2 / len(idx))
            logits = np.clip(delta[:, 1:], -30., 30.)
            delta[:, 1:] = (1 / (1 + np.exp(-logits)) - near_threshold[idx]) * (ordinal_weight / (3*len(idx)))
            back = (delta @ output.T) * (1 - hidden*hidden)
            gradients = (batch.T @ back, back.sum(0), hidden.T @ delta, delta.sum(0))
            step += 1
            for param, momentum, variance, grad in zip(params, first, second, gradients):
                momentum *= .9
                momentum += .1 * grad
                variance *= .999
                variance += .001 * grad**2
                param -= rate * momentum / (1 - .9**step) / (np.sqrt(variance / (1 - .999**step)) + 1e-8)
    # Keep deployment arithmetic in the same precision as existing checkpoints.
    weights = weights.astype(float)
    bias = bias.astype(float)
    state=dict(centers=centers, means=means, inv_scale=inv_scale,
                w_sensor=weights[:-len(centers)-1].copy(), w_base=weights[-len(centers)-1] / 60,
                bias=bias + weights[-len(centers):] - weights[-len(centers)-1] * (100 / 60),
                output=55 * output[:, 0].astype(float), intercept=55 * intercept[:1].astype(float))
    if keep_logits:
        state.update(ord_out=output[:,1:].astype(float),ord_bias=intercept[1:].astype(float))
    return state


def finish(model,state,x,y):
    ids=np.linspace(0,len(y)-1,min(len(y),2048),dtype=int)
    p=v4.predict_neural(model,state,x[ids],65.)
    a,b=np.quantile(p,[.35,.65])
    state.update(health_lo=np.array(a),health_hi=np.array(b),
                 health_amp=np.array(np.clip(np.std(y)/max(1.,np.std(p))-1.,.15,.7)),
                 extension_fraction=np.array(.50))
    return state


def tree_inputs(h,x,p):
    g=v4._regimes(x,h["centers"])
    z=np.clip((x[:,:-3]-h["means"][g])*h["inv_scale"][g],-5,5)
    return np.column_stack((z,p/55,np.eye(len(h["centers"]))[g]))


def tree_readout(route,model,h,x,y):
    # Neural training and residual calibration draw disjoint released rows when possible.
    ids=diagnostics.calibration_rows(len(y))[:12288]
    xx=x[ids]; labels=y[ids]
    prior=v4.predict(model,h,xx)
    design=tree_inputs(h,xx,prior)
    bins=12; depth=3; n,dim=design.shape
    splits=np.quantile(design,np.arange(1,bins)/bins,axis=0).T
    codes=np.stack([np.searchsorted(cuts,design[:,j],side="right") for j,cuts in enumerate(splits)])
    rounds=28 if route=="treehard" else 14
    features=np.zeros((rounds,depth),dtype=int)
    values=np.zeros_like(features,dtype=float)
    leaves=np.zeros((rounds,2**depth))
    residual=labels-prior
    for i in range(rounds):
        # Gentle bilateral emphasis and Huber controls keep the gradient bounded.
        w=np.ones(n)
        if route=="treerisk": w+=np.clip(np.where(residual<0,-residual/45,residual/58),0,2)*.65
        if route=="treerobust": w=np.minimum(1.,65/np.maximum(abs(residual),1))
        weighted=w*residual
        node=np.zeros(n,dtype=int)
        for d in range(depth):
            parents=2**d
            best=(-np.inf,0,0)
            for j in range(dim):
                index=node*bins+codes[j]
                mass=np.bincount(index,weights=w,minlength=parents*bins).reshape(parents,bins)
                total=np.bincount(index,weights=weighted,minlength=parents*bins).reshape(parents,bins)
                left=mass.cumsum(1)[:,:-1]; right=mass.sum(1)[:,None]-left
                upper=total.cumsum(1)[:,:-1]; lower=total.sum(1)[:,None]-upper
                score=(upper**2/(left+32)+lower**2/(right+32)).sum(0)
                k=int(np.argmax(score))
                if score[k]>best[0]: best=(score[k],j,k)
            _,j,k=best
            features[i,d]=j;values[i,d]=splits[j,k]
            node+=((design[:,j]>=splits[j,k]).astype(int)<<d)
        mass=np.bincount(node,weights=w,minlength=2**depth)
        total=np.bincount(node,weights=weighted,minlength=2**depth)
        leaves[i]=.20*total/(32+mass)
        residual-=leaves[i,node]
    h.update(tree_feature=features,tree_threshold=values,tree_leaves=leaves,tree_limit=np.array(14.))
    return h


def tree_predict(m,h,x,p):
    u=tree_inputs(h,x,p)
    f=h["tree_feature"]; cuts=h["tree_threshold"]; leaf=h["tree_leaves"]
    adj=np.zeros(len(x))
    for j in range(len(leaf)):
        node=((u[:,f[j]]>=cuts[j])@np.asarray([1,2,4]))
        adj+=leaf[j,node]
    return np.maximum(0,p+np.clip(adj,-float(h["tree_limit"]),float(h["tree_limit"])))


def ord_design(m,h,x,p):
    b=m.predict(x).reshape(-1)
    g=v4._regimes(x,h["centers"])
    z=np.clip((x[:,:-3]-h["means"][g])*h["inv_scale"][g],-6,6)
    hidden=np.tanh(z@h["w_sensor"]+b[:,None]*h["w_base"]+h["bias"][g])
    logits=hidden@h["ord_out"]+h["ord_bias"]
    q=1/(1+np.exp(-np.clip(logits,-30,30)))
    t=np.clip((p-100)/60,-2.,3.)
    return np.column_stack((np.ones(len(x)),t,q,q*q,q*t[:,None],np.eye(len(h["centers"]))[g]))


def ord_predict(m,h,x,p):
    design=ord_design(m,h,x,p)
    return np.maximum(0.,p+np.clip(design@h["ord_coef"],-10.,10.))


def ord_calibrate(route,m,h,x,y):
    ids=diagnostics.calibration_rows(len(y))
    xx=x[ids]; labels=y[ids]; prior=v4.predict(m,h,xx)
    design=ord_design(m,h,xx,prior); err=prior-labels
    n=len(ids); dim=design.shape[1]
    cov=design.T@design/n+np.eye(dim)*.015
    coef=np.linalg.solve(cov,-design.T@err/n)
    if route in ("ordrisk","ordguard"):
        # Project the squared-error descent direction onto the half-spaces for
        # the two signed PHM tails at two trimming levels, in the design metric.
        directions=[]
        for sign,scale in ((1.,10.),(-1.,13.)):
            exponent=sign*err/scale
            for q in (.97,.997):
                w=sign*np.where(exponent>0,np.exp(np.minimum(exponent,np.quantile(exponent,q))),0.)
                a=w@design/max(1.,np.abs(w).sum())
                d=np.linalg.solve(cov,a)
                directions.append((a,d,max(1e-9,a@d)))
        for k in range(12):
            for a,d,den in directions:
                coef-=max(0.,float(a@coef))/den*d
    candidates=[.7,.4,.2,0.] if route=="ordguard" else [.65]
    for frac in candidates:
        delta=np.clip(design@(coef*frac),-10.,10.)
        if route!="ordguard" or all(np.sum(np.expm1(np.clip(sign*(err+delta)/scale,0.,22.)))<=np.sum(np.expm1(np.clip(sign*err/scale,0.,22.)))*(1+1e-7) for sign,scale in ((1.,10.),(-1.,13.))):
            h["ord_coef"]=coef*frac
            break
    return h


def fit(route,model,x,y,idx,previous=None,stable_release=False):
    if route.startswith("tree"):
        return tree_readout(route,model,v4.fit(model,x,y,idx,previous,stable_release),x,y)
    if route=="base":
        return v4.fit(model,x,y,idx,previous,stable_release)
    long=stable_release and bool(previous)
    track=route in ("modest","coldguard")
    if track:
        prior_size=float((previous or {}).get("fit_count",0))
        modest=prior_size>0 and 0<=len(y)-prior_size<=.04*prior_size
        if route=="coldguard" and previous is None:
            modest=True
        route="ordguard" if (modest or long) else "joint6015"
    if route=="scheduled":
        route="ordguard" if long else "joint6015"
    if route=="ordjoint":
        route="ordjoint" if long else "joint6015"
    seed=145
    obj=route
    if route.startswith("renew") or route.startswith("cycle"):
        rotate=route.startswith("cycle") or long
        if rotate:
            seed=145+31*((idx-1)%5)
        obj=route[5:] or "base"
    if route.startswith("ord"):
        obj="joint6015" if route=="ordjoint" else "tilt15"
    h=neural(model,x,y,epochs=20 if long else 12,width=48 if long else 64,
             ordinal_weight=.2 if long else .7,objective=obj,seed=seed,keep_logits=route.startswith("ord"))
    h=finish(model,h,x,y)
    if route.startswith("ord"):
        h=ord_calibrate("ordguard" if route=="ordjoint" else route,model,h,x,y)
    if track:
        h["fit_count"]=np.array(len(y))
    return h


def predict(model,head,x):
    p=v4.predict(model,head,x)
    if "ord_coef" in head:
        return ord_predict(model,head,x,p)
    return tree_predict(model,head,x,p) if "tree_feature" in head else p


def bind(route):
    for name in ("baseline_model","industrial","guard","policy","serving","contest_submission"):
        sys.modules[name]=legacy(name,HERE.parent/"V4"/(name+".py"))
    if route!="frozen":
        engine.fit=lambda m,x,y,i,previous=None,stable_release=False:fit(route,m,x,y,i,previous,stable_release)
        engine.predict=predict
        engine.prepare_inference=lambda m,h:None


def replay(route,n,setting):
    if n not in range(11):
        raise ValueError("Develop only on published schedules 0 to 10")
    bind(route);benchmark.OUT=OUT/route
    benchmark.run("C","v5",n,setting,True)


def evaluation_data(dataset):
    x,y,audit,age,units,cruise,model=diagnostics.data(dataset)
    if dataset=="DS02":
        # Set external flight-regime bins using only the initial training release.
        # This is a separate domain fit with the compatible sensor representation.
        first=(~audit)&(age<=.55)
        with np.load(HERE.parent/"work/engine/external.npz") as src:
            settings=src["settings"]
        x[:,-3]=np.searchsorted(np.quantile(settings[first,0],[.25,.5,.75]),settings[:,0])
        x[:,-1]=settings[:,1]>np.median(settings[first,1])
        model=harness.Model("regression",model.meta["paths"],[],x[first],y[first],hidden=32,seed=42,batch_size=256)
        model.fit(x[first],y[first],15)
    return x,y,audit,age,units,cruise,model


def probe(dataset,routes):
    x,y,audit,age,units,cruise,model=evaluation_data(dataset)
    stages=[.55,.7,.85,1.] if dataset!="official" else [1.,1.]
    report=[]
    prior={}
    for i,cut in enumerate(stages,1):
        tr=~audit&(age<=cut);xx,yy=x[tr],y[tr]
        for route in routes:
            start=process_time()
            fn=engine.predict if route=="frozen" else predict
            h=(engine.fit if route=="frozen" else lambda *a,**kw:fit(route,*a,**kw))(model,xx,yy,i,previous=prior.get(route),stable_release=dataset=="official" and i>1)
            prior[route]=h
            fit_s=process_time()-start
            p=fn(model,h,x[audit])
            row=diagnostics.metrics(p,y[audit],age[audit],cruise[audit] if cruise is not None else None)
            row.update(round=i,release=cut,route=route,fit_s=fit_s)
            if route=="base":baseline=p
            if i==len(stages):
                ix=np.linspace(0,len(p)-1,min(300,len(p)),dtype=int)
                sample=x[audit][ix]
                start=process_time()
                single=np.asarray([fn(model,h,r[None])[0] for r in sample])
                row["scalar_us"]=1e6*(process_time()-start)/len(ix)
                np.testing.assert_allclose(single,p[ix],atol=1e-9,rtol=1e-11)
                g=units[audit]
                delta=(p-y[audit])**2-(baseline-y[audit])**2
                u=np.unique(g)
                sums=np.array([(delta[g==j].sum(),(g==j).sum()) for j in u])
                idx=np.random.default_rng(310).integers(len(u),size=(1000,len(u)))
                bs=sums[idx].sum(1)
                row["mse_ci95"]=np.quantile(bs[:,0]/bs[:,1],[.025,.975]).tolist()
            report.append(row)
    OUT.mkdir(exist_ok=True,parents=True)
    (OUT/(dataset+"-"+routes[1]+".json")).write_text(json.dumps(report,indent=2))


def grid(routes,settings,splits,workers=8):
    if any(n not in range(11) for n in splits):
        raise ValueError("Released development schedules only")
    def worker(request):
        route,n,setting=request
        dest=OUT/route/f"{setting}-v5-C-{n}.json"
        if dest.exists():
            return
        log=OUT/f"{route}-replay-{n}-{setting}.log"
        with log.open("w") as f:
            subprocess.run([sys.executable,"-Bu",str(Path(__file__).resolve()),"replay",route,str(n),setting],
                           check=True,stdout=f,stderr=subprocess.STDOUT)
    requests=[(r,n,setting) for n in splits for setting in settings for r in routes]
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        list(executor.map(worker,requests))


def verify(expected):
    x,y,audit,age,units,cruise,m=diagnostics.data("official")
    sample=x[audit][np.linspace(0,np.sum(audit)-1,768,dtype=int)]
    fit={}
    report=[]
    for i in (1,2):
        xx=x[~audit]; yy=y[~audit]
        candidate=engine.fit(m,xx,yy,i,previous=fit.get("prod"),stable_release=i==2)
        control=fit_reference(expected,m,xx,yy,i,fit.get("ref"),i==2)
        b=engine.predict(m,candidate,sample)
        alt=predict(m,control,sample)
        np.testing.assert_allclose(b,alt,atol=1e-9,rtol=1e-10)
        fn=engine.prepare_inference(m,candidate)
        scalar=np.array([fn(row) for row in sample])
        np.testing.assert_allclose(b,scalar,atol=1e-9,rtol=1e-10)
        report.append({"round":i,"rows":len(b),"reference_max_abs":float(np.max(abs(b-alt))),
                       "fast_max_abs":float(np.max(abs(b-scalar))),"head_bytes":sum(a.nbytes for a in candidate.values())})
        fit={"prod":candidate,"ref":control}
    (OUT/(expected+"-runtime.json")).write_text(json.dumps(report,indent=2))


fit_reference=fit


def speed():
    import os
    if hasattr(os,"sched_setaffinity"):
        os.sched_setaffinity(0,{max(os.sched_getaffinity(0))})
    x,y,audit,_,_,_,m=evaluation_data("official")
    xx,yy=x[~audit],y[~audit]; probe=x[audit][np.linspace(0,np.sum(audit)-1,1024,dtype=int)]
    result=[]
    for rep in range(3):
        for route,mod in ((("base",v4),("frozen",engine)) if rep%2==0 else (("frozen",engine),("base",v4))):
            prev=None
            for r in (1,2):
                start=process_time(); h=mod.fit(m,xx,yy,r,previous=prev,stable_release=r==2)
                fit_s=process_time()-start; prev=h
                single=mod.prepare_inference(m,h); target=mod.predict(m,h,probe)
                times=[]
                for _ in range(5):
                    start=process_time()
                    values=np.asarray([single(row) for row in probe])
                    times.append(1e6*(process_time()-start)/len(probe))
                np.testing.assert_allclose(values,target,atol=1e-9,rtol=1e-10)
                result.append(dict(rep=rep,route=route,round=r,fit_s=fit_s,single_us=float(np.median(times)),
                                   delta=float(np.max(abs(values-target))),head_bytes=sum(a.nbytes for a in h.values())))
    (OUT/"speed.json").write_text(json.dumps(result,indent=2))


def summarize():
    groups={}
    for dest in sorted(p for p in OUT.iterdir() if p.is_dir()):
        item={}
        for setting in ("drift","fresh","trace","official"):
            files=sorted(dest.glob(f"{setting}-v5-C-*.json"),key=lambda p:int(p.stem.split("-")[-1]))
            if not files: continue
            data=[json.loads(p.read_text()) for p in files]
            result={k:float(np.mean([d[k] for d in data])) for k in ("mean","worst","forget","delay")}
            result["lowest"]=min(d["worst"] for d in data)
            result["nasa_mean"]=float(np.mean([r["nasa_mean"] for d in data for r in d["rounds"]]))
            result["splits"]=[int(p.stem.split("-")[-1]) for p in files]
            result["selected"]=np.mean([sum(r["deployed"]=="candidate" for r in d["rounds"]) for d in data])
            result["fit_cpu_per_stream"]=float(np.mean([sum(r["cpu"] for r in d["rounds"]) for d in data]))
            item[setting]=result
        if item: groups[dest.name]=item
    (OUT/"summary.json").write_text(json.dumps(groups,indent=2))
    pairs={}
    for setting in ("drift","trace","official","fresh"):
        splits=range(11) if setting in ("drift","trace") else (0,5,10)
        result={}
        if any(not (OUT/route/f"{setting}-v5-C-{i}.json").exists() for route in ("base","scheduled") for i in splits):
            continue
        for route in ("base","scheduled"):
            streams=[json.loads((OUT/route/f"{setting}-v5-C-{i}.json").read_text()) for i in splits]
            stats={k:[d[k] for d in streams] for k in ("mean","worst","forget","delay")}
            stats["nasa"]=[np.mean([r["nasa_mean"] for r in d["rounds"]]) for d in streams]
            stats["selected"]=[sum(r["deployed"]=="candidate" for r in d["rounds"]) for d in streams]
            stats["last_accepted"]=[max([r["round"] for r in d["rounds"] if r["deployed"]=="candidate"],default=0) for d in streams]
            stats["cpu"]=[sum(r["cpu"] for r in d["rounds"]) for d in streams]
            result[route]={k:dict(values=values,mean=float(np.mean(values)),lowest=float(np.min(values))) for k,values in stats.items()}
        result["splits"]=list(splits)
        pairs[setting]=result
    (OUT/"paired.json").write_text(json.dumps(pairs,indent=2))


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("mode",choices=["probe","replay","grid","summary","verify","speed"])
    p.add_argument("name",nargs="?",default="");p.add_argument("args",nargs="*");a=p.parse_args()
    if a.mode=="probe":probe(a.name,a.args)
    elif a.mode=="grid":grid(a.name.split(","),a.args[0].split(","),list(map(int,a.args[1].split(","))),int(a.args[2]) if len(a.args)>2 else 8)
    elif a.mode=="speed":speed()
    elif a.mode=="verify":verify(a.name)
    elif a.mode=="summary":summarize()
    else:replay(a.name,int(a.args[0]),a.args[1])
