"""Paired tests of trajectory supervision and groupwise last-layer adjustments."""
from concurrent.futures import ThreadPoolExecutor
import importlib.util
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
from time import process_time

import numpy as np
HERE=Path(__file__).resolve().parent
V6=HERE.parent/"V6"
OUT=HERE.parent/"work/v7/engine"
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

# Every alternative receives the V6 controller, public release and selection rows.
sys.path.insert(0,str(V6))
import benchmark
import contest_submission
import engine
prior=load("diagnostic",HERE.parent/"V5/engine_v5_study.py")
original=load("original",V6/"engine.py")
live=load("deployed",HERE/"engine.py")


# The non-selected sensor features and losses are isolated in the offline study.
def _curvature_features(sensors, axes, gain=None):
    projections=sensors@axes
    if gain is not None:
        projections=projections*gain
    projections=np.clip(projections,-3.,3.)
    return .5*(projections**2-1.)

def _slope_axes(z, age_targets, device, covariance, size):
    """Directions that repeatedly change over observed within-engine trajectories."""
    _,group,counts=np.unique(device,return_inverse=True,return_counts=True)
    avg=np.bincount(group,weights=age_targets)/counts
    progress=avg[group]-age_targets
    scale=np.sqrt(np.bincount(group,weights=progress**2)/counts)
    reliable=(counts>=12)&(scale>3.)
    if reliable.sum()<12:
        return None
    norm=progress/np.maximum(1.,scale[group])
    slopes=np.zeros((len(counts),z.shape[1]))
    np.add.at(slopes,group,z*norm[:,None])
    slopes=slopes[reliable]/counts[reliable,None]
    eig,vectors=np.linalg.eigh(covariance)
    whitener=(vectors/np.sqrt(np.maximum(eig,0.)+.20))@vectors.T
    directional=slopes@whitener
    _,directions=np.linalg.eigh(directional.T@directional/len(directional))
    axes=whitener@directions[:,-size:]
    return axes/np.sqrt(np.sum(axes*(covariance@axes),axis=0)+.10)

def fit_neural(model, x, y, limit=24576, epochs=20, width=48, ordinal_weight=.7, stable=False, ordinal_y=None, condition_strength=0., track=None, slope_count=0, replace_curve=False, ordinal_scheme="default"):
    rng = np.random.default_rng(145)
    if len(x) > limit:
        ids = rng.choice(len(x), limit, replace=False)
        x, y = x[ids], y[ids]
        if track is not None:
            track=track[ids]
        if ordinal_y is not None:
            ordinal_y=ordinal_y[ids]
    centers, counts = np.unique(np.round(x[:, [-3, -1]]), axis=0, return_counts=True)
    centers = centers[np.argsort(-counts)[:8]]
    groups = original._regimes(x, centers)
    means = np.stack([x[groups == k, :-3].mean(0) for k in range(len(centers))])
    scales = np.stack([x[groups == k, :-3].std(0) for k in range(len(centers))])
    inv_scale = 1 / np.maximum(scales, 1e-4)
    sensors = np.clip((x[:, :-3] - means[groups]) * inv_scale[groups], -6, 6)
    # Low-variance sensor contrasts and the main common-mode directions encode
    # different degradation patterns. All axes are learned from released rows.
    sensor_count = sensors.shape[1]
    cov = sensors.T @ sensors / len(sensors)
    eigenvalues, axes = np.linalg.eigh(cov)
    active = np.flatnonzero(eigenvalues > .035)
    curve_axis = None
    curve_gain = None
    if len(active) >= 6:
        chosen = np.r_[active[:3], active[-3:]]
        curve_axis = axes[:, chosen] / np.sqrt(eigenvalues[chosen] + .10)
        if slope_count and track is not None:
            slopes=_slope_axes(sensors,y,track,cov,slope_count)
            if slopes is not None:
                curve_axis=np.column_stack((curve_axis[:,:-slope_count] if replace_curve else curve_axis,slopes))
        projection=sensors@curve_axis
        if condition_strength:
            pooled=np.mean(projection**2,axis=0)
            local=np.stack([np.mean(projection[groups==k]**2,axis=0) for k in range(len(centers))])
            curve_gain=np.sqrt((pooled+.15)/((1-condition_strength)*pooled+condition_strength*local+.15))
            projection=projection*curve_gain[groups]
        sensors=np.column_stack((sensors,.5*(np.clip(projection,-3.,3.)**2-1.)))
    base = model.predict(x).reshape(-1)
    design = np.column_stack((sensors, (base - 100) / 60, np.eye(len(centers))[groups])).astype(np.float32)
    target = ((y - base) / 55).astype(np.float32)
    ordinal_y=y if ordinal_y is None else ordinal_y
    levels={"default":(.20,.45,.70),"spread":(.15,.50,.85),"wide":(.10,.50,.90),
            "soft":(.20,.45,.70),"softspread":(.15,.50,.85),"softwide":(.10,.50,.90)}[ordinal_scheme]
    thresholds=np.quantile(ordinal_y,levels)
    if ordinal_scheme.startswith("soft"):
        width_y=max(2.,.10*float(np.quantile(ordinal_y,.75)-np.quantile(ordinal_y,.25)))
        delta=(thresholds-ordinal_y[:,None])/width_y
        near_threshold=(1/(1+np.exp(-np.clip(delta,-30.,30.)))).astype(np.float32)
    else:
        near_threshold = (ordinal_y[:, None] < thresholds).astype(np.float32)

    # The initial predictor and RNG state match the original head. Curvature
    # channels start at zero and are activated by gradients from released rows.
    weights = rng.normal(0, .22, (sensor_count + 1 + len(centers), width)).astype(np.float32)
    if curve_axis is not None:
        zeros = np.zeros((curve_axis.shape[1], width), dtype=np.float32)
        weights = np.vstack((weights[:sensor_count], zeros, weights[sensor_count:]))
    bias = np.zeros(width, dtype=np.float32)
    # Column zero predicts the standardised residual; other columns supervise
    # rank at three training-only lifetime thresholds for the optional stable-pool ordinal readout.
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
            delta[:, 0] = original._training_residual(delta[:, 0], target[idx], not stable) * (2 / len(idx))
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
    head = dict(centers=centers, means=means, inv_scale=inv_scale,
                w_sensor=weights[:-len(centers)-1].copy(), w_base=weights[-len(centers)-1] / 60,
                bias=bias + weights[-len(centers):] - weights[-len(centers)-1] * (100 / 60),
                output=55 * output[:, 0].astype(float), intercept=55 * intercept[:1].astype(float))
    if curve_axis is not None:
        head["curve_axis"] = curve_axis
        if curve_gain is not None:
            head["curve_gain"]=curve_gain
        head["curve_weight"] = head["w_sensor"][sensor_count:].copy()
        head["w_sensor"] = head["w_sensor"][:sensor_count].copy()
    if stable:
        head.update(ord_out=output[:,1:].astype(float), ord_bias=intercept[1:].astype(float))
    return head

def _residual(head, z, base, group):
    value = z @ head["w_sensor"]
    if "curve_axis" in head:
        value = value + _curvature_features(z, head["curve_axis"],head["curve_gain"][group] if "curve_gain" in head else None) @ head["curve_weight"]
    hidden = np.tanh(value + base * head["w_base"] + head["bias"][group])
    return hidden @ head["output"] + head["intercept"][0]

def _ordinal_hidden(model, head, x):
    base = model.predict(x).reshape(-1)
    group = original._regimes(x, head["centers"])
    z = np.clip((x[:,:-3]-head["means"][group])*head["inv_scale"][group], -6., 6.)
    value = z@head["w_sensor"]
    if "curve_axis" in head:
        value = value + _curvature_features(z, head["curve_axis"],head["curve_gain"][group] if "curve_gain" in head else None)@head["curve_weight"]
    hidden = np.tanh(value+base[:,None]*head["w_base"]+head["bias"][group])
    return group, hidden

research=load("research_base",V6/"engine.py")
research.fit_neural=fit_neural
research._residual=_residual
research._ordinal_hidden=_ordinal_hidden

def trajectories(y):
    good=np.isclose(np.diff(y),-1.,atol=1e-6)
    if len(y)<512 or good.mean()<.93:
        return None
    starts=np.r_[0,1+np.flatnonzero(~good)]
    ends=np.r_[starts[1:],len(y)]
    valid=ends-starts>=16
    if valid.sum()<8 or sum((ends-starts)[valid])<len(y)*.9:
        return None
    return np.repeat(np.arange(len(starts)),ends-starts),starts,valid


def trajectory_target(y,weight,local=False):
    seq=trajectories(y)
    if seq is None:
        return y
    ids,starts,valid=seq
    lifespan=y[starts]+1.
    typical=np.median(lifespan[valid])
    correction=(typical-lifespan[ids])*np.clip(y/np.maximum(1.,lifespan[ids]-1.),0.,1.)
    if local:
        # Shrink the noise only before the typical engine exhibits clear wear.
        correction*=np.clip((y-60.)/75.,0.,1.)
    return y+weight*np.clip(correction,-90.,90.)*valid[ids]


def device_readout(m,h,x,y,route):
    seq=trajectories(y)
    if seq is None: return h
    ids,_,_=seq
    # Published rows unused by both the V6 neural fit and its two calibrators.
    free=np.ones(len(y),dtype=bool)
    if len(y)<=24576: return h
    free[np.random.default_rng(145).choice(len(y),24576,replace=False)]=False
    free[original._reserve_rows(len(y))]=False
    free[original._reserve_rows(len(y),True)]=False
    pos=np.flatnonzero(free)
    if len(pos)<2048:return h
    pos=np.random.default_rng(713).choice(pos,min(len(pos),16384),replace=False)
    xx=x[pos]; target=y[pos]; unit=ids[pos]
    train=unit%4!=1; check=~train
    if min(train.sum(),check.sum())<256:return h
    err=original.predict(m,h,xx)-target
    base=m.predict(xx).reshape(-1)
    _,hidden=original._ordinal_hidden(m,h,xx)
    raw=hidden@h["output"]+h["intercept"][0]
    slope=(np.abs(raw)<65.)*np.where(raw>0.,float(h["extension_fraction"]),1.)
    j=np.column_stack((hidden,np.ones(len(xx))))*slope[:,None]
    v=j[train]; e=err[train]
    keys,group,ct=np.unique(unit[train],return_inverse=True,return_counts=True)
    vg=np.zeros((len(keys),v.shape[1]));np.add.at(vg,group,v);vg/=ct[:,None]
    eg=np.bincount(group,weights=e)/ct
    if route=="contrast":
        v=v-.7*vg[group];e=e-.7*eg[group]
        weight=np.ones(len(e))
    elif route=="worst":
        rms=np.sqrt(np.bincount(group,weights=e**2)/ct)
        ref=np.sqrt(np.bincount(group,weights=(base[train]-target[train])**2)/ct)
        weight=np.sqrt(np.mean(ct)/ct[group])*np.clip((rms/np.maximum(10.,ref))[group],.5,2.)
    else:raise ValueError(route)
    weight/=weight.mean()
    cov=v.T@(weight[:,None]*v)/len(v)+np.eye(v.shape[1])*.06
    delta=np.linalg.solve(cov,-v.T@(weight*e)/len(v))
    def risk(error):
        z=error[check]
        return np.array([np.mean(z*z),*(np.mean(np.expm1(np.clip(sign*z/scale,0,18)))for sign,scale in ((1,10),(-1,13)))])
    ceiling=risk(err)*np.array([1.0003,1.015,1.015])
    for frac in (.70,.40,.18):
        out=dict(h,output=h["output"]+delta[:-1]*frac,intercept=h["intercept"]+delta[-1]*frac)
        if np.all(risk(original.predict(m,out,xx)-target)<=ceiling):return out
    return h


def risk_projection(m,h,x,y,alpha):
    """Descend tempered log tail risk while the released held-out loss stays in budget."""
    if len(y)<25088:return h
    free=np.ones(len(y),dtype=bool)
    free[np.random.default_rng(145).choice(len(y),24576,replace=False)]=False
    free[original._reserve_rows(len(y))]=False
    free[original._reserve_rows(len(y),True)]=False
    idx=np.flatnonzero(free)
    if len(idx)<1024:return h
    ids=np.random.default_rng(791).choice(idx,min(16384,len(idx)),replace=False)
    xx=x[ids];target=y[ids]
    seq=trajectories(y)
    tr=np.arange(len(ids))%3!=1 if seq is None else seq[0][ids]%4!=1
    if min(tr.sum(),(~tr).sum())<256:return h
    err=original.predict(m,h,xx)-target
    base=m.predict(xx).ravel()
    g,hidden=original._ordinal_hidden(m,h,xx)
    raw=hidden@h["output"]+h["intercept"][0]
    jac=np.column_stack((hidden,np.ones(len(ids))))*((abs(raw)<65.)*np.where(raw>0.,float(h["extension_fraction"]),1.))[:,None]
    j=jac[tr];e=err[tr]
    def tails(z):
        return np.array([np.mean(np.expm1(np.clip(sign*z/scale,0.,20.))) for sign,scale in ((1,10),(-1,13))])
    ref=tails(err[~tr]); scale=np.sqrt(np.mean(e*e))
    gradient=e.copy()
    for sign,width in ((1.,10.),(-1.,13.)):
        exponent=np.maximum(0.,sign*e/(2*width))
        weight=np.where(exponent>0.,np.expm1(np.minimum(exponent,np.quantile(exponent,.998))),0.)
        gradient+=alpha*.5*scale*sign*weight/max(1e-8,weight.mean())
    H=j.T@j/len(j)+np.eye(j.shape[1])*.08
    delta=np.linalg.solve(H,-j.T@gradient/len(j))
    delta/=max(1.,np.sqrt(np.mean((j@delta)**2))/7.)
    allowed_mse=np.mean(err[~tr]**2)*1.002
    for fraction in (.8,.4,.18):
        out=dict(h,output=h["output"]+fraction*delta[:-1],intercept=h["intercept"]+fraction*delta[-1])
        e1=(original.predict(m,out,xx)-target)[~tr]
        r=tails(e1)
        if np.mean(e1*e1)<=allowed_mse and r.sum()<=ref.sum()*.985 and np.all(r<=ref*1.12):return out
    return h


def fit(route,m,x,y,i,previous=None,stable_release=False):
    if route.startswith(("joint","evidence","quant","focus","clinical","steady","start","initial","cohort")):
        seq=trajectories(y)
        neural=research.fit_neural
        # Only a head initialized on near-complete lives can keep partial pooling
        # on later releases. A severely truncated first release keeps raw targets.
        eligible=bool(previous.get("trajectory_support",False)) if previous else False
        if i==1 and seq is not None:
            _,s,v=seq
            last=np.r_[s[1:],len(y)]-1
            eligible=.005<=np.median((y[last]/np.maximum(1.,y[s]))[v])<.16
        def joint(m,x,y,**kw):
            if seq is None:
                plateaus=len(y)>=512 and np.mean(np.isclose(np.diff(y),0.,atol=1e-6))>=.50
                return neural(m,x,y,ordinal_scheme="softspread" if plateaus or not route.startswith("cohort") else "default",**kw)
            _,starts,valid=seq
            ends=np.r_[starts[1:],len(y)]-1
            missing=np.median((y[ends]/np.maximum(1.,y[starts]))[valid])
            if missing>=.005:
                if route.startswith(("focus","clinical","steady","start","initial","cohort")):
                    cap=int("".join(c for c in route if c.isdigit()))/100
                    factor=cap*float(np.clip((.10-missing)/.05,0.,1.))
                else:
                    factor=.25*float(np.clip((.16-missing)/.08,0.,1.))
                if route.startswith("steady") and not kw.get("stable"):factor=0.
                if route.startswith("start") and kw.get("stable"):factor=0.
                if route.startswith("initial") and i>1:factor=0.
                if route.startswith("cohort") and not eligible:factor=0.
                return neural(m,x,trajectory_target(y,factor),**kw)
            if route.startswith(("quant","focus","steady","start","initial","cohort")):
                return neural(m,x,y,ordinal_scheme="soft",**kw)
            if kw.get("stable"):
                return neural(m,x,y,ordinal_scheme="soft",ordinal_y=trajectory_target(y,.75) if route=="joint" else None,**kw)
            return neural(m,x,y,ordinal_scheme="soft",track=seq[0],slope_count=4,**kw)
        research.fit_neural=joint
        try:
            h=research.fit(m,x,y,i,previous,stable_release)
            if route.startswith("cohort"):h["trajectory_support"]=np.asarray(int(eligible))
            return h
        finally:research.fit_neural=neural
    if route in ("spread","wide","soft","softspread","softwide"):
        neural=research.fit_neural
        research.fit_neural=lambda m,x,y,**kw: neural(m,x,y,ordinal_scheme=route,**kw)
        try:return research.fit(m,x,y,i,previous,stable_release)
        finally:research.fit_neural=neural
    if route.startswith(("short","budget","paired")):
        amt=int("".join(c for c in route if c.isdigit()))
        seq=trajectories(y)
        missing=0. if seq is None else np.median((y[np.r_[seq[1][1:],len(y)]-1]/np.maximum(1.,y[seq[1]]))[seq[2]])
        neural=research.fit_neural
        def compressed(m,xx,yy,**kw):
            if route.startswith("budget") or route.startswith("short"):
                kw["epochs"]=max(1,int(round(kw["epochs"]*amt/100)))
            if route.startswith(("budget","paired")) and missing>=.005:
                yy=trajectory_target(yy,.25)
            elif route.startswith("paired") and seq is not None:
                if kw.get("stable"):
                    kw["ordinal_y"]=trajectory_target(yy,.75)
                else:
                    kw.update(track=seq[0],slope_count=4)
            return neural(m,xx,yy,**kw)
        research.fit_neural=compressed
        try:return research.fit(m,x,y,i,previous,stable_release)
        finally:research.fit_neural=neural
    if route.startswith("tilt"):
        h=original.fit(m,x,y,i,previous,stable_release)
        return risk_projection(m,h,x,y,int(route[4:])/100)
    if route=="cap125":
        neural=original.fit_neural
        original.fit_neural=lambda m,x,y,**kw: neural(m,x,np.minimum(y,125.),**kw)
        try:return original.fit(m,x,y,i,previous,stable_release)
        finally:original.fit_neural=neural
    if route=="frozen":return live.fit(m,x,y,i,previous,stable_release)
    if route.startswith(("censor","hybrid","graded","window","narrow","mature")):
        seq=trajectories(y)
        if seq is None:return original.fit(m,x,y,i,previous,stable_release)
        ids,starts,valid=seq
        ends=np.r_[starts[1:],len(y)]-1
        missing=np.median((y[ends]/np.maximum(1.,y[starts]))[valid])
        weight=int("".join(c for c in route if c.isdigit()))/100
        if missing >= .005:
            if route.startswith("graded"):weight*=min(1.,missing/.15)
            if route.startswith(("window","mature")):weight*=float(np.clip((.26-missing)/.16,0.,1.))
            if route.startswith("narrow"):weight*=float(np.clip((.16-missing)/.08,0.,1.))
            neural=original.fit_neural
            original.fit_neural=lambda m,x,y,**kw: neural(m,x,trajectory_target(y,weight),**kw)
            try:return original.fit(m,x,y,i,previous,stable_release)
            finally:original.fit_neural=neural
        if route.startswith(("hybrid","mature")):
            return fit("slope4" if not stable_release else "ord75",m,x,y,i,previous,stable_release)
        return original.fit(m,x,y,i,previous,stable_release)
    if route.startswith(("rev","ord","cond","slope","swap")):
        neural=research.fit_neural
        wt=int("".join(c for c in route if c.isdigit()))/100
        def altered(m,x,y,**kw):
            if route.startswith(("slope","swap")):
                seq=trajectories(y)
                return neural(m,x,y,track=None if seq is None else seq[0],slope_count=int(round(wt*100)),replace_curve=route.startswith("swap"),**kw)
            if route.startswith("cond"):
                return neural(m,x,y,condition_strength=wt,**kw)
            if route.startswith("ord"):
                return neural(m,x,y,ordinal_y=trajectory_target(y,wt),**kw)
            return neural(m,x,trajectory_target(y,-wt,route.startswith("revearly")),**kw)
        research.fit_neural=altered
        try:return research.fit(m,x,y,i,previous,stable_release)
        finally:research.fit_neural=neural
    neural=original.fit_neural
    if route.startswith(("health","early")):
        wt=int(''.join(c for c in route if c.isdigit()))/100
        original.fit_neural=lambda m,x,y,**kw: neural(m,x,trajectory_target(y,wt,route.startswith("early")),**kw)
    try:
        h=original.fit(m,x,y,i,previous,stable_release)
    finally:
        original.fit_neural=neural
    if route in ("contrast","worst"): h=device_readout(m,h,x,y,route)
    return h


def probe(dataset,routes):
    x,y,audit,age,units,cruise,m=prior.evaluation_data(dataset)
    stages=[1.,1.] if dataset=="official" else [.55,.7,.85,1.]
    result=[];previous={}
    for i,cut in enumerate(stages,1):
        tr=~audit&(age<=cut); xx,yy=x[tr],y[tr]
        for route in routes:
            tick=process_time()
            h=fit(route,m,xx,yy,i,previous.get(route),dataset=="official" and i>1)
            previous[route]=h
            val=(live if route=="frozen" else research if route.startswith("cond") else original).predict(m,h,x[audit])
            d=prior.diagnostics.metrics(val,y[audit],age[audit],cruise[audit] if cruise is not None else None)
            d.update(route=route,round=i,cpu=process_time()-tick)
            if route=="base":ref=val
            dif=(val-y[audit])**2-(ref-y[audit])**2
            u=units[audit]; z=np.array([(dif[u==k].sum(),sum(u==k))for k in np.unique(u)])
            picks=np.random.default_rng(411).integers(len(z),size=(600,len(z)))
            stat=z[picks].sum(1)
            d["paired_mse_ci"]=np.quantile(stat[:,0]/stat[:,1],[.025,.975]).tolist()
            result.append(d)
    (OUT/("probe-"+dataset+"-"+routes[1]+".json")).write_text(json.dumps(result,indent=2))


def replay(route,n,setting):
    assert 0<=n<20
    engine.fit=lambda m,x,y,i,previous=None,stable_release=False:fit(route,m,x,y,i,previous,stable_release)
    engine.predict=live.predict if route=="frozen" else research.predict if route.startswith("cond") else original.predict
    engine.prepare_inference=live.prepare_inference if route=="frozen" else original.prepare_inference
    benchmark.OUT=OUT/route
    benchmark.run("C","v6",n,setting)

def grid(routes,modes,seeds,workers):
    def one(q):
        r,m,n=q; folder=OUT/r;folder.mkdir(exist_ok=True)
        if (folder/f"{m}-v6-C-{n}.json").exists():return
        with (folder/f"{m}-{n}.log").open("w")as log:
            subprocess.run([sys.executable,"-B",__file__,"replay",r,m,str(n)],stdout=log,stderr=log,check=True)
    with ThreadPoolExecutor(workers) as pool:list(pool.map(one,[(r,m,n)for n in seeds for m in modes for r in routes]))

def performance(route):
    if hasattr(os,"sched_setaffinity"):os.sched_setaffinity(0,{max(os.sched_getaffinity(0))})
    x,y,audit,age,_,_,m=prior.evaluation_data("official")
    xx,yy=x[~audit],y[~audit]
    sample=x[audit][np.linspace(0,np.sum(audit)-1,1024,dtype=int)]
    stress=sample[:64].copy()
    stress[:,[-3,-1]]=np.rint(stress[:,[-3,-1]])+.5
    stress[:,:5]*=10.
    records=[]
    for train_route in ("base",route):
        prev=None
        for i in (1,2):
            t=process_time();h=fit(train_route,m,xx,yy,i,prev,i==2);cpu=process_time()-t
            prev=h
            for test in (sample,stress):
                arr=live.predict(m,h,test)
                fast=live.prepare_inference(m,h)
                batch=np.array([fast(row)for row in test])
                np.testing.assert_allclose(batch,arr,rtol=1e-10,atol=1e-9)
            t0=original.prepare_inference(m,h) if "curve_gain" not in h else None
            current=live.prepare_inference(m,h)
            timings={"old":[],"new":[]}
            for rep in range(7):
                fs=[("old",t0),("new",current)]
                for label,fn in (fs if rep%2==0 else reversed(fs)):
                    if fn is None:continue
                    clock=process_time()
                    for row in sample:fn(row)
                    timings[label].append((process_time()-clock)*1e6/len(sample))
            records.append(dict(route=train_route,round=i,cpu=cpu,bytes=sum(a.nbytes for a in h.values()),
                                single_us={k:statistics.median(v) for k,v in timings.items() if v},
                                max_abs=float(np.max(abs(batch-arr)))))
    (OUT/("speed-"+route+".json")).write_text(json.dumps(records,indent=2))

def verify():
    """Compare production against the frozen research variant and V6 checkpoint schema."""
    x,y,audit,age,_,_,m=prior.evaluation_data("official")
    released=~audit
    sample=x[audit][np.linspace(0,np.sum(audit)-1,384,dtype=int)]
    checks=[]
    previous={r:None for r in ("cohort25","frozen")}
    for index,(release,stable) in enumerate(((.955,False),(.97,False),(1.,False),(1.,True)),1):
        idx=released&(age<=release)
        out={}
        for r in previous:
            previous[r]=fit(r,m,x[idx],y[idx],index,previous[r],stable)
            out[r]=live.predict(m,previous[r],sample)
        np.testing.assert_array_equal(out["cohort25"],out["frozen"])
        for key in previous["cohort25"]:
            np.testing.assert_array_equal(previous["cohort25"][key],previous["frozen"][key])
        checks.append(dict(round=index,release=release,stable=stable,
                           support=int(previous["frozen"]["trajectory_support"]),max_abs=0.))
    # Unknown column ordering and undersized cohorts keep the raw regression target.
    rng=np.random.default_rng(402)
    scrambled=rng.permutation(y[released])[:4096]
    small=np.arange(350.,0.,-1.)
    for z in (scrambled,small):
        yy,levels,soft,init=live._released_targets(z,1,None)
        assert np.array_equal(z,yy) and not soft and not init
    (OUT/"equivalence.json").write_text(json.dumps(checks,indent=2))


def summarize():
    result={}
    for folder in sorted(OUT.iterdir()):
        if not folder.is_dir():continue
        mode_result={}
        for mode in ("schedule","trace","drift","fresh","official"):
            pairs=[]
            for f in sorted(folder.glob(mode+"-v6-*.json")):
                ref=OUT/"base"/f.name
                if ref.is_file():
                    pairs.append([int(f.stem.split("-")[-1]),json.loads(ref.read_text()),json.loads(f.read_text())])
            if not pairs:continue
            block={"seeds":[r[0] for r in pairs]}
            for metric in ("stage_mean","stage_floor","worst","forget","delay","nasa"):
                vals=[[statistics.mean(r["nasa_mean"] for r in q[i]["rounds"]) if metric=="nasa" else q[i][metric] for q in pairs] for i in (1,2)]
                block[metric]=dict(reference=statistics.mean(vals[0]),candidate=statistics.mean(vals[1]),
                    increased=sum(b>a+1e-10 for a,b in zip(*vals)),decreased=sum(b<a-1e-10 for a,b in zip(*vals)))
            mode_result[mode]=block
        result[folder.name]=mode_result
    (OUT/"paired.json").write_text(json.dumps(result,indent=2))

if __name__=="__main__":
    a=sys.argv[1:]
    OUT.mkdir(parents=True,exist_ok=True)
    if a[0]=="verify":verify()
    elif a[0]=="summary":summarize()
    elif a[0]=="speed":performance(a[1])
    elif a[0]=="replay":replay(a[1],int(a[3]),a[2])
    elif a[0]=="grid":grid(a[1].split(","),a[2].split(","),list(map(int,a[3].split(","))),int(a[4]))
    elif a[0]=="probe":probe(a[1],a[2].split(","))
