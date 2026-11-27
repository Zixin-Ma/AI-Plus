"""Ablations for engine longevity under an unchanged V5 controller."""
import argparse
import concurrent.futures
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from time import process_time

import numpy as np
import benchmark
import engine
HERE=Path(__file__).resolve().parent
OUT=HERE.parent/"work/v6/engine"


def load_module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


v5=load_module("reference_engine",HERE.parent/"V5/engine.py")
prior=load_module("past_engine_study",HERE.parent/"V5/engine_v5_study.py")
_regimes=v5._regimes
_training_residual=v5._training_residual

def curve_input(z,axis):
    return .5*(np.clip(z@axis,-3.,3.)**2-1.)


prior_residual=v5._residual
prior_hidden=v5._ordinal_hidden


def curved_residual(h,z,b,g):
    if "curve_axis" not in h:return prior_residual(h,z,b,g)
    hidden=np.tanh(z@h["w_sensor"]+curve_input(z,h["curve_axis"])@h["curve_weight"]+
                   b*h["w_base"]+h["bias"][g])
    return hidden@h["output"]+h["intercept"][0]


def curved_hidden(m,h,x):
    if "curve_axis" not in h:return prior_hidden(m,h,x)
    b=m.predict(x).reshape(-1);g=v5._regimes(x,h["centers"])
    z=np.clip((x[:,:-3]-h["means"][g])*h["inv_scale"][g],-6,6)
    return g,np.tanh(z@h["w_sensor"]+curve_input(z,h["curve_axis"])@h["curve_weight"]+
                    b[:,None]*h["w_base"]+h["bias"][g])


v5._residual=curved_residual
v5._ordinal_hidden=curved_hidden

def fit_neural(model, x, y, limit=24576, epochs=20, width=48, ordinal_weight=.7, stable=False, route="base", norm=None):
    rng = np.random.default_rng(145)
    step_rows=min(len(x),limit)
    if len(x)>limit:
        ids=rng.choice(len(x),limit,replace=False)
        if route.startswith(("cover","cwhite","cdro","late","twin")):
            # Leave exactly the V5 calibration reserve untouched. Broaden the
            # training pool while retaining the number of updates per epoch.
            free=np.ones(len(y),dtype=bool);free[ids]=False
            hold=prior.diagnostics.calibration_rows(len(y)) if len(y)>=limit+512 else np.empty(0,dtype=int)
            free[hold]=False
            other=np.flatnonzero(free)
            ratio=4 if "4" in route and not route.startswith("cwhite") else 2
            extra=np.random.default_rng(381).choice(other,min(len(other),limit*(ratio-1)),replace=False)
            ids=np.r_[ids,extra]
        x,y=x[ids],y[ids]
    centers, counts = np.unique(np.round(x[:, [-3, -1]]), axis=0, return_counts=True)
    centers = centers[np.argsort(-counts)[:8]]
    groups = _regimes(x, centers)
    means = np.stack([x[groups == k, :-3].mean(0) for k in range(len(centers))])
    scales = np.stack([x[groups == k, :-3].std(0) for k in range(len(centers))])
    inv_scale = 1 / np.maximum(scales, 1e-4)
    if norm is not None:
        centers,means,inv_scale=norm["centers"],norm["means"],norm["inv_scale"]
        groups=_regimes(x,centers)
    elif route.startswith("late"):
        centers0,cnt=np.unique(np.round(x[:step_rows,[-3,-1]]),axis=0,return_counts=True)
        centers=centers0[np.argsort(-cnt)[:8]];groups=_regimes(x,centers)
        means=np.stack([x[:step_rows,:-3][groups[:step_rows]==g].mean(0) for g in range(len(centers))])
        scales=np.stack([x[:step_rows,:-3][groups[:step_rows]==g].std(0) for g in range(len(centers))]);inv_scale=1/np.maximum(scales,1e-4)
    sensors = np.clip((x[:, :-3] - means[groups]) * inv_scale[groups], -6, 6)
    transform=None
    if route.startswith(("white","wavg","color","wdro","cwhite")):
        mix=.80 if "80" in route else .40
        # Pooled, condition-standardized correlation. Preserve rotation-invariant
        # geometry; fold the resulting linear map into the sensor readout.
        cov=(sensors.T@sensors)/len(sensors)
        eig,vec=np.linalg.eigh(cov)
        exponent=.5 if route.startswith("color") else -.5
        scale=np.clip(((1-mix)+mix*np.maximum(0,eig))**exponent,.45,2.5)
        transform=(vec*scale)@vec.T
        sensors=sensors@transform
    extra_axis=None
    raw_dim=sensors.shape[1]
    if route.startswith(("quad","mode","sir","sham","rank","nest")):
        C=sensors.T@sensors/len(sensors)
        eig,vec=np.linalg.eigh(C)
        if route.startswith("sir"):
            # Sliced inverse regression emphasizes combinations discriminating
            # released lifetime bands without querying age or ID at inference.
            cuts=np.quantile(y,np.linspace(.1,.9,9)); idx=np.searchsorted(cuts,y)
            mu=np.stack([sensors[idx==j].mean(0) if (idx==j).any() else np.zeros(raw_dim) for j in range(10)])
            share=np.bincount(idx,minlength=10)/len(y)
            inv=(vec/np.sqrt(np.maximum(eig,0)+.20))@vec.T
            _,basis=np.linalg.eigh(inv@(mu.T@(share[:,None]*mu))@inv)
            extra_axis=inv@basis[:,-4:]
        else:
            keep=np.flatnonzero(eig>.035)
            pick=keep[-6:] if route=="quad6" else np.unique(np.r_[keep[:3],keep[-3:]])
            extra_axis=vec[:,pick]/np.sqrt(eig[pick]+.10)
        if route=="modeweak":
            extra_axis=extra_axis*.65
        if route=="sham6":
            extra_axis=np.zeros_like(extra_axis)
        # A contrast subspace and a common-mode subspace require six disjoint
        # directions. A lower-rank training release retains the original input.
        if route.startswith(("rank","nest")) and len(keep)<6:
            extra_axis=None
        mix_curve=None
        if extra_axis is not None:
            feature=curve_input(sensors,extra_axis)
            if route=="nestpool":
                mixing=np.eye(len(pick))
                for left in (0,3):
                    block=range(left,min(left+3,len(pick)))
                    for i in block:
                        near=[j for j in block if abs(eig[pick[j]]-eig[pick[i]])<.25*max(.05,eig[pick[i]])]
                        if len(near)>1:
                            mixing[near,i]=1/len(near)
                mix_curve=mixing
                feature=feature@mix_curve
            sensors=np.column_stack((sensors,feature))
    base = model.predict(x).reshape(-1)
    design = np.column_stack((sensors, (base - 100) / 60, np.eye(len(centers))[groups])).astype(np.float32)
    target = ((y - base) / 55).astype(np.float32)
    thresholds = np.quantile(y, [.20, .45, .70])
    near_threshold = (y[:, None] < thresholds).astype(np.float32)

    if route.startswith("nest") and extra_axis is not None:
        old=rng.normal(0,.22,(raw_dim+1+len(centers),width)).astype(np.float32)
        weights=np.vstack((old[:raw_dim],np.zeros((extra_axis.shape[1],width),dtype=np.float32),old[raw_dim:]))
    else:
        weights=rng.normal(0,.22,(design.shape[1],width)).astype(np.float32)
    bias = np.zeros(width, dtype=np.float32)
    # Column zero predicts the standardised residual; other columns supervise
    # rank at three training-only lifetime thresholds for the optional stable-pool ordinal readout.
    output = np.zeros((width, 4), dtype=np.float32)
    output[:, 0] = rng.normal(0, .03, width)
    intercept = np.zeros(4, dtype=np.float32)
    params = [weights, bias, output, intercept]
    first = [np.zeros_like(a) for a in params]
    second = [np.zeros_like(a) for a in params]
    bins=np.searchsorted(thresholds,y)
    anchor=np.bincount(bins,weights=target**2,minlength=4)/np.maximum(1,np.bincount(bins,minlength=4))
    ema=np.ones(4)
    average=[np.zeros_like(v) for v in params]
    snapshots=0
    step = 0
    for epoch in range(epochs):
        initial=route.startswith("late") and epoch < epochs*(.5 if route=="late50" else .67)
        order=rng.permutation(step_rows if initial else len(y))[:step_rows]
        rate = .003 / (1 + epoch / 10)
        for idx in np.array_split(order, max(1, step_rows // 256)):
            batch = design[idx]
            hidden = np.tanh(batch @ weights + bias)
            delta = hidden @ output + intercept
            raw=delta[:,0]
            direction=_training_residual(raw,target[idx],not stable)
            if route.startswith(("dro","both","wdro","cdro")):
                actual=np.clip(raw,-65/55,65/55)*np.where(raw>0,.5,1.)
                err=actual-target[idx]
                groups_b=bins[idx]
                count=np.bincount(groups_b,minlength=4)
                ratio=np.bincount(groups_b,weights=err**2,minlength=4)/np.maximum(1,count)/(anchor+.10)
                ema=.85*ema+.15*np.where(count>0,ratio,ema)
                group_weights=np.exp(np.clip(2*(ema-ema.mean()),-1.5,1.5))
                w=group_weights[groups_b]; w/=w.mean()
                strength=.50 if "50" in route else .20
                direction=direction*(1+strength*(w-1))
            delta[:, 0] = direction * (2 / len(idx))
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
        if route.startswith(("swa","both","wavg","cover2avg")) and epoch>=epochs*2//3:
            snapshots+=1
            for dest,param in zip(average,params):
                dest+=param
    if snapshots:
        alpha=.50 if route=="swa50" else 1.
        for param,accum in zip(params,average):
            param[:]=(1-alpha)*param+alpha*accum/snapshots
    # Keep deployment arithmetic in the same precision as existing checkpoints.
    weights = weights.astype(float)
    if transform is not None:
        weights[:sensors.shape[1]]=transform@weights[:sensors.shape[1]]
    bias = bias.astype(float)
    head = dict(centers=centers, means=means, inv_scale=inv_scale,
                w_sensor=weights[:-len(centers)-1].copy(), w_base=weights[-len(centers)-1] / 60,
                bias=bias + weights[-len(centers):] - weights[-len(centers)-1] * (100 / 60),
                output=55 * output[:, 0].astype(float), intercept=55 * intercept[:1].astype(float))
    if extra_axis is not None:
        head["curve_axis"]=extra_axis
        head["curve_weight"]=head["w_sensor"][raw_dim:].copy()
        if mix_curve is not None:
            head["curve_weight"]=mix_curve@head["curve_weight"]
        head["w_sensor"]=head["w_sensor"][:raw_dim].copy()
    if stable:
        head.update(ord_out=output[:,1:].astype(float), ord_bias=intercept[1:].astype(float))
    return head



def neural_head(route,m,x,y,idx,previous,stable_release):
    stable=bool(stable_release and previous)
    h=fit_neural(m,x,y,epochs=20 if stable else 12,width=48 if stable else 64,
                 ordinal_weight=.2 if stable else .7,stable=stable,route=route)
    use=np.linspace(0,len(x)-1,min(len(x),2048),dtype=int)
    est=v5.predict_neural(m,h,x[use],65.)
    lo,hi=np.quantile(est,[.35,.65])
    h.update(health_lo=np.array(lo),health_hi=np.array(hi),
             health_amp=np.array(np.clip(np.std(y)/max(1.,np.std(est))-1.,.15,.7)),
             extension_fraction=np.array(.50))
    if stable:
        h=v5._calibrate_ordinal(m,h,x,y)
        if "ord_coef" not in h:
            h.pop("ord_out"); h.pop("ord_bias")
    return h


def blend_cells(m,h,x,y,route):
    if len(y)<25088:
        return h
    ids=prior.diagnostics.calibration_rows(len(y))
    p=v5.predict(m,h,x[ids]); b=m.predict(x[ids]).reshape(-1); yy=y[ids]
    cuts=np.quantile(b,[.25,.50,.75])
    bands=np.searchsorted(cuts,b)*2+(p>b)
    yc=np.quantile(yy,[.2,.45,.7]); actual=np.searchsorted(yc,yy)
    scales=np.ones(8)
    for band in range(8):
        mask=bands==band
        if mask.sum()<128:continue
        target=yy[mask]; ref=b[mask]; change=p[mask]-ref
        idx=np.arange(len(target))%2
        stages=actual[mask]
        def stats(f,keep):
            err=ref[keep]+f*change[keep]-target[keep]
            baseline=ref[keep]-target[keep]
            gain=((err**2).mean()-(baseline**2).mean())/max(100.,(baseline**2).mean())
            risks=[]
            for k in range(4):
                part=stages[keep]==k
                if part.sum()>=12:
                    risks.append(np.mean(err[part]**2-baseline[part]**2)/max(100.,np.mean(baseline[part]**2)))
            sign_risk=[np.mean(np.expm1(np.clip(sign*err/den,0,10.))) for sign,den in ((1.,10.),(-1.,13.))]
            return gain,max(risks,default=gain),np.array(sign_risk)
        results=[]
        for half in [0,1]:
            keep=idx==half
            g,w,ns=stats(1.,keep)
            best=(0.,1.)
            for f in (.40,.65,.85,1.,1.15,1.3):
                gg,ww,n=stats(f,keep)
                loss=(gg-g)+(.5 if route=="cellfloor" else .15)*(ww-w)+.08*np.maximum(0,np.log((n+1)/(ns+1))).sum()+.10*(f-1)**2
                if loss<best[0]:best=(loss,f)
            results.append(best[1])
        if (results[0]-1)*(results[1]-1)>0:
            scales[band]=1+.75*(min(results)-1 if results[0]>1 else max(results)-1)
    h.update(cell_cuts=cuts,cell_scale=scales)
    return h


def constrained_blend(m,h,x,y,route):
    if len(y)<25088:return h
    ids=prior.diagnostics.calibration_rows(len(y))
    xx=x[ids]; labels=y[ids]
    p=v5.predict(m,h,xx); b=m.predict(xx).reshape(-1); c=p-b; error=p-labels
    cuts=np.quantile(b,[.25,.50,.75]); cell=np.searchsorted(cuts,b)*2+(c>0)
    life=np.searchsorted(np.quantile(labels,[.15,.35,.60,.85]),labels)
    den=np.bincount(life,weights=(b-labels)**2,minlength=5)+1
    ratio=np.bincount(life,weights=error**2,minlength=5)/den
    if route=="safemean": weights=np.ones(len(labels))
    else:
        urgency=np.exp(np.clip(3*(ratio-np.mean(ratio)),-2,2)); urgency/=urgency.mean()
        weights=.50+.50*urgency[life]
    hessian=np.bincount(cell,weights=weights*c*c,minlength=8)+250*len(ids)/8
    direction=-np.bincount(cell,weights=weights*c*error,minlength=8)/hessian
    normals=[]
    for sign,scale in ((1.,10.),(-1.,13.)):
        exp=sign*error/scale
        for trim in [.97,.997]:
            w=sign*np.where(exp>0,np.exp(np.minimum(exp,np.quantile(exp,trim))),0.)
            a=np.bincount(cell,weights=w*c,minlength=8)
            a/=max(1.,np.linalg.norm(a))
            u=a/hessian
            normals.append((a,u,max(1e-12,a@u)))
    for _ in range(10):
        for a,u,denom in normals:
            direction-=max(0.,a@direction)*u/denom
    direction/=max(1.,np.max(abs(direction))/.45)
    def side(e,keep,sign,scale):
        return np.expm1(np.clip(sign*e[keep]/scale,0.,16.)).mean()
    scale=0.
    for step in (1.,.6,.3):
        adjusted=error+step*c*direction[cell]
        passes=[]
        for j in (0,1):
            ix=np.arange(len(error))%2==j
            prev=error[ix]
            passes.append(np.mean(adjusted[ix]**2)<=np.mean(prev**2)*1.0005 and
                all(side(adjusted,ix,sign,d)<=1.02*side(error,ix,sign,d) for sign,d in ((1.,10.),(-1.,13.))))
        if all(passes):
            scale=step
            break
    if scale and np.max(abs(direction))>1e-6:
        h.update(cell_cuts=cuts,cell_scale=1+scale*direction)
    return h


def readout(m,h,x,y,route,disjoint=False,expanded=False):
    if len(y)<25088:return h
    ids=prior.diagnostics.calibration_rows(len(y))
    if disjoint:
        free=np.ones(len(y),dtype=bool)
        use=np.random.default_rng(145).choice(len(y),24576,replace=False)
        free[use]=False; free[ids]=False
        if expanded:
            avail=np.flatnonzero(free)
            more=np.random.default_rng(381).choice(avail,min(24576,len(avail)),replace=False)
            free[more]=False
        idx=np.flatnonzero(free)
        if len(idx)<512:return h
        ids=np.random.default_rng(621).choice(idx,min(16384,len(idx)),replace=False)
    xx=x[ids]; yy=y[ids]; p=v5.predict(m,h,xx); b=m.predict(xx).reshape(-1)
    group,hidden=v5._ordinal_hidden(m,h,xx)
    raw=hidden@h["output"]+h["intercept"][0]; value=np.maximum(0,b+np.clip(raw,-65.,65.))
    tail=np.maximum(0,value-h["health_hi"])+np.minimum(0,value-h["health_lo"])
    amp=.2*float(h["health_amp"])
    pre=np.maximum(0,value+np.clip(amp*tail,-25.,25.))
    slope=((np.abs(raw)<65)&(value>0)).astype(float)
    slope*=1+amp*((tail!=0)&(np.abs(amp*tail)<25))
    slope*=np.where(pre>b,float(h["extension_fraction"]),1.)
    jac=np.column_stack((hidden,np.ones(len(p))))*slope[:,None]
    err=p-yy; labels=np.searchsorted(np.quantile(yy,[.2,.4,.6,.8]),yy)
    train=np.arange(len(yy))%2==0; valid=~train
    def ratios(e,ids):
        return np.bincount(labels[ids],weights=e[ids]**2,minlength=5)/(100.+np.bincount(labels[ids],weights=(b[ids]-yy[ids])**2,minlength=5))
    r=ratios(err,train)
    importance=np.exp(np.clip(2.5*(r-np.mean(r)),-1.5,1.5));importance/=importance.mean()
    weight=np.ones(len(err)) if route=="readmean" else .5+.5*importance[labels]
    j=jac[train];e=err[train];w=weight[train]
    H=j.T@(j*w[:,None])/len(j)+np.eye(j.shape[1])*.06
    d=np.linalg.solve(H,-j.T@(w*e)/len(j))
    normals=[]
    for sign,scale in ((1.,10.),(-1.,13.)):
        exponent=sign*e/scale
        for trim in (.975,.998):
            v=sign*np.where(exponent>0,np.exp(np.minimum(exponent,np.quantile(exponent,trim))),0.)
            a=v@j/max(1.,np.abs(v).sum())
            u=np.linalg.solve(H,a)
            normals.append((a,u,max(1e-10,a@u)))
    for _ in range(12):
        for a,u,den in normals:
            d-=max(0.,a@d)*u/den
    d/=max(1.,np.sqrt(np.mean((j@d)**2))/6.)
    base_tail=[np.mean(np.expm1(np.clip(s*err[valid]/v,0,16))) for s,v in ((1,10),(-1,13))]
    for step in (.8,.45,.2):
        out=dict(h,output=h["output"]+step*d[:-1],intercept=h["intercept"]+step*d[-1])
        test=v5.predict(m,out,xx)-yy
        ok=all(np.mean(np.expm1(np.clip(s*test[valid]/v,0,16)))<=a*1.025 for (s,v),a in zip(((1,10),(-1,13)),base_tail))
        if np.mean(test[valid]**2)<=np.mean(err[valid]**2)*1.0005 and (route!="readfloor" or ratios(test,valid).max()<=ratios(err,valid).max()+.002) and ok:
            return out
    return h


def twin(route,m,x,y,i,previous,stable_release):
    original=v5.fit(m,x,y,i,previous,stable_release)
    stable=bool(previous and stable_release)
    other=fit_neural(m,x,y,epochs=8 if stable else 6,width=32,stable=False,
                     ordinal_weight=.3,route="twin",norm=original)
    fraction=.50 if route=="twin50" else .25
    h=dict(original)
    h["w_sensor"]=np.column_stack((original["w_sensor"],other["w_sensor"]))
    h["w_base"]=np.r_[original["w_base"],other["w_base"]]
    h["bias"]=np.column_stack((original["bias"],other["bias"]))
    h["output"]=np.r_[(1-fraction)*original["output"],fraction*other["output"]]
    h["intercept"]=(1-fraction)*original["intercept"]+fraction*other["intercept"]
    if "ord_out" in h:
        h["ord_out"]=np.vstack((h["ord_out"],np.zeros((len(other["output"]),3))))
    return h


def apply(route,m,x,y,i,previous=None,stable_release=False):
    if route in ("modecell","nestcell","nestread"):
        inner="mode6" if route=="modecell" else "nest6"
        h=neural_head(inner,m,x,y,i,previous,stable_release)
        if route=="nestread":
            return readout(m,h,x,y,"readfloor",disjoint=bool(previous and stable_release))
        return blend_cells(m,h,x,y,"cellmean")
    if route in ("rankregime","rankfallback"):
        if previous and stable_release:
            return apply("auditcover",m,x,y,i,previous,stable_release)
        h=neural_head("rank6",m,x,y,i,previous,stable_release)
        if route=="rankfallback" and "curve_axis" not in h:
            h=readout(m,h,x,y,"readmean")
        return h
    if route in ("regime","regimebase"):
        if previous and stable_release:
            return apply("auditcover" if route=="regime" else "auditbase",m,x,y,i,previous,stable_release)
        return neural_head("mode6",m,x,y,i,previous,stable_release)
    if route in ("auditmode","auditcover","auditbase"):
        inner={"auditmode":"mode6","auditcover":"cover2","auditbase":"base"}[route]
        h=apply(inner,m,x,y,i,previous,stable_release)
        return readout(m,h,x,y,"readfloor",disjoint=bool(previous and stable_release),expanded=inner=="cover2")
    if route in ("moderead","quadread","coverread","tworead","baseread"):
        inner={"moderead":"mode6","quadread":"quad6","coverread":"cover2","tworead":"twin25","baseread":"base"}[route]
        h=apply(inner,m,x,y,i,previous,stable_release)
        return readout(m,h,x,y,"readfloor")
    if route.startswith("twin"):
        return twin(route,m,x,y,i,previous,stable_release)
    if route=="mix":
        route="cover2" if previous and stable_release else "cellmean"
    if route in ("base","cellfloor","cellmean","safefloor","safemean","readmean","readfloor","readrisk"):
        h=v5.fit(m,x,y,i,previous,stable_release)
    else:
        h=neural_head(route,m,x,y,i,previous,stable_release)
    if route.startswith("read"):
        h=readout(m,h,x,y,route)
    if route.startswith("safe"):
        h=constrained_blend(m,h,x,y,route)
    if route.startswith("cell"):
        h=blend_cells(m,h,x,y,route)
    return h


def predict(m,h,x):
    p=v5.predict(m,h,x)
    if "cell_scale" in h:
        b=m.predict(x).reshape(-1)
        ix=np.searchsorted(h["cell_cuts"],b)*2+(p>b)
        p=np.maximum(0.,b+(p-b)*h["cell_scale"][ix])
    return p


def install(route):
    for name in ("baseline_model","industrial","guard","policy","serving","contest_submission"):
        sys.modules[name]=load_module(name,HERE.parent/"V5"/(name+".py"))
    if route!="frozen":
        engine.fit=lambda m,x,y,i,previous=None,stable_release=False: apply(route,m,x,y,i,previous,stable_release)
        engine.predict=predict
        engine.prepare_inference=lambda m,h: None if ("cell_scale" in h or "curve_axis" in h) else v5.prepare_inference(m,h)


def replay(route,n,setting):
    if n not in range(14): raise ValueError("Only pre-existing development schedules 0–13")
    install(route)
    benchmark.OUT=OUT/route
    benchmark.run("C","v6",n,setting,False)


def probe(dataset,routes):
    x,y,audit,age,units,cruise,m=prior.evaluation_data(dataset)
    stages=[1.,1.] if dataset=="official" else [.55,.7,.85,1.]
    out=[]; old={}
    for i,cut in enumerate(stages,1):
        mask=~audit&(age<=cut); xx=x[mask]; yy=y[mask]
        for route in routes:
            fn=engine.predict if route=="frozen" else predict
            fit=engine.fit if route=="frozen" else lambda *args,**kw: apply(route,*args,**kw)
            t=process_time()
            h=fit(m,xx,yy,i,previous=old.get(route),stable_release=dataset=="official" and i>1)
            dt=process_time()-t;old[route]=h
            p=fn(m,h,x[audit])
            v=prior.diagnostics.metrics(p,y[audit],age[audit],cruise[audit] if cruise is not None else None)
            if route=="base": ref=p
            delta=(p-y[audit])**2-(ref-y[audit])**2
            u=units[audit]
            diffs=np.array([(delta[u==k].sum(),np.sum(u==k)) for k in np.unique(u)])
            picks=np.random.default_rng(411).integers(len(diffs),size=(1000,len(diffs)))
            bs=diffs[picks].sum(1)
            v.update(round=i,route=route,fit_cpu=dt,mse_ci=np.quantile(bs[:,0]/bs[:,1],[.025,.975]).tolist())
            out.append(v)
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/(dataset+"-"+routes[1]+".json")).write_text(json.dumps(out,indent=2))


def grid(routes,settings,splits,workers):
    if any(i not in range(14) for i in splits):raise ValueError("Only 0–13 are available")
    OUT.mkdir(parents=True,exist_ok=True)
    def worker(req):
        r,k,i=req; dest=OUT/r/f"{k}-v6-C-{i}.json"
        if dest.exists():return
        with (OUT/f"{r}-{k}-{i}.log").open("w") as log:
            subprocess.run([sys.executable,"-Bu",__file__,"replay",r,str(i),k],stdout=log,stderr=subprocess.STDOUT,check=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(worker,[(r,k,i) for i in splits for k in settings for r in routes]))


def summarize():
    methods={}
    for folder in sorted(p for p in OUT.iterdir() if p.is_dir()):
        groups={}
        for setting in ("drift","trace","schedule","fresh","official"):
            paired=[]
            for i in range(14):
                name=f"{setting}-v6-C-{i}.json";base=OUT/"base"/name;test=folder/name
                if not base.exists() or not test.exists():continue
                def summarize_stream(path):
                    d=json.loads(path.read_text())
                    row={key:d[key] for key in ("mean","worst","forget","delay")}
                    row.update(nasa=np.mean([r["nasa_mean"] for r in d["rounds"]]),
                               cpu=sum(r["cpu"] for r in d["rounds"]),
                               accepted=sum(r["deployed"]=="candidate" for r in d["rounds"]))
                    return row
                paired.append([i,summarize_stream(base),summarize_stream(test)])
            if paired:
                g={"splits":[r[0] for r in paired]}
                for j,key in [(1,"base"),(2,"candidate")]:
                    g[key]={m:float(np.mean([r[j][m] for r in paired])) for m in paired[0][j]}
                    g[key]["lowest"]=min(r[j]["worst"] for r in paired)
                g["wins"]={m:int(sum((r[2][m]-r[1][m])*(1 if m in ("mean","worst","accepted") else -1)>1e-10 for r in paired)) for m in paired[0][1]}
                groups[setting]=g
        if groups:methods[folder.name]=groups
    (OUT/"paired.json").write_text(json.dumps(methods,indent=2))


def verify(expected):
    import hashlib
    result=[]
    for dataset in ("official","DS02"):
        x,y,audit,age,_,_,m=prior.evaluation_data(dataset)
        cut=[1.,1.] if dataset=="official" else [.55,1.]
        ids=np.flatnonzero(audit)
        sample=x[ids[np.linspace(0,len(ids)-1,512,dtype=int)]]
        prev_ref=prev_live=None
        for i,fraction in enumerate(cut,1):
            mask=~audit&(age<=fraction);xx,yy=x[mask],y[mask];same=dataset=="official" and i==2
            h=engine.fit(m,xx,yy,i,previous=prev_live,stable_release=same)
            ref=apply(expected,m,xx,yy,i,previous=prev_ref,stable_release=same)
            live=engine.predict(m,h,sample);target=predict(m,ref,sample)
            fn=engine.prepare_inference(m,h);sc=np.asarray([fn(row) for row in sample])
            np.testing.assert_allclose(live,target,atol=1e-8,rtol=1e-10)
            np.testing.assert_allclose(sc,target,atol=1e-8,rtol=1e-10)
            result.append(dict(dataset=dataset,round=i,rows=len(sc),arrays_bytes=sum(a.nbytes for a in h.values()),
                               batch_delta=float(np.max(abs(live-target))),scalar_delta=float(np.max(abs(sc-target)))))
            prev_ref=ref;prev_live=h
    (OUT/(expected+"-runtime.json")).write_text(json.dumps(dict(sha256=hashlib.sha256(Path(engine.__file__).read_bytes()).hexdigest(),tests=result),indent=2))


def speed():
    if hasattr(os,"sched_setaffinity"):
        os.sched_setaffinity(0,{max(os.sched_getaffinity(0))})
    x,y,au,_,_,_,m=prior.evaluation_data("official")
    xx=x[~au];yy=y[~au];v=x[au][np.linspace(0,au.sum()-1,1024,dtype=int)]
    out=[]
    for rep in range(3):
        variants=[("base",v5),("frozen",engine)]
        for label,mod in (variants if rep%2==0 else reversed(variants)):
            prev=None
            for i in (1,2):
                start=process_time();h=mod.fit(m,xx,yy,i,previous=prev,stable_release=i==2);cpu=process_time()-start
                prev=h;fn=mod.prepare_inference(m,h);target=mod.predict(m,h,v)
                timings=[]
                for _ in range(5):
                    t=process_time();pred=np.array([fn(row) for row in v]);timings.append((process_time()-t)*1e6/len(v))
                np.testing.assert_allclose(pred,target,rtol=1e-10,atol=1e-9)
                out.append(dict(variant=label,repeat=rep,round=i,fit_s=cpu,single_us=float(np.median(timings)),
                                bytes=sum(a.nbytes for a in h.values())))
    (OUT/"speed.json").write_text(json.dumps(out,indent=2))


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("mode");p.add_argument("name",nargs="?",default="");p.add_argument("args",nargs="*");a=p.parse_args()
    if a.mode=="summary":summarize()
    elif a.mode=="verify":verify(a.name)
    elif a.mode=="speed":speed()
    elif a.mode=="replay":replay(a.name,int(a.args[0]),a.args[1])
    elif a.mode=="probe":probe(a.name,a.args)
    elif a.mode=="grid":grid(a.name.split(","),a.args[0].split(","),list(map(int,a.args[1].split(","))),int(a.args[2]))
