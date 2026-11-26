"""Development replays, independent fleet probes and timing for V4 engine heads."""
import argparse
import importlib.util
import json
import sys
from pathlib import Path
from time import process_time
import numpy as np
import benchmark
import engine

HERE = Path(__file__).resolve().parent
OUT = HERE.parent/"work/v4/engine"


def legacy(name, path):
    spec=importlib.util.spec_from_file_location(name,path)
    mod=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


v3=legacy("enterprise_v3_engine", HERE.parent/"V3/engine.py")
study=legacy("engine_diagnostics", HERE.parent/"V3/engine_v3_study.py")


def train_member(model, x, y, limit, epochs, width, seed, common=None, ordinal=0., balance=0.):
    rng = np.random.default_rng(seed)
    if len(x) > limit:
        if balance:
            rounded=np.round(x[:,[-3,-1]])
            _,g=np.unique(rounded,axis=0,return_inverse=True)
            # Equal-size lifetime intervals make rare long-lived regimes explicit.
            low,high=np.quantile(y,[.01,.99])
            life=np.minimum(7,np.maximum(0,((y-low)/max(1.,high-low)*8).astype(int)))
            bucket=g*8+life
            counts=np.bincount(bucket)
            importance=counts[bucket].astype(float)**(-balance)
            sample=rng.choice(len(x),limit,replace=False,p=importance/importance.sum())
        else:
            sample = rng.choice(len(x), limit, replace=False)
        x, y = x[sample], y[sample]
    if common is None:
        centers, counts = np.unique(np.round(x[:, [-3, -1]]), axis=0, return_counts=True)
        centers = centers[np.argsort(-counts)[:8]]
        groups = v3._regimes(x, centers)
        means = np.stack([x[groups == k, :-3].mean(0) for k in range(len(centers))])
        scales = np.stack([x[groups == k, :-3].std(0) for k in range(len(centers))])
        inv_scale = 1 / np.maximum(scales, 1e-4)
    else:
        centers, means, inv_scale = (common[k] for k in ("centers","means","inv_scale"))
        groups = v3._regimes(x, centers)
    sensors = np.clip((x[:, :-3] - means[groups]) * inv_scale[groups], -6, 6)
    base = model.predict(x).reshape(-1)
    design = np.column_stack((sensors, (base - 100) / 60, np.eye(len(centers))[groups]))
    target = (y - base) / 55

    weights = rng.normal(0, .22, (design.shape[1], width))
    bias = np.zeros(width)
    output = rng.normal(0, .03, width)
    intercept = np.zeros(1)
    params = [weights, bias, output, intercept]
    if ordinal:
        thresholds = np.quantile(y,[.20,.45,.70])
        near = (y[:,None] < thresholds).astype(float)
        ordinal_w = np.zeros((width,3))
        ordinal_b = np.zeros(3)
        params.extend((ordinal_w,ordinal_b))
    first = [np.zeros_like(p) for p in params]
    second = [np.zeros_like(p) for p in params]
    step = 0
    for epoch in range(epochs):
        order = rng.permutation(len(y))
        for idx in np.array_split(order, max(1, len(y) // 256)):
            h = np.tanh(design[idx] @ weights + bias)
            delta = (h @ output + intercept - target[idx]) * (2 / len(idx))
            back = delta[:, None] * output
            if ordinal:
                logits=np.clip(h @ ordinal_w+ordinal_b,-30,30)
                aux=(1/(1+np.exp(-logits))-near[idx])*(ordinal/(3*len(idx)))
                back=back+aux@ordinal_w.T
            back=back*(1-h*h)
            gradients = [design[idx].T @ back, back.sum(0), h.T @ delta, np.array([delta.sum()])]
            if ordinal:
                gradients.extend((h.T@aux,aux.sum(0)))
            step += 1
            for j, grad in enumerate(gradients):
                first[j] = .9 * first[j] + .1 * grad
                second[j] = .999 * second[j] + .001 * grad ** 2
                rate = .003 / (1 + epoch / 10)
                params[j] -= (rate * first[j] / (1 - .9 ** step) /
                              (np.sqrt(second[j] / (1 - .999 ** step)) + 1e-8))
    # Fold the regime intercept and output scaling once for fast single-row serving.
    return dict(centers=centers, means=means, inv_scale=inv_scale,
                w_sensor=weights[:-len(centers)-1].copy(), w_base=weights[-len(centers)-1] / 60,
                bias=bias + weights[-len(centers):] - weights[-len(centers)-1] * (100 / 60),
                output=55 * output, intercept=55 * intercept)

def twin_model(route,m,x,y,previous,stable):
    long=stable and bool(previous)
    if route.startswith(("balance","ordinal")):
        limit=16384 if "lean" in route else 24576
        aux=(.7 if "strong" in route else .2) if route.startswith("ordinal") else 0.
        balance=(.35 if "strong" in route else .13) if route.startswith("balance") else 0.
        state=train_member(m,x,y,limit,20 if long else 12,48 if long else 64,145,ordinal=aux,balance=balance)
    elif route=="twinblend":
        first=train_member(m,x,y,24576,20 if long else 12,48 if long else 64,145)
        second=train_member(m,x,y,12288,10,32,370,common=first)
        weight=.30
    else:
        full=route.startswith("twinfull")
        width=40 if full else 32
        limit=24576 if full else 16384
        epochs=(18 if long else 12) if full else (15 if long else 9)
        first=train_member(m,x,y,limit,epochs,width,145)
        second=train_member(m,x,y,limit,epochs,width,370,common=first)
        weight=.5
    if not route.startswith(("ordinal","balance")):
        state={name:first[name] for name in ("centers","means","inv_scale")}
        for name in ("w_sensor","w_base","bias"):
            state[name]=np.concatenate((first[name],second[name]),axis=-1)
        state["output"]=np.concatenate(((1-weight)*first["output"],weight*second["output"]))
        state["intercept"]=(1-weight)*first["intercept"]+weight*second["intercept"]
    ids=np.linspace(0,len(y)-1,min(len(y),2048),dtype=int)
    p=v3.predict_neural(m,state,x[ids],65.)
    lo,hi=np.quantile(p,[.35,.65])
    state.update(health_lo=np.array(lo),health_hi=np.array(hi),health_amp=np.array(np.clip(np.std(y)/max(1,np.std(p))-1,.15,.7)),
                 extension_fraction=np.array(.50))
    return state


def fit_head(model,x,y,i,previous,stable,limit):
    if limit==24576:
        return v3.fit(model,x,y,i,previous,stable)
    long_fit=stable and bool(previous)
    state=v3.fit_neural(model,x,y,limit=limit,epochs=20 if long_fit else 12,width=48 if long_fit else 64)
    ids=np.linspace(0,len(y)-1,min(len(y),2048),dtype=int)
    pred=v3.predict_neural(model,state,x[ids],65.)
    lo,hi=np.quantile(pred,[.35,.65])
    state.update(health_lo=np.array(lo),health_hi=np.array(hi),
                 health_amp=np.array(np.clip(np.std(y)/max(1,np.std(pred))-1,.15,.7)),
                 extension_fraction=np.array(.5))
    return state


def rbf_inputs(h,x,p,groups):
    z=np.clip((x[:,:-3]-h["means"][groups])*h["inv_scale"][groups],-3,3)
    return np.column_stack((z,(p-100)/40))


def rbf(route,m,h,x,y,p,groups):
    z=rbf_inputs(h,x,p,groups)
    k=16
    h.update(v4_rbf_center=np.zeros((len(h["centers"]),k,z.shape[1])),
             v4_rbf_width=np.ones(len(h["centers"])),v4_rbf_coef=np.zeros((len(h["centers"]),k+1)))
    for g in range(len(h["centers"])):
        u=z[groups==g]
        if len(u)<k*2:
            continue
        r=(y-p)[groups==g]
        centers=u[np.random.default_rng(71).choice(len(u),k,replace=False)].copy()
        for it in range(5):
            d=np.maximum(0.,(u*u).sum(1)[:,None]+(centers*centers).sum(1)-2*u@centers.T)
            group=d.argmin(1)
            for j in np.unique(group):
                centers[j]=u[group==j].mean(0)
        d=np.maximum(0.,(u*u).sum(1)[:,None]+(centers*centers).sum(1)-2*u@centers.T)
        scale=max(1.,np.quantile(d.min(1),.75)*2)
        basis=np.column_stack((np.exp(-d/scale),np.ones(len(u))))
        w=np.ones(len(u))
        if "risk" in route:
            w+=np.minimum(2.,abs(r)/40)
        gram=basis.T@(basis*w[:,None])+.018*len(u)*np.eye(k+1)
        h["v4_rbf_center"][g]=centers
        h["v4_rbf_width"][g]=scale
        h["v4_rbf_coef"][g]=np.linalg.solve(gram,basis.T@(r*w))
    return h


def rbf_prediction(h,x,p):
    g=v3._regimes(x,h["centers"])
    z=rbf_inputs(h,x,p,g)
    result=p.copy()
    for k in np.unique(g):
        subgroup=g==k
        u=z[subgroup];c=h["v4_rbf_center"][k]
        d=np.maximum(0.,(u*u).sum(1)[:,None]+(c*c).sum(1)-2*u@c.T)
        a=h["v4_rbf_coef"][k]
        result[subgroup]+=np.clip(.5*(np.exp(-d/h["v4_rbf_width"][k])@a[:-1]+a[-1]),-12,12)
    return np.maximum(0.,result)


def fit_calibration(route,m,h,x,y):
    if route.startswith(("base","twin","memory")):
        return h
    ids=study.calibration_rows(len(y))
    if len(ids)>8192:
        ids=ids[:8192]
    xx,yy=x[ids],y[ids]
    p=v3.predict(m,h,xx)
    groups=v3._regimes(xx,h["centers"])
    if route.startswith("rbf"):
        return rbf(route,m,h,xx,yy,p,groups)
    if "sensor" in route or route.startswith("learned"):
        z=np.clip((xx[:,:-3]-h["means"][groups])*h["inv_scale"][groups],-4,4)
        axes=np.zeros((len(h["centers"]),z.shape[1]))
        # Estimate a within-condition sensor axis between high and low service life.
        for j in range(len(axes)):
            mask=groups==j
            if np.count_nonzero(mask)<32:
                continue
            zy=z[mask];labels=yy[mask]
            bottom,top=np.quantile(labels,[.25,.75])
            diff=zy[labels>=top].mean(0)-zy[labels<=bottom].mean(0)
            zc=zy-zy.mean(0)
            covariance=zc.T@zc/len(zy)+.22*np.eye(zy.shape[1])
            axes[j]=np.linalg.solve(covariance,diff)
        h["v4_axes"]=axes
        health=np.sum(z*axes[groups],axis=1)
    else:
        health=p
    stops=[]
    for j in range(len(h["centers"])):
        values=health[groups==j]
        # Moderate life estimates have uncertain health and keep the V3 correction.
        stops.append(np.quantile(values,[.40,.78]) if len(values)>=16 else np.quantile(health,[.4,.78]))
    h["v4_stops"]=np.asarray(stops)
    if route.startswith("learned"):
        # Select a shrinkage for each health stratum on released rows not used by the head.
        b=m.predict(xx)
        target=yy-p
        gg=groups
        knots=np.zeros((len(stops),4))
        adjust=np.zeros_like(knots)
        for j in range(len(stops)):
            sel=np.flatnonzero(gg==j)
            if len(sel)<32:
                continue
            for k,rows in enumerate(np.array_split(sel[np.argsort(health[sel])],4)):
                knots[j,k]=np.median(health[rows])
                diff=np.maximum(0.,b[rows]-p[rows])
                adjust[j,k]=np.clip((diff@target[rows])/(diff@diff+len(rows)*60),0,1)
        h.update(v4_knots=knots,v4_adjust=adjust)
    # Primitive array states are safe to archive and use without device metadata.
    h["v4_down"]=np.array(0. if route.startswith(("up","contrast")) else .70 if "strong" in route else .35)
    h["v4_contrast"]=np.array(route.startswith("contrast"))
    h["v4_up"]=np.array(.60 if route.startswith("contrast") else .45 if route.startswith("up") else .30 if "lift" in route else 0.)
    return h


def corrected(m,h,x):
    p=v3.predict(m,h,x)
    if "previous_output" in h:
        old={k[9:]:v for k,v in h.items() if k.startswith("previous_")}
        before=v3.predict(m,old,x)
        change=before-p
        if h["smoothing_mode"]>=1:
            change=np.maximum(change,0)
        if h["smoothing_mode"]==2:
            change=change*np.clip((p-h["health_lo"])/max(1,float(h["health_hi"]-h["health_lo"])),0,1)
        return np.maximum(0,p+np.clip(.4*change,-12,12))
    if "v4_rbf_center" in h:
        return rbf_prediction(h,x,p)
    if "v4_stops" not in h:
        return p
    g=v3._regimes(x,h["centers"])
    if "v4_axes" in h:
        z=np.clip((x[:,:-3]-h["means"][g])*h["inv_scale"][g],-4,4)
        health=np.sum(z*h["v4_axes"][g],axis=1)
    else:
        health=p
    b=m.predict(x).reshape(-1)
    bounds=h["v4_stops"][g]
    ramp=np.clip((health-bounds[:,0])/np.maximum(1e-6,bounds[:,1]-bounds[:,0]),0,1)
    down=ramp*float(h["v4_down"])
    if "v4_adjust" in h:
        for j in np.unique(g):
            u=g==j
            down[u]=np.interp(health[u],h["v4_knots"][j],h["v4_adjust"][j])
    up=(2*ramp-1) if bool(h.get("v4_contrast",False)) else ramp
    delta=down*np.maximum(0,b-p)+up*float(h["v4_up"])*np.maximum(0,p-b)
    return np.maximum(0.,p+np.clip(delta,-14.,14.))


def fit(route,m,x,y,i,previous=None,stable_release=False):
    if route.startswith("memory"):
        h=v3.fit(m,x,y,i,previous,stable_release)
        if previous and "output" in previous:
            for k in tuple(h):
                if k in previous:
                    h["previous_"+k]=previous[k]
            h["smoothing_mode"]=np.array({"memoryall":0,"memorydown":1,"memoryhealth":2}[route])
        return h
    if route.startswith("schedule"):
        selected="ordinalstrong" if not stable_release or not previous else "ordinal" if route=="schedulelight" else "base"
        return fit(selected,m,x,y,i,previous,stable_release)
    if route.startswith(("ordinal","twin","balance")):
        return twin_model(route,m,x,y,previous,stable_release)
    core=route.split("-")[0]
    limit=int(route.split("-")[1]) if "-" in route else 24576
    return fit_calibration(core,m,fit_head(m,x,y,i,previous,stable_release,limit),x,y)


def replay(route,n,setting):
    if n not in range(8):
        raise ValueError("C development uses splits 0–7 only")
    # Isolate the task head: retain the V3 controller even after the V4 shared policy is frozen.
    for name in ("baseline_model","industrial","guard","policy","contest_submission"):
        sys.modules[name]=legacy(name,HERE.parent/"V3"/(name+".py"))
    benchmark.OUT=OUT/route
    if route!="frozen":
        if hasattr(engine,"fit_candidates"):
            del engine.fit_candidates
        if route.startswith("family"):
            def variants(m,x,y,i,previous=None,stable_release=False):
                current=v3.fit(m,x,y,i,previous,stable_release)
                tag=route[6:]
                return {"engine":current,"health_adapted":fit_calibration(tag,m,dict(current),x,y)}
            engine.fit_candidates=variants
        engine.fit=lambda m,x,y,i,previous=None,stable_release=False:fit(route,m,x,y,i,previous,stable_release)
        engine.predict=corrected
    benchmark.run("C","v4",n,setting,True)


def probe(dataset,routes):
    x,y,audit,age,units,cruise,m=study.data(dataset)
    stages=[.55,.7,.85,1.] if dataset!="official" else [1.,1.]
    out=[]
    previous={}
    for i,cut in enumerate(stages,1):
        train=~audit&(age<=cut)
        xx,yy=x[train],y[train]
        memo={}
        for route in routes:
            limit=int(route.split("-")[1]) if "-" in route else 24576
            if limit not in memo:
                start=process_time()
                memo[limit]=fit_head(m,xx,yy,i,previous.get(limit),dataset=="official" and i>1,limit)
                previous[limit]=memo[limit]
                memo["time"+str(limit)]=process_time()-start
            start=process_time()
            if route.startswith(("ordinal","twin","balance")):
                h=twin_model(route,m,xx,yy,previous.get(limit),dataset=="official" and i>1)
                fn=corrected;sec=process_time()-start
            elif route=="frozen":
                h=engine.fit(m,xx,yy,i,previous={} if i==1 else memo[limit],stable_release=dataset=="official" and i>1)
                fn=engine.predict
                sec=process_time()-start
            else:
                h=fit_calibration(route.split("-")[0],m,dict(memo[limit]),xx,yy)
                fn=corrected
                sec=memo["time"+str(limit)]+process_time()-start
            p=fn(m,h,x[audit])
            ref=v3.predict(m,memo[24576],x[audit])
            row=study.metrics(p,y[audit],age[audit],cruise[audit] if cruise is not None else None)
            row.update(route=route,round=i,release=cut,fit_s=sec)
            if i==len(stages):
                ids=np.linspace(0,len(p)-1,min(256,len(p)),dtype=int)
                values=x[audit][ids]
                start=process_time()
                single=np.array([fn(m,h,z[None])[0] for z in values])
                row["scalar_us"]=1e6*(process_time()-start)/len(ids)
                np.testing.assert_allclose(single,p[ids],atol=1e-9,rtol=1e-11)
                group=units[audit]
                du=(p-y[audit])**2-(ref-y[audit])**2
                means=np.array([(du[group==g].sum(),(group==g).sum()) for g in np.unique(group)])
                rng=np.random.default_rng(31);sums=means[rng.integers(len(means),size=(1000,len(means)))].sum(1)
                row["paired_mse_ci"]=np.quantile(sums[:,0]/sums[:,1],[.025,.975]).tolist()
            out.append(row)
    OUT.mkdir(exist_ok=True,parents=True)
    (OUT/(dataset+"-"+routes[1]+"-probe.json")).write_text(json.dumps(out,indent=2))


def precision_and_speed():
    """Compare both loss schedules on real rows; batch and prepared scalar must agree."""
    import os
    allowed=sorted(os.sched_getaffinity(0))
    os.sched_setaffinity(0,{allowed[-1]})
    x,y,audit,age,_,_,model=study.data("official")
    query=x[audit][np.linspace(0,np.count_nonzero(audit)-1,1024,dtype=int)]
    report=[]
    for stable in (False,True):
        previous={"prior":np.array(1)} if stable else {}
        gold=twin_model("ordinal" if stable else "ordinalstrong",model,x[~audit],y[~audit],previous,stable)
        timings={"v3":[],"v4":[]}
        states={}
        for i in range(3):
            versions=["v4","v3"] if i%2 else ["v3","v4"]
            for name in versions:
                module=engine if name=="v4" else v3
                start=process_time()
                states[name]=module.fit(model,x[~audit],y[~audit],2 if stable else 1,previous,stable)
                timings[name].append(process_time()-start)
        reference=v3.predict(model,gold,query)
        p=engine.predict(model,states["v4"],query)
        old_state=states["v3"]
        old_expected=v3.predict(model,old_state,query)
        for label,h,expected in [("v3",old_state,old_expected),("v4",states["v4"],p)]:
            fn=engine.prepare_inference(model,h)
            scalar=[];batch1=[]
            for i in range(7):
                t0=process_time()
                one=np.array([fn(z) for z in query])
                scalar.append((process_time()-t0)*1e6/len(query))
                t0=process_time()
                ref=np.array([v3.predict(model,h,z[None])[0] for z in query])
                batch1.append((process_time()-t0)*1e6/len(query))
            np.testing.assert_allclose(one,expected,atol=1e-9,rtol=1e-11)
            np.testing.assert_allclose(one,ref,atol=1e-9,rtol=1e-11)
            report.append(dict(stable=stable,head=label,fit_s=timings[label],
                    scalar_us=float(np.median(scalar)),old_scalar_us=float(np.median(batch1)),
                    scalar_batch_max=float(np.max(abs(one-expected))),
                    fp32_vs_double_max=float(np.max(abs(p-reference))),
                    fp32_vs_double_rmse=float(np.sqrt(np.mean((p-reference)**2)))))
    OUT.mkdir(exist_ok=True,parents=True)
    (OUT/"precision-speed.json").write_text(json.dumps(report,indent=2))


if __name__=="__main__":
    arg=argparse.ArgumentParser()
    arg.add_argument("mode",choices=["probe","replay","speed"])
    arg.add_argument("name",nargs="?")
    arg.add_argument("rest",nargs="*")
    args=arg.parse_args()
    if args.mode=="speed":
        precision_and_speed()
    elif args.mode=="probe":
        probe(args.name,args.rest)
    else:
        replay(args.name,int(args.rest[0]),args.rest[1])
