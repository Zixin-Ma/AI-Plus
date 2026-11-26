"""CPU-only task-C ablation on disjoint engines; external DS02 is a native-schema transfer."""
from pathlib import Path
import argparse
import json
import time
import sys
import numpy as np

import engine
import contest_submission as harness
from benchmark import partitions

HERE = Path(__file__).resolve().parent
WORK = HERE.parent / "work" / "engine"
SCHOOL = HERE.parents[1] / "校赛/work/C"


def save(name, data):
    WORK.mkdir(exist_ok=True, parents=True)
    (WORK/name).write_text(json.dumps(data,indent=2,allow_nan=False))


def load(split, m):
    with (SCHOOL/"public"/f"{split}.jsonl").open() as f:
        rows = [json.loads(line) for line in f]
    x,y,_ = harness.projection(rows,m,"C")
    age,phase,even = partitions(rows,"C",0)
    units = np.array([r["meta"]["unit"] for r in rows])
    fd = np.array([r["meta"]["source_subset"] for r in rows])
    return x,y,phase,age,even,units,fd


def score(y,p,phase,fd):
    err = p-y
    def rmse(mask):
        return float(np.mean(err[mask]**2)**.5)
    return dict(rmse=rmse(slice(None)),stage=[rmse(phase==j) for j in range(4)],
                worst_fd=max(rmse(fd==j) for j in np.unique(fd)),
                mae=float(np.mean(abs(err))),nasa=float(np.expm1(np.where(err>=0,err/10,-err/13)).sum()))


def compare(m,x,y,v,t,phase,fd):
    outputs = {"static": m.predict(v)}
    cpu={}
    for name,limit,epochs,width in (("neural48",24576,20,48),("neural64",32768,32,64)):
        t0=time.process_time()
        head=engine.fit_neural(m,x,y,limit,epochs,width)
        cpu[name]=time.process_time()-t0
        outputs[name]=engine.predict_neural(m,head,v,65.)
    t0=time.process_time()
    h=fit_spectral(m,x,y)
    cpu["spectral"]=time.process_time()-t0
    for bound in (40.,65.):
        outputs[f"spectral{bound:g}"]=spectral_predict(m,h,v,bound)
    outputs["blend"]=(outputs["neural48"]+outputs["spectral65"])/2
    for name in ["neural48","neural64","spectral40","spectral65"]:
        p=outputs[name]
        # The released targets supply the location and spread; each fixed
        # contrast amplifies healthy-versus-worn differences using current measurements.
        for factor in (.25,.5,1.):
            center=float(np.median(y))
            amp=min(1.,np.std(y)/max(1.,np.std(m.predict(x)))-1.)
            outputs[f"{name}-contrast-{factor}"]=np.maximum(0.,p+np.clip(factor*max(.35,amp)*(p-center),-35,35))
        outputs[f"{name}-healthgate"]=outputs["static"] + (p-outputs["static"])*(.4+.6*np.clip(abs(p-center)/max(25.,np.std(y)),0,1))
    result={k:score(t,p,phase,fd) for k,p in outputs.items()}
    print("scores", {k:round(v["rmse"],3) for k,v in result.items()},flush=True)
    return dict(metrics=result,fit_cpu_s=cpu)


def fit_spectral(model, x, y, limit=32768, width=144):
    """Fit a regularized residual on whitened sensor channels within each operating regime."""
    rng = np.random.default_rng(302)
    if len(y) > limit:
        take = rng.choice(len(y), limit, replace=False)
        x, y = x[take], y[take]
    settings, counts = np.unique(np.round(x[:, [-3,-1]]), axis=0, return_counts=True)
    centers = settings[np.argsort(-counts)[:8]]
    groups = engine._regimes(x, centers)
    d = x.shape[1]-3
    projection = rng.normal(0, 1/np.sqrt(d), (d,width))
    projection *= np.resize([.5,1.5,3.],width)
    bias = rng.uniform(-1.,1.,width)
    target_features = np.zeros((len(x),width+2*d+2))
    means, transforms = [], []
    base = model.predict(x).reshape(-1)
    for i in range(len(centers)):
        take = groups == i
        block = x[take,:d]
        mean = block.mean(0)
        scale = np.maximum(block.std(0), 1e-4)
        z = (block-mean)/scale
        val, axes = np.linalg.eigh(z.T@z/len(z))
        # Shrink the small eigenvalues: near-constant physical channels are common.
        trans = ((axes / np.sqrt(np.maximum(.20,val))) @ axes.T) / scale[:,None]
        z = np.clip((block-mean)@trans,-6,6)
        target_features[take] = _basis(z,base[take],projection,bias)
        means.append(mean)
        transforms.append(trans)
    gram = target_features.T @ target_features
    lam = len(y)*.001
    gram.flat[::len(gram)+1] += lam
    residual = y-base
    shared = np.linalg.solve(gram, target_features.T@residual)
    coefs = []
    for i in range(len(centers)):
        take = groups == i
        X = target_features[take]
        G = X.T@X
        G.flat[::len(G)+1] += len(X)*.03 + lam
        local = np.linalg.solve(G, X.T@(residual[take] - X@shared))
        coefs.append(shared+local)
    return dict(centers=centers,means=np.stack(means),transforms=np.stack(transforms),
                projection=projection, spectral_bias=bias, coefs=np.stack(coefs))


def _basis(z, base, proj, bias):
    return np.column_stack((np.ones(len(z)), base/100, z, z*z, np.tanh(z@proj+bias)))


def spectral_predict(model, head, x, limit=65.):
    base = model.predict(x).reshape(-1)
    group = engine._regimes(x, head["centers"])
    result = base.copy()
    for i in np.unique(group):
        idx = group==i
        z = np.clip((x[idx,:-3]-head["means"][i])@head["transforms"][i], -6,6)
        phi = _basis(z,base[idx],head["projection"],head["spectral_bias"])
        result[idx] += np.clip(phi @ head["coefs"][i],-limit,limit)
    return np.maximum(result,0.)

def health_fit(model, x, y, index, route):
    if route == "spectral":
        head = fit_spectral(model,x,y)
    elif route.startswith("ridge"):
        epochs = int(route[5:])
        head = engine.fit_neural(model,x,y,24576,epochs,48)
        head = refine_readout(model,head,x,y)
    elif route == "lean":
        head = engine.fit_neural(model,x,y,24576,12,48)
    else:
        head = engine.fit_neural(model,x,y,24576,20,48)
    return attach_contrast(head,model,x,y)


def attach_contrast(head, model, x, y):
    # Estimate a central health band from train-domain predictions. Tail adjustments
    # act outside the band and always have an absolute maximum in target units.
    ids = np.linspace(0,len(x)-1,min(len(x),2048),dtype=int)
    pred = raw_predict(model,head,x[ids])
    lo,hi = np.quantile(pred,[.35,.65])
    amp = np.clip(np.std(y)/max(1.,np.std(pred)) - 1., .15,.7)
    head.update(health_lo=np.array(lo),health_hi=np.array(hi),health_amp=np.array(amp))
    return head


def raw_predict(model, head, x):
    if "coefs" in head:
        return spectral_predict(model,head,x)
    return engine.predict_neural(model,head,x,65.)


def health_predict(model, head, x, mode):
    p = raw_predict(model,head,x)
    if "health_amp" not in head or mode == "none":
        return p
    correction = np.maximum(0.,p-head["health_hi"])
    if mode != "upper":
        correction += np.minimum(0.,p-head["health_lo"])
    factor = .2 if mode=="mild" else 1.
    return np.maximum(0.,p+np.clip(factor*correction*head["health_amp"],-25,25))

def refine_readout(model, head, x, y, limit=24576, strength=.0015):
    """Solve the final layer against released training residuals after Adam feature learning."""
    ids = np.random.default_rng(739).choice(len(y), min(limit,len(y)), replace=False)
    x, y = x[ids],y[ids]
    base = model.predict(x)
    group = engine._regimes(x,head["centers"])
    z = np.clip((x[:,:-3]-head["means"][group])*head["inv_scale"][group],-6,6)
    hidden = np.tanh(z@head["w_sensor"]+base[:,None]*head["w_base"]+head["bias"][group])
    residual = y - base - (hidden@head["output"]+head["intercept"][0])
    mean = hidden.mean(0)
    centered = hidden - mean
    gram = centered.T@centered
    gram.flat[::len(gram)+1] += len(x)*strength
    change = np.linalg.solve(gram,centered.T@residual)
    head["output"] += change
    head["intercept"] += residual.mean() - mean@change
    return head


def internal():
    official=harness.Model.load(SCHOOL/"checkpoint.npz")
    xt,yt,pt,age,ev,unit,fd=load("train",official)
    val=load("validation",official)
    results={}
    for split in ("development","train_fold"):
        if split=="development":
            hold=val[4]
            xv,yv,pv,fv=[val[j][hold] for j in (0,1,2,6)]
            available=np.ones(len(xt),dtype=bool)
        else:
            hold=unit%5==2
            xv,yv,pv,fv=[a[hold] for a in (xt,yt,pt,fd)]
            available=~hold
        first=available&(age<=.55)
        meta=official.meta
        m=harness.Model(meta["task"], meta["paths"], meta["classes"],xt[first],yt[first],
                        hidden=meta["hidden"],seed=meta["seed"],batch_size=meta["batch_size"])
        m.fit(xt[first],yt[first],15)
        for label,mask in (("early",first),("full",available)):
            print(split,label,flush=True)
            results[split+"-"+label]=compare(m,xt[mask],yt[mask],xv,yv,pv,fv)
    save("internal.json",results)


def extract_external():
    """One observation every 128 ticks. Align cruise intervals by engine and cycle."""
    import h5py
    from zipfile import ZipFile
    from tempfile import TemporaryDirectory
    import shutil
    from contextlib import ExitStack
    ext = HERE.parent / "work/external"
    src = HERE.parents[1] / "Supp_Datasets"
    maskfile, = (src/"C/02_CruiseBench_CPM").glob("*.h5")
    with h5py.File(maskfile) as f:
        cycles = f["cycles"][:]
        ranges = {(int(r["unit"]),int(r["cycle"])):(int(r["start_sample"]),int(r["end_sample"]))
                  for r in cycles if r["dataset"]==b"DS02-006" and r["split"]==b"dev" and r["detected"]}
    WORK.mkdir(parents=True,exist_ok=True)
    with ExitStack() as stack:
        ds = ext/"ds02.h5"
        if not ds.is_file():
            tmp = Path(stack.enter_context(TemporaryDirectory(dir=WORK)))
            inner = ext/"N-CMAPSS-inner.zip"
            if not inner.is_file():
                inner = tmp/"inner.zip"
                outer, = (src/"C/01_N_CMAPSS").glob("*.zip")
                with ZipFile(outer) as archive, archive.open(next(n for n in archive.namelist() if n.endswith(".zip"))) as source, inner.open("wb") as target:
                    shutil.copyfileobj(source,target,8*1024*1024)
            ds=tmp/"ds02.h5"
            with ZipFile(inner) as archive, archive.open("data_set/N-CMAPSS_DS02-006.h5") as source, ds.open("wb") as target:
                shutil.copyfileobj(source,target,8*1024*1024)
        with h5py.File(ds) as f:
            identity=f["A_dev"][:,:2].astype(np.int32)
            changes=np.r_[0,1+np.flatnonzero(np.any(identity[:-1]!=identity[1:],axis=1)),len(identity)]
            idx=np.arange(0,len(identity),128)
            cruise=np.zeros(len(idx),bool)
            for lo,hi in zip(changes[:-1],changes[1:]):
                u,c=identity[lo]
                if (u,c) in ranges:
                    a,b=ranges[u,c]
                    if not 0<=a<=b<hi-lo:
                        raise ValueError("inconsistent cruise mask")
                    cruise[np.searchsorted(idx,lo+a):np.searchsorted(idx,lo+b,side="right")] = True
            sensors=f["X_s_dev"][::128]
            settings=f["W_dev"][::128]
            y=f["Y_dev"][::128,0]
            np.savez_compressed(WORK/"external.npz",sensors=sensors,settings=settings,y=y,
                                cruise=cruise,unit=identity[idx,0],cycle=identity[idx,1])


def external():
    with np.load(WORK/"external.npz") as a:
        sen,w,y,cruise,u,cycle=[a[k] for k in ("sensors","settings","y","cruise","unit","cycle")]
    units=np.unique(u)
    fit=np.isin(u,units[:-2]); val=~fit
    # The DS02 dataset provides continuous altitude, Mach, throttle and inlet T.
    # Train-derived discrete altitude/Mach strata support the same normalization.
    alt=np.searchsorted(np.quantile(w[fit,0],[.25,.50,.75]),w[:,0])
    mach=(w[:,1]>np.median(w[fit,1])).astype(float)
    x=np.column_stack((sen,w,alt, w[:,1],mach))
    phase=np.empty(len(y),int)
    for unit in units:
        ii=u==unit
        phase[ii]=np.minimum(3,(4*(cycle[ii]-cycle[ii].min())/max(1,np.ptp(cycle[ii])+1)).astype(int))
    m=harness.Model("regression",[str(i) for i in range(x.shape[1])],[],x[fit],y[fit],hidden=32)
    m.fit(x[fit],y[fit],15)
    subsets={"all":val,"cruise":val&cruise,"noncruise":val&~cruise}
    models={"static":{},"neural48":engine.fit_neural(m,x[fit],y[fit]),"spectral":fit_spectral(m,x[fit],y[fit]),
            "enterprise":engine.fit(m,x[fit],y[fit],1)}
    result={}
    for name,h in models.items():
        pred=(m.predict(x) if name=="static" else spectral_predict(m,h,x) if name=="spectral"
              else engine.predict_neural(m,h,x,65) if name=="neural48" else engine.predict(m,h,x))
        result[name]={g:score(y[ix],pred[ix],phase[ix],u[ix]) for g,ix in subsets.items()}
    report=dict(train_units=units[:-2].tolist(),eval_units=units[-2:].tolist(),train=int(fit.sum()),
                eval=int(val.sum()),cruise_eval=int((val&cruise).sum()),scores=result)
    save("external.json",report)
    print({name:{g:round(s["rmse"],2) for g,s in d.items()} for name,d in result.items()})



def replay(route,mode,policy="default"):
    import benchmark
    if policy.startswith("always"):
        harness.WEIGHTS["C"]=(1.,.85)
        harness.OFFSET["C"]=0.
        harness.MEMORY["C"]=float(policy[6:])
    if policy == "strong":
        # Paired policy used by the enterprise training entry.
        harness.WEIGHTS["C"]=(1.,.85)
        harness.OFFSET["C"]=0.
        harness.MEMORY["C"]=.10
    old_fit, old_predict=engine.fit,engine.predict
    try:
        engine.fit=lambda model,x,y,index: health_fit(model,x,y,index,route)
        engine.predict=lambda model,head,x: health_predict(model,head,x,mode)
        benchmark.OUT=WORK/f"replay-{route}-{mode}-{policy}"
        benchmark.run("C","enterprise",0)
    finally:
        engine.fit,engine.predict=old_fit,old_predict


def confirmation():
    """Post-freeze paired unit holdouts. No held-out row updates either checkpoint."""
    official=harness.Model.load(SCHOOL/"checkpoint.npz")
    x,y,phase,age,_,unit,fd=load("train",official)
    rows=[]
    for fold in (0,4):
        held=unit%5==fold
        keep=~held
        m=harness.Model("regression",official.meta["paths"],[],x[keep],y[keep],
                        hidden=official.meta["hidden"],seed=official.meta["seed"],batch_size=official.meta["batch_size"])
        m.fit(x[keep],y[keep],15)
        heads={}; out={}; times={}
        for name,fit_fn,pred_fn in (("campus",engine.fit_neural,lambda m,h,x:engine.predict_neural(m,h,x,65.)),
                                    ("enterprise",lambda m,x,y:engine.fit(m,x,y,1),engine.predict)):
            t=time.process_time();heads[name]=h=fit_fn(m,x[keep],y[keep]);seconds=time.process_time()-t
            out[name]=pred_fn(m,h,x[held])
            bench=x[held][np.linspace(0,held.sum()-1,512,dtype=int)]
            elapsed=[]
            for _ in range(3):
                t=time.process_time();one=np.array([pred_fn(m,h,row[None])[0] for row in bench])
                elapsed.append(1e6*(time.process_time()-t)/len(bench))
            np.testing.assert_allclose(one,pred_fn(m,h,bench),rtol=1e-11,atol=1e-9)
            times[name]=dict(fit_s=seconds,single_row_us=float(np.median(elapsed)),head_bytes=sum(v.nbytes for v in h.values()))
        yv=y[held];domain=fd[held];stage=phase[held]
        _,inv=np.unique([f"{a}/{b}" for a,b in zip(domain,unit[held])],return_inverse=True)
        count=np.bincount(inv);draw=np.random.default_rng(1234).integers(len(count),size=(1000,len(count)))
        def bootstrap(pred):
            se=np.bincount(inv,weights=(pred-yv)**2)
            return np.sqrt(se[draw].sum(1)/count[draw].sum(1))
        ci=np.quantile(bootstrap(out["enterprise"])-bootstrap(out["campus"]),[.025,.975]).tolist()
        row=dict(fold=fold,units=len(count),n=len(yv),rmse_delta_ci=ci,times=times,
                 scores={n:score(yv,z,stage,domain) for n,z in out.items()})
        rows.append(row)
        print(fold,row,flush=True)
    save("confirmation.json",rows)


if __name__=="__main__":
    {"confirm":confirmation,"internal":internal,"extract":extract_external,"external":external,"replay":replay}[sys.argv[1]](*sys.argv[2:])
