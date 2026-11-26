"""Development-only engine head comparisons, paired public replays and held-out checks."""
import argparse
import json
from pathlib import Path
from time import process_time
import sys

import numpy as np
import benchmark
import engine
import contest_submission as harness

HERE=Path(__file__).resolve().parent
OUT=HERE.parent/"work/v2/engine"


def trial_neural(model, x, y, limit=24576, epochs=20, width=48, initial=None, average=0, batch_size=256):
    rng = np.random.default_rng(145)
    if len(x) > limit:
        sample = rng.choice(len(x), limit, replace=False)
        x, y = x[sample], y[sample]
    centers, counts = np.unique(np.round(x[:, [-3, -1]]), axis=0, return_counts=True)
    centers = centers[np.argsort(-counts)[:8]]
    if initial:
        centers = initial["centers"]
    groups = engine._regimes(x, centers)
    if initial:
        means,inv_scale = initial["means"],initial["inv_scale"]
    else:
        means = np.stack([x[groups == k, :-3].mean(0) for k in range(len(centers))])
        scales = np.stack([x[groups == k, :-3].std(0) for k in range(len(centers))])
        inv_scale = 1 / np.maximum(scales, 1e-4)
    sensors = np.clip((x[:, :-3] - means[groups]) * inv_scale[groups], -6, 6)
    base = model.predict(x).reshape(-1)
    design = np.column_stack((sensors, (base - 100) / 60, np.eye(len(centers))[groups]))
    target = (y - base) / 55

    weights = rng.normal(0, .22, (design.shape[1], width))
    bias = np.zeros(width)
    output = rng.normal(0, .03, width)
    intercept = np.zeros(1)
    if initial:
        weights = np.vstack((initial["w_sensor"], initial["w_base"]*60,
                             initial["bias"] + initial["w_base"]*100))
        bias = np.zeros(weights.shape[1])
        output=initial["output"].copy()/55
        intercept=initial["intercept"].copy()/55
    params = [weights, bias, output, intercept]
    first = [np.zeros_like(p) for p in params]
    second = [np.zeros_like(p) for p in params]
    step = 0
    averaged = [np.zeros_like(param) for param in params] if average else []
    for epoch in range(epochs):
        order = rng.permutation(len(y))
        for idx in np.array_split(order, max(1, len(y) // batch_size)):
            h = np.tanh(design[idx] @ weights + bias)
            delta = (h @ output + intercept - target[idx]) * (2 / len(idx))
            back = delta[:, None] * output * (1 - h * h)
            gradients = [design[idx].T @ back, back.sum(0), h.T @ delta, np.array([delta.sum()])]
            step += 1
            for j, grad in enumerate(gradients):
                first[j] = .9 * first[j] + .1 * grad
                second[j] = .999 * second[j] + .001 * grad ** 2
                rate = (.0015 if initial else .003) / (1 + epoch / 10)
                params[j] -= (rate * first[j] / (1 - .9 ** step) /
                              (np.sqrt(second[j] / (1 - .999 ** step)) + 1e-8))
        if average and epoch >= epochs-average:
            for accumulator, param in zip(averaged,params):
                accumulator += param/average
    if average:
        weights,bias,output,intercept = averaged
    # Fold the regime intercept and output scaling once for fast single-row serving.
    return dict(centers=centers, means=means, inv_scale=inv_scale,
                w_sensor=weights[:-len(centers)-1].copy(), w_base=weights[-len(centers)-1] / 60,
                bias=bias + weights[-len(centers):] - weights[-len(centers)-1] * (100 / 60),
                output=55 * output, intercept=55 * intercept)




def refresh_output(model, x, y, previous, penalty=.012, rate=.85):
    """Refit the small readout toward current residuals with a penalty around the parent."""
    core = ("centers","means","inv_scale","w_sensor","w_base","bias","output","intercept")
    if not previous or any(key not in previous for key in core):
        return None
    # A newly introduced operating center requires new centering and hidden features.
    settings = np.unique(np.round(x[:,[-3,-1]]),axis=0)
    if not set(map(tuple,settings)).issubset(map(tuple,previous["centers"])):
        return None
    ids = np.random.default_rng(302).choice(len(y),min(16384,len(y)),replace=False)
    xs, ys = x[ids], y[ids]
    g = engine._regimes(xs,previous["centers"])
    base = model.predict(xs)
    z = np.clip((xs[:,:-3]-previous["means"][g])*previous["inv_scale"][g],-6,6)
    h = np.tanh(z@previous["w_sensor"] + base[:,None]*previous["w_base"] + previous["bias"][g])
    diff = ys - base - h @ previous["output"] - previous["intercept"][0]
    mu = h.mean(0)
    centered = h - mu
    gram = centered.T @ centered
    gram.flat[::len(gram)+1] += penalty*len(ys)
    dw = np.linalg.solve(gram,centered.T@diff)
    state = {key:previous[key] for key in core}
    state["output"] = previous["output"] + rate*dw
    state["intercept"] = previous["intercept"] + rate*(diff.mean()-mu@dw)
    return state


def calibrate(model, x, y, state):
    idx = np.linspace(0,len(x)-1,min(len(x),2048),dtype=int)
    estimate = engine.predict_neural(model,state,x[idx],65.)
    lo,hi = np.quantile(estimate,[.35,.65])
    # Least squares shrinks extremes toward the mean. Correct the tails mildly
    # when released training lifetimes vary more than the model predictions.
    amp = np.clip(np.std(y)/max(1.,np.std(estimate)) - 1., .15,.7)
    state.update(health_lo=np.array(lo),health_hi=np.array(hi),health_amp=np.array(amp))
    return state




def alternate(route, model, x, y, index, previous=None, stable_release=False):
    if route.startswith("scheduled"):
        # Both alternatives use only past and present publicly released counts.
        regular = (stable_release and bool(previous)) or (index==1 and route=="scheduledlong")
        h = engine.fit_neural(model,x,y,epochs=20 if regular else 12,width=48 if regular else 64)
        return calibrate(model,x,y,h)
    epoch = 12 if route.startswith("lean") else 20
    width = 64 if route.startswith("leanwide") else 48
    if route in ("swa","large","swalarge","compact"):
        epochs,width,batch,avg = {"swa":(12,64,256,4), "large":(18,64,512,0),
                                  "swalarge":(18,64,512,6), "compact":(12,40,256,4)}[route]
        h=trial_neural(model,x,y,epochs=epochs,width=width,average=avg,batch_size=batch)
        return calibrate(model,x,y,h)
    state = None
    if previous and route not in ("cold","leanwide"):
        if "periodic" in route and index%4==1:
            previous=None
        if previous:
            state=refresh_output(model,x,y,previous,penalty=.012,
                                        rate=.5 if "damped" in route else .85)
            if state and "tune" in route:
                state=trial_neural(model,x,y,epochs=int(route.split("tune")[-1]),initial=state)
    if state is None:
        state=trial_neural(model,x,y,epochs=epoch,width=width)
    return calibrate(model,x,y,state)


def replay(route,split,setting,series=""):
    # Every route uses the same real train, submit, reload and predict path.
    # The pinned local harness supplies deterministic development model responses.
    if split not in (0,1,2):
        raise ValueError("Use only published development splits here.")
    bench=OUT/series/route
    bench.mkdir(exist_ok=True,parents=True)
    benchmark.OUT=bench
    if route != "frozen":
        engine.fit=lambda m,x,y,i,previous=None,stable_release=False: alternate(route,m,x,y,i,previous,stable_release)
    benchmark.run("C","v2",split,setting,True)


def unit_check(dataset):
    """Two staged releases. Hold entire units out of both updates; score all life quarters."""
    import importlib.util
    old_path=HERE.parent/"V1/engine.py"
    spec=importlib.util.spec_from_file_location("enterprise_v1_engine",old_path)
    old=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old)
    if dataset == "DS02":
        raw=HERE.parent/"work/engine/external.npz"
        if not raw.exists():
            raise FileNotFoundError("Run V1/engine_study.py extract to prepare DS02.")
        with np.load(raw) as z:
            sensor,settings,target,cycle,unit,cruise=[z[k] for k in ("sensors","settings","y","cycle","unit","cruise")]
        domain=np.repeat("DS02",len(unit));valid=np.isin(unit,[18,20])
        alt=np.searchsorted(np.quantile(settings[~valid,0],[.25,.5,.75]),settings[:,0])
        mach=settings[:,1]>np.median(settings[~valid,1])
        features=np.column_stack((sensor,settings,alt,settings[:,1],mach))
        paths=[str(i) for i in range(features.shape[1])]
    else:
        base=benchmark.WORK/"C"
        initial=harness.Model.load(base/"checkpoint.npz")
        rows=[json.loads(s) for s in (base/"public/train.jsonl").open()]
        features,target,_=harness.projection(rows,initial,"C")
        paths=initial.meta["paths"]
        cycle=np.array([r["meta"]["cycle"] for r in rows])
        unit=np.array([r["meta"]["unit"] for r in rows])
        domain=np.array([r["meta"]["source_subset"] for r in rows]);del rows
        valid=unit%5==(1 if dataset=="fold1" else 2)
        cruise=None
    identities=np.array([f"{a}/{b}" for a,b in zip(domain,unit)])
    age=np.zeros(len(target))
    for identity in np.unique(identities):
        mask=identities==identity;cc=cycle[mask];age[mask]=(cc-cc.min()+.5)/max(1.,np.ptp(cc)+1)
    first=(age<=.55)&~valid
    model=harness.Model("regression",paths,[],features[first],target[first],hidden=32,seed=42,batch_size=256)
    model.fit(features[first],target[first],15)
    out={}
    for version in ["v1","v2"]:
        prev=None
        results=[]
        for i,cut in enumerate([.55,.7,.85,1.],1):
            use=~valid&(age<=cut)
            start=process_time()
            prev=old.fit(model,features[use],target[use],i) if version=="v1" else engine.fit(model,features[use],target[use],i,previous=prev)
            sec=process_time()-start
            pred=(old.predict if version=="v1" else engine.predict)(model,prev,features[valid])
            err=pred-target[valid]
            rm=lambda e:float(np.mean(e**2)**.5)
            item=dict(n=len(pred),fit_s=sec,rmse=rm(err),phase=[rm(err[np.minimum(3,(4*age[valid]).astype(int))==p]) for p in range(4)],
                      nasa_sum=float(np.expm1(np.where(err>0,err/10,-err/13)).sum()))
            if cruise is not None:
                item.update(cruise=rm(err[cruise[valid]]),noncruise=rm(err[~cruise[valid]]))
            else:
                item.update(worst_fd=max(rm(err[domain[valid]==j]) for j in np.unique(domain)))
            if i==4:
                ix=np.flatnonzero(valid)[::max(1,valid.sum()//300)]
                fn=old.predict if version=="v1" else engine.predict
                a=fn(model,prev,features[ix]);b=np.array([fn(model,prev,features[j:j+1])[0] for j in ix])
                np.testing.assert_allclose(a,b,atol=1e-9,rtol=1e-11)
                item["row_batch_diff"]=float(np.max(abs(a-b)))
            results.append(item)
        out[version]=results
    OUT.mkdir(parents=True,exist_ok=True)
    path=OUT/(dataset+"-confirmation.json")
    path.write_text(json.dumps(out,indent=2))
    print(path)


if __name__=="__main__":
    if sys.argv[1] == "check":
        unit_check(sys.argv[2]);sys.exit()
    p=argparse.ArgumentParser()
    p.add_argument("route")
    p.add_argument("split",type=int)
    p.add_argument("setting",nargs="?",default="drift")
    p.add_argument("--series",default="")
    args=p.parse_args()
    replay(args.route,args.split,args.setting,args.series)
