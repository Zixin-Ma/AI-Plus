"""Group-disjoint bearing experiments. Run from the school competition directory."""
import copy
import json
import time
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))
import bearing
from industrial import Model, adjusted, dataset, experiments

DATA = ROOT / "work" / "B"
OUTPUT = ROOT / "work" / "v2" / "bearing"
OUTPUT.mkdir(exist_ok=True, parents=True)


def load(split, model):
    with (DATA / "public" / f"{split}.jsonl").open() as stream:
        rows = [json.loads(line) for line in stream]
    return (*dataset(rows, model), np.array([r["meta"]["bearing"] for r in rows]))


def scores(y, pred):
    delta = np.maximum(0, pred) - y
    return {"rmse": float(np.mean(delta ** 2) ** .5), "mae": float(np.mean(abs(delta))),
            "bias": float(delta.mean()), "n": len(y)}


def views(x, kind):
    # Official extraction schema: band energies 0:3, RMS 11, kurtosis 6, centroid 14.
    z = np.log(np.maximum(x.reshape(-1, 2, 17), 1e-8))
    amp = z[:, :, 11]
    peak = z[:, :, 9]
    bands = z[:, :, :3]
    if kind == "rms":
        return np.max(amp, axis=1)[:, None]
    shape = np.column_stack((amp, peak, z[:, :, 6], bands.reshape(len(x), -1)))
    if kind == "physical":
        return shape
    return np.column_stack((shape, z[:, :, 14], z[:, :, 4]))


def kernel_head(x, y, kind):
    feat = views(x,kind)
    loc = np.median(feat,0)
    spread = np.maximum(np.percentile(feat,75,axis=0)-np.percentile(feat,25,axis=0), .1)
    z = np.clip((feat-loc)/spread,-6,6)
    # Random centers and soft Nadaraya-Watson interpolation. Cap stored support.
    ids=np.random.default_rng(2026).choice(len(z),min(512,len(z)),replace=False)
    return loc,spread,z[ids],y[ids]


def kernel_predict(x, head, kind):
    loc, spread, anchors, y=head
    z=np.clip((views(x,kind)-loc)/spread,-6,6)
    d=np.maximum((z*z).sum(1)[:,None] + (anchors*anchors).sum(1) - 2*z@anchors.T,0) / z.shape[1]
    w=np.exp(-2*(d-d.min(axis=1,keepdims=True)))
    return (w@y)/w.sum(1)


def evaluate(m,x,y,xv,yv,gs):
    outputs={}
    begin=time.process_time()
    pred=m.predict(xv)
    outputs["start"]=(pred,time.process_time()-begin)
    begin=time.process_time()
    head={}
    traditional=copy.deepcopy(m)
    for ix in [1,2]:
        # V1 deployed the 'full' residual twice. Fix that recipe without scoring the held-out fold.
        _,heads=experiments("B",traditional,head,(x,y),(x[:32],y[:32]),ix)
        head=heads["full"]
    outputs["v1"]= (adjusted(traditional,head,"B",xv)[:,0],time.process_time()-begin)
    begin=time.process_time()
    m=copy.deepcopy(m)
    m.meta["learning_rate"]=.0008
    subset=np.random.default_rng(42).choice(len(y),min(1024,len(y)),replace=False)
    m.fit(x[subset],y[subset],1)
    base=m.predict(xv)
    base_cost=time.process_time()-begin
    outputs["continued"]=(base,base_cost)
    for view in ["rms","physical","spectral"]:
        begin=time.process_time()
        head=kernel_head(x,y,view)
        out=kernel_predict(xv, head,view)
        secs=time.process_time()-begin+base_cost
        for blend in [.35,.75]:
            outputs[f"kernel_{view}_{blend}"]=((1-blend)*base+blend*out,secs)
    for blend in [.35,.75]:
        outputs[f"prior_{blend}"]=((1-blend)*base+blend*np.mean(y),base_cost)
    begin=time.process_time()
    fitted=bearing.fit(m,x,y,1)
    deployed=bearing.predict(m,fitted,xv)
    deployed_seconds=base_cost+time.process_time()-begin
    outputs["deployed"]=(deployed,deployed_seconds)
    raw_pred=bearing.corrected(m,fitted,xv)
    prior_values={"mean":np.mean(y), "root":np.mean(np.sqrt(y))**2, "trim":np.minimum(y,np.quantile(y,.75)).mean()}
    secs=base_cost+time.process_time()-begin
    outputs["ridge"]=(raw_pred,secs)
    for nm,prior in prior_values.items():
        for a in [.3,.55,.75]:
            outputs[f"ridge_{nm}_{a}"]=((1-a)*raw_pred+a*prior,secs)
    for a in [.35,.55,.75]:
        outputs[f"base_root_{a}"]=((1-a)*base+a*prior_values["root"],base_cost)
    # Strong vibration late in life conflicts with a population prior that is constant in RUL.
    train_amp=views(x,"rms")[:,0]
    val_amp=views(xv,"rms")[:,0]
    boundaries=np.unique(np.quantile(train_amp,np.linspace(0,1,9)[1:-1]))
    bins=np.searchsorted(boundaries,train_amp)
    conditional=[]
    for b in range(len(boundaries)+1):
        root=np.sqrt(y[bins==b])
        conditional.append(float(root.mean()**2) if len(root) else prior_values["root"])
    for tail in [.75,.90]:
        low,high=np.quantile(train_amp,[tail,.99])
        # Gradually withdraw the global prior as amplitude approaches extreme training events.
        gate=1-np.clip((val_amp-low)/max(high-low,1e-9),0,1)
        intensity=.60*gate
        outputs[f"tail_{tail}"]=((1-intensity)*raw_pred+intensity*prior_values["root"],secs)
    conditional=np.interp(val_amp,[np.median(train_amp[bins==b]) for b in range(len(conditional))],conditional)
    assert np.allclose(deployed,outputs["tail_0.75"][0])
    for b in [.25,.50]:
        prior=(1-b)*prior_values["root"]+b*conditional
        outputs[f"severity_{b}"]=(.4*raw_pred+.6*prior,secs)
    return {k:{"cpu_s":v[1],"pooled":scores(yv,v[0]),
               "groups":{g:scores(yv[gs==g],v[0][gs==g]) for g in np.unique(gs)}} for k,v in outputs.items()}


def timing(official,x,y,xv,yv):
    measurements={}
    for method in ("v1","deployed"):
        training=[]; single=[]
        for _ in range(5):
            model=copy.deepcopy(official)
            start=time.process_time()
            if method=="v1":
                _,heads=experiments("B",model,{},(x,y),(xv,yv),1)
                head=heads["full"]
            else:
                ids=np.random.default_rng(42).choice(len(y),min(1024,len(y)),replace=False)
                model.meta["learning_rate"]=.0008
                model.fit(x[ids],y[ids],1)
                head=bearing.fit(model,x,y,1)
                bearing.predict(model,head,xv)
            training.append(time.process_time()-start)
            if method == "deployed":
                batch=bearing.predict(model,head,xv[:20])
                singles=np.array([bearing.predict(model,head,row[None])[0] for row in xv[:20]])
                np.testing.assert_allclose(singles,batch,rtol=1e-12,atol=1e-9)
            start=time.process_time()
            if method=="v1":
                for row in xv[:1000]:
                    adjusted(model,head,"B",row[None])
            else:
                for row in xv[:1000]:
                    bearing.predict(model,head,row[None])
            single.append((time.process_time()-start)*1000)
        measurements[method]={"round_cpu_seconds":training,
                              "prediction_microseconds_per_row":single}
    return measurements


def main():
    official=Model.load(DATA/"checkpoint.npz")
    x,y,groups=load("train",official)
    folds={}
    for g in sorted(set(groups)):
        use=groups!=g
        fold=Model("regression",official.meta["paths"],[],x[use],y[use],hidden=24)
        fold.fit(x[use],y[use],15)
        folds[g]=evaluate(fold,x[use],y[use],x[~use],y[~use],groups[~use])
    xv,yv,gv=load("validation",official)
    public=evaluate(official,x,y,xv,yv,gv)
    # As a post-selection stress test, withhold one unit from each operating setting together.
    # Unlike the nine LOO folds these refit the base on six whole bearings only.
    grouped={}
    for suffix in ("1", "2", "3"):
        test=np.array([g.split("_")[-1]==suffix for g in groups])
        fold=Model("regression",official.meta["paths"],[],x[~test],y[~test],hidden=24)
        fold.fit(x[~test],y[~test],15)
        estimates=evaluate(fold,x[~test],y[~test],x[test],y[test],groups[test])
        grouped[suffix]={k:estimates[k] for k in ("start","v1","deployed")}
    results={"folds":folds,"public":public,"grouped_stress":grouped,
             "timing":timing(official,x,y,xv,yv)}
    (OUTPUT/"benchmark.json").write_text(json.dumps(results,indent=2,sort_keys=True))
    for method, val in public.items():
        parts=[f[method]["pooled"] for f in folds.values()]
        rm=np.array([g["rmse"] for g in parts])
        count=np.array([g["n"] for g in parts])
        print(method,"CV macro/pool/worst",*(round(v,1) for v in [rm.mean(),np.average(rm**2,weights=count)**.5,rm.max()]),
              "public RMSE/MAE/worst",*(round(v,1) for v in [val["pooled"]["rmse"],val["pooled"]["mae"],max(g["rmse"] for g in val["groups"].values())]))

if __name__=="__main__":
    main()
