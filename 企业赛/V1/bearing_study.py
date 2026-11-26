"""Whole-bearing holdouts for vibration and spectral lifetime priors (CPU, train only)."""
import argparse
import copy
import importlib.util
import io
import json
from pathlib import Path
from time import process_time
import zipfile

import numpy as np

import bearing
from industrial import Model,dataset

ROOT=Path(__file__).resolve().parents[2]
WORK=ROOT/"企业赛/work/bearing"
DATA=ROOT/"校赛/work/B"
spec=importlib.util.spec_from_file_location("campus_bearing",ROOT/"校赛/V2/bearing.py")
campus=importlib.util.module_from_spec(spec)
spec.loader.exec_module(campus)


def load(split, m):
    with (DATA/"public"/f"{split}.jsonl").open() as f:
        rows=[json.loads(line) for line in f]
    return (*dataset(rows,m), np.array([r["meta"]["bearing"] for r in rows]))


def extract_features(a, sample_rate):
    out={}
    for axis, vals in zip(("horizontal","vertical"),a.T):
        vals=vals-vals.mean()
        mag=np.abs(np.fft.rfft(vals))/len(vals)
        hz=np.fft.rfftfreq(len(vals),1/sample_rate)
        power=mag**2
        peak=np.max(abs(vals)); rms=np.sqrt(np.mean(vals**2))
        ma=np.mean(abs(vals)); rs=np.mean(np.sqrt(abs(vals)))**2
        z=vals/max(rms,1e-12)
        out[axis]=dict(rms=rms,peak=peak,peak_to_peak=np.ptp(vals),mean_abs=ma,
                     kurtosis=np.mean(z**4),skewness=np.mean(z**3),crest_factor=peak/max(rms,1e-12),
                     shape_factor=rms/max(ma,1e-12),impulse_factor=peak/max(ma,1e-12),
                     margin_factor=peak/max(rs,1e-12),spectral_centroid=np.sum(hz*power)/max(power.sum(),1e-12),
                     spectral_rms=np.sqrt(np.mean(power)),
                     spectral_kurtosis=np.mean(((mag-mag.mean())/max(mag.std(),1e-12))**4),
                     dominant_freq=hz[np.argmax(power[1:])+1],
                     band1_energy_0_2k=power[hz<2000].sum(),band2_energy_2k_6k=power[(hz>=2000)&(hz<6000)].sum(),
                     band3_energy_6k_12k=power[(hz>=6000)&(hz<=12000)].sum())
    return {"features":out}


def femto(m):
    cache=WORK/"femto-stride10-v1.npz"
    if cache.exists():
        with np.load(cache,allow_pickle=False) as data:return tuple(data[k] for k in ("x","y","g"))
    outer,=(ROOT/"Supp_Datasets/B/01_FEMTO_PRONOSTIA").glob("*.zip")
    # Development staging is optional. The standalone entry also reads the original release.
    mid=ROOT/"企业赛/work/external/FEMTO_Bearing-inner.zip"
    if not mid.is_file():
        with zipfile.ZipFile(outer) as a: mid=io.BytesIO(a.read(next(n for n in a.namelist() if n.endswith(".zip"))))
    rows=[]; groups=[]
    with zipfile.ZipFile(mid) as a,zipfile.ZipFile(io.BytesIO(a.read("Training_set.zip"))) as z:
        sources={}
        for name in sorted(z.namelist()):
            if "/acc_" in name and name.endswith(".csv"):
                sources.setdefault(name.split("/")[-2],[]).append(name)
        for group,names in sources.items():
            selected=sorted(set(names[::10]+[names[-1]]))
            end=int(Path(names[-1]).stem.split("_")[-1])
            for name in selected:
                raw=z.read(name)
                delimiter=";" if b";" in raw.splitlines()[0] else ","
                frame=np.loadtxt(io.BytesIO(raw),delimiter=delimiter)
                sample=extract_features(frame[:,4:6],25600.)
                tick=int(Path(name).stem.split("_")[-1])
                rul=(end-tick)*10./60.
                rows.append({"inputs":sample,"answer":rul})
                groups.append(group)
    x,y=dataset(rows,m);g=np.array(groups)
    np.savez_compressed(cache,x=x,y=y,g=g)
    return x,y,g


def frequency_table(x, y, columns, width=16., conditioned=True):
    """Empirical lifetime prior indexed by vibration severity and local spectral peak."""
    rms = x[:, columns[:2]].max(axis=1)
    limits = np.quantile(rms, [.40, .75, .90]) if conditioned else np.array([])
    severity = np.searchsorted(limits, rms)
    # The grid caps out-of-support frequencies while keeping memory bounded.
    last = int(min(2048, np.ceil(x[:, columns[2:]].max() / width) + 1))
    count = np.zeros((len(limits)+1, max(last,1)+1))
    root = np.zeros_like(count)
    for col in columns[2:]:
        ix = np.clip(np.rint(x[:, col] / width).astype(int), 0, count.shape[1]-1)
        np.add.at(count, (severity, ix), 1.)
        np.add.at(root, (severity, ix), np.sqrt(np.maximum(y,0)))
    weights = np.array([.2,.6,1.,.6,.2])
    def smooth(mat):
        return sum(w*np.pad(mat,((0,0),(2,2)))[:,k:k+mat.shape[1]] for k,w in enumerate(weights))
    count, root = smooth(count),smooth(root)
    prior = np.sqrt(np.maximum(y,0)).mean()
    target = ((root + 12*prior)/(count + 12))**2
    return dict(freq_cols=columns, freq_width=np.asarray(width),freq_limits=limits,
                freq_target=target, freq_support=count)


def frequency_prior(head, x):
    columns=head["freq_cols"]
    severity=np.searchsorted(head["freq_limits"], x[:, columns[:2]].max(axis=1))
    ix=np.clip(np.rint(x[:, columns[2:]] / head["freq_width"]).astype(int), 0, head["freq_target"].shape[1]-1)
    support=head["freq_support"][severity[:,None],ix]
    values=head["freq_target"][severity[:,None],ix]
    # Regularize confidence in peaks from a different resonant mode.
    w=support/(support+12.)
    return (w*values).sum(1)/np.maximum(w.sum(1),1e-10),w.mean(1)




def evaluate(model,x,y,xt,yt,gt):
    model=copy.deepcopy(model)
    model.meta["learning_rate"]=.0004
    ids=np.random.default_rng(2027).choice(len(x),min(768,len(x)),replace=False)
    model.fit(x[ids],y[ids],1)
    campus_head=campus.fit(model,x,y,1)
    base=campus.predict(model,campus_head,xt)
    variants={"campus":base,"static_adam":model.predict(xt)}
    rms=campus_head["rms_columns"]
    amp=bearing._amplitude(x,rms)
    bins=np.array_split(np.argsort(amp,kind="stable"),min(12,len(y)))
    levels=np.array([amp[ix].mean() for ix in bins])
    roots=np.array([np.sqrt(np.maximum(0,y[ix])).mean()**2 for ix in bins])
    levels, ids=np.unique(levels,return_index=True)
    val_amp=bearing._amplitude(xt,rms)
    low,high=campus_head["amplitude_bounds"]
    gate=1-np.clip((val_amp-low)/max(1e-8,high-low),0,1)
    variants["amplitude_bins"]=base+.6*.55*gate*(np.interp(val_amp,levels,roots[ids])-campus_head["prior"])
    cols=np.asarray([i for name in ("rms","dominant_freq") for i,path in enumerate(model.meta["paths"])
                     if json.loads(path)[-1][1]==name])
    for width in (8.,24.,64.):
        for condition in (True,False):
            table=frequency_table(x,y,cols,width,condition)
            prior,trust=frequency_prior(table,xt)
            for strength in (.35,.70):
                name=f"peak-{width:g}-{'strata' if condition else 'pooled'}-{strength:g}"
                blend=strength*trust
                variants[name]=base+blend*(prior-base)
    order=bearing.order_prior(x,y,cols[2:])
    geo=bearing.order_predict(xt,order)
    amp=bearing._amplitude(xt,cols[:2])
    low,high=campus_head["amplitude_bounds"]
    gate=1-np.clip((amp-low)/max(1e-8,high-low),0,1)
    for strength in (.4,.7,1.):
        variants[f"order-{strength:g}"]=np.maximum(0,base+strength*gate*(geo-campus_head["prior"]))
    means=order["order_means"]
    for label,ordered in (("ranked",np.sort(means)),("pooled",np.array([means[0],min(means[1],means[2]),max(means)]))):
        modified={**order,"order_means":ordered}
        geo=bearing.order_predict(xt,modified)
        for strength in (.5,.85):
            variants[f"{label}-{strength:g}"]=np.maximum(0,base+strength*gate*(geo-campus_head["prior"]))
    release=bearing.fit(model,x,y,1)
    deployed=bearing.predict(model,release,xt)
    np.testing.assert_allclose(deployed,variants["ranked-0.5"],rtol=1e-11,atol=1e-7)
    singles=np.array([bearing.predict(model,release,row[None])[0] for row in xt[:64]])
    np.testing.assert_allclose(singles,deployed[:len(singles)],rtol=1e-11,atol=1e-7)
    return {k:{str(g):{"n":int((gt==g).sum()),"rmse":float(np.sqrt(np.mean((pred[gt==g]-yt[gt==g])**2)))}
               for g in np.unique(gt)} for k,pred in variants.items()}


def run(use_external):
    WORK.mkdir(parents=True,exist_ok=True)
    template=Model.load(DATA/"checkpoint.npz")
    x,y,g=femto(template) if use_external else load("train",template)
    paths=template.meta["paths"]
    folds={}
    for group in np.unique(g):
        keep=g!=group
        m=Model("regression",paths,[],x[keep],y[keep],hidden=24)
        m.fit(x[keep],y[keep],15)
        folds[group]=evaluate(m,x[keep],y[keep],x[~keep],y[~keep],g[~keep])
    summary={}
    for name in next(iter(folds.values())):
        errors=[v[name][k]["rmse"] for k,v in folds.items()]
        summary[name]={"avg":float(np.mean(errors)),"max":float(np.max(errors))}
    result={"folds":folds,"summary":summary,"rows":len(x)}
    if not use_external:
        xv,yv,gv=load("validation",template)
        public=evaluate(template,x,y,xv,yv,gv)
        result["public"]=public
        for k,v in public.items():
            summary[k]["public_pooled"]=float(np.sqrt(sum(g["rmse"]**2*g["n"] for g in v.values())/len(yv)))
    if not use_external:
        compute={}
        for name,algo in (("campus",campus),("enterprise",bearing)):
            fit_times=[];single_times=[]
            for _ in range(4):
                t=process_time();head=algo.fit(template,x,y,1);fit_times.append(process_time()-t)
                t=process_time()
                for row in xv:
                    algo.predict(template,head,row[None])
                single_times.append((process_time()-t)/len(xv))
            compute[name]=dict(median_fit_seconds=float(np.median(fit_times)),
                        median_single_us=float(np.median(single_times)*1e6),
                        tensor_bytes=sum(a.nbytes for a in head.values()))
        result["cost"]=compute
        # Three whole devices per fold, one per operating condition. These are
        # additional combinations of development devices, not new equipment.
        results={}
        for index in ("1","2","3"):
            test=np.array([label.endswith("_"+index) for label in g])
            m=Model("regression",paths,[],x[~test],y[~test],hidden=24)
            m.fit(x[~test],y[~test],15)
            rows=evaluate(m,x[~test],y[~test],x[test],y[test],g[test])
            results[index]={k:rows[k] for k in ("campus","ranked-0.5")}
        result["three_device_holdouts"]=results
    keys=list(folds)
    diffs=np.array([folds[k]["ranked-0.5"][k]["rmse"]-folds[k]["campus"][k]["rmse"] for k in keys])
    sampled=np.random.default_rng(123).choice(diffs,size=(4000,len(keys)),replace=True).mean(axis=1)
    result["leave_one_device_difference"]={"reduced":int((diffs<0).sum()),"total":len(keys),
             "avg":float(diffs.mean()),"paired_device_percentiles":np.percentile(sampled,[2.5,97.5]).tolist()}
    target=WORK/("femto.json" if use_external else "xjtu.json")
    target.write_text(json.dumps(result,indent=2))
    for k,v in summary.items(): print(k,dict((s,round(t,2)) for s,t in v.items()),flush=True)


if __name__=="__main__":
    arg=argparse.ArgumentParser();arg.add_argument("--femto",action="store_true")
    run(arg.parse_args().femto)
