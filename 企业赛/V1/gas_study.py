"""A: supervised ablations, blocked retraining, and device-native chronological transfer."""
import argparse
import csv
import importlib.util
import hashlib
import io
import json
from pathlib import Path
from time import process_time
import zipfile

import numpy as np

import gas
import prototype
from baseline_model import Model
from industrial import dataset, metric

ROOT = Path(__file__).resolve().parents[2]
PUBLIC = ROOT / "校赛/work/A"
OUT = ROOT / "企业赛/work/gas-study"
EXTERNAL = ROOT / "Supp_Datasets/A/01_Long_Term_Drift_2025/Dataset.zip"


def summarize(y, outputs, blocks):
    return {k:dict(macro=metric("A",y,p),
                   blocks=[metric("A",y[mask],p[mask]) for mask in blocks],
                   recall=[float((p[y==j].argmax(1)==j).mean()) for j in np.unique(y)])
            for k,p in outputs.items()}


def ablations(b,a,r,k,s):
    return {"MLP":b, "V2":b+a+r, "kernel":b+.4*(a+r)+k,
            "subspace":b+.4*(a+r)+s, "enterprise":b+.4*(a+r)+.5*(k+s)}


def quarters(y, n):
    labels = np.empty(len(y),dtype=int)
    for j in np.unique(y):
        for k,ix in enumerate(np.array_split(np.flatnonzero(y==j),n)):
            labels[ix]=k
    return labels


def evaluate(model, x, y, v, target):
    model.meta["learning_rate"]=.0005
    model.fit(x,y,1)
    head=gas.fit(model,x,y,1)
    absolute,relative=gas.signatures(v)
    b,a,r=gas.components(model,head,v,(absolute,relative))
    k,s=prototype.scores(absolute,np.log(np.maximum(abs(v),.02)),head)
    outputs=ablations(b,a,r,k,s)
    np.testing.assert_allclose(gas.predict(model,head,v),outputs["enterprise"],atol=1e-8)
    np.testing.assert_allclose(np.stack([gas.predict(model,head,row[None])[0] for row in v[:16]]),
                               outputs["enterprise"][:16],atol=1e-8)
    phase=quarters(target,4)
    return summarize(target,outputs,[phase==i for i in range(4)])


def primary():
    starter=Model.load(PUBLIC/"checkpoint.npz")
    def load(split):
        with (PUBLIC/"public"/(split+".jsonl")).open() as stream:
            rows=sorted((json.loads(line) for line in stream),key=lambda r:r["meta"]["source_index"])
        return dataset(rows,starter)
    x,y=load("train");v,target=load("validation")
    phase=quarters(y,4)
    folds=[]
    for g in range(4):
        m=starter.meta
        model=Model("classification",m["paths"],m["classes"],x[phase!=g],y[phase!=g],
                    hidden=m["hidden"],seed=m["seed"],batch_size=m["batch_size"])
        model.fit(x[phase!=g],y[phase!=g],15)
        folds.append(evaluate(model,x[phase!=g],y[phase!=g],x[phase==g],y[phase==g]))
    return dict(train=len(y),valid=len(target),classes=starter.meta["classes"],folds=folds,
                public=evaluate(starter,x,y,v,target))


def cross_day():
    with zipfile.ZipFile(EXTERNAL) as z:
        data=z.read("extracted_features.csv")
    sha=hashlib.sha256(data).hexdigest()
    assert sha=="9933dca26ad0837c25eecca188d9a0499d848bb08e4c27574538e9e8fc69a38d"
    rows=list(csv.DictReader(io.StringIO(data.decode()),delimiter=" "))
    days=np.array([r["Date[yyyy-mm-dd]"] for r in rows])
    cls,y=np.unique([r["Class"].split()[0] for r in rows],return_inverse=True)
    features=["mean_R{}[Ohm]","rel_diff_R{}","sample_startslope_R{}","rec_lvl_R{}"]
    native=np.array([[[float(row[k.format(i)]) for k in features] for i in range(1,63)] for row in rows])
    dates=np.unique(days)
    train=np.isin(days,dates[:12])
    x=native.reshape(len(y),-1)
    native=native.copy()
    native[:,:,0]=np.log(np.maximum(native[:,:,0],1e-6))
    absolute=native.reshape(len(y),-1)
    relative=(native-native.mean(1,keepdims=True)).reshape(len(y),-1)
    logread=np.sign(x)*np.log1p(abs(x))
    model=Model("classification",[str(i) for i in range(x.shape[1])],list(cls),x[train],y[train])
    model.fit(x[train],y[train],20)
    model.meta["learning_rate"]=.001
    model.fit(x[train],y[train],1)
    h=prototype.fit(absolute[train],logread[train],y[train],len(cls))
    h.update({"absolute_"+k:v for k,v in gas._discriminant(absolute[train],y[train],len(cls),48,0,.005).items()})
    h.update({"relative_"+k:v for k,v in gas._discriminant(relative[train],y[train],len(cls),24,.8,.005).items()})
    b,a,r=gas.components(model,h,x[~train],(absolute[~train],relative[~train]))
    k,s=prototype.scores(absolute[~train],logread[~train],h)
    blocks=[np.isin(days[~train],t) for t in np.array_split(dates[12:],3)]
    outputs=ablations(b,a,r,k,s)
    result=summarize(y[~train],outputs,blocks)
    # Pair by acquisition day, preserving within-day dependence.
    test_days=days[~train]
    test_y=y[~train]
    matrices=[]
    for name in ("V2","enterprise"):
        guess=outputs[name].argmax(1)
        matrices.append(np.array([np.bincount(len(cls)*test_y[test_days==d]+guess[test_days==d],
                             minlength=len(cls)**2).reshape(len(cls),len(cls)) for d in dates[12:]]))
    rng=np.random.default_rng(20261126)
    draw=rng.integers(0,len(dates)-12,size=(4000,len(dates)-12))
    measures=[]
    for table in matrices:
        counts=table[draw].sum(1)
        scores=2*np.diagonal(counts,axis1=1,axis2=2)/np.maximum(1,counts.sum(1)+counts.sum(2))
        measures.append(scores.mean(1))
    interval=np.quantile(measures[1]-measures[0],[.025,.975]).tolist()
    return dict(source=str(EXTERNAL.relative_to(ROOT)),csv_sha256=sha,classes=cls.tolist(),
                train=int(train.sum()),test=int((~train).sum()),train_days=dates[:12].tolist(),
                test_days=dates[12:].tolist(),results=result,day_bootstrap_enterprise_minus_v2=interval)


def costs():
    spec=importlib.util.spec_from_file_location("campus_gas",ROOT/"校赛/V2/gas.py")
    campus=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(campus)
    model=Model.load(PUBLIC/"checkpoint.npz")
    with (PUBLIC/"public/train.jsonl").open() as stream:
        x,y=dataset((json.loads(line) for line in stream),model)
    with (PUBLIC/"public/validation.jsonl").open() as stream:
        xv,_=dataset((json.loads(line) for line in stream),model)
    result={}
    for name,module in [("V2",campus),("enterprise",gas)]:
        train=[]; serve=[]
        for _ in range(5):
            t=process_time()
            head=module.fit(model,x,y,1)
            train.append(process_time()-t)
            t=process_time()
            for row in xv[:256]:
                module.predict(model,head,row[None])
            serve.append((process_time()-t)/256*1e6)
        buf=io.BytesIO()
        np.savez_compressed(buf,**head)
        result[name]=dict(cpu_train_s_median=float(np.median(train)),
                          cpu_predict_us_median=float(np.median(serve)),
                          head_npz_bytes=buf.getbuffer().nbytes)
    return result


def ds_ps():
    from scipy.io import loadmat

    archive = ROOT / "Supp_Datasets/A/02_DS_PS_2026/DS-PS.zip"
    labels, repetitions, ppm, vectors = [], [], [], []
    def summarize_periods(signal):
        # Release indexes as [heater cycle, 100 times, 12 sensor copies, 16 loads, 8 types].
        signal=signal[:,:1536].reshape(-1,4,25,12,4,4,8)
        return signal.mean(axis=(0,2,3,5))
    with zipfile.ZipFile(archive) as z:
        for folder in range(1,31):
            for concentration in range(2,8):
                var=f"M{concentration}"
                payload=z.read(f"DS-PS/{folder}/{var}.mat")
                trial=loadmat(io.BytesIO(payload),squeeze_me=True,struct_as_record=False)[var]
                base=summarize_periods(trial.Bl)
                trans=summarize_periods(trial.Tr)-base
                steady=summarize_periods(trial.St)-base
                vectors.append(np.stack((trans,steady)))
                labels.append((folder-1)//10)
                repetitions.append((folder-1)%10)
                ppm.append((concentration-1)*20)
    y,rep,ppm=np.asarray(labels),np.asarray(repetitions),np.asarray(ppm)
    native=np.array(vectors)
    x=native.reshape(len(y),-1)
    relative=(native-native.mean(axis=-1,keepdims=True)).reshape(len(y),-1)
    train=rep<6
    logread=np.sign(x)*np.log1p(abs(x))
    cls=["ethanol","acetone","butanone"]
    model=Model("classification",[str(i) for i in range(x.shape[1])],cls,x[train],y[train])
    model.fit(x[train],y[train],20)
    model.meta["learning_rate"]=.001
    model.fit(x[train],y[train],1)
    h=prototype.fit(x[train],logread[train],y[train],len(cls))
    h.update({"absolute_"+k:v for k,v in gas._discriminant(x[train],y[train],len(cls),48,0,.005).items()})
    h.update({"relative_"+k:v for k,v in gas._discriminant(relative[train],y[train],len(cls),24,.8,.005).items()})
    b,a,r=gas.components(model,h,x[~train],(x[~train],relative[~train]))
    k,s=prototype.scores(x[~train],logread[~train],h)
    output=ablations(b,a,r,k,s)
    blocks=[(ppm[~train]>=lo)&(ppm[~train]<=lo+20) for lo in (20,60,100)]
    return dict(source=str(archive.relative_to(ROOT)),train=int(train.sum()),test=int((~train).sum()),
                repetitions_train=[1,2,3,4,5,6],repetitions_test=[7,8,9,10],
                concentration_blocks_ppm=[[20,40],[60,80],[100,120]],classes=cls,
                results=summarize(y[~train],output,blocks))


if __name__=="__main__":
    OUT.mkdir(exist_ok=True,parents=True)
    args=argparse.ArgumentParser()
    args.add_argument("--ds-ps",action="store_true",help="run the separate MAT-file replication experiment")
    if args.parse_args().ds_ps:
        result=ds_ps()
        (OUT/"ds-ps.json").write_text(json.dumps(result,indent=2,allow_nan=False))
        print({k:v["macro"] for k,v in result["results"].items()})
        raise SystemExit(0)
    data={"primary":primary(),"external":cross_day(),"cost":costs()}
    (OUT/"gas-ablation.json").write_text(json.dumps(data,indent=2,allow_nan=False))
    for k in data["primary"]["public"]:
        print(k,"public",round(data["primary"]["public"][k]["macro"],4),
              "folds",[round(f[k]["macro"],4) for f in data["primary"]["folds"]],
              "external",round(data["external"]["results"][k]["macro"],4),
              "external blocks",np.round(data["external"]["results"][k]["blocks"],4))
