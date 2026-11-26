"""Gas ablations, held-out concentration repetitions and long-term acquisition dates."""
import argparse
import csv
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import zipfile

import numpy as np

import benchmark
import gas
import prototype
from baseline_model import Model
from industrial import dataset, metric

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "企业赛/work/v2/gas"
PUBLIC = ROOT / "校赛/work/A"


def load_v1():
    spec = importlib.util.spec_from_file_location("reference_gas",ROOT/"企业赛/V1/gas.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def route_replay(route):
    original_fit, original_predict, original_candidates = gas.fit, gas.predict, gas.fit_candidates
    prior=load_v1()
    if route=="full":
        pass
    elif route=="curve":
        def fit(model,x,y,index):
            head=original_fit(model,x,y,index)
            if "log_means" in head:
                head["log_means"][:]=0.
                head["log_axes"][:]=0.
            return head
        # The remaining raw-norm term is common to classes and does not affect argmax.
        gas.fit=fit
    elif route=="union":
        def fit(model,x,y,index):
            return {**prior.fit(model,x,y,index),**original_fit(model,x,y,index)}
        def predict(model,head,x):
            if "curve_coeff" not in head:
                return original_predict(model,head,x)
            f,v,q=gas.curve_inputs(x)
            full=prototype.score_curve(f,v,q,head)
            z=prototype._normalize(v,head["log_center"],head["log_scale"])
            d=z[:,None,:]-head["log_means"]
            aligned=np.einsum("ncd,ckd->nck",d,head["log_axes"])
            sub=np.maximum(0.,(d**2).sum(2)-.94*(aligned**2).sum(2))/v.shape[1]
            b=np.log(np.maximum(model.predict(x),1e-12))
            return .5*(prior.predict(model,head,x)+b)+full+3*sub
        gas.fit,gas.predict=fit,predict
    gas.fit_candidates=lambda m,x,y,i:{route:gas.fit(m,x,y,i)}
    benchmark.OUT=OUT/("study-"+route)
    try:
        for s in (0,1,2):
            benchmark.run("A","v2",s,profile=True)
    finally:
        gas.fit,gas.predict,gas.fit_candidates=original_fit,original_predict,original_candidates


def native_compare(x,absolute,relative,raw,q,classes,y,train,blocks,cluster=None, inner=None, return_scores=False):
    m=Model("classification",[str(i) for i in range(x.shape[1])],list(classes),x[train],y[train])
    m.fit(x[train],y[train],20)
    m.meta["learning_rate"]=.001
    m.fit(x[train],y[train],1)
    h=prototype.fit(absolute[train],raw[train],y[train],len(classes))
    for prefix,feat,rank,weight in (("absolute_",absolute,48,0.),("relative_",relative,24,.8)):
        old=gas._discriminant(feat[train],y[train],len(classes),rank,weight,.005)
        h.update({prefix+k:v for k,v in old.items()})
    test=~train
    b,a,r=gas.components(m,h,x[test],(absolute[test],relative[test]))
    k,s=prototype.scores(absolute[test],raw[test],h)
    curve=prototype.fit_curve(absolute[train],raw[train],q[train],y[train],len(classes))
    feeds=[absolute[test],raw[test],q[test]]
    scores={"mlp":b,"v1":b+.4*(a+r)+.5*(k+s),"v2":b+prototype.score_curve(*feeds,curve)}
    if return_scores:
        return scores
    success={}
    if inner is not None:
        calibrate=train & ~inner
        trials=native_compare(x,absolute,relative,raw,q,classes,y,inner,[],return_scores=True)
        signed={}
        for name in ("v1","v2"):
            pred=trials[name][calibrate[~inner]]
            signed[name]=metric("A",y[calibrate],pred)
        choice=max(signed,key=signed.get)
        scores["heldout_choice"]=scores[choice]
        success={"chosen":choice,"validation_macro":signed,"fit_n":int(inner.sum()),"selection_n":int(calibrate.sum())}
    results={n:{"macro":metric("A",y[test],o),"blocks":[metric("A",y[mask],o[mask[test]]) for mask in blocks]}
            for n,o in scores.items()}
    summary=dict(train=int(train.sum()),test=int(test.sum()),results=results,inner_selection=success)
    if cluster is not None:
        days=cluster[test]
        diff=[]
        rng=np.random.default_rng(20261126)
        unique=np.unique(days)
        boot=rng.integers(len(unique),size=(4000,len(unique)))
        for n in ("v1","v2"):
            pred=scores[n].argmax(1)
            table=np.array([np.bincount(len(classes)*y[test][days==d]+pred[days==d],
                              minlength=len(classes)**2).reshape(len(classes),-1) for d in unique])
            counts=table[boot].sum(1)
            diff.append((2*np.diagonal(counts,axis1=1,axis2=2)/np.maximum(1,counts.sum(1)+counts.sum(2))).mean(1))
        summary["paired_day_delta_ci95"]=np.quantile(diff[1]-diff[0],[.025,.975]).tolist()
    return summary


def cross_day():
    archive=ROOT/"Supp_Datasets/A/01_Long_Term_Drift_2025/Dataset.zip"
    with zipfile.ZipFile(archive) as z:
        data=z.read("extracted_features.csv")
    assert hashlib.sha256(data).hexdigest()=="9933dca26ad0837c25eecca188d9a0499d848bb08e4c27574538e9e8fc69a38d"
    rows=list(csv.DictReader(io.StringIO(data.decode()),delimiter=" "))
    days=np.array([r["Date[yyyy-mm-dd]"] for r in rows])
    classes,y=np.unique([r["Class"].split()[0] for r in rows],return_inverse=True)
    keys=["mean_R{}[Ohm]","rel_diff_R{}","sample_startslope_R{}","rec_lvl_R{}"]
    obs=np.array([[[float(r[k.format(i)]) for k in keys] for i in range(1,63)] for r in rows])
    q=np.log1p(abs(obs[:,:,1])).mean(1)
    native=obs.copy();native[:,:,0]=np.log(np.maximum(native[:,:,0],1e-6))
    x=obs.reshape(len(y),-1)
    raw=np.sign(x)*np.log1p(abs(x))
    dates=np.unique(days)
    blocks=[np.isin(days,t) for t in np.array_split(dates[12:],3)]
    return native_compare(x,native.reshape(len(y),-1),(native-native.mean(1,keepdims=True)).reshape(len(y),-1),
                          raw,q,classes,y,np.isin(days,dates[:12]),blocks,days,inner=np.isin(days,dates[:8]))


def dsps():
    from scipy.io import loadmat

    archive=ROOT/"Supp_Datasets/A/02_DS_PS_2026/DS-PS.zip"
    y,rep,ppm,vectors=[],[],[],[]
    def summary(sig):
        return sig[:,:1536].reshape(-1,4,25,12,4,4,8).mean(axis=(0,2,3,5))
    with zipfile.ZipFile(archive) as z:
        for folder in range(1,31):
            for c in range(2,8):
                key=f"M{c}"
                mat=loadmat(io.BytesIO(z.read(f"DS-PS/{folder}/{key}.mat")),squeeze_me=True,struct_as_record=False)[key]
                base=summary(mat.Bl)
                vectors.append(np.stack((summary(mat.Tr)-base,summary(mat.St)-base)))
                y.append((folder-1)//10);rep.append((folder-1)%10);ppm.append(20*(c-1))
    y,rep,ppm=np.asarray(y),np.asarray(rep),np.asarray(ppm)
    native=np.array(vectors); x=native.reshape(len(y),-1)
    q=np.log1p(abs(native[:,1])).reshape(len(y),-1).mean(1)
    train=rep<6
    blocks=[~train & (ppm>=low) & (ppm<=low+20) for low in (20,60,100)]
    return native_compare(x,x,(native-native.mean(-1,keepdims=True)).reshape(len(y),-1),
                          np.sign(x)*np.log1p(abs(x)),q,["ethanol","acetone","butanone"],y,train,blocks,inner=rep<4)


def public():
    prior=load_v1()
    starter=Model.load(PUBLIC/"checkpoint.npz")
    def load(split):
        with (PUBLIC/"public"/(split+".jsonl")).open() as file:
            rows=sorted((json.loads(l) for l in file),key=lambda r:r["meta"]["source_index"])
        return dataset(rows,starter)
    x,y=load("train");v,t=load("validation")
    def eval_set(model,x,y,v,t):
        model.meta["learning_rate"]=.0005;model.fit(x,y,1)
        result={}
        for name,module in (("v1",prior),("v2",gas)):
            head=module.fit(model,x,y,1)
            score=module.predict(model,head,v)
            scalar=np.vstack([module.predict(model,head,row[None]) for row in v[:16]])
            np.testing.assert_allclose(score[:16],scalar,atol=1e-7,rtol=1e-7)
            buffer=io.BytesIO();np.savez_compressed(buffer,**head)
            result[name]={"macro":metric("A",t,score),"head_bytes":buffer.getbuffer().nbytes}
        return result
    fold=np.zeros(len(y),int)
    for cls in np.unique(y):
        for i,ids in enumerate(np.array_split(np.flatnonzero(y==cls),4)):
            fold[ids]=i
    out=[]
    for i in range(4):
        use=fold!=i
        config=starter.meta
        m=Model("classification",config["paths"],config["classes"],x[use],y[use],hidden=config["hidden"],
                seed=config["seed"],batch_size=config["batch_size"])
        m.fit(x[use],y[use],15)
        out.append(eval_set(m,x[use],y[use],x[~use],y[~use]))
    return dict(official=eval_set(starter,x,y,v,t),folds=out)


if __name__=="__main__":
    arg=argparse.ArgumentParser()
    arg.add_argument("study",choices=["public","cross-day","ds-ps","full","curve","union"])
    args=arg.parse_args()
    OUT.mkdir(exist_ok=True,parents=True)
    if args.study in ("full","curve","union"):
        route_replay(args.study)
    else:
        data={"public":public,"cross-day":cross_day,"ds-ps":dsps}[args.study]()
        (OUT/(args.study+".json")).write_text(json.dumps(data,indent=2,allow_nan=False))
        print(json.dumps(data,ensure_ascii=False))
