"""Re-run C quality and single-process CPU comparisons using only the official public projection."""
import json
import sys
import time
from copy import deepcopy
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))
from industrial import Model, adjusted, experiments, load_head
from engine import fit_neural, predict_neural
from bench_engine_routes import fit_shared,predict_shared,fit_trees,predict_trees,fit_pool, predict_pool, predict, train_head, fit_density, predict_density

WORK = ROOT / "work"
C = WORK / "C"


def collect(file):
    x, y, units, subsets = [], [], [], []
    with file.open() as stream:
        for line in stream:
            item = json.loads(line)
            meta = item["meta"]
            x.append(item["inputs"]["sensors"] + item["inputs"]["settings"])
            y.append(item["answer"])
            units.append(meta["unit"])
            subsets.append(meta["source_subset"])
    return tuple(np.asarray(a) for a in (x, y, units, subsets))


def metrics(y, pred, domain):
    err = pred - y
    rmse = lambda v: float(np.sqrt(np.mean(v ** 2)))
    groups = {k: rmse(err[domain == k]) for k in np.unique(domain)}
    return dict(rmse=rmse(err), mae=float(np.mean(np.abs(err))),
                nasa=float(np.expm1(np.where(err > 0, err/10, -err/13)).sum()),
                group_rmse=groups, worst_rmse=max(groups.values()), n=len(y))


def v1_replay(model, train, valid):
    parent, head = deepcopy(model), {}
    for index in range(1, 11):
        candidate = deepcopy(parent)
        scores, heads = experiments("C", candidate, head, train, valid, index)
        method = min(heads, key=scores.get)
        if scores[method] < scores["parent"] - 1e-8:
            parent, head = candidate, heads[method]
    return np.maximum(0, adjusted(parent, head, "C", valid[0])[:, 0])


def experiment(model, train, valid, name, add_v1_final=False):
    x,y = train[:2]
    xv,yv,_,domain = valid
    records = {}
    def record(label, out, start, size=0):
        records[label]=dict(**metrics(yv, out, domain), cpu=time.process_time()-start, bytes=size)
    start=time.process_time()
    record("starting_mlp",model.predict(xv),start)
    if add_v1_final:
        start = time.process_time()
        archived = WORK / "validation10/C/round-10/record.json"
        if archived.exists():
            final = WORK / json.loads(archived.read_text())["selected"]
            preds = adjusted(Model.load(final / "checkpoint.npz"), load_head(final / "residual.npz"), "C", xv)[:, 0]
        else:
            preds = v1_replay(model, (x, y), (xv, yv))
        record("v1_ten_round", np.maximum(preds, 0), start)
    start=time.process_time()
    full=deepcopy(model)
    scores,heads=experiments("C",full,{},(x,y),(xv,yv),1)
    choice=min(heads,key=lambda k:scores[k])
    record("v1_first",np.maximum(0,adjusted(full,heads[choice],"C",xv)[:,0]),start)
    short=deepcopy(model)
    start=time.process_time()
    rng=np.random.default_rng(851)
    sample=rng.choice(len(y),min(len(y),2048),replace=False)
    short.meta["learning_rate"]=.0003
    short.fit(x[sample],y[sample],1)
    common_cpu=time.process_time()-start
    for method in ("linear","quadratic","kernel"):
        for limit in ([8192, 16384, 32768] if method=="kernel" else [16384]):
            start=time.process_time()-common_cpu
            head=train_head(short,x,y,method,limit)
            out=predict(short,head,xv)
            record(f"{method}_{limit}",out,start,sum(v.nbytes for v in head.values()))
    for shrink in (.08,.30):
        start=time.process_time()-common_cpu
        head=fit_density(short,x,y,32768,shrink)
        posterior=predict_density(short,head,xv)
        for weight in (.6,.85,1.):
            pred=weight*posterior+(1-weight)*short.predict(xv)
            record(f"posterior_{shrink}_{weight}",pred,start,sum(a.nbytes for a in head.values()))
    for width in (64,160,320):
        for penalty in (.0001,.001):
            start=time.process_time()-common_cpu
            head=fit_pool(short,x,y,32768,width,penalty)
            pred=predict_pool(short,head,xv)
            record(f"pool_{width}_{penalty}",pred,start,sum(a.nbytes for a in head.values()))
    start=time.process_time()-common_cpu
    head=fit_trees(short,x,y,24576,144)
    for span in (32,80,144):
        sliced=dict(head)
        for key in ["dims","leaves","thresholds"]:
            sliced[key]=head[key][:span]
        record(f"trees_{span}",predict_trees(short,sliced,xv),start,sum(a.nbytes for a in sliced.values()))
    for size,w,sup in ((16384,96,False),(16384,96,True),(32768,128,True),(32768,256,True)):
        start=time.process_time()-common_cpu
        head=fit_shared(short,x,y,size,w,sup)
        pred=predict_shared(short,head,xv)
        record(f"shared_{size}_{w}_{sup}",pred,start,sum(a.nbytes for a in head.values()))
    for width,epochs in ((40,8),(48,20),(64,35)):
        start=time.process_time()-common_cpu
        h=fit_neural(short,x,y,24576,epochs,width)
        pred=predict_neural(short,h,xv)
        record(f"neural_{width}_{epochs}",pred,start,sum(a.nbytes for a in h.values()))
    print(name,{k:tuple(round(r[d],3) for d in ["rmse","worst_rmse","cpu"]) for k,r in records.items()},flush=True)
    return records


def main():
    train=collect(C/"public/train.jsonl")
    valid=collect(C/"public/validation.jsonl")
    official=Model.load(C/"checkpoint.npz")
    # For independent engine holdouts, refit all model parameters on the retained engines.
    # Stratified deterministic unit folds preserve all four C-MAPSS subsets.
    result={}
    for fold in (0,1,2):
        hold=train[2]%5==fold
        sub=tuple(a[~hold] for a in train)
        audit=tuple(a[hold] for a in train)
        m=Model("regression",official.meta["paths"],[],*sub[:2],hidden=32,
                seed=42,learning_rate=.002,batch_size=256)
        m.fit(*sub[:2],15)
        result[f"engine_fold_{fold}"]=experiment(m,sub,audit,f"engine_fold_{fold}")
    result["public"]=experiment(official,train,valid,"public",True)
    path=WORK/"v2/engine/benchmark.json"
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(result,indent=2))
if __name__=="__main__":
    main()
