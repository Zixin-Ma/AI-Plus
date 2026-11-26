"""Train-only engine head comparisons and paired, whole-engine diagnostic checks."""
import argparse
import importlib.util
import json
from pathlib import Path
from time import process_time

import numpy as np
import benchmark
import contest_submission as harness
import engine

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "work/v3/engine"
spec = importlib.util.spec_from_file_location("reference_engine", HERE.parent/"V2/engine.py")
v2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v2)


def calibration_rows(n):
    # Reproduce the deterministic training draw, then calibrate on different released rows.
    rng = np.random.default_rng(145)
    if n > 24576:
        unseen = np.ones(n, dtype=bool)
        unseen[rng.choice(n, 24576, replace=False)] = False
        rows = np.flatnonzero(unseen)
        if len(rows) >= 512:
            return rng.choice(rows, min(len(rows), 16384), replace=False)
    return rng.choice(n, min(n, 16384), replace=False)


def tail_table(route, model, state, x, y):
    idx=calibration_rows(len(y))
    xs=x[idx]; yy=y[idx]
    p=v2.predict(model,state,xs)
    groups=v2._regimes(xs,state["centers"])
    bins=8
    knots=np.zeros((len(state["centers"]),bins))
    shifts=knots.copy()
    for j in range(len(knots)):
        use=groups==j
        if not use.any():
            continue
        pp=p[use]; residual=yy[use]-pp
        for t, ids in enumerate(np.array_split(np.argsort(pp),min(bins,len(pp)))):
            knots[j,t]=np.median(pp[ids])
            rr=residual[ids]
            if route.startswith("exp"):
                # Twice the NASA loss temperature reduces single-observation influence.
                temp=2 if "warm" in route else 1
                lo,hi=np.quantile(rr,[.015,.985])
                r=np.clip(rr,lo,hi)
                l=13*temp; h=10*temp
                # Closed form for the symmetric extension of each exponential branch.
                a=np.log(np.mean(np.exp(np.clip(r/l,-45,45))))-np.log(l)
                b=np.log(np.mean(np.exp(np.clip(-r/h,-45,45))))-np.log(h)
                shift=(a-b)/(1/l+1/h)
            else:
                q=.995 if "995" in route else .98
                lo,hi=np.quantile(rr,[1-q,q])
                shift=(13*lo+10*hi)/23
            shifts[j,t]=np.clip(shift,-24,24)
        if len(pp)<bins:
            knots[j,len(pp):]=knots[j,len(pp)-1]
            shifts[j,len(pp):]=shifts[j,len(pp)-1]
    state.update(r_knots=knots, r_shift=shifts, r_scale=np.array(.50 if "half" in route else .25))
    return state


def earned_trust(route,model,state,x,y):
    idx=calibration_rows(len(y))
    xs,labels=x[idx], y[idx]
    b=model.predict(xs).reshape(-1)
    p=v2.predict(model,state,xs)
    g=v2._regimes(xs,state["centers"])
    knots=np.zeros((len(state["centers"]),5))
    trust=np.ones_like(knots)
    for j in range(len(knots)):
        rows=np.flatnonzero((g==j)&(p>b))
        if len(rows)<40:
            continue
        chunks=np.array_split(rows[np.argsort(p[rows])],5)
        for k,ids in enumerate(chunks):
            knots[j,k]=np.median(p[ids])
            h=p[ids]-b[ids]; t=labels[ids]-b[ids]
            t=np.clip(t,*np.quantile(t,[.015,.985]))
            scale=(.35 if "strong" in route else .6)
            grid=np.linspace(scale,1,8)
            e=(grid[:,None]*h-t)
            if "mse" in route:
                loss=np.where(e>0,1.6,1.)*e**2
            else:
                loss=np.expm1(np.clip(np.where(e>0,e/20,-e/26),0,32))
            w=grid[np.argmin(loss.mean(axis=1))]
            trust[j,k]=(w*len(ids)+.8*32)/(len(ids)+32)
    state.update(t_knots=knots,t_values=trust)
    return state


def fit_correction(route, model, state, x, y):
    if route.startswith("earned"):
        return earned_trust(route,model,state,x,y)
    if route.startswith("trust"):
        state["trust"] = np.asarray(float(route[5:])/100)
        return state
    if route.startswith(("exp","qt")):
        return tail_table(route,model,state,x,y)
    idx = calibration_rows(len(y))
    xs, ys = x[idx], y[idx]
    prior = v2.predict(model, state, xs)
    g = v2._regimes(xs, state["centers"])
    z = np.clip((xs[:, :-3]-state["means"][g])*state["inv_scale"][g], -4, 4)
    z = np.column_stack((z, np.clip((prior-100)/65, -3, 3)))
    k, dim = len(state["centers"]), z.shape[1]
    first, second = np.triu_indices(dim)
    quad = route not in ("linear","bias")
    if not quad:
        first, second = first[:0], second[:0]
    target_lo, target_hi = np.quantile(ys, [.18,.82])
    state.update(c_lin=np.zeros((k,dim)), c_quad=np.zeros((k,dim,dim)), c_bias=np.zeros(k))
    for j in range(k):
        use = g==j
        if np.count_nonzero(use)<40:
            continue
        u, labels, score = z[use],ys[use],prior[use]
        basis = np.column_stack((u, u[:,first]*u[:,second]))
        resid = labels-score
        weights = np.ones(len(labels))
        if route in ("tails", "tailsoft", "twosided"):
            weights += (0.35 if route=="tailsoft" else 1.0)*((labels<=target_lo)|(labels>=target_hi))
        if route=="risk":
            weights += .3*((labels<target_lo)&(resid<0))
        if route=="twosided":
            weights += .5*((labels<target_lo)&(resid<0))
        if route=="bias":
            state["c_bias"][j]=resid.mean()
            continue
        mu = np.average(basis,axis=0,weights=weights)
        fit_x = basis-mu
        gram = fit_x.T @ (weights[:,None]*fit_x)
        gram.flat[::len(gram)+1] += .08*len(u)
        correction = np.linalg.solve(gram,fit_x.T@(weights*resid))
        state["c_lin"][j] = correction[:dim]
        state["c_quad"][j,first,second] = correction[dim:]
        state["c_bias"][j] = np.average(resid,weights=weights)-mu@correction
    state["c_scale"] = np.array(.65 if route!="mild" else .32)
    state["c_limit"] = np.array(20.)
    return state


def predict_route(model, state, x):
    p = v2.predict(model, state, x)
    if "t_values" in state:
        base=model.predict(x).reshape(-1)
        groups=v2._regimes(x,state["centers"])
        for j in np.unique(groups):
            use=groups==j
            w=np.interp(p[use],state["t_knots"][j],state["t_values"][j])
            p[use]-=(1-w)*np.maximum(0.,p[use]-base[use])
        return p
    if "trust" in state or "downtrust" in state:
        base=model.predict(x).reshape(-1)
        d=p-base
        return p-(1.-float(state.get("trust",1.)))*np.maximum(0.,d)-(1.-float(state.get("downtrust",1.)))*np.minimum(0.,d)
    if "r_shift" in state:
        g=v2._regimes(x,state["centers"])
        for j in np.unique(g):
            mask=g==j
            p[mask]+=float(state["r_scale"])*np.interp(p[mask],state["r_knots"][j],state["r_shift"][j])
        if "post_trust" in state:
            p=p-(1.-float(state["post_trust"]))*np.maximum(0.,p-model.predict(x).reshape(-1))
        return np.maximum(0,p)
    if "c_lin" not in state:
        return p
    g = v2._regimes(x, state["centers"])
    z = np.clip((x[:, :-3]-state["means"][g])*state["inv_scale"][g], -4, 4)
    z = np.column_stack((z, np.clip((p-100)/65, -3, 3)))
    delta = np.zeros(len(x))
    for j in np.unique(g):
        m = g==j
        u=z[m]
        delta[m]=u@state["c_lin"][j]+np.sum((u@state["c_quad"][j])*u,axis=1)+state["c_bias"][j]
    return np.maximum(0,p+np.clip(state["c_scale"]*delta,-state["c_limit"],state["c_limit"]))


def fitted(route, model, x, y, index, previous=None, stable_release=False):
    state = v2.fit(model,x,y,index,previous,stable_release)
    if route.startswith("steady"):
        route=("trust65" if route=="steady" else "earned") if stable_release and previous else "expwarm"
    if route.startswith("hybrid"):
        state["post_trust"]=np.array(.65 if route=="hybrid" else .8)
        route="expwarm"
    return state if route=="v2" else fit_correction(route,model,state,x,y)


def data(dataset):
    if dataset == "DS02":
        with np.load(HERE.parent/"work/engine/external.npz") as z:
            sensor,settings,y,cycle,unit,cruise = [z[k] for k in ("sensors","settings","y","cycle","unit","cruise")]
        domain=np.repeat("DS02",len(unit))
        audit=np.isin(unit,[18,20])
        alt=np.searchsorted(np.quantile(settings[~audit,0],[.25,.5,.75]),settings[:,0])
        mach=settings[:,1]>np.median(settings[~audit,1])
        x=np.column_stack((sensor,settings,alt,settings[:,1],mach))
        paths=[str(i) for i in range(x.shape[1])]
        model=None
    else:
        base=benchmark.WORK/"C"
        model=harness.Model.load(base/"checkpoint.npz")
        def read(name):
            return [json.loads(s) for s in (base/"public"/(name+".jsonl")).open()]
        rows=read("train")
        n=len(rows)
        if dataset=="official" or dataset=="drift":
            rows+=read("validation")
            audit=np.arange(len(rows))>=n
        else:
            audit=np.array([r["meta"]["unit"]%5==int(dataset[-1]) for r in rows])
        x,y,_=harness.projection(rows,model,"C")
        paths=model.meta["paths"]
        cycle,unit,domain=[np.array([r["meta"][k] for r in rows]) for k in ("cycle","unit","source_subset")]
        cruise=None
    identities = np.array([f"{a}/{b}" for a,b in zip(domain,unit)])
    age = np.zeros(len(y))
    for identity in np.unique(identities):
        for offset in (False,True):
            # Validation and train sets may restart unit numbering.
            ids = np.flatnonzero((identities==identity)&(audit==offset))
            if len(ids):
                age[ids]=(cycle[ids]-cycle[ids].min()+.5)/(np.ptp(cycle[ids])+1)
    first=(age<=.55)&~audit
    if dataset!="official":
        model=harness.Model("regression",paths,[],x[first],y[first],hidden=32,seed=42,batch_size=256)
        model.fit(x[first],y[first],15)
    return x,y,audit,age,identities,cruise,model


def metrics(pred, y, age, cruise):
    e=pred-y
    phase=np.minimum(3,(4*age).astype(int))
    rm=lambda err:float(np.sqrt(np.mean(err**2)))
    out=dict(rmse=rm(e),phase=[rm(e[phase==i]) for i in range(4)],late_gt30=float((e>30).mean()),
             terminal_over30=float(np.mean(e[phase==3]>30)),q90_abs=float(np.quantile(abs(e),.9)),
             nasa_late=float(np.expm1(np.maximum(e,0)/10).sum()), nasa_early=float(np.expm1(np.maximum(-e,0)/13).sum()),
             nasa=float(np.expm1(np.where(e>0,e/10,-e/13)).sum()))
    if cruise is not None:
        out.update(cruise=rm(e[cruise]),noncruise=rm(e[~cruise]))
    return out


def probe(dataset,routes,series=""):
    x,y,audit,age,units,cruise,model=data(dataset)
    target=y[audit]
    cuts = [.55,.7,.85,1.] if dataset!="official" else [1.,1.]
    prev=None
    rows=[]
    for i,cut in enumerate(cuts,1):
        use=~audit & (age<=cut)
        start=process_time()
        shared=v2.fit(model,x[use],y[use],i,previous=prev,stable_release=dataset=="official" and i>1)
        vanilla_s=process_time()-start
        prev=shared
        ref=v2.predict(model,shared,x[audit])
        for route in routes:
            start=process_time()
            if route=="frozen":
                h=dict(shared,extension_fraction=np.asarray(.50))
            else:
                h=shared if route=="v2" else fit_correction(route,model,dict(shared),x[use],y[use])
            extra=process_time()-start
            fn=engine.predict if route=="frozen" else v2.predict if route=="v2" else predict_route
            pred=fn(model,h,x[audit])
            item=metrics(pred,target,age[audit],cruise[audit] if cruise is not None else None)
            if cut==1.:
                ids=np.linspace(0,len(pred)-1,min(256,len(pred)),dtype=int)
                xx=x[audit][ids]
                if route=="frozen":
                    expected=predict_route(model,dict(shared,trust=np.asarray(.50)),x[audit])
                    np.testing.assert_allclose(expected,pred,atol=1e-9,rtol=1e-12)
                tick=process_time()
                single=np.array([fn(model,h,xx[j:j+1])[0] for j in range(len(xx))])
                item["sample_us"]=1e6*(process_time()-tick)/len(ids)
                np.testing.assert_allclose(single,pred[ids],rtol=1e-9,atol=1e-8)
                group_ids=units[audit]
                sums=np.array([np.sum((pred[group_ids==u]-target[group_ids==u])**2-
                                      (ref[group_ids==u]-target[group_ids==u])**2) for u in np.unique(group_ids)])
                counts=np.array([np.sum(group_ids==u) for u in np.unique(group_ids)])
                idx=np.random.default_rng(31).integers(len(sums),size=(1000,len(sums)))
                item["paired_mse_ci"]=np.quantile(sums[idx].sum(1)/counts[idx].sum(1),[.025,.975]).tolist()
            item.update(release=cut,round=i,route=route,head_fit_s=vanilla_s+extra)
            rows.append(item)
    OUT.mkdir(exist_ok=True,parents=True)
    (OUT/(dataset+"-probe"+series+".json")).write_text(json.dumps(rows,indent=2))


def replay(route,split,setting):
    if split not in range(5):
        raise ValueError("Engine development is limited to splits 0 through 4")
    benchmark.OUT=OUT/route
    if route=="frozen":
        return benchmark.run("C","v3",split,setting,True)
    if hasattr(engine,"fit_candidates"):
        del engine.fit_candidates
    if route.startswith("family"):
        def family(model,x,y,i,previous=None,stable_release=False):
            original=v2.fit(model,x,y,i,previous,stable_release)
            scale=.15 if "strong" in route else .45
            heads={"engine":original,"limit_extension":dict(original,trust=np.array(scale))}
            if route in ("familyband","familywide"):
                # Guard the complete candidate set against volatile life extensions.
                low,high=(.45,.65) if route=="familyband" else (.30,.65)
                heads={"moderate_extension":dict(original,trust=np.array(high)),
                       "conservative_extension":dict(original,trust=np.array(low))}
            if "both" in route:
                heads["limit_shortening"]=dict(original,downtrust=np.array(scale))
            if "warm" in route:
                heads["tail_risk"]=tail_table("expwarm",model,dict(original),x,y)
            return heads
        engine.fit_candidates=family
    engine.fit=lambda m,x,y,i,previous=None,stable_release=False:fitted(route,m,x,y,i,previous,stable_release)
    engine.predict=v2.predict if route=="v2" else predict_route
    benchmark.run("C","v3",split,setting,True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("mode",choices=["probe","replay"])
    p.add_argument("name")
    p.add_argument("args",nargs="*")
    a=p.parse_args()
    if a.mode=="probe":
        probe(a.name,a.args,series=("-"+a.args[1] if len(a.args)>1 else ""))
    else:
        replay(a.name,int(a.args[0]),a.args[1])
