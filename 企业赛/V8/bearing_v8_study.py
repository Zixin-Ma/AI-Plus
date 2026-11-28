"""Public held-bearing, source validation and phase replays for bounded calibration of the V5 head."""
from importlib.util import module_from_spec, spec_from_file_location
import json
import os
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "work/v8/bearing"
OUT = OUT / os.environ.get("BEARING_WORK", "")
OUT.mkdir(exist_ok=True, parents=True)
sys.path.insert(0, str(HERE))


def load(name, path):
    spec = spec_from_file_location(name, path)
    mod = module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


v5 = load("bearing_v5_fixed", HERE.parent/"V5/bearing.py")
v7 = load("bearing_v7_fixed", HERE.parent/"V7/bearing.py")
ROUTES = ["v5", "v7", "dense", "pair", "loss", "trim", "spectral", "ordered",
          "offset", "upper", "cover", "pool", "lowbias", "root", "isolate", "early", "late"]


def proposal(model, h, x, y, route):
    if route == "v5":
        return h
    if route.startswith(("rank","delta","shape","envelope","gate","health","stretch","repeat","conditional","coherent","terminal","combine","hinge","lower","narrow","centered","taper","upperwing","adaptive","guarded","balanced","compass","fusion","robust","tight","observed","complete","bounded","qualified","certified","evidence","verified")):
        return lifecycle(model,h,x,y,route)
    if route.startswith(("cal","diag","map","prob","shrink","shift","prior","separate","imbalance","span","risk","cross")):
        return response_fit(model,h,x,y,route)
    spans = v7._training_runs(y)
    if len(spans) < 3:
        return h
    residual = y - model.predict(x)
    d = v5._design(x, h)
    sizes = np.array([s.stop-s.start for s in spans])
    # Equal temporal quantiles cover each lifetime. Partition widths vary between devices.
    sample = np.unique(np.concatenate([np.linspace(s.start, s.stop-1, min(s.stop-s.start, 480), dtype=int) for s in spans]))
    z = d[sample]
    t = residual[sample]
    gram = z.T @ z
    rhs = z.T @ t
    if route in ("pair", "ordered", "late", "early"):
        for s in spans:
            step = max(1,(s.stop-s.start)//8)
            ends = np.linspace(s.start+step, s.stop-1, min(480, s.stop-s.start-step), dtype=int)
            diff = d[ends] - d[ends-step]
            weight = (1 if route != "ordered" else 4.)
            gram += weight*(diff.T @ diff)
            rhs += weight*(diff.T @ (residual[ends]-residual[ends-step]))
    if route in ("loss","trim"):
        error = residual - d @ h["coef"]
        for s in spans:
            ids = sample[(sample>=s.start)&(sample<s.stop)]
            err = error[ids]
            if route=="trim":
                weight = np.minimum(1., np.quantile(abs(err),.70)/np.maximum(abs(err),.001))
            else:
                weight = np.full(len(ids), np.clip(np.sqrt(np.mean(err**2))/np.maximum(np.sqrt(np.mean(error**2)),1e-5),.25,2.))
            zz=d[ids]
            gram += zz.T @ ((weight-1)[:,None]*zz)
            rhs += zz.T @ ((weight-1)*residual[ids])
    if route == "spectral":
        mid = len(h["center"])+1
        gram[mid:,mid:] += len(z)*.16*np.eye(len(gram)-mid)
    gram.flat[::len(gram)+1] += .008*len(z)
    coef = np.linalg.solve(gram,rhs)
    if route == "early" and np.mean(y<=.02*np.quantile(y,.9)) > .025: return h
    if route == "late" and np.mean(y<=.02*np.quantile(y,.9)) <= .025: return h
    if route in ("dense", "pair","loss","trim","spectral","ordered","early","late"):
        return {**h,"coef":coef}
    # Conditional mean lifetime follows the sampling mass. Estimate on common quantiles per
    # observed machine. Incomplete spans retain original scale, avoiding a hard lifespan reset.
    if "curve_spans" not in h: return h
    old = v5.predict(model, h, x)
    med = np.array([np.mean(y[s]-old[s]) for s in spans])
    if route in ("offset","upper","lowbias"):
        offset = np.quantile(med,{"offset":.5,"upper":.75,"lowbias":.25}[route])
        c=h["coef"].copy()
        c[0] += offset
        return {**h,"coef":c}
    q=np.array([np.quantile(y[s],[.25,.5,.75]) for s in spans])
    if route=="pool": target=np.mean(q[:,1])*2
    elif route=="root": target=(np.mean(np.sqrt(q[:,1])))**2*2
    elif route=="isolate": target=np.quantile(q[:,1],.75)*2
    else: target=np.median(q[:,1])*2
    return {**h,"curve_spans":.5*h["curve_spans"]+.5*target,
            "curve_short":.5*h["curve_short"]+.25*target}


def lifecycle(model,h,x,y,route):
    import re
    spans=v7._training_runs(y)
    if not spans or "curve_spans" not in h: return h
    shape=re.match(r"[a-z]+",route).group()
    amount=float(route[len(shape):] or 1.)
    if shape in ("certified","evidence","verified"):
        complete=[s for s in spans if abs(y[s.stop-1])<=1e-8]
        terminal=[s for s in spans if y[s.stop-1]<=max(1.,.10*np.max(y[s]))]
        if len(terminal)<max(3,np.ceil(2.*len(spans)/3.)): return h
        alt=proposal(model,h,x,y,"upper")
        change=alt["coef"][0]-h["coef"][0]
        limit=np.median(abs(y-v5.predict(model,h,x)))
        mature = len(complete)>=max(3,np.ceil(2.*len(spans)/3.))
        gain=.5 if mature or shape=="evidence" else 1.
        if shape=="verified" and mature: gain=0.
        rebuilt={**h,"coef":h["coef"].copy()}
        rebuilt["coef"][0]+=gain*np.clip(change,-limit,limit)
        if not mature: return rebuilt
        return lifecycle(model,rebuilt,x,y,"lower"+str(amount))
    if shape in ("observed","complete","bounded","qualified"):
        complete=[s for s in spans if abs(y[s.stop-1])<=1e-8]
        alt=proposal(model,h,x,y,"upper")
        change=alt["coef"][0]-h["coef"][0]
        limit=np.quantile(abs(y-v5.predict(model,h,x)),.5)
        gain=.5 if shape=="bounded" else 1.
        rebuilt={**h,"coef":h["coef"].copy()}
        rebuilt["coef"][0]+=gain*np.clip(change,-limit,limit)
        if len(complete) < max(3, np.ceil(.67*len(spans))): return rebuilt
        if shape=="qualified": return lifecycle(model,h,x,y,"lower0.75")
        return lifecycle(model,rebuilt if shape!="complete" else h,x,y,"lower"+str(amount))
    if shape in ("fusion","robust","tight"):
        alt=proposal(model,h,x,y,"upper")
        h={**h,"coef":h["coef"]+amount*(alt["coef"]-h["coef"])}
        return lifecycle(model,h,x,y,{"fusion":"lower1","robust":"coherent0.5","tight":"lower0.5"}[shape])
    if shape in ("adaptive","guarded","balanced","compass"):
        t=sum(y[s.stop-1] <= max(1.,.02*np.max(y[s])) for s in spans)
        if t < .70*len(spans):
            alt=proposal(model,h,x,y,"upper")
            return {**h,"coef":h["coef"]+.65*(alt["coef"]-h["coef"])}
        return lifecycle(model,h,x,y,{"adaptive":"coherent","guarded":"taper","balanced":"lower","compass":"hinge"}[shape]+str(amount))
    if shape=="combine":
        # Preserve complete releases; give the upper-quartile unit residual a bounded vote on prefixes.
        h=lifecycle(model,h,x,y,"rank"+str(amount))
        if sum(y[s.stop-1] <= max(1.,.02*np.max(y[s])) for s in spans) < .7*len(spans):
            h=proposal(model,h,x,y,"upper")
        return h
    bands=np.log(np.maximum(x[:,h["curve_bands"]],1e-12))
    lo=bands[:,:2].mean(1); hi=bands[:,2:].mean(1)
    ev=v5.order_evidence(x,h["order_cols"])
    wt=np.exp(7.*(ev-ev.max(1)[:,None]));wt/=wt.sum(1)[:,None]
    lev=np.linspace(.025,.975,13)
    trajectories=[s for s in spans if y[s.stop-1]<=max(1.,.02*np.max(y[s])) and np.ptp(y[s])>0]
    if len(trajectories)<3: return h
    if shape in ("gate","terminal","envelope","health"):
        if not h["curve_mature"]: return h
        # Guarding for impact is calibrated at equal fractions of each complete source run.
        qs=[.20,.75] if shape=="gate" else [.25,.9]
        curves=[np.quantile(hi[s],qs) for s in trajectories]
        sizes=np.array([s.stop-s.start for s in trajectories])
        if shape=="gate" or shape=="envelope":
            bounds=np.average(curves,axis=0,weights=np.sqrt(sizes))
        else:
            template=[]
            for s in trajectories:
                q=y[s]/max(y[s]);template.append([np.median(hi[s][q>=.75]),np.median(hi[s][q<=.25])])
            bounds=np.average(template,axis=0,weights=np.sqrt(sizes))
            if bounds[1]<=bounds[0]: return h
        return {**h,"curve_impact":h["curve_impact"]+amount*(bounds-h["curve_impact"])}
    all_life=[]; all_energy=[]; all_prob=[]; all_weight=[]
    for s in trajectories:
        life=y[s]/max(y[s]);en=lo[s]
        levels=np.linspace(.025,.975,13)
        if shape in ("shape","stretch"):
            # Invert unit's monotone profile after pooling adjacent 7.9%-life bins.
            cuts=np.array([np.median(en[abs((1-life)-j)<=.055]) if np.any(abs((1-life)-j)<=.055) else np.median(en) for j in levels])
            cuts=np.maximum.accumulate(cuts)
        else: cuts=np.quantile(en,levels)
        all_energy.append(cuts)
        all_prob.append(wt[s].mean(0))
        all_weight.append(s.stop-s.start)
    all_energy=np.stack(all_energy)
    vote=np.array(all_prob);sz=np.asarray(all_weight)
    if shape in ("rank","shape","repeat","coherent","hinge","lower","narrow","centered","taper","upperwing"): vote*=np.sqrt(sz[:,None])
    if shape in ("stretch","delta"): vote*=sz[:,None]
    if shape in ("conditional","repeat"): vote=vote**2
    new=np.stack([np.average(all_energy,axis=0,weights=np.maximum(wj,1e-9)) for wj in vote.T])
    if shape in ("coherent","hinge","lower","narrow","centered","taper","upperwing"):
        # Match source center; only reshape the empirical within-run spread.
        new+=h["curve_cuts"][:,6:7]-new[:,6:7]
        if shape=="lower": new[:,6:]=h["curve_cuts"][:,6:]
        if shape=="upperwing": new[:,:7]=h["curve_cuts"][:,:7]
        if shape=="hinge": new=h["curve_cuts"]+np.maximum(0.,new-h["curve_cuts"])
        if shape=="narrow": new=h["curve_cuts"]+np.minimum(0.,new-h["curve_cuts"])
        if shape=="taper":
            gain=1.-abs(np.linspace(-1.,1.,len(lev)))
            new=np.maximum.accumulate(h["curve_cuts"]+gain*(new-h["curve_cuts"]),axis=1)
        if shape=="centered":
            offset=np.clip(new-h["curve_cuts"],-.25*np.ptp(h["curve_cuts"],axis=1)[:,None],.25*np.ptp(h["curve_cuts"],axis=1)[:,None])
            new=np.maximum.accumulate(h["curve_cuts"]+offset,axis=1)
    return {**h,"curve_cuts":h["curve_cuts"]+amount*(new-h["curve_cuts"])}


def response_fit(model,h,x,y,route):
    import re
    spans=v7._training_runs(y)
    if not spans or "curve_spans" not in h: return h
    tag=re.match(r"[a-z]+",route).group()
    opt=float(route[len(tag):] or 1)
    d=v5._design(x,h)
    old=v5.predict(model,h,x)
    base=model.predict(x)
    delta=y-old
    ids=np.linspace(0,len(y)-1,min(2400,len(y)),dtype=int)
    if tag in ("diag","cal","map","prob","cross","shrink","risk"):
        bands=np.log(np.maximum(x[:,h["curve_bands"]],1e-12))
        lo=bands[:,:2].mean(1); hi=bands[:,2:].mean(1)
        freq=v5.order_evidence(x,h["order_cols"])
        weights=np.exp(7*(freq-freq.max(1)[:,None]));weights/=weights.sum(1)[:,None]
        if tag=="shrink":
            comp=np.c_[base-old,(base-old)*weights[:,0],(base-old)*weights[:,2]]
        elif tag=="map":
            comp=np.c_[hi-lo,bands[:,0]-bands[:,1],bands[:,2]-bands[:,3],hi,lo,old,base]
        elif tag=="prob":
            comp=np.c_[weights[:,0],weights[:,2],old,base]
        elif tag=="diag":
            comp=np.c_[old,base,old-base,lo,hi]
        elif tag=="cal":
            comp=np.c_[old,base]
        else:
            comp=np.c_[hi-lo,bands[:,0]-bands[:,1],bands[:,2]-bands[:,3],weights[:,0],weights[:,2],old,base]
        scaled=np.clip((comp-np.median(comp,axis=0))/np.maximum(np.ptp(np.quantile(comp,[.15,.85],axis=0),axis=0),1e-6),-4,4)
        design=np.c_[np.ones(len(y)),scaled]
        def calibrate(idx):
            dd=design[idx]
            return np.linalg.solve(dd.T@dd+.075*len(idx)*np.eye(dd.shape[1]),dd.T@delta[idx])
        if tag in ("risk","cross"):
            coef=np.array([calibrate(np.arange(s.start,s.stop)) for s in spans])
            w=np.sqrt([s.stop-s.start for s in spans])
            if tag=="risk": w=np.ones_like(w)
            correction=design@(np.average(coef,axis=0,weights=w))
        else:
            correction=design@calibrate(ids)
        correction=np.clip(correction,-np.quantile(abs(delta),.70),np.quantile(abs(delta),.70))
        zz=d[ids]; G=zz.T@zz
        G.flat[::len(G)+1]+=.025*len(zz)
        direction=np.linalg.solve(G,zz.T@correction[ids])
        return {**h,"coef":h["coef"]+opt*direction}
    freq=v5.order_evidence(x,h["order_cols"])
    weights=np.exp(7*(freq-freq.max(1)[:,None]));weights/=weights.sum(1)[:,None]
    if tag=="span":
        maxlife=np.array([np.max(y[s]) for s in spans])
        devw=np.stack([weights[s].mean(0) for s in spans])
        # Normalize each trajectory to one vote, then estimate scale of its lifetime.
        target=np.average(maxlife[:,None]*np.ones((1,3)),weights=devw,axis=0)
        return {**h,"curve_spans":h["curve_spans"]+opt*(target-h["curve_spans"])}
    if tag=="shift":
        condition=weights.T@delta/weights.sum(0)
        return {**h,"order_means":np.maximum(0,h["order_means"]+opt*condition)}
    if tag=="prior":
        mu=np.mean(y)
        return {**h,"prior":h["prior"]+opt*(mu-h["prior"]),"order_means":h["order_means"]+opt*(weights.T@y/weights.sum(0)-h["order_means"])}
    if tag=="separate":
        # Decouple soft assignment means from the presorted frequency-ordered baseline.
        effective=weights.T@np.sqrt(np.maximum(y,0))/weights.sum(0)
        return {**h,"order_means":h["order_means"]+opt*(effective**2-h["order_means"])}
    if tag=="imbalance":
        estimate=y.mean()+weights.T@delta/weights.sum(0)
        return {**h,"prior":h["prior"]+opt*delta.mean(),"order_means":np.maximum(0,h["order_means"]+opt*(estimate-y.mean()))}
    raise ValueError(route)


def impl(route):
    if route=="v5": return v5
    if route=="v7": return v7
    if route=="production": return load("bearing_v8_frozen", HERE/"bearing.py")
    mod = load("bearing_candidate",HERE.parent/"V5/bearing.py")
    def fit(model,x,y,index):
        head=v5.fit(model,x,y,index)
        if ":" in route:
            name,weight=route.split(":")
            alt=proposal(model,head,x,y,name)
            return {k:(v+float(weight)*(alt[k]-v) if k in ("coef","curve_spans","curve_short") else v) for k,v in head.items()}
        return proposal(model,head,x,y,route)
    mod.fit=fit
    return mod


def metrics(yhat,t,group):
    block=[]
    for gid in np.unique(group):
        ix=np.where(group==gid)[0]
        block.extend(x for x in np.array_split(ix,4) if len(x))
    error=yhat-t
    return dict(rmse=float(np.mean(error**2)**.5),mae=float(np.abs(error).mean()),worst_quarter=float(max(np.mean(error[j]**2)**.5 for j in block)))


def folds(routes):
    from industrial import Model
    import contest_submission as cs
    helpers=load("replay_data_b3", HERE.parent/"V3/bearing_v3_study.py")
    base=Model.load(helpers.loader.DATA/"checkpoint.npz")
    result={}
    def evaluate(key,m,x,y,z,t,grp):
        h5=v5.fit(m,x,y,1)
        h7=v7.fit(m,x,y,1)
        rows={}
        for name in routes:
            source=v7 if name=="v7" else v5
            if name=="production":
                source=impl(name); h=source.fit(m,x,y,1)
            elif name=="v7": h=h7
            elif ":" in name:
                h=impl(name).fit(m,x,y,1)
            else: h=proposal(m,h5,x,y,name)
            rows[name]=metrics(source.predict(m,h,z),t,grp)
        result[key]=rows
    for src,(x,y,grp) in [("XJTU",helpers.loader.load("train",base)),("FEMTO",helpers.external(base))]:
        for gid in np.unique(grp):
            tr=grp!=gid
            m=Model("regression",base.meta["paths"],[],x[tr],y[tr],hidden=24)
            m.fit(x[tr],y[tr],15)
            cs.continue_mlp("B",1,m,x[tr],y[tr],"half")
            evaluate(src+"/"+str(gid),m,x[tr],y[tr],x[~tr],y[~tr],grp[~tr])
    x,y,_=helpers.loader.load("train",base)
    z,t,grp=helpers.loader.load("validation",base)
    cs.continue_mlp("B",1,base,x,y,"half")
    evaluate("public",base,x,y,z,t,grp)
    for gid in np.unique(grp):
        ix=grp==gid
        evaluate("public/"+str(gid),base,x,y,z[ix],t[ix],grp[ix])
    ext=load("independent_b4",HERE.parent/"V4/bearing_v4_study.py")
    ext.OUT=OUT
    z,t=ext.kaist_samples(base,helpers)
    evaluate("KAIST",base,x,y,z,t,np.zeros(len(t)))
    (OUT/("folds-"+"-".join(routes)+".json")).write_text(json.dumps(result,indent=2))
    print("route XJTU, FEMTO, public, KAIST, public worst quarter")
    for name in routes:
        out=[np.mean([r[name]["rmse"] for k,r in result.items() if k.startswith(s+"/")]) for s in ("XJTU","FEMTO")]
        print(name, *[round(z,3) for z in out+[result["public"][name]["rmse"],result["KAIST"][name]["rmse"],result["public"][name]["worst_quarter"]]])


def opening(routes):
    sys.path.insert(0,str(HERE.parent/os.environ.get("BEARING_CONTROL","V7")))
    sys.path.insert(1,str(HERE.parent/"V7"))
    import benchmark, serving_study as svc, contest_submission as agent
    svc.ASSETS=OUT/"artifacts"
    svc.WORK=OUT
    static=agent.Model.load(benchmark.WORK/"B"/"checkpoint.npz")
    valid=svc.rows("B","validation")
    xv,yv,_=agent.projection(valid,static,"B")
    groups=np.asarray([r["meta"]["bearing"] for r in valid])
    summary={}
    for route in routes:
        sys.modules["bearing"]=impl(route)
        svc.make_artifact("B",route)
        dest=svc.ASSETS/route/"B"
        m,h,_=agent.artifact(dest)
        pred=agent.values("B",m,h,xv).reshape(-1)
        summary[route]=dict(**metrics(pred,yv,groups),choice=json.loads((dest/"record.json").read_text()))
        summary[route]["devices"] = {str(g): metrics(pred[groups==g],yv[groups==g],groups[groups==g]) for g in np.unique(groups)}
        print(route,summary[route],flush=True)
    (OUT/("opening-"+"-".join(routes)+".json")).write_text(json.dumps(summary,indent=2))


def verify():
    import time
    import contest_submission as cs
    from industrial import Model
    fixture=load("public_b3",HERE.parent/"V3/bearing_v3_study.py")
    model=Model.load(fixture.loader.DATA/"checkpoint.npz")
    x,y,grp=fixture.loader.load("train",model)
    z,_,_=fixture.loader.load("validation",model)
    cs.continue_mlp("B",1,model,x,y,"half")
    last=impl("production")
    legacy=v5.fit(model,x,y,1)
    report={}
    rng=np.random.default_rng(715)
    for tag,ids in [("complete",np.arange(len(y))),("early",np.concatenate([np.where(grp==g)[0][:max(1,(grp==g).sum()//2)] for g in np.unique(grp)])),("shuffled",rng.permutation(len(y)))]:
        main=last.fit(model,x[ids],y[ids],1)
        base=v5.fit(model,x[ids],y[ids],1)
        old=last.predict(model,base,z)
        tuned=last.predict(model,main,z)
        other=impl("certified0.75")
        alt=other.fit(model,x[ids],y[ids],1)
        scalar=last.prepare_inference(model,main)
        report[tag]=dict(vs_design=float(np.max(abs(tuned-other.predict(model,alt,z)))),scalar_max_abs=float(np.max(abs(tuned-np.array([scalar(row) for row in z])))),vs_v5=float(np.max(abs(tuned-old))),nonnegative=bool(np.min(tuned)>=0))
    report["legacy_reads_max_abs"]=float(np.max(abs(last.predict(model,legacy,z)-v5.predict(model,legacy,z))))
    calls={"v5":v5.prepare_inference(model,legacy),"production":last.prepare_inference(model,last.fit(model,x,y,1))}
    times={k:[] for k in calls}
    for rep in range(9):
        rows=z[rng.permutation(len(z))[:800]]
        for k in (list(calls) if rep%2 else list(reversed(calls))):
            start=time.perf_counter_ns()
            for row in rows: calls[k](row)
            times[k].append((time.perf_counter_ns()-start)/len(rows)/1000.)
    report["scalar_us_runs"]=times
    (OUT/"verify_bearing.json").write_text(json.dumps(report,indent=2))
    print(json.dumps(report))


def replay(route,mode,seed):
    if seed not in range(20): raise ValueError("Reserved confirmation seed")
    sys.path.insert(0,str(HERE.parent/os.environ.get("BEARING_CONTROL","V7")))
    sys.path.insert(1,str(HERE.parent/"V7"))
    bench=load("bearing_v8_replay", HERE.parent/"V7/benchmark.py")
    sys.modules["bearing"]=impl(route)
    bench.OUT=OUT/route
    bench.OUT.mkdir(exist_ok=True)
    bench.run("B","v7",seed,mode)


if __name__=="__main__":
    if sys.argv[1]=="folds": folds(sys.argv[2:] or ROUTES)
    elif sys.argv[1]=="opening": opening(sys.argv[2:] or ROUTES)
    elif sys.argv[1]=="verify": verify()
    else: replay(sys.argv[2],sys.argv[3],int(sys.argv[4]))
