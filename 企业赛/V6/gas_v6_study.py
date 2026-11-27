"""Public-only gas model study, V6. Each run is isolated in a fresh process."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

import benchmark
import gas
import prototype

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "work/v6/gas"
ROUTES = ("final", "production", "base", "prior", "ordinal", "contrast", "temporal", "uncertain", "optional", "confident", "weak", "optionalprior", "weakprior", "balanced", "smallprior", "smallordinal", "smallboost", "smalltemper", "smallweak", "smallbalanced", "localtemp", "localboost", "localprior", "localweak", "extrap", "indtemp", "indweak", "indprior", "robust", "huber", "spread", "tangent", "medoid", "dose", "mixland")
SCREEN = (0, 3, 5, 6, 9, 11, 12)
CHECK = (1, 2, 4, 7, 8, 10, 13)


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def represent(z, state, route):
    if route in ("ordinal","uncertain","optional","confident","weak","smallordinal","smallweak","localweak","indweak"):
        s = np.arcsinh((z-state["q_center"])/state["q_scale"])
        order = s.argsort(axis=1, kind="stable").argsort(axis=1, kind="stable")
        # Soft ranks preserve distinctions between response ties.
        return (order/max(z.shape[1]-1,1)-.5).reshape(len(z),-1)
    log = np.arcsinh((z-state["q_center"])/state["q_scale"])
    # Project away a row-wise, per-measurement common electronics offset.
    common = log.mean(1,keepdims=True)
    return (log-common).reshape(len(z),-1)


def learn(z, y, classes, route):
    counts = np.bincount(y,minlength=classes)
    if route in ("prior","optionalprior","weakprior","balanced","smallprior","smallboost","smalltemper","smallbalanced","localtemp","localprior","localboost","extrap","indtemp","indprior"):
        logprior = np.log(counts+1.)
        return {"q_prior": (logprior.mean()-logprior).astype(np.float32)}
    middle = np.median(z,axis=0)
    state = {"q_center": middle,"q_scale": np.maximum(1.4826*np.median(abs(z-middle),axis=0),1e-6)}
    f = represent(z, state, route)
    means = np.stack([f[y==k].mean(0) for k in range(classes)])
    pooled = np.zeros((f.shape[1],f.shape[1]))
    shifts = []
    for k in range(classes):
        group = f[y==k]
        res = group-means[k]
        pooled += res.T@res / (classes*max(1,len(res)-1))
        for block in np.array_split(group,min(4,len(group))):
            shifts.append(block.mean(0)-means[k])
    variance = max(np.trace(pooled)/len(pooled),1e-4)
    cov = .7*pooled+.3*variance*np.eye(len(pooled))
    if route == "temporal":
        shifts = np.stack(shifts)
        cov += 2.0 * shifts.T@shifts/len(shifts)
    w = np.linalg.solve(cov,means.T)
    b = -.5*np.sum(w.T*means,axis=1)
    # Distances in units of the corresponding pair of source centroids.
    scores = f@w+b
    correct = scores[np.arange(len(y)),y]
    scores[np.arange(len(y)),y] = -np.inf
    typical = max(1.,np.median(correct-scores.max(1)))
    state.update(q_weights=w/typical,q_bias=b/typical)
    return {k:np.asarray(v,dtype=np.float32) for k,v in state.items()}


def adjust(z, state, route):
    if route in ("prior","optionalprior","weakprior","balanced","smallprior","smallboost","smalltemper","smallbalanced","localtemp","localprior","localboost","extrap","indtemp","indprior"):
        return state["q_prior"]
    a = represent(z,state,route)@state["q_weights"]+state["q_bias"]
    return np.clip(2.*(a-a.max(1,keepdims=True)),-4.,0.)


def weighted_adjust(scores, addition, route):
    if route == "uncertain":
        order=np.partition(scores,-2,axis=1)[:,-2:]
        return addition*np.clip(1.-(order[:,1]-order[:,0])/3.,0.,1.)[:,None]
    if route == "confident":
        order=np.partition(addition,-2,axis=1)[:,-2:]
        return addition*np.clip((order[:,1]-order[:,0])/.7,0.,1.)[:,None]
    if route in ("weak","weakprior","smallweak"):
        return .35*addition
    return addition


def balance_bias(x,y,predict):
    # Two predeclared gains emulate systematic sensitivity changes. Equal weight
    # per class prevents this calibration from reproducing the release histogram.
    perturbations=[predict(x*gain) for gain in (.4,1.,2.5)]
    scores=np.concatenate(perturbations,axis=0)
    target=np.tile(y,len(perturbations))
    n=scores.shape[1]
    mass=1./(n*np.bincount(target)[target])
    offset=np.zeros(n)
    for _ in range(8):
        centered=scores+offset
        e=np.exp(centered-centered.max(1,keepdims=True))
        probs=e/e.sum(1,keepdims=True)
        freq=mass@probs
        difference=(np.log(1./n)-np.log(np.maximum(freq,1e-8)))
        offset=np.clip(.9*offset+difference-difference.mean(),-2.,2.)
    return {"q_prior":offset.astype(np.float32)}


def support_state(q,y,classes):
    levels=np.linspace(0.,1.,9)
    return {"q_quantiles":np.stack([np.quantile(q[y==k],levels) for k in range(classes)]).astype(np.float32),
            "q_counts":np.bincount(y,minlength=classes).astype(np.float32)}


def support(q, state, label, route="localtemp"):
    result=np.zeros(len(q))
    for cls,knots in enumerate(state["q_quantiles"]):
        idx=label==cls
        if idx.any():
            p=np.interp(q[idx],knots,np.linspace(0.,1.,len(knots)))
            if route=="extrap":
                fraction=np.minimum(p,1.-p)
                small=np.clip(4./np.sqrt(np.min(state["q_counts"]))-1.,0.,1.)
                result[idx]=np.maximum(small,np.clip(1.-fraction/.05,0.,1.))
                continue
            effective=state["q_counts"][cls]*2.*np.minimum(p,1.-p)
            result[idx]=np.clip(4./np.sqrt(np.maximum(effective,1.))-1.,0.,1.)
    return result[:,None]


RESIDUAL_ROUTES = ("robust","huber","spread","tangent")


LANDMARKS = ("medoid","dose","mixland")


def physical_landmarks(z,q,y,h,route):
    f=prototype.invariant_features(z,h["phys_loc"],h["phys_scale"])
    f=np.clip((f-h["phys_mid"])/h["phys_std"],-8.,8.)
    nodes=[]
    classes=len(h["sub_means"]) if "sub_means" in h else int(np.max(y))+1
    for cls in range(classes):
        ids=np.flatnonzero(y==cls);v=f[ids]
        n=min(6,max(1,len(ids)//3))
        if route=="dose":
            ordered=v[np.argsort(q[ids],kind="stable")]
            centers=np.stack([g.mean(0) for g in np.array_split(ordered,n)])
        else:
            total=n if route=="medoid" else max(1,n//2)
            center=v.mean(0)
            first=((v-center)**2).sum(1).argmin()
            seeds=[v[first]];dist=np.full(len(v),np.inf)
            for _ in range(1,total):
                dist=np.minimum(dist,((v-seeds[-1])**2).sum(1))
                seeds.append(v[dist.argmax()])
            centers=np.stack(seeds)
            for _ in range(3):
                near=((v[:,None,:]-centers)**2).sum(2).argmin(1)
                centers=np.stack([v[near==j].mean(0) if np.any(near==j) else old for j,old in enumerate(centers)])
            if route=="mixland" and total<n:
                centers=np.concatenate([centers,np.stack([g.mean(0) for g in np.array_split(v,n-total)])])
        nodes.extend(centers)
    nodes=np.stack(nodes)
    norm=(nodes*nodes).mean(1)
    def kernel(a):
        sq=(a*a).mean(1)[:,None]+norm-(2./a.shape[1])*(a@nodes.T)
        return np.exp(-.85*np.maximum(sq,0.)/float(h["phys_width"]))
    k=kernel(f)
    wt=len(y)/(classes*np.bincount(y,minlength=classes)[y])
    coef=np.linalg.solve((k.T*wt)@k+.003*len(y)*kernel(nodes)+.001*np.eye(len(nodes)),(k.T*wt)@np.eye(classes)[y])
    return {key:v.astype(np.float32) for key,v in {"phys_nodes":nodes,"phys_norm":norm,"phys_coef":coef}.items()}


def fit_residual(x, y, h, route):
    classes=len(h["sub_means"])
    if route=="tangent":
        q=np.clip((gas._response_amplitude(x)-h["amp_center"])/h["amp_scale"],-3.,3.)
        return {"q_bounds":np.stack([np.quantile(q[y==k],[.05,.95]) for k in range(classes)]).astype(np.float32)}
    raw=np.log(np.maximum(abs(x),.02))
    d=prototype.raw_distance(raw,h["sub_center"],h["sub_scale"],h["sub_means"],h["sub_axes"])
    true=d[np.arange(len(y)),y]
    if route=="spread":
        mean=max(np.mean(true),.01)
        scale=np.array([(np.sum(true[y==k])+16*mean)/(np.sum(y==k)+16)/mean for k in range(classes)])
        return {"q_scale_residual":np.clip(scale,.5,2.).astype(np.float32)}
    return {"q_knee":np.asarray(max(1e-3,np.quantile(true,.9)) if route=="huber" else 0.,dtype=np.float32)}


def residual_adjust(x, h, route):
    raw=np.log(np.maximum(abs(x),.02))
    d=prototype.raw_distance(raw,h["sub_center"],h["sub_scale"],h["sub_means"],h["sub_axes"])
    f,q=gas.signatures(x)[0],gas._response_amplitude(x)
    kinetic=prototype.curve_log_score(prototype.curve_distance(f,q,h))
    old=kinetic-3*d
    if route=="tangent":
        new=prototype.curve_log_score(prototype.curve_distance(f,q,{**h,"amplitude_bounds":h["q_bounds"]},True))-3*d
        shift=0.
    else:
        if route=="spread":
            scale=h["q_scale_residual"];new_d=d/scale+np.log(scale)
        else:
            pivot=h["q_knee"];new_d=np.minimum(d,pivot)+np.log1p(np.maximum(d-pivot,0.))
        shift=3*(d-new_d);new=kinetic-3*new_d
    limit=float(h["sensor_evidence_cap"])
    return shift + np.clip(new-new.max(1,keepdims=True),-limit,0.)-np.clip(old-old.max(1,keepdims=True),-limit,0.)


def patch(route):
    if route in ("production","final"):
        return
    parent=load("previous_gas",HERE.parent/"V5/gas.py")
    fit0,pred0=parent.fit_candidates,parent.predict
    gas.fit_candidates,gas.predict=fit0,pred0
    if route == "base":
        return
    def fit(model,x,y,index):
        heads = fit0(model,x,y,index)
        if "tail" in heads:
            if route in LANDMARKS:
                h=heads["independent"]
                h.update(physical_landmarks(prototype.response_shape(x),gas._response_amplitude(x),y,h,route))
                return heads
            if route in RESIDUAL_ROUTES:
                heads["independent"].update(fit_residual(x,y,heads["independent"],route))
                return heads
            if route.startswith("ind"):
                h=heads["independent"]
                h.update(learn(prototype.response_shape(x),y,len(model.meta["classes"]),route))
                h.update(support_state(gas._response_amplitude(x),y,len(model.meta["classes"])))
                return heads
            extra = learn(prototype.response_shape(x),y,len(model.meta["classes"]),route)
            # Preserve the independent candidate as a chemically different fallback.
            if route.startswith("local") or route=="extrap":
                extra.update(support_state(gas._response_amplitude(x),y,len(model.meta["classes"])))
            if route.startswith("small"):
                extra["q_scarce"]=np.asarray(np.clip(4./np.sqrt(np.bincount(y).min())-1.,0.,1.),dtype=np.float32)
                if extra["q_scarce"]==0:
                    return heads
            if route in ("balanced","smallbalanced"):
                extra.update(balance_bias(x,y,lambda rows: pred0(model,heads["tail"],rows)))
            if route in ("optional","optionalprior"):
                heads["ordinal"] = {**heads["tail"], **extra}
            else:
                heads["tail"].update(extra)
        return heads
    def predict(model,head,x):
        scores = pred0(model,head,x)
        if route in RESIDUAL_ROUTES and any(k in head for k in ("q_knee","q_scale_residual","q_bounds")):
            return scores+residual_adjust(x,head,route)
        if "q_weights" in head or "q_prior" in head:
            if route.startswith(("local","ind")) or route=="extrap":
                base=np.log(np.maximum(model.predict(x),1e-12))
                weight=support(gas._response_amplitude(x),head,base.argmax(1),route)
                if route in ("localtemp","extrap","indtemp"):
                    correction=-.5*(base-base.mean(1,keepdims=True))
                elif route=="localboost":
                    correction=scores-base
                elif route in ("localweak","indweak"):
                    correction=.35*adjust(prototype.response_shape(x),head,route)
                else:
                    correction=head["q_prior"]
                scores += weight*correction
            elif route in ("smallboost","smalltemper"):
                base=np.log(np.maximum(model.predict(x),1e-12))
                scores += float(head["q_scarce"])*(scores-base if route=="smallboost" else -.5*(base-base.mean(1,keepdims=True)))
            else:
                scores += float(head.get("q_scarce",1.))*weighted_adjust(scores,adjust(prototype.response_shape(x),head,route),route)
        return scores
    gas.fit_candidates, gas.predict = fit,predict


def replay(route, part, settings):
    patch(route)
    benchmark.OUT=OUT/"replays"/route
    for setting in settings:
        for split in SCREEN if part=="screen" else CHECK:
            output=benchmark.OUT/f"{setting}-v6-A-{split}.json"
            if not output.exists():
                benchmark.run("A","v6",split,setting)


def external(route):
    src = load("external5",HERE.parent/"V5/gas_v5_study.py")
    src.OUT = HERE.parent/"work/v5/gas"
    native = src.native
    def compare_native(args,fit,query,unused):
        scores=native(args,fit,query,"angular")
        z, classes, y=args[5:8]
        tail=scores["v5_tail"]
        if route in LANDMARKS or route=="final":
            groups=args[-1];q=args[4]
            ref=prototype.fit_physical(z[fit],y[fit],len(classes),groups=groups[fit])
            new={**ref,**physical_landmarks(z[fit],q[fit],y[fit],ref,"dose" if route=="final" else route)}
            def vote(h):
                feats=prototype.invariant_features(z[query],h["phys_loc"],h["phys_scale"])
                t=np.clip((feats-h["phys_mid"])/h["phys_std"],-8.,8.)
                d=(t*t).mean(1)[:,None]+h["phys_norm"]-(2./t.shape[1])*(t@h["phys_nodes"].T)
                k=np.exp(-.85*np.maximum(0.,d)/h["phys_width"])
                return prototype.physical_vote(k@h["phys_coef"])
            independent=scores["v5_independent"]
            new_tail=tail
            if route=="final":
                scarce=np.clip(4./np.sqrt(np.bincount(y[fit]).min())-1.,0.,1.)
                if scarce:
                    from baseline_model import Model
                    x=args[0]
                    m=Model("classification",[str(i) for i in range(x.shape[1])],list(classes),x[fit],y[fit])
                    m.fit(x[fit],y[fit],20);m.meta["learning_rate"]=.001;m.fit(x[fit],y[fit],1)
                    base=np.log(np.maximum(m.predict(x[query]),1e-12))
                    new_tail=tail-.5*scarce*(base-base.mean(1,keepdims=True))
            return {"v4_tail":tail,"v5_tail":new_tail,"v4_independent":independent,
                    "v5_independent":independent-vote(ref)+vote(new)}
        if route in ("smalltemper","localtemp","localboost","localweak","localprior","smallboost","extrap"):
            from baseline_model import Model
            x, _, _, _, q = args[:5]
            m=Model("classification",[str(i) for i in range(x.shape[1])],list(classes),x[fit],y[fit])
            m.fit(x[fit],y[fit],20);m.meta["learning_rate"]=.001;m.fit(x[fit],y[fit],1)
            base=np.log(np.maximum(m.predict(x[query]),1e-12))
            scarce=np.clip(4./np.sqrt(np.bincount(y[fit]).min())-1.,0.,1.)
            if route.startswith("local") or route=="extrap":
                state=support_state(q[fit],y[fit],len(classes))
                scarce=support(q[query],state,base.argmax(1),route)
            if route in ("localboost","smallboost"):
                correction=scarce*(tail-base)
            elif route=="localweak":
                correction=.35*scarce*adjust(z[query],learn(z[fit],y[fit],len(classes),route),route)
            elif route=="localprior":
                correction=scarce*learn(z[fit],y[fit],len(classes),route)["q_prior"]
            else:
                correction=-.5*scarce*(base-base.mean(1,keepdims=True))
        else:
            correction=weighted_adjust(tail,adjust(z[query],learn(z[fit],y[fit],len(classes),route),route),route)
            if route.startswith("small"):
                correction*=np.clip(4./np.sqrt(np.bincount(y[fit]).min())-1.,0.,1.)
        return {"v4_tail":tail,"v5_tail":tail+correction,
                "v4_independent":scores["v5_independent"],"v5_independent":scores["v5_independent"]}
    src.native=compare_native
    data={"one_year":src.forward(route),"dsps":src.dsps(route,False),
          "concentration":src.dsps(route,True),"home":src.home(route)}
    (OUT/f"external-{route}.json").write_text(json.dumps(data,indent=2))


def public_pair():
    import contest_submission as harness
    from industrial import Model
    baseline=load("prev_gas_full",HERE.parent/"V5/gas.py")
    origfit,origpredict=gas.fit_candidates,gas.predict
    full=load("full_pair",HERE.parent/"V4/gas_v4_study.py");full.OUT=OUT
    orig_train=harness.train
    data=benchmark.WORK/"A";predict={}
    def collect(r):
        orig_train(r)
        m,h,_=harness.artifact(r.candidate)
        predict[tag]=harness.values("A",m,h,r.selection[0]).argmax(1)
    try:
        harness.train=collect
        for tag in ("v5","v6"):
            if tag=="v5":
                gas.fit_candidates,gas.predict=baseline.fit_candidates,baseline.predict
            else:
                gas.fit_candidates,gas.predict=origfit,origpredict
            full.full_round("public-pair-"+tag)
    finally:
        harness.train=orig_train;gas.fit_candidates,gas.predict=origfit,origpredict
    rows=[json.loads(r) for r in (data/"public/validation.jsonl").read_text().splitlines()]
    model=Model.load(data/"checkpoint.npz");_,y,_=harness.projection(rows,model,"A")
    names=model.meta["classes"];n=len(names);strata=[]
    for k in range(n):
        ids=np.flatnonzero(y==k)
        ids=np.array(sorted(ids,key=lambda j:rows[j]["meta"]["source_index"]))
        strata.append(np.array_split(ids,max(1,round(len(ids)/16))))
    def cm(idx,p):
        return np.bincount(n*y[idx]+p[idx],minlength=n*n).reshape(n,n)
    def f1(m):
        return (2*np.diagonal(m,axis1=-2,axis2=-1)/np.maximum(1,m.sum(-1)+m.sum(-2))).mean(-1)
    rng=np.random.default_rng(20261127);rep=4000
    mats=[np.zeros((rep,n,n)) for _ in range(2)]
    for blocks in strata:
        choices=rng.integers(len(blocks),size=(rep,len(blocks)))
        for pos,tag in enumerate(("v5","v6")):
            mats[pos]+=np.stack([cm(idx,predict[tag]) for idx in blocks])[choices].sum(1)
    totals={k:cm(np.arange(len(y)),v) for k,v in predict.items()}
    result={"rows":len(y),"classes":names,"changed":int(np.sum(predict["v5"]!=predict["v6"])),
            "blocks_per_class":[len(b) for b in strata],"paired_macro_f1_ci95":np.quantile(f1(mats[1])-f1(mats[0]),[.025,.975]).tolist(),
            "recalls":{k:(np.diag(v)/v.sum(1)).tolist() for k,v in totals.items()},
            "f1":{k:float(f1(v)) for k,v in totals.items()}}
    (OUT/"public-pair.json").write_text(json.dumps(result,indent=2))


def verify_deployment():
    from industrial import Model, dataset
    import os
    import time
    os.sched_setaffinity(0,{max(os.sched_getaffinity(0))})
    source=benchmark.WORK/"A";mdl=Model.load(source/"checkpoint.npz")
    parent=load("gas_previous",HERE.parent/"V5/gas.py")
    def rows(split):
        return dataset([json.loads(r) for r in (source/"public"/(split+".jsonl")).read_text().splitlines()],mdl)
    x,y=rows("train");v,_=rows("validation")
    stress=np.vstack((v,np.zeros(128),v[:4]*120,v[5:9]*.001,-v[10:14]))
    result={}
    for size in (4,9,16,10000):
        ids=np.sort(np.concatenate([np.flatnonzero(y==k)[:size] for k in np.unique(y)]))
        prev=parent.fit_candidates(mdl,x[ids],y[ids],1)
        curr=gas.fit_candidates(mdl,x[ids],y[ids],1)
        for family in curr:
            old=prev[family];new=curr[family]
            stock=parent.predict(mdl,old,stress)
            raw=gas.predict(mdl,new,stress)
            fast=gas.prepare_inference(mdl,new)
            one=np.stack([fast(row) for row in stress])
            expect=stock.copy();weight=float(new.get("backbone_shrink",0.))
            if family=="independent":
                expected_nodes=physical_landmarks(prototype.response_shape(x[ids]),gas._response_amplitude(x[ids]),y[ids],old,"dose")
                np.testing.assert_allclose(new["phys_nodes"],expected_nodes["phys_nodes"],atol=8e-5,rtol=8e-5)
                expect+=prototype.physical_score(stress,new)-prototype.physical_score(stress,old)
            if weight:
                base=np.log(np.maximum(mdl.predict(stress),1e-12))
                expect-=weight*(base-base.mean(1,keepdims=True))
            np.testing.assert_allclose(raw,expect,rtol=1e-12,atol=1e-12)
            np.testing.assert_allclose(raw,one,rtol=1e-9,atol=1e-9)
            np.testing.assert_array_equal(stock,gas.predict(mdl,old,stress))
            calls=[parent.prepare_inference(mdl,old),fast];times=[[],[]]
            for r in range(7):
                for m in ([0,1] if r%2==0 else [1,0]):
                    t=time.process_time()
                    for row in v[:350]:
                        calls[m](row)
                    times[m].append(1e6*(time.process_time()-t)/350)
            result[f"{len(ids)}-{family}"]={"n":len(ids),"nmin":int(np.bincount(y[ids]).min()),"n_eval":len(stress),
                 "shrink":weight,"batch_single_max":float(np.abs(raw-one).max()),"prediction_disagreements":int(np.sum(raw.argmax(1)!=one.argmax(1))),
                 "v5_v6_changed_labels":int(np.sum(raw.argmax(1)!=stock.argmax(1))),"bytes_added":sum(v.nbytes for k,v in new.items() if k not in old),
                 "parent_us":float(np.median(times[0])),"frozen_us":float(np.median(times[1]))}
    (OUT/"deployment-final.json").write_text(json.dumps(result,indent=2))
    official=load("official_gas",HERE.parent/"V4/gas_v4_study.py");official.OUT=OUT
    official.full_round("final")


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("route",choices=ROUTES)
    parser.add_argument("part",choices=("screen","check","external","verify","full","pair"))
    parser.add_argument("settings",nargs="*",default=["trace","schedule","fresh","official"])
    args=parser.parse_args()
    OUT.mkdir(exist_ok=True,parents=True)
    if args.part=="pair":
        public_pair()
    elif args.part=="full":
        patch(args.route)
        full=load("gas_round",HERE.parent/"V4/gas_v4_study.py");full.OUT=OUT;full.full_round(args.route)
    elif args.part=="verify":
        verify_deployment()
    elif args.part=="external":
        external(args.route)
    else:
        replay(args.route,args.part,args.settings)
