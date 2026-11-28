"""Gas A paired development studies, all descriptors fitted on released training rows."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "work/v8/gas"
sys.path.insert(0, str(HERE))
import gas
import prototype
# Use one immutable controller for every head ablation, even after future V8 changes.
sys.path.insert(0, str(HERE.parent/"V7"))
import contest_submission as harness

def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    obj = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obj)
    return obj

bench = module(HERE.parent/"V7/benchmark.py", "gas_dev_benchmark")
sys.modules["benchmark"] = bench
old = module(HERE.parent/"V5/prototype.py", "gas_v5_reference")
gas5 = module(HERE.parent/"V5/gas.py", "gas_v5_head")
gas5.prototype = old

def geometry(features, raw, y, count, route):
    state = original_fit(features, raw, y, count)
    x = old._normalize(features, state["kernel_center"], state["kernel_scale"])
    z = old._normalize(raw, state["sub_center"], state["sub_scale"])
    labels = [np.flatnonzero(y==c) for c in range(count)]
    if "cluster" in route or "half" in route or "anchor" in route:
        nodes = []
        for ids in labels:
            g = x[ids]
            size = min(16,max(1,len(ids)//4))
            blocks = np.stack([b.mean(0) for b in np.array_split(g,size)])
            if "cluster" in route or "half" in route:
                # Initialize by robust within-class PC1 quantiles; then refine locally in full space.
                _,_,vt = np.linalg.svd(g-g.mean(0),full_matrices=False)
                by_pc = np.argsort((g-g.mean(0))@vt[0],kind="stable")
                k = size if "cluster" in route else (size+1)//2
                centers = np.stack([g[b].mean(0) for b in np.array_split(by_pc,k)])
                for _ in range(3):
                    which = ((centers*centers).sum(1)-2*g@centers.T).argmin(1)
                    centers = np.stack([g[which==i].mean(0) if np.any(which==i) else c
                                        for i,c in enumerate(centers)])
                blocks = centers if k==size else np.concatenate((blocks[1::2],centers))
            nodes.extend(blocks)
        nodes = np.stack(nodes)
        norm = (nodes*nodes).mean(1)
        k = old._kernel(x,nodes,norm)
        reg = .001*len(y)*old._kernel(nodes,nodes,norm) + .0001*np.eye(len(nodes))
        weights = 1/np.maximum(1.,np.bincount(y,minlength=count)[y]/40)
        if "anchor" in route:
            # Protect against shifts of local mean response within each chemical.
            for ids in labels:
                for seg in np.array_split(ids,min(6,max(1,len(ids)//4))):
                    delta = k[seg].mean(0)-k[ids].mean(0)
                    reg += len(seg)*weights[seg[0]]*np.outer(delta,delta)
        coef = np.linalg.solve((k.T*weights)@k+reg,(k.T*weights)@np.eye(count)[y])
        state.update(kernel_landmarks=nodes,kernel_norm=norm,kernel_coef=coef)
    if "shared" in route or "pool" in route or "protect" in route:
        group = [z[ids]-z[ids].mean(0) for ids in labels]
        covs = [g.T@g/max(len(g)-1,1) for g in group]
        shared = np.mean(covs,axis=0)
        if "protect" in route:
            centers = state["sub_means"]
            _,_,vt = np.linalg.svd(centers-centers.mean(0),full_matrices=False)
            proj = np.eye(z.shape[1])-vt[:count-1].T@vt[:count-1]
            shared = proj@shared@proj
        axes = []
        for c,g in enumerate(group):
            effective = (8 if "pool" in route else 24) / (len(g)+ (8 if "pool" in route else 24))
            cov = (1-effective)*covs[c] + effective*shared
            axes.append(np.linalg.eigh(cov)[1][:,-3:].T)
        state["sub_axes"]=np.stack(axes)
    return state

def physics(z,y,n,q,route):
    # Preserve the common-mode feature. Blend order-stratified and concentration-stratified
    # released templates using deterministic per-class quantile barycenters.
    geo=old.fit_physical(z,y,n)
    u=old.invariant_features(z,geo["phys_loc"],geo["phys_scale"])
    x=np.clip((u-geo["phys_mid"])/geo["phys_std"],-8.,8.)
    nodes=[]
    for cls in range(n):
        ids=np.flatnonzero(y==cls)
        size=min(6,max(1,len(ids)//3))
        order=ids[np.argsort(q[ids],kind="stable")]
        mean=lambda it:np.stack([x[b].mean(0) for b in np.array_split(it,size)])
        if "pblend" in route:
            source=mean(ids)
            # Sort the old nodes by their empirical mean response level before interpolation.
            source=source[np.argsort([q[b].mean() for b in np.array_split(ids,size)],kind="stable")]
            fraction=.25 if "pblend25" in route else .5
            nodes.extend((1-fraction)*source+fraction*mean(order))
        else:
            nodes.extend(mean(order))
    nodes=np.stack(nodes)
    norm=(nodes*nodes).mean(1)
    def kernel(rows):
        d=(rows*rows).mean(1)[:,None]+norm-2*(rows@nodes.T)/x.shape[1]
        return np.exp(-.85*np.maximum(0,d)/geo["phys_width"])
    k=kernel(x)
    weight=len(y)/(n*np.bincount(y,minlength=n)[y])
    reg=.003*len(y)*kernel(nodes)+.001*np.eye(len(nodes))
    coef=np.linalg.solve((k.T*weight)@k+reg,(k.T*weight)@np.eye(n)[y])
    geo.update(phys_nodes=nodes,phys_norm=norm,phys_coef=coef)
    return geo

def install(route):
    if route=="production":
        return
    if route=="v7":
        other = module(HERE.parent/"V7/gas.py","gas_v7_head")
        other.prototype = module(HERE.parent/"V7/prototype.py","gas_v7_reference")
        gas.fit_candidates,gas.predict,gas.prepare_inference=other.fit_candidates,other.predict,other.prepare_inference
        return
    if "sparse" in route or "temper" in route:
        other = module(HERE.parent/"V7/gas.py","gas_sparse_head")
        ref = module(HERE.parent/"V7/prototype.py","gas_sparse_proto")
        ref.fit = lambda f,r,y,n:geometry(f,r,y,n,route)
        ref.fit_physical = lambda z,y,n,amplitude=None:physics(z,y,n,amplitude,route) if "pblend" in route or "pdose" in route else old.fit_physical(z,y,n)
        if "temper" in route:
            ref.fit_curve = lambda f,r,q,y,n,raw_state=None,pooled_tail=False:old.fit_curve(f,r,q,y,n,raw_state)
        other.prototype=ref
        gas.fit_candidates,gas.predict,gas.prepare_inference=other.fit_candidates,other.predict,other.prepare_inference
        return
    gas.fit_candidates,gas.predict,gas.prepare_inference=gas5.fit_candidates,gas5.predict,gas5.prepare_inference
    if route!="v5":
        old.fit = lambda f,r,y,n:geometry(f,r,y,n,route)

def external(route):
    src=module(HERE.parent/"V5/gas_v5_study.py","gas_native")
    # fit one model per instrument and released window; both heads use the same backbone.
    def native(args, train, test, unused):
        x,f,relative,raw,q,physical,classes,y,dates=args
        n=len(classes)
        model=harness.Model("classification",[str(i) for i in range(x.shape[1])],list(classes),x[train],y[train])
        model.fit(x[train],y[train],20)
        model.meta["learning_rate"]=.001
        model.fit(x[train],y[train],1)
        st={}
        for name,fmat,rank,mix in [("absolute_",f,48,0.),("relative_",relative,24,.8)]:
            st.update({name+k:v for k,v in gas5._discriminant(fmat[train],y[train],n,rank,mix,.005).items()})
        b,a,r=gas5.components(model,st,x[test],(f[test],relative[test]))
        original=original_fit(f[train],raw[train],y[train],n)
        cand=prototype.fit(f[train],raw[train],y[train],n) if route=="production" else geometry(f[train],raw[train],y[train],n,route)
        scores={}
        def physical_vote(geom):
            t=old.invariant_features(physical[test],geom["phys_loc"],geom["phys_scale"])
            t=np.clip((t-geom["phys_mid"])/geom["phys_std"],-8.,8.)
            d=(t*t).mean(1)[:,None]+geom["phys_norm"]-2*(t@geom["phys_nodes"].T)/t.shape[1]
            return old.physical_vote(np.exp(-.85*np.maximum(0,d)/geom["phys_width"])@geom["phys_coef"])
        feed=f[test],raw[test],q[test]
        for nm,h in [("v4",original),("v5",cand)]:
            geom=physics(physical[train],y[train],n,q[train],route) if nm=="v5" and ("pblend" in route or "pdose" in route) else old.fit_physical(physical[train],y[train],n)
            phys=physical_vote(geom)
            k,sub=old.scores(f[test],raw[test],h)
            c=old.fit_curve(f[train],raw[train],q[train],y[train],n,h)
            sensor=old.score_curve(*feed,c)
            scores[nm+"_independent"]=b+.4*(a+r)+.5*(k+sub)+np.clip(sensor-sensor.max(1,keepdims=True),-4,0)+phys
            low=np.bincount(y[train],minlength=n).min()
            log_base=b
            if nm=="v5" and low<16 and ("sparse" in route or "temper" in route or route=="production"):
                if "sparse" in route:
                    ref=module(HERE.parent/"V7/prototype.py","sparse_external_reference")
                    c=ref.fit_curve(f[train],raw[train],q[train],y[train],n,h,pooled_tail=True)
                shrink=.5*np.clip(4./np.sqrt(low)-1.,0.,1.)
                log_base=b-shrink*(b-b.mean(1,keepdims=True))
            u=np.clip((q[train]-c["amp_center"])/c["amp_scale"],-3,3)
            c["amplitude_bounds"]=np.stack([np.quantile(u[y[train]==j],[.05,.95]) for j in range(n)])
            c["agreement_fraction"]=np.array(.30)
            scores[nm+"_tail"]=log_base+old.robust_score(*feed,c)
        return scores
    src.native=native
    results={"year":src.forward(route),"dsps":src.dsps(route,False),"dose":src.dsps(route,True),"home":src.home(route)}
    (OUT/f"external-{route}.json").write_text(json.dumps(results,indent=2))

original_fit=old.fit


def stress(route):
    """Unseen sensor gains are applied only to the public validation waveform summaries."""
    import tempfile
    from types import SimpleNamespace
    src = bench.WORK/"A"
    def load(name):
        return [json.loads(s) for s in (src/"public"/(name+".jsonl")).read_text().splitlines()]
    train,valid=load("train"),load("validation")
    base=harness.Model.load(src/"checkpoint.npz")
    x,y,_=harness.projection(valid,base,"A")
    audit=module(HERE.parent/"V7/opening_study.py","group_ids")
    group=audit.domain_ids(valid,"A")
    result={}
    with tempfile.TemporaryDirectory(dir=OUT) as tmp:
        r=bench.PublicRound();r.agent,r.dataset_key,r.round_index=harness,"A",1
        r.parent,r.records,r.call_ids=bench.Parent(src),[],[]
        r.artifacts=[];r.prompt_bytes=0;r.plan="refit|half|test sensor evidence"
        r.data=SimpleNamespace(training_rows=lambda:iter(train),validation_rows=lambda:iter(valid))
        r.selection=x,y
        r.root=Path(tmp);os.environ["CONTEST_WORK"]=tmp
        harness.train(r)
        model,h,_=harness.artifact(r.candidate)
        for mode in ("clean","channel_low","channel_high","pair_low","pair_high","common_down","common_up"):
            bank=np.arange(16)//2
            if mode.startswith("channel"):
                vec=np.cos(2*np.pi*bank/8)
            elif mode.startswith("pair"):
                vec=(-1.)**np.arange(16)
            else:
                vec=np.ones(16)
            # column 1 is a relative steady-state measure; the other summaries carry gain.
            sign=-1 if mode.endswith(("low","down")) else 1
            factor=np.exp(sign*.30*vec) if mode!="clean" else np.ones(16)
            mult=np.repeat(factor[:,None],8,axis=1);mult[:,1]=1.
            obs=x*mult.ravel()
            pred=harness.values("A",model,h,obs); cls=pred.argmax(1)
            result[mode]=dict(f1=harness.metric("A",y,pred),accuracy=float(np.mean(y==cls)),
                           group_floor=min(harness.metric("A",y[group==i],pred[group==i]) for i in np.unique(group)))
    (OUT/f"stress-{route}.json").write_text(json.dumps(result,indent=2))


def verify():
    from hashlib import sha256
    import time
    data=bench.WORK/"A"
    model=harness.Model.load(data/"checkpoint.npz")
    def get(split):
        return harness.projection([json.loads(i) for i in (data/"public"/(split+".jsonl")).read_text().splitlines()],model,"A")[:2]
    x,y=get("train");valid,_=get("validation")
    check=np.vstack((valid,np.zeros(128),valid[:4]*120.,valid[4:8]*.001,-valid[8:12]))
    prod=module(HERE/"gas.py","verified_production")
    prod.prototype=module(HERE/"prototype.py","verified_prototypes")
    selected="cluster+anchor+temper"
    install(selected)
    cases={}
    cpu=max(os.sched_getaffinity(0));os.sched_setaffinity(0,{cpu})
    for count in [2,5,15,16,999]:
        ids=np.concatenate([np.flatnonzero(y==k)[:count] for k in range(len(model.meta["classes"]))])
        expected=gas.fit_candidates(model,x[ids],y[ids],1)
        got=prod.fit_candidates(model,x[ids],y[ids],1)
        previous=gas5.fit_candidates(model,x[ids],y[ids],1)
        for family,h in got.items():
            actual=prod.predict(model,h,check)
            want=gas.predict(model,expected[family],check)
            batch=gas5.predict(model,previous[family],check)
            old_hot=prod.prepare_inference(model,previous[family])
            hot=prod.prepare_inference(model,h)
            single=np.stack([hot(row) for row in check])
            np.testing.assert_allclose(want,actual,rtol=1e-7,atol=1e-6)
            np.testing.assert_allclose(single,actual,rtol=1e-9,atol=1e-9)
            np.testing.assert_allclose([old_hot(row) for row in check],batch,rtol=1e-9,atol=1e-9)
            funcs=[gas5.prepare_inference(model,previous[family]),hot]
            times=[[],[]]
            for rep in range(7):
                for j in ((0,1) if rep%2==0 else (1,0)):
                    start=time.process_time()
                    for row in valid[:160]: funcs[j](row)
                    times[j].append(1e6*(time.process_time()-start)/160)
            cases[f"{count}-{family}"]=dict(rows=len(check),train=len(ids),experimental_diff=float(abs(want-actual).max()),
                hot_diff=float(abs(single-actual).max()),bytes=[sum(v.nbytes for v in st.values()) for st in (previous[family],h)],
                median_us=[float(np.median(a)) for a in times])
    (OUT/"deployment.json").write_text(json.dumps(dict(cases=cases,cpu=cpu,sha256={n:sha256((HERE/n).read_bytes()).hexdigest() for n in ("gas.py","prototype.py")}),indent=2))


def fit_cost():
    import time
    data=bench.WORK/"A"
    model=harness.Model.load(data/"checkpoint.npz")
    rows=[json.loads(row) for row in (data/"public/train.jsonl").read_text().splitlines()]
    x,y,_=harness.projection(rows,model,"A")
    variants=[gas5,module(HERE.parent/"V7/gas.py","cost_v7"),module(HERE/"gas.py","cost_v8")]
    variants[1].prototype=module(HERE.parent/"V7/prototype.py","cost_v7_proto")
    variants[2].prototype=module(HERE/"prototype.py","cost_v8_proto")
    cpu=max(os.sched_getaffinity(0));os.sched_setaffinity(0,{cpu})
    times=[[],[],[]]
    for h in variants:h.fit_candidates(model,x,y,1)
    for rep in range(12):
        for step in range(3):
            j=(rep+step)%3
            tic=time.process_time()
            variants[j].fit_candidates(model,x,y,1)
            times[j].append((time.process_time()-tic)*1000)
    (OUT/"fit-cost.json").write_text(json.dumps(dict(cpu=cpu,training_rows=len(y),ms=dict(zip(("v5","v7","v8"),times))),indent=2))

def main():
    pa=argparse.ArgumentParser()
    pa.add_argument("route")
    pa.add_argument("action",choices=("full","opening","stream","external","stress","verify","fitcost"))
    pa.add_argument("settings",nargs="*")
    args=pa.parse_args()
    OUT.mkdir(exist_ok=True,parents=True)
    name = args.route
    install(name)
    if args.action=="fitcost":
        fit_cost()
    elif args.action=="verify":
        verify()
    elif args.action=="full":
        m=module(HERE.parent/"V4/gas_v4_study.py","gas_first_round")
        m.OUT=OUT
        m.full_round(name)
    elif args.action=="opening":
        m=module(HERE.parent/"V7/opening_study.py","gas_opening")
        m.OUT=OUT/"opening"/name
        m.run("A","v8")
    elif args.action=="stream":
        bench.OUT=OUT/"streams"/name
        for case in args.settings[1:]:
            for split in map(int,args.settings[0].split(",")):
                assert 0<=split<20
                bench.run("A","v8",split,case)
    elif args.action=='stress':
        stress(name)
    else:
        external(name)

if __name__=="__main__":
    main()
