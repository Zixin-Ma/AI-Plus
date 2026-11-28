"""Paired gas head experiments. Reuse the V6 CPU benchmark and native-device holdouts."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
import numpy as np

HERE=Path(__file__).resolve().parent
V6=HERE.parent/"V6"
OUT=HERE.parent/"work/v7/gas"
sys.path.insert(0,str(HERE))
import gas
import prototype
import contest_submission as harness

def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    obj=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obj)
    return obj

bench=module("v6bench_gas",V6/"benchmark.py")
reference=module("v6proto",V6/"prototype.py")
oldgas=module("v6gas",V6/"gas.py")
oldgas.prototype=reference

def contrast_curve(features,raw,amplitude,y,classes,raw_state=None,route="blend"):
    result=reference.fit_curve(features,raw,amplitude,y,classes,raw_state)
    z=reference._normalize(features,result["curve_center"],result["curve_scale"])
    q=np.clip((amplitude-result["amp_center"])/result["amp_scale"],-3.,3.)
    design=np.column_stack((np.ones(len(q)),q,q*q))
    co=[]; errors=[]; covs=[]
    for k in range(classes):
        ids=y==k; a=design[ids]
        c=np.linalg.solve(a.T@a+np.diag([.01,1.,2.]),a.T@z[ids])
        co.append(c); err=z[ids]-a@c; errors.append(err);covs.append(err.T@err/max(1,len(err)-1))
    co=np.stack(co)
    ev,basis=np.linalg.eigh(np.mean(covs,axis=0))
    fixed=result["curve_proj"]
    # Contrast class curves only within each class's supported amplitude range.
    queries=np.quantile(q,[.2,.5,.8])
    planted=[]
    for k in range(classes):
        t=np.clip(queries,*np.quantile(q[y==k],[.1,.9]))
        planted.append(co[k,0]+t[:,None]*(co[k,1]+t[:,None]*co[k,2]))
    contrast=np.stack(planted)
    contrast-=contrast.mean(axis=0,keepdims=True)
    S=basis/np.sqrt(np.maximum(ev,0)+.30)
    _,sing,axes=np.linalg.svd(contrast.reshape(-1,z.shape[1])@S,full_matrices=False)
    nr={"blend":6,"tight":3,"discrim":12,"weak":6,"anti":6}.get(route,6)
    nr=min(nr,np.count_nonzero(sing>1e-8),fixed.shape[1]-1)
    if nr<1:return result
    directions=S@axes[:nr].T
    scale={"weak":.35,"tight":.7,"anti":1.}.get(route,.7)
    if route=="anti":
        directions=basis[:,:nr]/np.sqrt(np.maximum(ev[:nr],0)+.3)
    proj=np.concatenate((fixed[:,nr:],scale*directions),axis=1)
    local=[];gain=[]
    for e in errors:
        local_z=e@proj
        v,u=np.linalg.eigh(local_z.T@local_z/max(1,len(e)-1))
        local.append(u[:,-4:].T);gain.append(v[-4:]/(1+v[-4:]))
    result.update(curve_proj=proj,curve_coeff=co@proj,curve_axes=np.stack(local),curve_gain=np.stack(gain))
    return result

def kernel_dose(features,raw,y,n,route,amplitude=None):
    h=reference.fit(features,raw,y,n)
    f=reference._normalize(features,h["kernel_center"],h["kernel_scale"])
    if amplitude is None:amplitude=features[:,::12].mean(1)
    nodes=[];weights=np.ones(len(y))
    for k in range(n):
        ids=np.flatnonzero(y==k)
        order=np.argsort(amplitude[ids],kind="stable")
        for b in np.array_split(ids[order],min(16,max(1,len(ids)//4))):
            nodes.append(f[b].mean(0))
    nodes=np.stack(nodes)
    norms=np.mean(nodes**2,axis=1);design=reference._kernel(f,nodes,norms)
    base=1/np.maximum(1.,np.bincount(y,minlength=n)[y]/40)
    reg=.001*len(y)*reference._kernel(nodes,nodes,norms)+.0001*np.eye(len(nodes))
    onehot=np.eye(n)[y]
    for step in range(2 if route=="hardnode" else 1):
        wt=weights*base
        coef=np.linalg.solve((design.T*wt)@design+reg,(design.T*wt)@onehot)
        err=np.linalg.norm(design@coef-onehot,axis=1)
        weights=np.clip(err/max(np.mean(err),1e-5),.5,2.)
    h.update(kernel_landmarks=nodes,kernel_norm=norms,kernel_coef=coef)
    return h

def phys_metric(z,y,n,groups=None,amplitude=None,route="reliable"):
    h=reference.fit_physical(z,y,n,groups,amplitude)
    f=reference.invariant_features(z,h["phys_loc"],h["phys_scale"])
    f=(f-h["phys_mid"])/h["phys_std"]
    means=np.stack([f[y==k].mean(0) for k in range(n)])
    within=np.mean([f[y==k].var(0) for k in range(n)],0)
    rel=np.clip((means.var(0)+.1)/(within+.1),.10,10.)**.25
    if route=="shapeonly":rel[-z.shape[-1]:]=.25
    rel/=np.sqrt(np.mean(rel**2))
    std=h["phys_std"]/rel
    f=np.clip((reference.invariant_features(z,h["phys_loc"],h["phys_scale"])-h["phys_mid"])/std,-8,8)
    nodes=[]
    for k in range(n):
        ids=np.flatnonzero(y==k)
        order=amplitude if amplitude is not None else groups
        if order is not None:ids=ids[np.argsort(order[ids],kind="stable")]
        nodes.extend(f[b].mean(0) for b in np.array_split(ids,min(6,max(1,len(ids)//3))))
    nodes=np.stack(nodes);norms=np.mean(nodes**2,1);width=max(np.mean((f-f.mean(0))**2),1e-5)
    def kernel(x):
        d=(x**2).mean(1)[:,None]+norms-2*x@nodes.T/f.shape[1]
        return np.exp(-.85*np.maximum(0,d)/width)
    D=kernel(f);wt=len(y)/(n*np.bincount(y,minlength=n)[y])
    coef=np.linalg.solve((D.T*wt)@D+.003*len(y)*kernel(nodes)+.001*np.eye(len(nodes)),(D.T*wt)@np.eye(n)[y])
    h.update(phys_std=std,phys_width=np.asarray(width),phys_nodes=nodes,phys_norm=norms,phys_coef=coef)
    return h

def counterfactual(x,y,route):
    cube=x.reshape(-1,16,8)
    steady=np.maximum(abs(cube[:,:,0]),1.)
    ratio=np.maximum(abs(cube[:,:,1]-1.),.10)
    baseline=np.log(steady/ratio)
    # Match gas and approximate dose, then extract correlated shifts across sensors.
    dose=np.log(steady).mean(1)
    delta=np.zeros_like(baseline)
    for c in np.unique(y):
        ids=np.flatnonzero(y==c)
        ids=ids[np.argsort(dose[ids],kind="stable")]
        for bin_ids in np.array_split(ids,max(1,min(5,len(ids)//8))):
            delta[bin_ids]=baseline[bin_ids]-np.median(baseline[bin_ids],axis=0)
    delta-=np.median(delta,axis=0)
    rng=np.random.default_rng(8497)
    picked=delta[rng.permutation(len(y))]
    amplitude=.75 if route in ("gain","offset","both","replace") else 1.5
    shift=np.exp(np.clip(amplitude*picked,-.7,.7))
    changed=cube.copy()
    if route in ("gain","heavy","both","replace"):
        changed[:,:,0]*=shift
        changed[:,:,2:]*=shift[:,:,None]
    if route in ("offset","both","offset2"):
        changed[:,:,1]=1.+(changed[:,:,1]-1.)/shift
    if route=="replace":
        changed[::2]=cube[::2]
        return changed.reshape(x.shape),y
    return np.concatenate((x,changed.reshape(x.shape))),np.tile(y,2)

def augmentation_fit(model,x,y,index,sub):
    old=oldgas.fit_candidates(model,x,y,index)
    xx,yy=counterfactual(x,y,sub)
    expanded=oldgas.fit_candidates(model,xx,yy,index)
    if "independent" in expanded and "independent" in old:
        old["independent"]=expanded["independent"]
    return old

def raw_tangents(features,raw,y,n,route):
    h=(kernel_dose if route.startswith("node") else None)
    h=kernel_dose(features,raw,y,n,"node") if h else reference.fit(features,raw,y,n)
    z=reference._normalize(raw,h["sub_center"],h["sub_scale"])
    dose=features[:,::12].mean(1)
    for k in range(n):
        ids=np.flatnonzero(y==k)
        sorted_ids=ids[np.argsort(dose[ids],kind="stable")]
        chunks=np.array_split(sorted_ids,min(5,len(ids)))
        groups=np.stack([np.median(z[idx],axis=0) for idx in chunks])
        center=np.mean(groups,axis=0)
        e=z[ids]-center
        norm=np.sqrt(np.mean(e*e,1));cap=np.median(norm)+np.median(abs(norm-np.median(norm)))
        e*=np.minimum(1.,cap/np.maximum(norm,1.e-6))[:,None]
        if "doseaxis" in route:
            chunk=(groups[-1]-groups[0]);chunk/=max(1e-6,np.linalg.norm(chunk))
            e-=np.outer(e@chunk,chunk)
            ax=np.vstack((chunk,np.linalg.svd(e,full_matrices=False)[2][:2]))
        elif "lowrank" in route:
            ax=np.linalg.svd(e,full_matrices=False)[2][:1]
        else:
            ax=np.linalg.svd(e,full_matrices=False)[2][:3]
        h["sub_means"][k]=center
        h["sub_axes"][k]=0.
        h["sub_axes"][k,:len(ax)]=ax
    return h

def discriminant_route(f,y,n,rank,mix,ridge,route):
    original=oldgas._discriminant
    if route=="coupled":
        mix=min(mix,.4)
        ridge=max(ridge,.04)
    elif route=="adapt":
        smallest=np.bincount(y,minlength=n).min()
        ridge=max(ridge,min(.25,rank/max(1,4*smallest)))
        mix=mix*smallest/(smallest+rank/2)
    elif route.startswith("ridge"):
        ridge=float(route[5:])/100
    return original(f,y,n,rank,mix,ridge)

def kinetic_signature(x,kind):
    sig=np.log(np.maximum(abs(x),.02)).reshape(len(x),16,8)
    # Rising and decaying transients are measured at the same three EMA rates.
    pair=sig[:,:,2:5]-sig[:,:,5:8]
    if kind=="scale":
        return np.concatenate((np.diff(sig[:,:,2:5],axis=2),np.diff(sig[:,:,5:8],axis=2),pair),axis=2).reshape(len(x),-1)
    if kind=="relative":pair-=pair.mean(1,keepdims=True)
    return pair.reshape(len(x),-1)

def kinetic_readout(x,y,classes,kind):
    features=kinetic_signature(x,kind)
    counts=np.bincount(y,minlength=classes)
    w=len(y)/(classes*counts[y])
    mid=np.average(features,axis=0,weights=w)
    z=features-mid
    sigma=np.sqrt(np.maximum(np.average(z*z,axis=0,weights=w),.025**2))
    z=z/sigma
    one=np.eye(classes)[y]-1/classes
    wt=z.T*w
    reg=.12 if kind=="scale" else .08
    coef=np.linalg.solve(wt@z+np.eye(z.shape[1])*(reg*len(y)),wt@one)/sigma[:,None]
    basis=coef.reshape(16,-1,classes)
    full=np.zeros((16,8,classes)); pair=basis[:,-3:]
    if kind=="relative":
        pair=pair-pair.mean(0)
    full[:,2:5]+=pair;full[:,5:8]-=pair
    if kind=="scale":
        full[:,2:4]-=basis[:,:2];full[:,3:5]+=basis[:,:2]
        full[:,5:7]-=basis[:,2:4];full[:,6:8]+=basis[:,2:4]
    return {"kin_w":full.reshape(128,classes).astype(np.float32),"kin_b":(-mid@coef).astype(np.float32)}

def add_vote(raw,state):
    v=raw@state["kin_w"]+state["kin_b"]
    return float(state["kin_strength"])*np.clip(v-v.max(axis=-1,keepdims=True),-.85,0.)

def install_kinetic(route):
    prev_fit=gas.fit_candidates;prev_pred=gas.predict
    def fit(m,x,y,i):
        heads=prev_fit(m,x,y,i)
        if "tail" not in heads:return heads
        state=kinetic_readout(x,y,len(m.meta["classes"]),"scale" if "scale" in route else "relative" if "relative" in route else "pair")
        raw=float(route[-1]) if route[-1].isdigit() else 4.
        state["kin_strength"]=np.asarray(raw,dtype=np.float32)
        for family in ("tail","independent"):
            if ("tail" in route and family!="tail") or ("ind" in route and family!="independent"):continue
            heads[family].update(state)
        return heads
    gas.fit_candidates=fit
    def pred(m,h,x):
        base=prev_pred(m,h,x)
        if "kin_w" in h:
            base+=add_vote(np.log(np.maximum(abs(x),.02)),h)
        return base
    gas.predict=pred

def pooled_curve(features,raw,amp,y,n,raw_state=None,route="pool8"):
    h=reference.fit_curve(features,raw,amp,y,n,raw_state)
    z=reference._normalize(features,h["curve_center"],h["curve_scale"])
    q=np.clip((amp-h["amp_center"])/h["amp_scale"],-3.,3.)
    qt=np.empty_like(q);zt=np.empty_like(z)
    for k in range(n):
        ids=y==k;qt[ids]=q[ids]-q[ids].mean();zt[ids]=z[ids]-z[ids].mean(0)
    slope=qt@zt/(qt@qt+1.)
    strength=float(route.split("pool")[-1])
    smallest=np.bincount(y,minlength=n).min()
    coef=[]; errors=[];cov=[]
    for k in range(n):
        ids=y==k
        center=q[ids].mean();t=q[ids]-center
        basis=np.column_stack((np.ones(len(t)),t,t*t))
        penalty=np.diag([.01,1.+strength,2.+strength])
        prior=np.zeros((3,z.shape[1]));prior[1]=slope
        c=np.linalg.solve(basis.T@basis+penalty,basis.T@z[ids]+penalty@prior)
        global_basis=np.column_stack((np.ones(len(t)),q[ids],q[ids]**2))
        adjusted=np.stack((c[0]-center*c[1]+center*center*c[2],c[1]-2*center*c[2],c[2]))
        if route.startswith(("soft","local","precision")):
            previous=np.linalg.solve(global_basis.T@global_basis+np.diag([.01,1.,2.]),global_basis.T@z[ids])
            if route.startswith("soft"):
                blend=np.clip((24-smallest)/18,0,1)
            elif route.startswith("local"):
                blend=min(1.,(12./len(t))**2)
            else:
                blend=4./(4.+t@t)
            adjusted=previous+blend*(adjusted-previous)
        e=z[ids]-global_basis@adjusted
        errors.append(e);cov.append(e.T@e/max(1,len(e)-1))
        coef.append(adjusted)
    ev,basis=np.linalg.eigh(np.mean(cov,axis=0))
    proj=basis[:,-24:]/np.sqrt(ev[-24:]+.12)
    axes=[];gains=[]
    for e in errors:
        u=e@proj
        value,ax=np.linalg.eigh(u.T@u/max(1,len(u)-1))
        axes.append(ax[:,-4:].T);gains.append(value[-4:]/(1+value[-4:]))
    h.update(curve_proj=proj,curve_coeff=np.stack(coef)@proj,curve_axes=np.stack(axes),curve_gain=np.stack(gains))
    return h

def split_curve(model,x,y,index,prior_fit,route):
    heads=prior_fit(model,x,y,index)
    if "tail" not in heads:
        return heads
    n=len(model.meta["classes"])
    smallest=np.bincount(y,minlength=n).min()
    if route.startswith("small") and smallest>=16:return heads
    f,raw,q=gas.curve_inputs(x)
    h=pooled_curve(f,raw,q,y,n,heads["independent"],route)
    heads["tail"].update({k:a.astype(np.float32) for k,a in h.items() if k.startswith("curve_")})
    return heads

def install(route):
    if route=="production":return
    for key in ["fit","fit_curve","fit_physical","physical_vote"]:
        setattr(prototype,key,getattr(reference,key))
    live=module("v6study_gas",V6/"gas.py")
    live.prototype=prototype
    gas.fit_candidates=oldgas.fit_candidates if route=="base" else live.fit_candidates
    gas.predict=oldgas.predict if route=="base" else live.predict
    if route not in ("base","production"):
        for sub in route.split("+"):
            if sub in ("blend","weak","tight","discrim","anti"):
                prototype.fit_curve=lambda *a,sub=sub,**kw:contrast_curve(*a,route=sub,**kw)
            elif sub in ("node","hardnode"):
                prototype.fit=lambda *a,sub=sub,**kw:kernel_dose(*a,route=sub,**kw)
            elif sub in ("reliable","shapeonly"):
                prototype.fit_physical=lambda *a,sub=sub,**kw:phys_metric(*a,route=sub,**kw)
            elif sub in ("rawrobust","doseaxis","lowrank","nodedoseaxis","noderawrobust"):
                prototype.fit=lambda *a,sub=sub,**kw:raw_tangents(*a,route=sub,**kw)
            elif sub in ("coupled","adapt") or sub.startswith("ridge"):
                live._discriminant=lambda *a,sub=sub,**kw:discriminant_route(*a,route=sub,**kw)
            elif sub.startswith(("tailpool","smallpool","softpool","localpool","precisionpool")):
                old_fit=gas.fit_candidates
                gas.fit_candidates=lambda m,x,y,i,of=old_fit,sub=sub:split_curve(m,x,y,i,of,sub)
            elif sub.startswith("pool"):
                prototype.fit_curve=lambda *a,sub=sub,**kw:pooled_curve(*a,route=sub,**kw)
            elif sub.startswith("kin"):
                install_kinetic(sub)
            elif sub in ("gain","offset","both","heavy","replace","offset2"):
                gas.fit_candidates=lambda m,x,y,i,sub=sub:augmentation_fit(m,x,y,i,sub)
            else:raise ValueError(sub)

def external(route):
    """Apply the same training rule after defining each external instrument's native views."""
    sys.modules["benchmark"]=bench
    src=module("native_legacy",HERE.parent/"V5/gas_v5_study.py")
    src.OUT=HERE.parent/"work/v5/gas"
    from industrial import Model
    def comparison(args,train,test,unused):
        x,f,relative,raw,q,physical,classes,y,dates=args
        n=len(classes)
        model=Model("classification",[str(i) for i in range(x.shape[1])],list(classes),x[train],y[train])
        model.fit(x[train],y[train],20)
        model.meta["learning_rate"]=.001
        model.fit(x[train],y[train],1)
        st={}
        for name,fmat,rank,cmix in [("absolute_",f,48,0.),("relative_",relative,24,.8)]:
            st.update({name+k:v for k,v in oldgas._discriminant(fmat[train],y[train],n,rank,cmix,.005).items()})
        base,a,b=oldgas.components(model,st,x[test],(f[test],relative[test]))
        scores={}
        for tag,kind in [("v4","base"),("v5",route)]:
            parts=kind.split("+")
            kin=reference.fit(f[train],raw[train],y[train],n)
            if "node" in parts:
                kin=kernel_dose(f[train],raw[train],y[train],n,"node",amplitude=q[train])
            k,sub=reference.scores(f[test],raw[test],kin)
            c=reference.fit_curve(f[train],raw[train],q[train],y[train],n,kin)
            feed=(f[test],raw[test],q[test])
            independent=base+.4*(a+b)+.5*(k+sub)
            sensor=reference.score_curve(*feed,c)
            independent+=np.clip(sensor-sensor.max(1,keepdims=True),-4.,0.)
            if "shapeonly" in parts:
                ph=phys_metric(physical[train],y[train],n,amplitude=q[train],route="shapeonly")
            else:
                ph=reference.fit_physical(physical[train],y[train],n,amplitude=q[train])
            t=reference.invariant_features(physical[test],ph["phys_loc"],ph["phys_scale"])
            t=np.clip((t-ph["phys_mid"])/ph["phys_std"],-8.,8.)
            d=(t*t).mean(1)[:,None]+ph["phys_norm"]-2*(t@ph["phys_nodes"].T)/t.shape[1]
            votes=np.exp(-.85*np.maximum(0,d)/ph["phys_width"])@ph["phys_coef"]
            scores[tag+"_independent"]=independent+reference.physical_vote(votes)
            smallest=np.bincount(y[train],minlength=n).min()
            for variant in parts:
                if variant.startswith(("tailpool","smallpool","softpool","localpool","precisionpool","pool")) and (not variant.startswith("small") or smallest<16):
                    c=pooled_curve(f[train],raw[train],q[train],y[train],n,kin,route=variant)
            u=np.clip((q[train]-c["amp_center"])/c["amp_scale"],-3.,3.)
            c["amplitude_bounds"]=np.stack([np.quantile(u[y[train]==j],[.05,.95]) for j in range(n)])
            c["agreement_fraction"]=np.array(.3)
            shrink=.5*np.clip(4./np.sqrt(smallest)-1.,0.,1.)
            scores[tag+"_tail"]=base-shrink*(base-base.mean(1,keepdims=True))+reference.robust_score(*feed,c)
        return scores
    src.native=comparison
    reports={"one_year":src.forward(route),"dsps":src.dsps(route,False),"concentration":src.dsps(route,True),"home":src.home(route)}
    def relabel(node):
        if not isinstance(node,dict):return
        if "families" in node:
            for fld in ("inner_scores","head_f1"):
                node[fld]={("v6" if k[:2]=="v4" else "v7")+k[2:]:v for k,v in node[fld].items()}
            for s in node["families"].values():
                s["family"]=("v6" if s["family"][:2]=="v4" else "v7")+s["family"][2:]
            node["families"]={("v6" if k=="v4" else "v7"):v for k,v in node["families"].items()}
        else:
            for child in node.values():relabel(child)
    relabel(reports)
    (OUT/f"external-{route}.json").write_text(json.dumps(reports,indent=2))

def verify(route):
    """Cross-check final production states, optimizer-free prediction, and the V6 ABI."""
    import hashlib
    import time
    chosen="node+shapeonly+smallpool8"
    data=bench.WORK/"A"
    model=harness.Model.load(data/"checkpoint.npz")
    def get(split):
        return harness.projection([json.loads(i)for i in (data/"public"/(split+".jsonl")).read_text().splitlines()],model,"A")[:2]
    x,y=get("train");valid,_=get("validation")
    check=np.vstack((valid,np.zeros(128),valid[:4]*120.,valid[4:8]*.001,-valid[8:12]))
    result={}
    prod=module("live_gas7",HERE/"gas.py")
    prod.prototype=module("live_proto7",HERE/"prototype.py")
    for n in [4,9,16,999]:
        ids=np.concatenate([np.flatnonzero(y==k)[:n]for k in range(len(model.meta["classes"]))])
        install(chosen)
        want=gas.fit_candidates(model,x[ids],y[ids],1)
        got=prod.fit_candidates(model,x[ids],y[ids],1)
        old=oldgas.fit_candidates(model,x[ids],y[ids],1)
        for family,state in got.items():
            target=gas.predict(model,want[family],check)
            bat=prod.predict(model,state,check)
            single=prod.prepare_inference(model,state)
            hot=np.stack([single(row)for row in check])
            np.testing.assert_allclose(bat,target,rtol=1e-7,atol=1e-6)
            np.testing.assert_allclose(bat,hot,rtol=1e-9,atol=1e-9)
            old_hot=prod.prepare_inference(model,old[family])
            np.testing.assert_allclose([old_hot(row)for row in check],oldgas.predict(model,old[family],check),rtol=1e-9,atol=1e-9)
            funcs=[oldgas.prepare_inference(model,old[family]),single];times=[[],[]]
            for repeat in range(7):
                for j in ((0,1) if repeat%2==0 else (1,0)):
                    start=time.process_time()
                    for row in valid[:160]:funcs[j](row)
                    times[j].append((time.process_time()-start)*1e6/160)
            result[f"{n}-{family}"]=dict(n=len(ids),rows=len(check),pred_diff=float(np.max(abs(bat-target))),single_diff=float(np.max(abs(bat-hot))),
                 v6_us=float(np.median(times[0])),v7_us=float(np.median(times[1])),state_bytes=[sum(a.nbytes for a in ss.values()) for ss in [old[family],state]])
    (OUT/"deployment.json").write_text(json.dumps({"hash":{k:hashlib.sha256((HERE/(k+".py")).read_bytes()).hexdigest() for k in ("gas","prototype")},"cases":result},indent=2))


def summarize():
    """Summarize matched protocols from the pre-registered research decision."""
    record = OUT/"choice.json"
    # A clean checkout can replay the frozen decision without fabricating an
    # original timestamp. The archived record is preferred on the study server.
    choice = json.loads(record.read_text()) if record.is_file() else {
        "route":"node+shapeonly+smallpool8",
        "check":[2,4,6,7,10,11,13,14,15,16,17,18,19],
        "check_primary":["schedule","trace","official","fresh","drift"],
        "origin":"published V7 gas protocol replay"
    }
    recipe = choice["route"]
    def read(rel):
        return json.loads((OUT/rel).read_text())
    def stream(recipe, protocol, order):
        return read(f"streams/{recipe}/{protocol}-v6-A-{order}.json")
    def panel(orders):
        result = {}
        for protocol in choice["check_primary"]:
            base = [stream("base", protocol, n) for n in orders]
            candidate = [stream(recipe, protocol, n) for n in orders]
            rows = {}
            for key in ("stage_mean","stage_floor","forget","delay"):
                left = np.array([r[key] for r in base])
                right = np.array([r[key] for r in candidate])
                delta = right-left
                rows[key] = {"base":float(left.mean()),"new":float(right.mean()),
                             "higher":int((delta>1e-9).sum()),"lower":int((delta< -1e-9).sum()),
                             "tied":int((abs(delta)<=1e-9).sum())}
            result[protocol] = rows
        return result
    matched = []
    prompt_bytes = {"experimental_driver":0, "production_driver":0}
    for path in sorted((OUT/"streams"/"production").glob("*.json")):
        prod = json.loads(path.read_text())
        candidate = read(f"streams/{recipe}/{path.name}")
        assert len(prod["rounds"]) == len(candidate["rounds"]) == 10
        for r,s in zip(prod["rounds"],candidate["rounds"]):
            prompt_bytes["experimental_driver"] += s["prompt_bytes"]
            prompt_bytes["production_driver"] += r["prompt_bytes"]
            for key in ("deployed","expert_weight","head_family","calls"):
                assert r[key] == s[key], (path,key)
            for key in ("gains","metrics","static"):
                np.testing.assert_allclose(r[key],s[key],atol=1e-8,rtol=1e-8)
        matched.append(path.name)
    result = {"choice":choice, "held_out":panel(choice["check"]), 
              "deployment":read("deployment.json"), "production_stream_parity":matched, "paired_driver_prompt_bytes":prompt_bytes,
              "opening":{}, "first_round":{}, "external":read(f"external-{recipe}.json")}
    for route in ("base","node","shapeonly","node+shapeonly","node+shapeonly+tailpool8",
                  "node+shapeonly+localpool8","node+shapeonly+kin4","production"):
        rows = read(f"opening/{route}/v6-A.json")
        gains = [r["gain"] for r in rows]
        result["opening"][route]={"gains":gains,"mean":float(np.mean(gains)),"floor":float(min(gains))}
        result["first_round"][route]=read(f"full-{route}.json")
    (OUT/"summary.json").write_text(json.dumps(result,indent=2)+"\n")

def main():
    pa=argparse.ArgumentParser()
    pa.add_argument("route");pa.add_argument("action",choices=("full","opening","stream","external","verify","summary"))
    pa.add_argument("params",nargs="*")
    a=pa.parse_args();OUT.mkdir(exist_ok=True,parents=True)
    install(a.route)
    if a.action=="full":
        sys.modules["benchmark"]=bench
        full=module("full_gas",HERE.parent/"V4/gas_v4_study.py");full.OUT=OUT;full.full_round(a.route)
    elif a.action=="opening":
        sys.modules["benchmark"]=bench
        m=module("opening_gas",V6/"opening_study.py");m.OUT=OUT/"opening"/a.route;m.run("A","v6")
    elif a.action=="stream":
        orders=[int(v) for v in a.params[0].split(",")]
        assert all(0<=v<20 for v in orders)
        bench.OUT=OUT/"streams"/a.route
        for setting in a.params[1:]:
            for n in orders:
                if not (bench.OUT/f"{setting}-v6-A-{n}.json").exists():bench.run("A","v6",n,setting)
    elif a.action=="external":
        external(a.route)
    elif a.action=="verify":
        verify(a.route)
    else:
        summarize()

if __name__=="__main__":main()
