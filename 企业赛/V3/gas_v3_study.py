"""Gas V3 experiments on released development schedules and native external instruments."""
import argparse
import importlib.util
import io
import json
from pathlib import Path
import time
import numpy as np
import benchmark
import gas
import prototype
from baseline_model import Model
from industrial import dataset, metric

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
OUT=HERE.parent/"work/v3/gas"
old_spec=importlib.util.spec_from_file_location("gas_v2_reference",HERE.parent/"V2/gas.py")
old_gas=importlib.util.module_from_spec(old_spec);old_spec.loader.exec_module(old_gas)
BASE_FIT=old_gas.fit_candidates
BASE_PRED=old_gas.predict


def views(x):
    # All eight features of the same sensor experience the same hardware path.
    log=np.log(np.maximum(abs(x.reshape(len(x),16,8)),.025))
    channels=log.reshape(len(x),8,2,8).mean(2)
    trans=channels[:,:,2:]-channels[:,:,:1]
    shape=np.concatenate((channels[:,:,0:1]-channels[:,:,0:1].mean(1,keepdims=True),
              channels[:,:,1:2],trans-trans.mean(2,keepdims=True),trans.mean(2,keepdims=True)),axis=2)
    return shape.reshape(len(x),-1),np.log(np.maximum(abs(x),.025)),channels[:,:,0].mean(1)


def geometry(f,q,y,nclass,kind):
    # Fit class trajectories with a global slope prior and independent residual covariance.
    mu=f.mean(0);unit=np.maximum(f.std(0),.04);z=np.clip((f-mu)/unit,-8.,8.)
    if kind=="sphere":
        z=z/np.maximum(np.sqrt(np.mean(z*z,axis=1,keepdims=True)),.08)
    center=q.mean();scale=max(q.std(),.1);t=np.clip((q-center)/scale,-3.,3.)
    X=np.stack((np.ones(len(y)),t),axis=1)
    xc=X.copy();zc=z.copy()
    for i in range(nclass):
        mask=y==i
        xc[mask]-=X[mask].mean(0);zc[mask]-=z[mask].mean(0)
    pooled=np.linalg.solve(xc.T@xc+np.eye(2),xc.T@zc)
    means=[];res=[];cov=[]
    for i in range(nclass):
        g=y==i;reg=np.diag([.0001,12.])
        beta=np.linalg.solve(X[g].T@X[g]+reg,X[g].T@z[g]+reg@pooled)
        if kind=="sphere":
            beta[0]=z[g].mean(0);beta[1]=0.
        diff=z[g]-X[g]@beta
        means.append(beta);res.append(diff);cov.append(diff.T@diff/max(1,len(diff)-1))
    pooled_cov=np.mean(cov,axis=0)
    # Supervised projection includes between-class directions and conditional residual geometry.
    vals,vec=np.linalg.eigh(z.T@z/len(z))
    P=vec[:,-20:]/np.sqrt(np.maximum(.15,vals[-20:]))
    C0=P.T@pooled_cov@P
    H=[]
    for m,r in zip(cov,res):
        n=len(r);mix=n/(n+50.)
        H.append((1-mix)*C0+mix*(P.T@m@P)+.16*np.eye(P.shape[1]))
    H=np.stack(H)
    return dict(v3_center=mu,v3_scale=unit,v3_q=np.array([center,scale]),v3_proj=P,
                v3_sphere=np.asarray(kind=="sphere"),v3_means=np.stack(means)@P,
                v3_prec=np.linalg.inv(H),v3_logdet=np.linalg.slogdet(H)[1])


def geom_score(f,q,h):
    z=np.clip((f-h["v3_center"])/h["v3_scale"],-8.,8.)
    if h["v3_sphere"]:
        z=z/np.maximum(np.sqrt(np.mean(z*z,axis=1,keepdims=True)),.08)
    t=np.clip((q-h["v3_q"][0])/h["v3_q"][1],-3.,3.)
    mu=h["v3_means"]
    r=(z@h["v3_proj"])[:,None,:]-(mu[:,0]+t[:,None,None]*mu[:,1])
    d=np.einsum("ncd,cde,nce->nc",r,h["v3_prec"],r)+h["v3_logdet"]
    return np.clip(-.5*(d-d.min(1,keepdims=True)),-16.,0.)


def kernel_views(x):
    logs=np.log(np.maximum(abs(x.reshape(-1,16,8)),.025))
    a=logs.reshape(-1,8,2,8).mean(2)
    contrast=a-a.mean(1,keepdims=True)
    trans=a[:,:,2:]-a[:,:,:1]
    rate=trans-trans.mean(2,keepdims=True)
    return np.stack((a,contrast,np.concatenate((a[:,:,1:2],trans.mean(2,keepdims=True),rate),axis=2)),axis=1).reshape(len(x),3,-1)


def radial_kernel(z,nodes):
    # Each radial kernel sees one physical response representation.
    cross=np.einsum("nvd,cvd->nvc",z,nodes)/z.shape[-1]
    d=(z*z).mean(2)[:,:,None]+(nodes*nodes).mean(2).T[None]-2*cross
    return np.exp(-.65*np.maximum(d,0.)).mean(1)


def radial_head(x,y,classes):
    v=kernel_views(x); c=v.mean(0); s=np.maximum(.03,v.std(0));z=np.clip((v-c)/s,-7.,7.)
    amp=np.log(np.maximum(abs(x[:,::8]),1.)).mean(1)
    nodes=[]
    for i in range(classes):
        ids=np.flatnonzero(y==i);ordered=ids[np.argsort(amp[ids],kind="stable")]
        for piece in np.array_split(ordered,min(12,max(1,len(ids)//3))):
            nodes.append(z[piece].mean(0))
    nodes=np.stack(nodes)
    K=radial_kernel(z,nodes)
    w=1./np.maximum(np.bincount(y,minlength=classes)[y]/40,1.)
    R=.001*len(x)*radial_kernel(nodes,nodes)+.001*np.eye(len(nodes))
    b=np.linalg.solve((K.T*w)@K+R,(K.T*w)@np.eye(classes)[y])
    return dict(rad_center=c,rad_scale=s,rad_nodes=nodes,rad_coef=b)


def radial_scores(x,h):
    z=np.clip((kernel_views(x)-h["rad_center"])/h["rad_scale"],-7.,7.)
    return radial_kernel(z,h["rad_nodes"])@h["rad_coef"]


def component_mix(m,h,x):
    der=gas.signatures(x); base,a,r=gas.components(m,h,x,der)
    k,sub=prototype.scores(der[0],np.log(np.maximum(abs(x),.02)),h)
    extra=radial_scores(x,h)
    old=.4*(a+r)+.5*(k+sub)
    if int(h["rad_mode"])==1:
        # Bounded margin gate: agreement only increases the existing class margin.
        margin=np.sort(extra,axis=1)[:,-1]-np.sort(extra,axis=1)[:,-2]
        w=np.clip(margin,0.,.65)[:,None]
        return base+(1-w)*old+w*(18*extra+.5*sub)
    return base+.4*(a+r)+.5*sub+10*extra


def hybrid_head(m,x,y,index):
    h=gas.fit_legacy(m,x,y,index); curve=gas.fit(m,x,y,index)
    # Shared raw log-response subspace is identical in both parent models.
    for k,v in curve.items():
        if k.startswith(("curve_","amp_")):
            h[k]=v
    return h


def hybrid(m,h,x,mode):
    feat=gas.signatures(x)
    b,a,r=gas.components(m,h,x,feat)
    k,sub=prototype.scores(feat[0],np.log(np.maximum(abs(x),.02)),h)
    legacy=b+.4*(a+r)+.5*(k+sub)
    mode=int(mode)
    top=np.partition(legacy,-2,axis=1)[:,-2:]
    margin=top.max(1)-top.min(1)
    gate=np.clip((6.-margin)/3.,0.,1.) if mode==1 else np.ones(len(x))
    use=gate>0
    if not use.any(): return legacy
    f=feat[0][use]
    z=prototype._normalize(f,h["curve_center"],h["curve_scale"])@h["curve_proj"]
    q=np.log(np.maximum(abs(x[use,::8]),1.)).mean(1)
    q=np.clip((q-h["amp_center"])/h["amp_scale"],-3.,3.)
    c=h["curve_coeff"];res=z[:,None,:]-(c[:,0]+q[:,None,None]*(c[:,1]+q[:,None,None]*c[:,2]))
    ali=np.einsum("ncd,ckd->nck",res,h["curve_axes"])
    d=(res*res).sum(2)-(h["curve_gain"]*ali**2).sum(2)
    curve=b[use]+.5*sub[use]+np.clip(-.3*(d-d.min(1,keepdims=True)),-12.,0.)
    if mode==3:
        qlo,qhi=h["clamped_support"].T
        anchor=np.clip(q[:,None],qlo,qhi)
        extra=np.clip(q[:,None]-anchor,-.5,.5)
        mu=c[:,0]+anchor[:,:,None]*(c[:,1]+anchor[:,:,None]*c[:,2])+extra[:,:,None]*(c[:,1]+2*anchor[:,:,None]*c[:,2])
        res=z[:,None,:]-mu; ali=np.einsum("ncd,ckd->nck",res,h["curve_axes"])
        d=24*np.log1p(np.maximum(0.,(res**2).sum(2)-(h["curve_gain"]*ali**2).sum(2))/24)
        curve=b[use]-3*np.log1p(np.maximum(0.,-sub[use]/6))+np.clip(-.3*(d-d.min(1,keepdims=True)),-12.,0.)
    if mode==2:
        return curve+np.clip(legacy-legacy.max(1,keepdims=True),-4,0)
    if mode in (4,5):
        curve=curve-b[use]
    cutoff=np.log(19.) if mode==5 else 4.
    bounded=np.clip(curve-curve.max(1,keepdims=True),-cutoff,0.)
    legacy[use]+=gate[use,None]*bounded
    return legacy


def range_head(m,x,y,mode):
    h=gas.fit(m,x,y,1)
    q=gas.curve_inputs(x)[2];q=np.clip((q-h["amp_center"])/h["amp_scale"],-3.,3.)
    h["amplitude_range"]=np.stack([np.quantile(q[y==cls],[.05,.95]) for cls in range(len(m.meta["classes"]))])
    h["curve_tail"]=np.array(mode)
    return h


def guarded_curve(m,h,x):
    f,raw,q=gas.curve_inputs(x)
    z=prototype._normalize(f,h["curve_center"],h["curve_scale"])@h["curve_proj"]
    q=np.clip((q-h["amp_center"])/h["amp_scale"],-3.,3.)
    lo,hi=h["amplitude_range"].T
    anchor=np.clip(q[:,None],lo,hi)
    extra=np.clip(q[:,None]-anchor,-.5,.5)
    co=h["curve_coeff"]
    trajectory=co[:,0]+anchor[:,:,None]*(co[:,1]+anchor[:,:,None]*co[:,2])+extra[:,:,None]*(co[:,1]+2*anchor[:,:,None]*co[:,2])
    u=z[:,None,:]-trajectory
    a=np.einsum("ncd,ckd->nck",u,h["curve_axes"])
    d=np.maximum(0.,(u*u).sum(2)-(a*a*h["curve_gain"]).sum(2))
    r=prototype._normalize(raw,h["log_center"],h["log_scale"])[:,None,:]-h["log_means"]
    ar=np.einsum("ncd,ckd->nck",r,h["log_axes"])
    dr=np.maximum(0.,(r*r).sum(2)-.94*(ar*ar).sum(2))/raw.shape[1]
    if int(h["curve_tail"]):
        d=24*np.log1p(d/24);dr=np.log1p(dr)
    return np.log(np.maximum(m.predict(x),1e-12))+np.clip(-.3*(d-d.min(1,keepdims=True)),-12,0)-3*dr


def alternate(m,x,y,route):
    classes=len(m.meta["classes"])
    if route in ("range","heavy"):
        return range_head(m,x,y,route=="heavy")
    if route in ("tempered","support","uncertain"):
        h=gas.fit_legacy(m,x,y,1);h["backbone_cap"]=np.array(("tempered","support","uncertain").index(route));return h
    if route in ("bounded","margin","reverse"):
        h=hybrid_head(m,x,y,1);h["odds_cap"]=np.asarray(("bounded","margin","reverse").index(route));return h
    if route in ("radial","radmix","radreplace"):
        head=radial_head(x,y,classes)
        if route!="radial":
            head.update(gas.fit_legacy(m,x,y,1));head["rad_mode"]=np.asarray(1 if route=="radmix" else 2)
        return head
    f,raw,q=views(x)
    if route in ("sphere","pool","geometry"):
        return geometry(f,q,y,classes,"sphere" if route=="sphere" else "pool")
    if route=="augment":
        rng=np.random.default_rng(73201)
        tensor=x.reshape(-1,16,8)
        # Perturb shared channel gain, preserving the dimensionless steady-state ratio.
        factor=rng.normal(0.,.10,(len(x),16,1))
        direction=np.array([1.,0.,1.,1.,1.,1.,1.,1.])
        expanded=np.concatenate([tensor,tensor*np.exp(factor*direction),tensor*np.exp(-factor*direction)]).reshape(-1,128)
        return gas.fit_legacy(m,expanded,np.tile(y,3),1)
    raise ValueError(route)


def install(route):
    def fits(m,x,y,i):
        if np.bincount(y,minlength=len(m.meta["classes"])).min()<2:
            return BASE_FIT(m,x,y,i)
        if route in ("independent","odds95"):
            tail=range_head(m,x,y,True);alt=gas.fit_legacy(m,x,y,i)
            alt.update({k:v for k,v in tail.items() if k.startswith(("amp_","curve_"))})
            alt["odds_cap"]=np.asarray(4 if route=="independent" else 5)
            return {"tail":tail,route:alt}
        if route=="shared":
            tail=range_head(m,x,y,True);alt=gas.fit_legacy(m,x,y,i)
            alt.update({k:v for k,v in tail.items() if k.startswith(("amp_","curve_"))})
            alt["odds_cap"]=np.asarray(3);alt["clamped_support"]=tail["amplitude_range"]
            return {"tail":tail,"shared":alt}
        if route=="combined":
            return {"tail":range_head(m,x,y,True),"bounded":alternate(m,x,y,"bounded")}
        ext=alternate(m,x,y,route)
        if route=="geometry":
            ext.update(gas.fit(m,x,y,i));ext["v3_mix"]=np.asarray(True)
        if route in ("augment","radmix","radreplace","bounded","margin","tempered","support","uncertain"):
            return {"curve":gas.fit(m,x,y,i),route:ext}
        return {route:ext,"v1":gas.fit_legacy(m,x,y,i)}
    def pred(m,h,x):
        if "amplitude_range" in h: return guarded_curve(m,h,x)
        if "backbone_cap" in h:
            feat=gas.signatures(x);b,a,r=gas.components(m,h,x,feat)
            k,sub=prototype.scores(feat[0],np.log(np.maximum(abs(x),.02)),h)
            mode=int(h["backbone_cap"])
            vote=b if mode==0 else b+.5*sub
            legacy=b+.4*(a+r)+.5*(k+sub)
            if mode==2:
                # The distance to the nearest class subspace attenuates the extra source vote.
                support=np.exp(np.minimum(0.,sub.max(1,keepdims=True))/6.)
            else: support=1.
            return legacy+support*np.clip(vote-vote.max(1,keepdims=True),-4,0)
        if "odds_cap" in h:
            return hybrid(m,h,x,h["odds_cap"])
        if "rad_center" in h:
            if "rad_mode" in h: return component_mix(m,h,x)
            return np.log(np.maximum(m.predict(x),1e-12))+18*radial_scores(x,h)
        if "v3_prec" not in h:
            return BASE_PRED(m,h,x)
        f,_,q=views(x); s=geom_score(f,q,h)
        if "v3_mix" in h:
            cs=prototype.score_curve(*gas.curve_inputs(x),h)
            # Margin of the discriminant regulates correction on weak evidence.
            top=np.partition(s,-2,axis=1)[:,-2:]
            w=np.clip((top.max(1)-top.min(1))/8.,.15,.65)[:,None]
            s=(1-w)*cs+w*s
        return np.log(np.maximum(m.predict(x),1e-12))+s
    gas.fit_candidates=fits
    gas.predict=pred


def configure(route):
    import contest_submission as harness
    # Fix control policy to the supervised head ablation throughout these comparisons.
    harness.CONTEXT_TASKS=set()
    harness.ANCHOR_TASKS=set()
    harness.SWITCH_SE=0.
    harness.BLOCK_WEIGHT=0.
    harness.CLASS_DROP_LIMIT=None
    if route=="v2":
        gas.fit_candidates,gas.predict=BASE_FIT,BASE_PRED
    elif route=="packed":
        frozen_fit=gas.fit_candidates
        def packed(*args):
            return {k:{n:v.astype(np.float32) if v.dtype==np.float64 else v for n,v in h.items()} for k,h in frozen_fit(*args).items()}
        gas.fit_candidates=packed
    elif route!="frozen":
        install(route)


def resource_pairs():
    import os
    import subprocess
    import sys
    allowed=sorted(os.sched_getaffinity(0))
    os.sched_setaffinity(0,{allowed[-1]})
    for i in range(3):
        for setting in ("drift","official"):
            for route in (("v2","frozen") if i%2==0 else ("frozen","v2")):
                subprocess.run([sys.executable,"-B",__file__,"cost-"+route,setting,str(i)],check=True)


def run(route,settings):
    configure(route)
    benchmark.OUT=OUT/route
    for setting in settings:
        for i in range(5):
            benchmark.run("A","v3",i,setting,profile=True)


def physical_metrics(y,s,temperature=1.):
    exponent=s/temperature;exponent-=exponent.max(1,keepdims=True)
    p=np.exp(exponent);p/=p.sum(1,keepdims=True)
    confidence=p.max(1);pred=p.argmax(1)
    bins=np.minimum((confidence*10).astype(int),9)
    ece=sum(np.sum(bins==j)*abs(np.mean((pred==y)[bins==j])-np.mean(confidence[bins==j]))
            for j in np.unique(bins))/len(y)
    return {"f1":metric("A",y,s),"ece10":float(ece),"nll":float(-np.log(np.maximum(p[np.arange(len(y)),y],1e-12)).mean()),
            "brier":float(np.mean(np.sum((p-np.eye(s.shape[1])[y])**2,axis=1)))}


def native_test(name):
    path=HERE.parent/"V2/gas_v2_study.py"
    spec=importlib.util.spec_from_file_location("external_extract",path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    base_native=module.native_compare
    def compare(x,absolute,relative,raw,q,classes,y,train,blocks,cluster=None,inner=None):
        def per(use):
            raw_scores=base_native(x,absolute,relative,raw,q,classes,y,use,[],return_scores=True)
            b,v1,curve=(raw_scores[k] for k in ("mlp","v1","v2"))
            teacher=curve-b
            vote=teacher-teacher.max(1,keepdims=True)
            uv=prototype.fit_curve(absolute[use],raw[use],q[use],y[use],len(classes))
            standardized=np.clip((q[use]-uv["amp_center"])/uv["amp_scale"],-3.,3.)
            uv["amplitude_bounds"]=np.stack([np.quantile(standardized[y[use]==j],[.05,.95]) for j in range(len(classes))])
            tail=b+prototype.robust_score(absolute[~use],raw[~use],q[~use],uv)
            return {"legacy":v1,"curve":curve,"independent":v1+np.clip(vote,-4.,0.),"tail":tail}
        inner_scores=per(inner);outer=per(train);test=~train;valid=train&~inner
        cal={k:v[valid[~inner]] for k,v in inner_scores.items()}
        choices={"v2":max(("curve","legacy"),key=lambda k:metric("A",y[valid],cal[k])),
                 "v3":max(("tail","independent"),key=lambda k:metric("A",y[valid],cal[k]))}
        result={"train":int(train.sum()),"selection":int(valid.sum()),"test":int(test.sum()),"selected":choices,
                "candidates":{},"selection_f1":{}}
        for k,v in outer.items():
            temp=min((.5,1.,2.,4.),key=lambda t:physical_metrics(y[valid],cal[k],t)["nll"])
            result["selection_f1"][k]=metric("A",y[valid],cal[k])
            result["candidates"][k]={"raw":physical_metrics(y[test],v),"calibrated":physical_metrics(y[test],v,temp),
                "temperature_from_inner":temp,"blocks":[metric("A",y[b],v[b[test]]) for b in blocks]}
        if cluster is not None:
            exposed=cluster[test];days=np.unique(exposed);C=len(classes)
            rng=np.random.default_rng(20261127);ix=rng.integers(len(days),size=(4000,len(days)))
            boot=[]
            for pick in choices.values():
                t=y[test];p=outer[pick].argmax(1)
                mat=np.stack([np.bincount(C*t[exposed==day]+p[exposed==day],minlength=C*C).reshape(C,C) for day in days])
                mat=mat[ix].sum(1)
                boot.append((2*np.diagonal(mat,axis1=1,axis2=2)/np.maximum(1,mat.sum(1)+mat.sum(2))).mean(1))
            result["selected_delta_day_ci95"]=np.quantile(boot[1]-boot[0],[.025,.975]).tolist()
        return result
    module.native_compare=compare
    result=(module.cross_day if name=="cross-day" else module.dsps)()
    (OUT/(name+".json")).write_text(json.dumps(result,indent=2,allow_nan=False))
    print(json.dumps(result))


def all_public(which):
    import os
    import tempfile
    from types import SimpleNamespace
    import contest_submission as harness
    from benchmark import Parent,PublicRound,WORK
    if which=="v2":
        gas.fit_candidates,gas.predict=BASE_FIT,BASE_PRED
    data=WORK/"A"
    def rows(name):
        with (data/"public"/(name+".jsonl")).open() as inf:
            for line in inf:
                r=json.loads(line);yield {"inputs":r["inputs"],"answer":r["answer"]}
    run=PublicRound();run.agent,run.dataset_key,run.round_index=harness,"A",1
    run.parent,run.records,run.call_ids=Parent(data),[],[]
    run.artifacts=[];run.prompt_bytes=0;run.plan="refit|half|calibrate gas responses"
    run.data=SimpleNamespace(training_rows=lambda:rows("train"),validation_rows=lambda:rows("validation"))
    run.selection=harness.projection(rows("validation"),harness.Model.load(run.parent.path("checkpoint.npz")),"A")[:2]
    with tempfile.TemporaryDirectory(dir=OUT) as tmp:
        os.environ["CONTEST_WORK"]=tmp;run.root=Path(tmp);harness.train(run)
        model,head,_=harness.artifact(run.candidate)
        xv,yv=run.selection
        subset=xv[np.linspace(0,len(xv)-1,64,dtype=int)]
        batch=harness.values("A",model,head,subset)
        scalar=np.vstack([harness.values("A",model,head,z[None]) for z in subset])
        np.testing.assert_allclose(batch,scalar,atol=1e-7,rtol=1e-7)
        cfg=run.record["configuration"]
        name=cfg["deployed"]
        result={"plan":run.plan,"public_rows":len(xv),"family":cfg["head_family"],"weight":cfg["expert_weight"],
                "deployed":name,"macro":cfg["scores"][name]["primary"],"worst":cfg["scores"][name]["worst"],
                "parent":cfg["scores"]["parent"]["primary"],"spread":float(np.abs(batch-scalar).max()),
                "checkpoint":sum(p.stat().st_size for p in run.candidate.glob("*.npz"))}
    (OUT/("full-"+which+".json")).write_text(json.dumps(result,indent=2))
    print(result)


def summarize():
    for base in sorted(OUT.glob("*")):
        if not base.is_dir(): continue
        for setting in ("drift","official"):
            res=[json.loads(p.read_text()) for p in sorted(base.glob(setting+"-*.json"))]
            if res:
                print(base.name,setting,len(res),*(round(np.mean([r[k] for r in res]),4) for k in ["mean","worst","forget","delay"]),
                      "A4",round(res[-1]["forget"],4),"ms,bytes",
                      round(np.median([r["prediction_us"] for r in res if "prediction_us" in r]),1),
                      round(np.mean([np.mean([t["checkpoint_bytes"] for t in r["rounds"]]) for r in res])))

if __name__=="__main__":
    cli=argparse.ArgumentParser()
    cli.add_argument("route");cli.add_argument("setting",nargs="?",default="all")
    cli.add_argument("repeat",nargs="?",type=int,default=0)
    a=cli.parse_args()
    if a.route=="cost": resource_pairs()
    elif a.route.startswith("cost-"):
        route=a.route.split("-")[1]; configure(route)
        benchmark.OUT=OUT/"cost"/str(a.repeat)/route
        benchmark.run("A","v3",0,a.setting,profile=True)
    elif a.route.startswith("full-"): all_public(a.route.split("-")[1])
    elif a.route in ("cross-day","ds-ps"): native_test(a.route)
    elif a.route=="summary": summarize()
    else: run(a.route,("drift","official") if a.setting=="all" else (a.setting,))
