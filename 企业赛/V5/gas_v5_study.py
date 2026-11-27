"""V5 gas route evaluations. Only development schedules 0 through 10 are supported."""
import argparse
import importlib.util
import json
from pathlib import Path
import numpy as np
import gas
import prototype
import benchmark

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
OUT=HERE.parent/"work/v5/gas"
_spec=importlib.util.spec_from_file_location("v4_gas_reference",HERE.parent/"V4/gas.py")
_previous=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(_previous)
BASE_FIT=_previous.fit_candidates
BASE_PRED=_previous.predict
ROUTES=("scale","support","trajectory","pca","both","consensus","bounded")
PILOT=(0,3,5,6,9)


def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    return m


def learn(f,v,a,y,head,mode):
    labels=range(len(head["curve_coeff"]))
    h={}
    if mode in ("trajectory","both","pca","rank4","rank3","adaptive"):
        z=prototype._normalize(v,head["log_center"],head["log_scale"])
        q=np.clip((a-head["amp_center"])/head["amp_scale"],-3.,3.)
        means=[];slopes=[];axes=[];gains=[]
        for c in labels:
            use=y==c;rows=z[use];t=q[use]
            slope=((t-t.mean())@(rows-rows.mean(0)))/(np.sum((t-t.mean())**2)+4.)
            if mode!="trajectory" and mode!="both": slope[:]=0.
            center=rows.mean(0)-t.mean()*slope
            res=rows-center-t[:,None]*slope
            _,s,V=np.linalg.svd(res,full_matrices=False)
            rank=3 if mode=="rank3" else 5 if mode=="adaptive" else 4
            n=min(rank,len(s));ev=s*s/max(1,len(t)-1)
            noise=max(.05,float(ev[n:].sum()/max(1,z.shape[1]-n)))
            shrink=np.maximum(0.,1.-noise/np.maximum(ev[:n],noise))
            if mode=="rank4": shrink[:]=.94
            if mode=="adaptive":
                # Finite-sample discount on directions competing with residual noise.
                shrink=np.maximum(0.,1.-noise*(1.+np.sqrt(z.shape[1]/len(t)))**2/np.maximum(ev[:n],noise))
            basis=np.zeros((rank,z.shape[1]));basis[:n]=V[:n]
            g=np.zeros(rank);g[:n]=shrink
            axes.append(basis);gains.append(g);means.append(center);slopes.append(slope)
        h.update(g5_inter=np.stack(means),g5_slope=np.stack(slopes),g5_axes=np.stack(axes),g5_gain=np.stack(gains))
    if mode in ("scale","support","both"):
        d=prototype.curve_distance(f,a,head,extrapolate=True)
        pooled=np.median(d[np.arange(len(y)),y])
        centers=[]
        for c in labels:
            m=y==c;n=m.sum()
            centers.append((n*np.median(d[m,c])+24*pooled)/(n+24))
        h["g5_noise"]=np.clip(np.asarray(centers)/max(pooled,.01),.55,1.8)
        h["g5_support"]=np.array(mode=="support")
    return h


def evidence(f,v,a,head):
    d=prototype.curve_distance(f,a,head,extrapolate=True);dim=head["curve_proj"].shape[1]
    if "g5_noise" in head:
        noise=head["g5_noise"]
        d=d/noise+dim*np.log(noise)
        if head["g5_support"]:
            # Bound correction when every class is far outside released response geometry.
            d=d/np.maximum(1.,d.min(1,keepdims=True)/dim)
    d=np.maximum(d,0.)
    if "g5_axes" in head:
        z=prototype._normalize(v,head["log_center"],head["log_scale"])
        q=np.clip((a-head["amp_center"])/head["amp_scale"],-3.,3.)
        u=z[:,None,:]-(head["g5_inter"]+q[:,None,None]*head["g5_slope"])
        al=np.einsum("ncd,ckd->nck",u,head["g5_axes"])
        raw=np.maximum(0.,(u*u).sum(2)-(head["g5_gain"]*al**2).sum(2))/v.shape[1]
    else:
        raw=prototype.raw_distance(v,head["log_center"],head["log_scale"],head["log_means"],head["log_axes"])
    return prototype.curve_log_score(dim*np.log1p(d/dim))-3*np.log1p(raw)


def logprob(x):
    t=x-x.max(1,keepdims=True)
    return t-np.log(np.exp(t).sum(1,keepdims=True))


def fuse(f,v,a,h,mode):
    d=prototype.curve_distance(f,a,h,extrapolate=True)
    dim=h["curve_proj"].shape[1]
    k=prototype.curve_log_score(dim*np.log1p(d/dim))
    raw=prototype.raw_distance(v,h["log_center"],h["log_scale"],h["log_means"],h["log_axes"])
    r=-3*np.log1p(raw)
    if mode=="softcap":
        z=k+r
        return 8*np.tanh((z-z.max(1,keepdims=True))/8)
    if mode=="novel":
        support=np.clip(2.*dim/np.maximum(dim,d.min(1,keepdims=True)),.4,1.)
        return (k+r)*support
    if mode=="angular":
        # Discount the kinetic vote when it and the shape vote support different chemicals.
        sv=r-r.mean(1,keepdims=True);kv=k-k.mean(1,keepdims=True)
        dot=np.sum(kv*sv,axis=1,keepdims=True)
        rho=dot/np.maximum(np.linalg.norm(kv,axis=1,keepdims=True)*np.linalg.norm(sv,axis=1,keepdims=True),1e-8)
        return (.7+.3*np.maximum(rho,0))*k+r
    lk=logprob(2*k);lr=logprob(2*r)
    union=np.logaddexp(lk,lr)-np.log(2.)
    if mode=="mixture": return union
    js=.5*(np.sum(np.exp(lk)*(lk-union),axis=1,keepdims=True)+np.sum(np.exp(lr)*(lr-union),axis=1,keepdims=True))
    strength=.45*np.clip(js/np.log(2.),0,1)
    return (1-strength)*(k+r)+strength*union


def phys_kernel(z,h):
    f=prototype.invariant_features(z,h["phys_loc"],h["phys_scale"])
    v=np.clip((f-h["phys_mid"])/h["phys_std"],-8.,8.)
    sq=(v*v).mean(1)[:,None]+h["phys_norm"]-(2./v.shape[1])*(v@h["phys_nodes"].T)
    return np.exp(-.85*np.maximum(sq,0.)/h["phys_width"])


def calibrate(z,y,h):
    K=phys_kernel(z,h);c=K.max(1)
    counts=[min(6,max(1,int(np.sum(y==i))//3)) for i in range(int(np.max(y))+1)]
    ids=np.repeat(np.arange(len(counts)),counts)
    return {"vote_support":np.quantile(c,[.15,.55]),"landmark_class":ids}


def physical_adjust(K,h,mode):
    g=K@h["phys_coef"]
    old=prototype.physical_vote(g)
    sim=K.max(1,keepdims=True);lo,hi=h["vote_support"]
    trust=np.clip((sim-lo)/max(hi-lo,.01),0.,1.)
    if mode=="trust":
        return (.35+.65*trust)*old
    if mode=="normalize":
        mod=g/np.sqrt(np.clip(sim/max(hi,.01),.25,1.)) 
        return prototype.physical_vote(mod)
    if mode=="nearest":
        nearest=np.stack([K[:,h["landmark_class"]==j].max(1) for j in range(g.shape[1])],axis=1)
        prior=np.log(np.maximum(nearest,1e-8))
        return old+(1.-trust)*np.clip(2*(prior-prior.max(1,keepdims=True)),-2.,0.)
    raise ValueError(mode)


def install(route):
    def fits(m,x,y,i):
        heads=BASE_FIT(m,x,y,i)
        if "tail" not in heads: return heads
        f,v,a=gas.curve_inputs(x)
        head=heads["tail"]
        if route in ("trust","normalize","nearest"):
            h=heads["independent"]
            h.update(calibrate(prototype.response_shape(x),y,h))
            h["phys_rule"]=np.asarray(("trust","normalize","nearest").index(route))
        elif route in ("softcap","novel","angular","mixture","js"):
            head["fusion5"]=np.asarray(("softcap","novel","angular","mixture","js").index(route))
        elif route in ("anchor","recent","amplitude","jackpair","kinetics"):
            C=len(m.meta["classes"])
            if route in ("anchor","recent","amplitude"):
                ids=[]
                for cls in range(C):
                    ii=np.flatnonzero(y==cls);n=len(ii)
                    if route=="anchor": ids.extend(ii[:max(1,n//3)])
                    elif route=="recent": ids.extend(ii[-max(1,n//3):])
                    else:
                        cuts=np.quantile(a,[.15,.35,.65,.85]);bins=np.searchsorted(cuts,a[ii])
                        for j in np.unique(bins):
                            grp=ii[bins==j];add=min(len(grp),max(0,int(n/5)-len(grp)))
                            if add: ids.extend(grp[np.linspace(0,len(grp)-1,add,dtype=int)])
                ii=np.r_[np.arange(len(y)),np.asarray(ids,dtype=int)]
                cx,cy=x[ii],y[ii]
            else:
                pair=x.reshape(-1,8,2,8)
                avg=pair.mean(2,keepdims=True)
                noise=(pair[:,:,0:1]-pair[:,:,1:2])*.22
                if route=="kinetics":
                    noise=noise.copy();noise[:,:,:,:2]=0.
                cx=np.concatenate((x,(pair+noise).reshape(x.shape),(pair-noise).reshape(x.shape)))
                cy=np.tile(y,3)
            ff,vv,aa=gas.curve_inputs(cx)
            new=prototype.fit_curve(ff,vv,aa,cy,C)
            qq=np.clip((aa-new["amp_center"])/new["amp_scale"],-3.,3.)
            new["amplitude_bounds"]=np.stack([np.quantile(qq[cy==j],[.05,.95]) for j in range(C)])
            heads["tail"]={k:np.asarray(v,dtype=np.float32) for k,v in new.items()}
        elif route in ("consensus","bounded"):
            geom=prototype.fit_physical(prototype.response_shape(x),y,len(m.meta["classes"]))
            head.update(geom)
            head["g5_vote"]=np.asarray(1. if route=="consensus" else .35)
        else:
            head.update(learn(f,v,a,y,head,route))
        return heads
    def predict(m,h,x):
        if "phys_rule" in h:
            kk=phys_kernel(prototype.response_shape(x),h)
            nm=("trust","normalize","nearest")[int(h["phys_rule"])]
            return BASE_PRED(m,h,x)-prototype.physical_vote(kk@h["phys_coef"])+physical_adjust(kk,h,nm)
        if "fusion5" in h:
            name=("softcap","novel","angular","mixture","js")[int(h["fusion5"])]
            return np.log(np.maximum(m.predict(x),1e-12))+fuse(*gas.curve_inputs(x),h,name)
        if "g5_inter" in h or "g5_noise" in h:
            return np.log(np.maximum(m.predict(x),1e-12))+evidence(*gas.curve_inputs(x),h)
        if "g5_vote" in h:
            return BASE_PRED(m,h,x)+float(h["g5_vote"])*prototype.physical_score(x,h)
        return BASE_PRED(m,h,x)
    gas.fit_candidates=fits;gas.predict=predict
    gas.prepare_inference=lambda m,h:None


def replay(route,full):
    if route=="v4":
        gas.fit_candidates=BASE_FIT;gas.predict=BASE_PRED
    elif route!="frozen": install(route)
    benchmark.OUT=OUT/"replay"/route
    for case in (("fresh","drift","official","fresh-measured") if full else ("fresh","drift","official")):
        setting=case.split("-")[0];review="measured" if case.endswith("measured") else "controlled"
        for n in (range(11) if full else PILOT):
            path=benchmark.OUT/f"{case}-v5-A-{n}.json"
            if path.exists(): continue
            benchmark.run("A","v5",n,setting,profile=False,review=review)


def summary():
    for path in sorted((OUT/"replay").iterdir()):
        fields=[]
        for case in ["fresh","drift","official","fresh-measured"]:
            files=sorted(path.glob(f"{case}-v5-A-*.json"))
            rows=[json.loads(f.read_text()) for f in files]
            if rows: fields.append([case,len(rows),*[round(100*np.mean([r[k] for r in rows]),2) for k in ("mean","worst","forget")]])
        print(path.name,fields)


def native(args,fit,query,route):
    x,f,relative,raw,q,z,classes,y,groups=args
    union=fit|query
    v2=module("prev21",HERE.parent/"V2/gas_v2_study.py")
    scores=v2.native_compare(x[union],f[union],relative[union],raw[union],q[union],classes,y[union],
                             fit[union],[],return_scores=True)
    b,old,curve=(scores[k] for k in ("mlp","v1","v2"))
    sensor=curve-b
    independent=old+np.clip(sensor-sensor.max(1,keepdims=True),-4,0)
    ph=prototype.fit_physical(z[fit],y[fit],len(classes),groups=groups[fit])
    pf=prototype.invariant_features(z[query],ph["phys_loc"],ph["phys_scale"])
    v=np.clip((pf-ph["phys_mid"])/ph["phys_std"],-8.,8.)
    sq=(v*v).mean(1)[:,None]+ph["phys_norm"]-(2./v.shape[1])*(v@ph["phys_nodes"].T)
    K=np.exp(-.85*np.maximum(sq,0.)/ph["phys_width"])
    guide=K@ph["phys_coef"]
    independent+=prototype.physical_vote(guide)
    h=prototype.fit_curve(f[fit],raw[fit],q[fit],y[fit],len(classes))
    amount=np.clip((q[fit]-h["amp_center"])/h["amp_scale"],-3.,3.)
    h["amplitude_bounds"]=np.stack([np.quantile(amount[y[fit]==c],[.05,.95]) for c in range(len(classes))])
    old_tail=b+prototype.robust_score(f[query],raw[query],q[query],h)
    new_independent=independent
    if route in ("trust","normalize","nearest"):
        ph.update(calibrate(z[fit],y[fit],ph))
        new_independent=independent-prototype.physical_vote(guide)+physical_adjust(K,ph,route)
        new_tail=old_tail
    elif route in ("softcap","novel","angular","mixture","js"):
        new_tail=b+fuse(f[query],raw[query],q[query],h,route)
    else:
        h.update(learn(f[fit],raw[fit],q[fit],y[fit],h,route))
        new_tail=b+evidence(f[query],raw[query],q[query],h)
    return {"v4_tail":old_tail,"v5_tail":new_tail,"v4_independent":independent,"v5_independent":new_independent}


def compare(args,pre,valid,train,test,cluster,route):
    from industrial import metric
    sel=native(args,pre,valid,route)
    all_scores=native(args,train,test,route)
    y=args[-2];classes=len(args[-3]);days=args[-1][test]
    def scores(name,sets,labels):
        return float(metric("A",labels,sets[name]))
    family_scores={k:scores(k,sel,y[valid]) for k in sel}
    result={"pre_n":int(pre.sum()),"validation_n":int(valid.sum()),"fit_n":int(train.sum()),"test_n":int(test.sum()),
            "inner_scores":family_scores,"families":{}}
    choices={}
    boots=[]
    rng=np.random.default_rng(20261127)
    groups=np.unique(days)
    ix=rng.integers(len(groups),size=(4000,len(groups)))
    for ver in ("v4","v5"):
        fam=max((ver+"_tail",ver+"_independent"),key=family_scores.get)
        choices[ver]=fam
        result["families"][ver]={"family":fam,"f1":scores(fam,all_scores,y[test])}
        p=all_scores[fam].argmax(1);t=y[test]
        cm=np.stack([np.bincount(classes*t[days==g]+p[days==g],minlength=classes*classes).reshape(classes,classes) for g in groups])
        mat=cm[ix].sum(1)
        boots.append((2*np.diagonal(mat,axis1=1,axis2=2)/np.maximum(1,mat.sum(1)+mat.sum(2))).mean(1))
    if cluster:
        result["paired_date_ci95"]=np.quantile(boots[1]-boots[0],[.025,.975]).tolist()
    result["head_f1"]={k:scores(k,all_scores,y[test]) for k in all_scores}
    return result


def forward(route):
    source=module("external4",HERE.parent/"V4/gas_v4_study.py")
    args=source.instrument(); days=args[-1]
    out={}
    for name,(a,b,c) in {"historical":(8,12,36),"early":(6,8,12),"middle":(12,14,18),
                         "late":(18,20,24),"terminal":(24,28,36)}.items():
        out[name]=compare(args,days<a,(days>=a)&(days<b),days<b,(days>=b)&(days<c),True,route)
    return out


def dsps(route,terminal):
    src=module("ds_reader",HERE.parent/"V2/gas_v2_study.py")
    def fn(x,absolute,relative,raw,q,classes,y,train,blocks,cluster=None,inner=None):
        n=len(y);pair=x.reshape(n,2,-1)
        z=np.stack((pair[:,1],pair[:,0]/np.maximum(abs(pair[:,1]),1e-6)),axis=2)
        reps=np.array([(i//6)%10 for i in range(n)])
        ppm=20*(np.arange(n)%6+1)
        args=(x,absolute,relative,raw,q,z,classes,y,reps)
        if terminal:
            low=ppm<=80
            return compare(args,low&(reps<6),low&(reps>=6)&(reps<8),low&(reps<8),(ppm>=100)&(reps>=8),False,route)
        return compare(args,inner,train&~inner,train,~train,False,route)
    src.native_compare=fn
    return src.dsps()


def home(route):
    from datetime import datetime
    import hashlib
    import io
    import zipfile
    path=OUT/"home.zip"
    if hashlib.sha256(path.read_bytes()).hexdigest()!="7c143b9f4402a8205ebe8072c2ce0967c25741f1865e23d650e981053294f395":
        raise ValueError("UCI 362 source checksum mismatch")
    with zipfile.ZipFile(path) as bundle:
        meta=[r.split() for r in bundle.read("HT_Sensor_metadata.dat").decode().splitlines()[1:] if r.strip()]
        with zipfile.ZipFile(io.BytesIO(bundle.read("HT_Sensor_dataset.zip"))) as sub:
            with sub.open("HT_Sensor_dataset.dat") as f:
                observations=np.loadtxt(f,skiprows=1,usecols=range(10))
    ids=observations[:,0].astype(int);rows=[];names=[];dates=[]
    edges=np.flatnonzero(np.diff(ids))+1
    recordings={int(block[0,0]):block[:,1:] for block in np.split(observations,edges) if len(block)}
    for code,date,name,onset,length in meta:
        if int(code) not in recordings:
            continue
        data=recordings[int(code)]
        times=data[:,0]*3600;signal=data[:,1:]
        warm=signal[(times<0)&(times>=-900)]
        mask=(times>=0)&(times<=float(length)*3600)
        if not len(warm) or mask.sum()<20: raise ValueError("invalid exposure window")
        base=np.median(warm,axis=0);seq=signal[mask];clock=times[mask]
        delta=seq[int(.8*len(seq)):].mean(0)-base
        cols=[delta,delta/np.maximum(abs(base),1e-5)];up=[];down=[]
        for lag in (2.,10.,45.):
            index=np.searchsorted(clock,clock-lag);good=clock>=clock[0]+lag
            deriv=(seq[good]-seq[index[good]])/lag
            up.append(np.quantile(deriv,.95,axis=0));down.append(np.quantile(deriv,.05,axis=0))
        rows.append(np.stack(cols+up+down,axis=1));names.append(name)
        dates.append(datetime.strptime(date,"%m-%d-%y").date().isoformat())
    obs=np.stack(rows);x=obs.reshape(len(obs),-1)
    # The multirate features describe finished exposures on this native eight-sensor device.
    log=np.log(np.maximum(abs(obs),1e-6))
    f=np.concatenate((log[:,:,:2],log[:,:,2:]-log[:,:,:1],np.diff(log[:,:,2:5],axis=2),np.diff(log[:,:,5:8],axis=2)),axis=2)
    raw=np.log(np.maximum(abs(x),1e-6));q=np.log(np.maximum(abs(obs[:,:,0]),1e-5)).mean(1)
    z=np.concatenate([obs[:,:,1:2],obs[:,:,2:]/np.maximum(abs(obs[:,:,:1]),1e-5)],axis=2)
    cls,y=np.unique(names,return_inverse=True)
    ds=np.unique([datetime.strptime(r[1],"%m-%d-%y").date().isoformat() for r in meta]);day=np.searchsorted(ds,dates)
    args=(x,f.reshape(len(y),-1),(f-f.mean(1,keepdims=True)).reshape(len(y),-1),raw,q,z,cls,y,day)
    a,b=int(.55*len(ds)),int(.75*len(ds))
    result=compare(args,day<a,(day>=a)&(day<b),day<b,day>=b,True,route)
    result.update(days=len(ds),missing_record_ids=[int(r[0]) for r in meta if int(r[0]) not in recordings],cut_dates=[str(ds[a]),str(ds[b])],classes=cls.tolist(),counts=np.bincount(y).tolist())
    return result


def external(kind,route):
    if kind=="cross-day": data=forward(route)
    elif kind=="home": data=home(route)
    else: data=dsps(route,terminal=kind=="cross-concentration")
    (OUT/(kind+"-"+route+".json")).write_text(json.dumps(data,indent=2,allow_nan=False))
    print(json.dumps(data))


def resources():
    import os
    import sys
    import subprocess
    os.sched_setaffinity(0,{max(os.sched_getaffinity(0))})
    for n in range(3):
        for setting in ("drift","fresh","official"):
            for ver in (("v4","v5") if n%2==0 else ("v5","v4")):
                subprocess.run([sys.executable,"-B",str(HERE/"benchmark.py"),"A",ver,"0",
                                "--setting",setting,"--profile","--output",str(OUT/"cost"/str(n))],check=True)


def official(ver):
    if ver=="v4":
        gas.fit_candidates,gas.predict=BASE_FIT,BASE_PRED
    old=module("full4",HERE.parent/"V4/gas_v4_study.py")
    old.OUT=OUT; old.full_round(ver)


def equivalence():
    from industrial import Model,dataset
    root=ROOT/"校赛/work/A";m=Model.load(root/"checkpoint.npz")
    def rows(k):
        return dataset([json.loads(r) for r in (root/"public"/(k+".jsonl")).read_text().splitlines()],m)
    x,y=rows("train");v,_=rows("validation")
    rows=np.vstack((v,np.zeros(128),v[:4]*120,v[5:9]*.001,-v[10:14]))
    result={}
    for ver,fn in (("v4",BASE_FIT),("v5",gas.fit_candidates)):
        for k,h in fn(m,x,y,1).items():
            plan=gas.prepare_inference(m,h)
            batch=gas.predict(m,h,rows);single=np.stack([plan(row) for row in rows])
            np.testing.assert_allclose(batch,single,atol=1e-9,rtol=1e-9)
            info={"rows":len(rows),"max_score_error":float(np.abs(batch-single).max()),
                  "class_disagreements":int(np.sum(batch.argmax(1)!=single.argmax(1)))}
            if ver=="v4":
                info["old_checkpoint_max_error"]=float(np.abs(batch-BASE_PRED(m,h,rows)).max())
            result[ver+"-"+k]=info
    (OUT/"single.json").write_text(json.dumps(result,indent=2));print(result)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("route");parser.add_argument("--full",action="store_true")
    args=parser.parse_args()
    if args.route=="summary": summary()
    elif args.route=="cost": resources()
    elif args.route=="single": equivalence()
    elif args.route.startswith("full-"): official(args.route.split("-")[1])
    elif ":" in args.route: external(*args.route.split(":"))
    else: replay(args.route,args.full)
