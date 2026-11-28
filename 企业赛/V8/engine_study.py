"""Compare fixed-budget engine training, risk-constrained calibration and incumbent heads."""
from pathlib import Path
import importlib.util
import sys
import json
import numpy as np

HERE=Path(__file__).resolve().parent
V5=HERE.parent/"V5"
OUT=HERE.parent/"work/v8/engine"
sys.path.insert(0,str(V5))
import engine as baseline
import benchmark
import engine_v5_study as lab
from time import process_time


def load(name, path):
    spec=importlib.util.spec_from_file_location(name,path)
    mod=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def sensor_state(head,x,y,route):
    """Choose sensor directions with high between-row and low cycle-to-cycle variance."""
    g=baseline._regimes(x,head["centers"])
    z=np.clip((x[:,:-3]-head["means"][g])*head["inv_scale"][g],-6,6)
    ids=np.linspace(0,len(z)-1,min(len(z),32768),dtype=int)
    z0=z[ids]
    total=z0.T@z0/len(z0)
    innovations=[]
    for j in range(len(head["centers"])):
        subset=np.flatnonzero(g==j)
        gaps=np.diff(subset)
        # Equal regime and known causal label decrement reject joins between units.
        edges=(gaps<=32)&np.isclose(np.diff(y[subset]),-gaps)
        if edges.any():
            delta=np.diff(z[subset],axis=0)[edges]
            innovations.append(delta[::max(1,len(delta)//4096)])
    noise=np.eye(z.shape[1])
    if innovations and "pca" not in route:
        changes=np.concatenate(innovations)
        if len(changes)>=128:
            noise=.5*changes.T@changes/len(changes)+np.eye(z.shape[1])*.08
    val,axes=np.linalg.eigh(noise)
    white=axes/np.sqrt(np.maximum(val,.04))
    strengths,signal=np.linalg.eigh(white.T@total@white)
    rank=6 if "rank6" in route else 4
    use=np.flatnonzero(strengths>.08)[-rank:]
    if not len(use):return {}
    proj=white@signal[:,use]/np.sqrt(strengths[use]+.15)
    a=np.clip(z0@proj,-5,5)
    # Farthest-point cover of the occupied subspace across operating regimes.
    rng=np.random.default_rng(3741)
    sample=a[rng.choice(len(a),min(2048,len(a)),replace=False)]
    if "dense" in route:ncenters=64
    else:ncenters=32
    anchors=[np.median(sample,axis=0)]
    dist=np.sum((sample-anchors[0])**2,axis=1)
    for j in range(ncenters-1):
        # Distance sampling avoids devoting all centers to extreme noise.
        weights=np.minimum(dist,np.quantile(dist,.90))
        if weights.sum()<1e-8:break
        anchor=sample[rng.choice(len(sample),p=weights/weights.sum())]
        anchors.append(anchor)
        dist=np.minimum(dist,np.sum((sample-anchor)**2,axis=1))
    centers=np.asarray(anchors)
    if len(centers)<2:return {}
    dd=np.sum((centers[:,None,:]-centers[None,:,:])**2,axis=-1)
    width=max(.15,np.median(np.sort(dd,axis=1)[:,min(4,len(centers)-1)]))
    return dict(slow_axis=proj,slow_centers=centers,slow_width=np.asarray(width))


def basis(head,x,prior):
    g=baseline._regimes(x,head["centers"])
    z=np.clip((x[:,:-3]-head["means"][g])*head["inv_scale"][g],-6,6)
    a=np.clip(z@head["slow_axis"],-5,5)
    centers=head["slow_centers"]
    dd=np.sum(a*a,axis=1)[:,None]+np.sum(centers*centers,axis=1)-2*a@centers.T
    k=np.exp(-np.maximum(dd,0.)/head["slow_width"])
    t=np.clip((prior-100)/60,-2,4)
    return np.column_stack([np.ones(len(x)),a,k,k*t[:,None],t,t*t,np.eye(len(head["centers"]))[g]])


def calibrate(model,h,x,y,route):
    state=sensor_state(h,x,y,route)
    if not state:return h
    h.update(state)
    rng=np.random.default_rng(145)
    idx=np.arange(len(y))
    if len(y)>24576+1024:
        remain=np.ones(len(y),bool)
        remain[rng.choice(len(y),24576,replace=False)]=False
        idx=idx[remain]
    idx=rng.choice(idx,min(24576,len(idx)),replace=False)
    yy=y[idx];xx=x[idx];pred=old.predict(model,h,xx)
    design=basis(h,xx,pred)
    residual=yy-pred
    ridge=.003 if "loose" in route else .012
    weight=np.ones(len(idx))
    if "tail" in route:
        # Protect both signs of the exponential cost with capped empirical influence.
        cost=np.where(residual>0,residual/13.,-residual/10.)
        weight+=.16*np.exp(np.minimum(3.,cost/3.))
    gram=design.T@(weight[:,None]*design)/len(idx)+ridge*np.eye(design.shape[1])
    h["slow_coef"]=np.linalg.solve(gram,design.T@(weight*residual)/len(idx))
    h["slow_limit"]=np.array(24. if "wide" in route else 12.)
    return h


def prediction(model,h,x):
    p=old.predict(model,h,x)
    if "slow_coef" in h:
        adjustment=basis(h,x,p)@h["slow_coef"]
        p=np.maximum(0,p+np.clip(adjustment,-h["slow_limit"],h["slow_limit"]))
    return p


def fit(route,model,x,y,i,previous=None,stable_release=False):
    if route.startswith("maturity"):return maturity(model,x,y,i,previous,stable_release,route)
    if "joint" in route:
        initial=resampling("avg",model,x,y,i,previous,stable_release) if "avg" in route else v7.fit(model,x,y,i,previous,stable_release)
        return refine(model,initial,x,y,"wide" in route)
    if any(t in route for t in ("rotate","avg","nocurve","noise")):return resampling(route,model,x,y,i,previous,stable_release)
    if route=="v7":return v7.fit(model,x,y,i,previous,stable_release)
    if route=="frozen":return deployed.fit(model,x,y,i,previous,stable_release)
    h=old.fit(model,x,y,i,previous,stable_release)
    return h if route=="v5" else calibrate(model,h,x,y,route)


def score(route,m,h,x):
    if route=="frozen":return deployed.predict(m,h,x)
    if route.startswith("maturity") or route=="v7" or "joint" in route or any(t in route for t in ("rotate","avg","nocurve","noise")):return v7.predict(m,h,x)
    return prediction(m,h,x)


def probe(dataset,routes):
    x,y,hold,age,units,cruise,m=lab.evaluation_data(dataset)
    cuts=[1.,1.] if dataset=="official" else [.55,.7,.85,1.]
    prev={};rows=[]
    for i,cut in enumerate(cuts,1):
        use=~hold&(age<=cut)
        for r in routes:
            tick=process_time()
            prev[r]=fit(r,m,x[use],y[use],i,prev.get(r),dataset=="official" and i==2)
            pred=score(r,m,prev[r],x[hold]);e=pred-y[hold]
            met=lab.diagnostics.metrics(pred,y[hold],age[hold],cruise[hold] if cruise is not None else None)
            rows.append(dict(route=r,round=i,cpu=process_time()-tick,mae=float(np.mean(abs(e))),**met))
    (OUT/f"probe-{dataset}-{routes[-1]}.json").write_text(json.dumps(rows,indent=2))


def refine(m,head,x,y,wide=False):
    """Jointly calibrate four groups of hidden input weights on separate released rows."""
    ids=v7._reserve_rows(len(y),separate_ordinal=True)
    if len(ids)<256:return head
    xx,yy=x[ids],y[ids]
    g,hidden=v7._ordinal_hidden(m,head,xx)
    z=np.clip((xx[:,:-3]-head["means"][g])*head["inv_scale"][g],-6,6)
    base=m.predict(xx).reshape(-1)
    design=np.column_stack((z,(base-100.)/60,np.eye(len(head["centers"]))[g]))
    output=head["output"]
    moves=np.zeros((len(output),4))
    for j,neurons in enumerate(np.array_split(np.argsort(output),4)):
        moves[neurons,j]=output[neurons]/max(1.,np.linalg.norm(output[neurons]))
    raw=hidden@output+head["intercept"][0]
    slope=(np.abs(raw)<65.)*np.where(raw>0.,float(head["extension_fraction"]),1.)
    sens=((1.-hidden**2)@(moves*output[:,None]))*slope[:,None]
    jac=(sens[:,:,None]*design[:,None,:]).reshape(len(yy),-1)
    error=v7.predict(m,head,xx)-yy
    train=np.arange(len(ids))%2==0
    check=~train
    j,e=jac[train],error[train]
    scale=np.maximum(np.sqrt(np.mean(j*j,axis=0)),.25)
    j=j/scale
    gram=j.T@j/len(j)+np.eye(j.shape[1])*(.14 if wide else .38)
    step=np.linalg.solve(gram,-j.T@e/len(j))
    # First-order projections preserve both extreme positive and extreme negative tails.
    con=[]
    for sign,denom in ((1.,10.),(-1.,13.)):
        ex=sign*e/denom
        for q in (.98,.998):
            weight=sign*np.where(ex>0,np.exp(np.minimum(ex,np.quantile(ex,q))),0.)
            normal=weight@j/max(1.,np.abs(weight).sum())
            dual=np.linalg.solve(gram,normal)
            con.append((normal,dual,max(1e-10,normal@dual)))
    for _ in range(12):
        for normal,dual,d in con:step-=max(0.,normal@step)*dual/d
    step/=max(1.,np.sqrt(np.mean((j@step)**2))/6.)
    update=(step/scale).reshape(4,-1).T@moves.T
    n=z.shape[1]
    def risks(err):
        groups=np.searchsorted(np.quantile(yy[check],[.25,.5,.75]),yy[check])
        checkerr=err[check]
        return np.r_[[np.mean(checkerr[groups==i]**2) for i in np.unique(groups)],
                     [np.mean(np.expm1(np.minimum(20.,np.maximum(0.,sgn*checkerr/den))))
                      for sgn,den in ((1.,10.),(-1.,13.))]]
    safe=risks(error)
    for fraction in ((1.,.5,.25) if wide else (.6,.3,.15)):
        d=update*fraction
        trial=dict(head,w_sensor=head["w_sensor"]+d[:n],
                   w_base=head["w_base"]+d[n]/60.,bias=head["bias"]+d[n+1:]-d[n]*100./60.)
        e=v7.predict(m,trial,xx)-yy
        risk=risks(e)
        if (np.mean(e[check]**2)<np.mean(error[check]**2) and
            np.mean(abs(e[check]))<=np.mean(abs(error[check]))*1.001 and
            np.all(risk[:-2]<=safe[:-2]*1.018) and np.all(risk[-2:]<=safe[-2:]*1.015)):
            return trial
    return head


def maturity(m,x,y,i,previous,stable_release,route):
    # Only nearly complete observed trajectories warrant weighting late iterates:
    # aggressively truncated and high-frequency repeated-cycle streams retain the last iterate.
    steps=np.diff(y)
    consecutive=np.isclose(steps,-1.,atol=1e-6)
    blend=False
    if len(y)>=512 and consecutive.mean()>=.93:
        starts=np.r_[0,1+np.flatnonzero(~consecutive)]
        ends=np.r_[starts[1:],len(y)]
        valid=(ends-starts)>=16
        if valid.sum()>=8:
            fraction=np.median((y[ends-1]/np.maximum(1.,y[starts]))[valid])
            blend=.005<=fraction<.16 or (i==1 and fraction<.005)
    if stable_release:blend=False
    if blend:head=resampling('avg' if 'full' in route else 'halfavg',m,x,y,i,previous,stable_release)
    else:head=v7.fit(m,x,y,i,previous,stable_release)
    if 'cal' in route and stable_release:return refine(m,head,x,y)
    return head


def resampling(route,m,x,y,i,previous,stable_release):
    """Fixed-budget minibatches, fresh examples per epoch, optional averaging of late iterates."""
    import inspect
    source=inspect.getsource(v7.fit_neural)
    if 'rotate' in route:
        source=source.replace('ids = rng.choice(len(x), limit, replace=False)','''first = rng.choice(len(x), limit, replace=False)
        allowed = np.ones(len(x), dtype=bool)
        allowed[_reserve_rows(len(x))] = False
        if stable:
            allowed[_reserve_rows(len(x), separate_ordinal=True)] = False
        ids = np.flatnonzero(allowed)''')
        source=source.replace('order = rng.permutation(len(y))','order = rng.choice(len(y), min(len(y), limit), replace=False)')
        source=source.replace('len(y) // 256','len(order) // 256')
    if 'avg' in route:
        source=source.replace('step = 0','averaged = [np.zeros_like(a) for a in params]\n    snapshots = 0\n    step = 0')
        source=source.replace('    # Keep deployment','''        if epoch >= epochs * .50:
            for avg, current in zip(averaged, params):
                avg += current
            snapshots += 1
    for param, avg in zip(params, averaged):
        param[:] = avg / snapshots
    # Keep deployment''')
    if 'half' in route:source=source.replace('param[:] = avg / snapshots','param[:] = .5*param + .5*(avg/snapshots)')
    if 'nocurve' in route:
        source=source.replace('if len(active) >= 6:', 'if False:')
    if 'noise' in route:
        source=source.replace('batch = design[idx]', '''batch = design[idx].copy()
            batch[:, :sensor_count] += rng.normal(0., .08, (len(idx), sensor_count)).astype(np.float32)''')
    scope=dict(v7.__dict__)
    exec(source,scope)
    neural=v7.fit_neural
    v7.fit_neural=scope['fit_neural']
    try:return v7.fit(m,x,y,i,previous,stable_release)
    finally:v7.fit_neural=neural


def replay(route,n,mode):
    assert 0<=n<120
    b=load('v7benchmark',HERE.parent/'V7'/'benchmark.py')
    baseline.fit=lambda m,x,y,i,previous=None,stable_release=False:fit(route,m,x,y,i,previous,stable_release)
    baseline.predict=lambda m,h,x:score(route,m,h,x)
    baseline.prepare_inference=v7.prepare_inference if route!='frozen' else deployed.prepare_inference
    b.OUT=OUT/route
    b.OUT.mkdir(exist_ok=True)
    b.run('C','v6',n,mode)

def verify(route):
    """Check frozen-vs-research predictions and scalar serving including inherited heads."""
    import os
    from statistics import median
    if hasattr(os,'sched_setaffinity'):os.sched_setaffinity(0,{max(os.sched_getaffinity(0))})
    x,y,hold,age,units,_,m=lab.evaluation_data("official")
    test=x[hold][np.linspace(0,hold.sum()-1,600,dtype=int)]
    stress=test[:48].copy()
    stress[:,[-3,-1]]=np.rint(stress[:,[-3,-1]])+.5
    stress[:,:5]*=10.
    prior={};out=[]
    for i,(limit,stable) in enumerate(((.96,False),(.96,True),(1.,False),(1.,True)),1):
        tr=(~hold)&(age<=limit)
        predictions={}
        for r in ('v5','v7',route,'frozen'):
            tick=process_time()
            prior[r]=fit(r,m,x[tr],y[tr],i,prior.get(r),stable)
            elapsed=process_time()-tick
            h=prior[r]
            ref=score(r,m,h,test)
            fast=deployed.prepare_inference(m,h)
            err=0.
            for features in (test,stress):
                expected=score(r,m,h,features)
                actual=np.array([fast(row) for row in features])
                np.testing.assert_allclose(actual,expected,rtol=1e-10,atol=1e-8)
                err=max(err,float(np.max(abs(expected-actual))))
            timers={'frozen_serving':[],'original_serving':[]}
            oldfn=(old if r=='v5' else v7).prepare_inference(m,h)
            for j in range(7):
                steps=[('original_serving',oldfn),('frozen_serving',fast)]
                for k,fn in (steps if j%2 else reversed(steps)):
                    start=process_time()
                    for row in test:fn(row)
                    timers[k].append((process_time()-start)*1e6/len(test))
            predictions[r]=ref
            out.append(dict(round=i,route=r,train=len(y[tr]),fit_s=elapsed,scalar_us={k:median(a) for k,a in timers.items()},max_abs=err,bytes=sum(np.asarray(v).nbytes for v in h.values())))
        np.testing.assert_allclose(predictions[route],predictions['frozen'],rtol=0,atol=1e-10)
    (OUT/'verification.json').write_text(json.dumps(out,indent=2))


def paired_units(dataset):
    """Cluster-bootstrap final errors by whole validation engine."""
    x,y,hold,age,units,_,model=lab.evaluation_data(dataset)
    cuts=[1.,1.] if dataset=='official' else [.55,.7,.85,1.]
    group=np.unique(units[hold],return_inverse=True)[1]
    size=np.bincount(group)
    draws=np.random.default_rng(4930).integers(0,len(size),(1200,len(size)))
    result={};fitted={}
    for route in ('v5','v7','frozen'):
        previous=None
        for i,cut in enumerate(cuts,1):
            use=(~hold)&(age<=cut)
            previous=fit(route,model,x[use],y[use],i,previous,dataset=='official' and i==2)
        err=score(route,model,previous,x[hold])-y[hold]
        statistics=np.column_stack([np.bincount(group,weights=element) for element in (err*err,abs(err),np.expm1(np.maximum(err/10.,-err/13.)))])
        sample=statistics[draws].sum(axis=1)/size[draws].sum(axis=1)[:,None]
        sample[:,0]=np.sqrt(sample[:,0])
        fitted[route]=sample
    for comparator in ('v5','v7'):
        delta=fitted['frozen']-fitted[comparator]
        delta[:,2]=fitted['frozen'][:,2]/fitted[comparator][:,2]-1.
        result[comparator]={key:np.quantile(delta[:,i],[.025,.5,.975]).tolist() for i,key in enumerate(('rmse_difference','mae_difference','nasa_relative_change'))}
    (OUT/('bootstrap-'+dataset+'.json')).write_text(json.dumps(dict(units=len(size),rows=int(hold.sum()),comparisons=result),indent=2))


if __name__=="__main__":
    OUT.mkdir(parents=True,exist_ok=True)
    old=load("oldengine",V5/"engine.py")
    v7=load("prevengine",HERE.parent/"V7/engine.py")
    deployed=load("deployengine",HERE/"engine.py")
    if sys.argv[1]=="verify":verify(sys.argv[2])
    if sys.argv[1]=="ci":paired_units(sys.argv[2])
    if sys.argv[1]=="replay":replay(sys.argv[2],int(sys.argv[4]),sys.argv[3])
    if sys.argv[1]=="probe":probe(sys.argv[2],sys.argv[3].split(","))
