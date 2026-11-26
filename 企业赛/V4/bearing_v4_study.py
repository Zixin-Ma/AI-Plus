"""Paired B regression check, deployment profiling and public drift ablations for V4."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np


HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
OUT=HERE.parent/"work/v4/bearing"
OUT.mkdir(exist_ok=True, parents=True)


def load(name, path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module



def student_fit(model,x,y,index,teacher_fit,teacher_predict):
    """Compress only the released-data teacher residual into a single affine feature layer."""
    teacher=teacher_fit(model,x,y,index)
    ids=np.random.default_rng(2027).choice(len(y),size=min(len(y),1536),replace=False)
    subset=x[ids]
    z=(np.sign(subset)*np.log1p(abs(subset))-teacher["center"])/teacher["scale"]
    design=np.column_stack((np.ones(len(z)),np.clip(z,-6.,6.)))
    residual=teacher_predict(model,teacher,subset)-model.predict(subset)
    gram=design.T@design+.008*len(z)*np.eye(design.shape[1])
    beta=np.linalg.solve(gram,design.T@residual)
    return {"student_center":teacher["center"],"student_scale":teacher["scale"],"student_beta":beta}


def student_predict(model,head,x):
    z=np.clip((np.sign(x)*np.log1p(abs(x))-head["student_center"])/head["student_scale"],-6.,6.)
    beta=head["student_beta"]
    return np.maximum(0.,model.predict(x)+beta[0]+z@beta[1:])


def profile_student():
    from time import process_time_ns
    import bearing
    import contest_submission as cs
    from industrial import Model
    fixtures=load("student_data",HERE.parent/"V3/bearing_v3_study.py")
    model=Model.load(fixtures.loader.DATA/"checkpoint.npz")
    x,y,_=fixtures.loader.load("train",model); q,t,groups=fixtures.loader.load("validation",model)
    cs.continue_mlp("B",1,model,x,y,"half")
    teacher=bearing.fit(model,x,y,1); full=bearing.prepare_inference(model,teacher)
    h=student_fit(model,x,y,1,bearing.fit,bearing.predict)
    beta=h["student_beta"]
    def student(a):
        z=(np.copysign(np.log1p(abs(a)),a)-h["student_center"])/h["student_scale"]
        np.maximum(z,-6.,out=z);np.minimum(z,6.,out=z)
        return max(0.,model.predict_one(a)+beta[0]+z@beta[1:])
    out={}
    for name,fn,state in (("teacher",full,teacher),("student",student,h)):
        pred=np.array([fn(r) for r in q])
        elapsed=[]
        for _ in range(7):
            start=process_time_ns()
            for r in q:fn(r)
            elapsed.append((process_time_ns()-start)/1e3/len(q))
        out[name]={"hot_us":float(np.median(elapsed)),"head_bytes":sum(a.nbytes for a in state.values()),
                   "public_rmse":float(np.mean((pred-t)**2)**.5),
                   "per_device_rmse":{str(g):float(np.mean((pred[groups==g]-t[groups==g])**2)**.5)
                                      for g in np.unique(groups)}}
    (OUT/"student.json").write_text(json.dumps(out,indent=2))


def replay(route, split, setting):
    if split not in range(8):
        raise ValueError("The technical study is limited to previously opened splits 0–7.")
    bench=load("v4_benchmark",HERE/"benchmark.py")
    # Isolate this ablation from concurrent changes to the V4 controller.
    sys.path.insert(0,str(HERE.parent/"V3"))
    import contest_submission as cs
    source=(HERE.parent/"V3/bearing.py") if route=="reference" else (HERE/"bearing.py")
    expert=load("candidate_bearing",source)
    sys.modules["bearing"]=expert
    if route=="student":
        fit, predict=expert.fit,expert.predict
        expert.fit=lambda m,x,y,k:student_fit(m,x,y,k,fit,predict)
        expert.predict=lambda m,h,x:student_predict(m,h,x) if "student_beta" in h else predict(m,h,x)
    elif route=="weight":
        cs.WEIGHTS["B"]=(1.,.85,.65,.35,.15)
    elif route=="support":
        orig=expert.predict
        def predict(model,head,x):
            current=orig(model,head,x)
            if "curve_mature" not in head or not head["curve_mature"]:
                return current
            # Partial pooling weakens the correction in the region between normal and high impacts.
            cols=head["curve_bands"][2:]
            impact=np.log(np.maximum(x[:,cols],1e-12)).mean(1)
            lo,hi=head["curve_impact"]
            gate=np.clip((impact-lo)/max(hi-lo,1e-8),0,1)
            prior=expert._legacy_prediction(model,head,x,model.predict(x))
            return current+.5*gate*(prior-current)
        expert.predict=predict
    elif route not in ("reference","frozen"):
        raise ValueError(route)
    bench.OUT=OUT/route
    bench.run("B","v4",split,setting)


def sweep(routes):
    def job(request):
        route,split,setting=request
        path=OUT/route/f"{setting}-v4-B-{split}.json"
        if path.exists():
            return
        with (OUT/f"{route}-{setting}-{split}.log").open("w") as log:
            subprocess.run([sys.executable,"-B",str(Path(__file__).resolve()),"replay",route,str(split),setting],
                           stdout=log,stderr=subprocess.STDOUT,check=True)
    jobs=[(route,split,setting) for route in routes for split in range(8)
          for setting in ("drift","official","fresh")]
    with ThreadPoolExecutor(max_workers=4) as executor:
        list(executor.map(job,jobs))




def kaist_samples(model,fixtures):
    import io
    from datetime import datetime
    from itertools import islice
    from zipfile import ZipFile
    from industrial import dataset
    local=OUT/"kaist-finite-hourly.npz"
    previous=HERE.parent/"work/v3/bearing/kaist-finite-hourly.npz"
    for cache in (local,previous):
        if cache.is_file():
            with np.load(cache,allow_pickle=False) as data:
                return data["x"],data["y"]
    archive=next((ROOT/"Supp_Datasets/B/02_KAIST_Run_to_Failure").glob("*.zip"))
    rows=[]; timestamps=[]
    with ZipFile(archive) as z:
        for name in sorted(n for n in z.namelist() if n.endswith(".csv")):
            timestamps.append(datetime.strptime(Path(name).stem,"LogFile_%Y-%m-%d-%H-%M-%S"))
            with z.open(name) as member:
                wave=np.loadtxt(io.BytesIO(b"".join(islice(member,16384))),delimiter=",",usecols=(0,1))
            edges=np.flatnonzero(np.diff(np.r_[False,np.isfinite(wave).all(1),False]))
            start=next(a for a,b in edges.reshape(-1,2) if b-a>=8192)
            rows.append({"inputs":fixtures.loader.extract_features(wave[start:start+8192],25600.),"answer":0.})
    x,_=dataset(rows,model)
    y=np.array([(timestamps[-1]-t).total_seconds()/60 for t in timestamps])
    np.savez_compressed(local,x=x,y=y)
    return x,y


def audit():
    import hashlib
    from time import process_time_ns
    import bearing as fast
    import guard
    import serving
    import contest_submission as agent
    from industrial import Model
    old=load("v3_bearing_reference",HERE.parent/"V3/bearing.py")
    fixtures=load("bearing_study_v3",HERE.parent/"V3/bearing_v3_study.py")
    fixtures.OUT=OUT
    template=Model.load(fixtures.loader.DATA/"checkpoint.npz")
    model=Model.load(fixtures.loader.DATA/"checkpoint.npz")
    train,life,ids=fixtures.loader.load("train",model)
    valid,y,dev=fixtures.loader.load("validation",model)
    agent.continue_mlp("B",1,model,train,life,"half")
    xf,yf,gf=fixtures.external(model)
    # Reuse the V3 source-side extraction. Do not refit using target-platform endpoints.
    xk,yk=kaist_samples(model,fixtures)
    data={"train":train,"public":valid,"FEMTO":xf,"KAIST":xk}
    result={"sha256":hashlib.sha256((HERE/"bearing.py").read_bytes()).hexdigest(),
            "samples":{k:len(v) for k,v in data.items()},"heads":{}}

    def check(name,head,samples):
        runner=fast.prepare_inference(model,head)
        report={}
        for part,query in samples.items():
            if not len(query):continue
            batch=old.predict(model,head,query)
            actual=np.array([runner(row) for row in query])
            delta=abs(actual-batch)
            np.testing.assert_allclose(actual,batch,atol=2e-9,rtol=1e-11)
            report[part]={"rows":len(query),"max_abs":float(delta.max())}
        result["heads"][name]=report
        return runner

    show=train[::41]
    subs={"mature":life>=0,"partial":life>55,"unseen_long_device":ids!="Bearing3_2"}
    for label,mask in subs.items():
        state=fast.fit(model,train[mask],life[mask],1)
        v3state=old.fit(model,train[mask],life[mask],1)
        assert set(state)==set(v3state) and all(np.array_equal(state[k],v3state[k]) for k in state)
        check(label,state,data)
        result["heads"][label]["fit_arrays_identical"]=True
    head=fast.fit(model,train,life,1)
    check("femto_trained",fast.fit(model,xf,yf,1),{"FEMTO":xf,"XJTU":show,"KAIST":xk})
    for label,src in (("campus",ROOT/"校赛/V2/bearing.py"),("v1",HERE.parent/"V1/bearing.py"),
                      ("v2",HERE.parent/"V2/bearing.py")):
        legacy=load("legacy_bearing_"+label,src).fit(model,train,life,1)
        check(label,legacy,{"public":valid,"FEMTO":xf,"KAIST":xk})
    assert fast.prepare_inference(model,{}) is None
    assert fast.prepare_inference(model,{"backbone_only":np.asarray(1)}) is None

    # Repeated curve cut values, fitted-tail boundaries, zero and negative bands,
    # spectral half-harmonic rounding, and a constant high-impact interval.
    edge=[np.zeros(train.shape[1]),np.full(train.shape[1],-1.)]
    for base in (valid[0],valid[-1]):
        for peak in (0.,-10.,1.7,17.5,35.,52.5,75.,99.999,10000.):
            for freq in (np.nextafter(peak,-np.inf),peak,np.nextafter(peak,np.inf)):
                row=base.copy()
                row[head["order_cols"]]=[freq,freq]
                edge.append(row)
        for value in head["curve_impact"]:
            for offset in (-1e-9,0.,1e-9):
                row=base.copy();row[head["curve_bands"][2:]]=np.exp(value+offset)
                edge.append(row)
        for knots in head["curve_cuts"]:
            for value in knots:
                row=base.copy();row[head["curve_bands"][:2]]=np.exp(value)
                edge.append(row)
    edge=np.asarray(edge)
    check("finite_edges",head,{"synthetic":edge})
    deg=dict(head)
    deg["curve_cuts"]=np.repeat(head["curve_cuts"][:,:1],13,axis=1)
    lo=float(head["curve_impact"][0])
    for gap in (0.,1e-10):
        deg["curve_impact"]=np.array([lo,lo+gap])
        check("degenerate_"+str(gap),deg,{"synthetic":edge})

    result["serving"]={}
    for wt in (0.,.15,.65,1.):
        candidate=guard.freeze(dict(head,life_offset=np.asarray(-12.5)),template,wt)
        infer=serving.prepare("B",model,candidate)
        expected=agent.values("B",model,candidate,valid)[:,0]
        value=np.array([infer(z)["value"] for z in valid])
        np.testing.assert_allclose(value,expected,rtol=1e-11,atol=2e-9)
        result["serving"][str(wt)]=float(abs(value-expected).max())
    rejected=0
    for wrong in (np.r_[valid[0],1.],np.full(valid.shape[1],np.nan),np.full(valid.shape[1],np.inf)):
        try:infer(wrong)
        except ValueError:rejected+=1
    assert rejected==3
    result["serving_invalid_rejected"]=rejected
    result["resource"]={}
    def clearing(module):
        for name in ("_feature_map","_single_evidence","_pair_weights","_scalar_order_scores","_scalar_order_weights"):
            if hasattr(module,name):getattr(module,name).cache_clear()
    queries=valid[np.random.default_rng(503).permutation(len(valid))]
    for label,mod in (("v3",old),("v4",fast)):
        fits=[];cold=[];hot=[];cost_full=[];load_cost=[]
        for repeat in range(7):
            tic=process_time_ns(); hh=mod.fit(model,train,life,1);fits.append((process_time_ns()-tic)/1e6)
            for i in range(24):
                clearing(mod)
                tic=process_time_ns()
                run=(mod.prepare_inference(model,hh) if label=="v4" else lambda z:float(mod.predict(model,hh,z[None])[0]))
                load_cost.append((process_time_ns()-tic)/1e3)
                run(queries[repeat+i])
                cold.append((process_time_ns()-tic)/1e3)
            for z in queries:run(z)
            tic=process_time_ns()
            for z in queries:run(z)
            hot.append((process_time_ns()-tic)/1e3/len(queries))
            # Time the shared row-level validation and a reference mixture with / without the B hook.
            mixed=guard.freeze(dict(hh,life_offset=np.asarray(-12.5)),template,.65)
            infer=serving.prepare("B",model,mixed,use_expert_hook=(label=="v4"))
            for z in queries:infer(z)
            tic=process_time_ns()
            for z in queries:infer(z)
            cost_full.append((process_time_ns()-tic)/1e3/len(queries))
        result["resource"][label]={"fit_ms":float(np.median(fits)),
                                   "prepare_us":float(np.median(load_cost)),
                                   "cold_prepare_plus_first_us":float(np.median(cold)),
                                   "hot_us":float(np.median(hot)),
                                   "checked_mixed_us":float(np.median(cost_full)),
                                   "head_bytes":sum(x.nbytes for x in hh.values())}
    (OUT/"audit.json").write_text(json.dumps(result,indent=2))
    print(json.dumps({k:result[k] for k in ("samples","resource","serving","sha256")},indent=2))





def entry_cost():
    import tempfile
    from time import process_time_ns
    from types import SimpleNamespace
    import bearing
    import contest_submission as v4
    import guard
    from industrial import Model
    seeds=load("endpoint_data",HERE.parent/"V3/bearing_v3_study.py")
    v3=load("old_endpoint",HERE.parent/"V3/contest_submission.py")
    old_head=load("old_endpoint_head",HERE.parent/"V3/bearing.py")
    anchor=Model.load(seeds.loader.DATA/"checkpoint.npz");model=Model.load(seeds.loader.DATA/"checkpoint.npz")
    x,y,_=seeds.loader.load("train",model)
    v4.continue_mlp("B",1,model,x,y,"half")
    state=bearing.fit(model,x,y,1)
    with (seeds.loader.DATA/"public"/"validation.jsonl").open() as file:
        rows=[SimpleNamespace(inputs=a["inputs"],task="regression",dataset_key="B") for a in map(json.loads,file)]
    mixed=guard.freeze(dict(state,life_offset=np.asarray(-12.5)),anchor,.65)
    out={}
    with tempfile.TemporaryDirectory(dir=OUT) as tmp:
        folder=Path(tmp)
        model.save(folder/"checkpoint.npz");np.savez_compressed(folder/"head.npz",**mixed)
        expected=None
        for name,agent,head in (("v3",v3,old_head),("v4",v4,bearing)):
            sys.modules["bearing"]=head
            cold=[];hot=[]
            for repeat in range(7):
                agent.artifact.cache_clear()
                if hasattr(agent,"prepared_artifact"):agent.prepared_artifact.cache_clear()
                for fn in ("_feature_map","_scalar_order_scores","_scalar_order_weights","_single_evidence","_pair_weights"):
                    if hasattr(head,fn):getattr(head,fn).cache_clear()
                start=process_time_ns()
                json.dumps(agent.predict(rows[repeat],folder),allow_nan=False)
                cold.append((process_time_ns()-start)/1e3)
                for row in rows:agent.predict(row,folder)
                start=process_time_ns()
                predictions=[agent.predict(row,folder) for row in rows]
                for output in predictions:json.dumps(output,allow_nan=False)
                hot.append((process_time_ns()-start)/1e3/len(rows))
            vector=np.asarray([entry["value"] for entry in predictions])
            if expected is None:expected=vector
            np.testing.assert_allclose(vector,expected,atol=2e-9,rtol=1e-11)
            us=float(np.median(hot))
            out[name]={"rounds":7,"samples":len(rows),"hot_us":us,"samples_per_cpu_second":1e6/us,
                       "cold_artifact_and_first_us":float(np.median(cold)),
                       "max_abs_delta":float(abs(vector-expected).max())}
    (OUT/"entry.json").write_text(json.dumps(out,indent=2))


def external_scores():
    import bearing
    import contest_submission as cs
    from industrial import Model
    old=load("fold_reference",HERE.parent/"V3/bearing.py")
    fixtures=load("fold_data",HERE.parent/"V3/bearing_v3_study.py")
    fixtures.OUT=OUT
    template=Model.load(fixtures.loader.DATA/"checkpoint.npz")
    results={}
    for tag,data in (("XJTU",fixtures.loader.load("train",template)),("FEMTO",fixtures.external(template))):
        x,y,device=data
        folds={}
        for name in np.unique(device):
            training=device!=name
            model=Model("regression",template.meta["paths"],[],x[training],y[training],hidden=24)
            model.fit(x[training],y[training],15)
            cs.continue_mlp("B",1,model,x[training],y[training],"half")
            h3=old.fit(model,x[training],y[training],1)
            h4=bearing.fit(model,x[training],y[training],1)
            assert all(np.array_equal(h3[k],h4[k]) for k in h3)
            fn=bearing.prepare_inference(model,h4)
            previous=old.predict(model,h3,x[~training])
            current=np.array([fn(row) for row in x[~training]])
            np.testing.assert_allclose(current,previous,atol=2e-9,rtol=1e-11)
            folds[str(name)]={"v3_rmse":float(np.mean((previous-y[~training])**2)**.5),
                              "v4_rmse":float(np.mean((current-y[~training])**2)**.5),
                              "max_abs_prediction_diff":float(abs(previous-current).max())}
            if tag=="FEMTO":
                observed=y[training]
                record=folds[str(name)]
                record["near_end_fraction"]=float(np.mean(observed<=.02*np.quantile(observed,.9)))
                record["deployed_mature"]=bool(h3["curve_mature"])
                alt=fixtures.route_fit(model,x[training],observed,1,"impact-full-0.75")
                alt_pred=fixtures.route_predict(model,alt,x[~training])
                record["old_research_rmse"]=float(np.mean((alt_pred-y[~training])**2)**.5)
        keys=("v3_rmse","v4_rmse")
        results[tag]={"folds":folds,"mean":[float(np.mean([f[k] for f in folds.values()])) for k in keys],
                      "worst":[max(f[k] for f in folds.values()) for k in keys]}
    model=Model.load(fixtures.loader.DATA/"checkpoint.npz")
    x,y,_=fixtures.loader.load("train",model)
    cs.continue_mlp("B",1,model,x,y,"half")
    h=bearing.fit(model,x,y,1); predict=bearing.prepare_inference(model,h)
    xx,yy=kaist_samples(model,fixtures)
    v3=old.predict(model,h,xx);v4=np.array([predict(r) for r in xx])
    results["KAIST"]={"samples":len(yy),"v3_rmse":float(np.mean((v3-yy)**2)**.5),
                      "v4_rmse":float(np.mean((v4-yy)**2)**.5),"max_abs_prediction_diff":float(abs(v3-v4).max())}
    (OUT/"external.json").write_text(json.dumps(results,indent=2))


def confirm():
    fixtures=load("v3_study",HERE.parent/"V3/bearing_v3_study.py")
    fixtures.OUT=OUT
    import bearing
    old=load("v3_reference",HERE.parent/"V3/bearing.py")
    result={"complete":{},"replays":{}}
    for label,module in (("v3",old),("v4",bearing)):result["complete"][label]=fixtures.fixed_round(module)
    for setting in ("drift","official","fresh"):
        max_diff=0.
        equal_selection=0
        for split in range(8):
            prior=json.loads((OUT/"reference"/f"{setting}-v4-B-{split}.json").read_text())
            revised=json.loads((OUT/"frozen"/f"{setting}-v4-B-{split}.json").read_text())
            for a,b in zip(prior["rounds"],revised["rounds"],strict=True):
                equal_selection += int((a["expert_weight"],a["head_family"],a["deployed"])==
                                       (b["expert_weight"],b["head_family"],b["deployed"]))
                max_diff=max(max_diff, max(abs(np.asarray(a["metrics"])-b["metrics"])))
            for k in ("mean","worst","forget","delay"):
                np.testing.assert_allclose(prior[k],revised[k],rtol=1e-12,atol=1e-12)
        assert equal_selection==80 and max_diff<1e-8
        result["replays"][setting]={"matching_round_choices":equal_selection,"max_stage_rmse_difference":max_diff}
    (OUT/"confirmed.json").write_text(json.dumps(result,indent=2))


if __name__ == "__main__":
    if sys.argv[1]=="entry": entry_cost()
    elif sys.argv[1]=="student": profile_student()
    elif sys.argv[1]=="external": external_scores()
    elif sys.argv[1]=="confirm": confirm()
    elif sys.argv[1]=="audit": audit()
    elif sys.argv[1]=="sweep": sweep(sys.argv[2].split(","))
    elif sys.argv[1]=="replay": replay(sys.argv[2],int(sys.argv[3]),sys.argv[4])
