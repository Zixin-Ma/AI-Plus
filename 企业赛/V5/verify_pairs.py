"""Independently reconstruct version-paired public outcomes from four audit panels per round."""
import collections
import importlib.util
import json
import math
from pathlib import Path
import statistics

import qualify

HERE=Path(__file__).resolve().parent
WORK=qualify.OUT
spec=importlib.util.spec_from_file_location('v4_independent_auditor',HERE.parent/'V4/verify_pairs.py')
old=importlib.util.module_from_spec(spec)
spec.loader.exec_module(old)
ORDERS=(*old.DEVELOPMENT_ORDERS,(2,0,2,3,1,1,0,3,2,1),(1,3,3,0,2,1,0,2,3,0),(0,3,1,2,2,0,1,3,1,0),
        (2,2,0,3,2,1,0,0,3,1),(3,1,1,0,2,1,3,0,2,2),(1,0,3,3,2,1,0,2,0,1))

def main():
    old.self_test()
    counts=collections.Counter()
    b_different=0
    max_b_delta=0.
    residual=0.
    rows=[]
    for part in ('dev','confirm'):
        for key,ver,idx,setting in qualify.matrix(part):
            if ver!='v4':
                continue
            records=[json.loads(qualify.path_for(setting,v,key,idx).read_text()) for v in ('v4','v5')]
            scores=[]
            for data in records:
                summary,error=old.stream_audit(data,key,ORDERS[idx])
                residual=max(error,residual)
                scores.append(summary)
            for left,right in zip(*(d['rounds'] for d in records)):
                if any(left[k]!=right[k] for k in ('stage','train','selection','audit','static')):
                    raise ValueError('Input or baseline mismatch')
                if key=='B':
                    delta=max(abs(x-y) for x,y in zip(left['metrics'],right['metrics']))
                    max_b_delta=max(max_b_delta,delta)
                    b_different+=int(delta>1e-8 or left['deployed']!=right['deployed'])
            counts[part]+=10
            row=dict(part=part,setting=setting,task=key,split=idx,v4=scores[0],v5=scores[1])
            if key=='C':
                row['mean_per_round_nasa']=[statistics.mean(r['nasa_mean'] for r in d['rounds']) for d in records]
            rows.append(row)
    result=dict(status='passed',paired_rounds=dict(counts),recorded_rounds=2*sum(counts.values()),
                pairs=len(rows),max_metric_residual=residual,bearing_changed_pairs_of_rounds=b_different,
                bearing_max_metric_difference=max_b_delta,streams=rows)
    (WORK/'verification.json').write_text(json.dumps(result,indent=2))
    print({k:v for k,v in result.items() if k!='streams'})

if __name__=='__main__':
    main()
