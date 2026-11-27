"""Summarize paired single-CPU resource tests, complete release metrics and real SDK latency."""
import json
from pathlib import Path
import statistics as st
import numpy as np

import benchmark
import contest_submission as harness
import serving_study

OUT = Path(__file__).resolve().parents[1]/"work/v6"

def read(path):
    return json.loads(path.read_text())

def main():
    sdk=read(OUT/"sdk-latency.json")
    result={}
    for k in "ABC":
        val=serving_study.rows(k,"validation")
        result[k]={}
        for v in ("v5","v6"):
            # The serving ABI is backward compatible, verified by the paired SDK check.
            folder=OUT/"serving-assets"/v/k
            model,head,_=harness.artifact(folder)
            x,y,_=harness.projection(val,model,k)
            pred=harness.values(k,model,head,x)
            rows=[r for r in sdk["rows"] if r["task"]==k and r["source"]==v and r["runtime"]==v]
            assert len(rows)==5
            runs=[read(OUT/"resources"/str(i)/f"drift-{v}-{k}-0.json")["rounds"] for i in range(3)]
            data=dict(first_release=read(folder/"record.json"), metric=harness.metric(k,y,pred),
                      **{m:st.median(r[m]for r in rows) for m in ("cold_process_ms","p50_us","p95_us","p99_us")},
                      **{m:st.median(sum(r[m]for r in run) for run in runs) for m in ("cpu","checkpoint_bytes","calls","prompt_bytes")})
            if k=="C":
                er=pred[:,0]-y
                data.update(nasa_sum=float(np.expm1(np.maximum(er/10.,-er/13.)).sum()),mae=float(np.abs(er).mean()))
            result[k][v]=data
    (OUT/"resource-summary.json").write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))

if __name__=="__main__":
    main()
