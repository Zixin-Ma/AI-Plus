"""Alternate otherwise matched images and measure process creation to first SDK reply."""
import json
import os
from pathlib import Path
from statistics import median
from tempfile import TemporaryDirectory

import numpy as np
from sdk_latency import WORK, PUBLIC, export_image, warm_and_measure

def main():
    os.sched_setaffinity(0,{sorted(os.sched_getaffinity(0))[-2]})
    files={'plain':WORK/'v5/image-controls/plain/harness.tar','cache':WORK/'v5/delivery/harness.tar'}
    rows=[]
    with TemporaryDirectory(dir=WORK/'v5') as tmp:
        roots={k:Path(tmp)/k for k in files}
        for k,root in roots.items():
            export_image('v5',root,files[k])
        for key in 'ABC':
            task='classification' if key=='A' else 'regression'
            with (PUBLIC/key/'public/validation.jsonl').open() as stream:
                lines=[json.dumps({'task':task,'inputs':json.loads(next(stream))['inputs']}).encode()+b'\n' for _ in range(40)]
            for rep in range(7):
                both={}
                for variant in (('plain','cache') if rep%2==0 else ('cache','plain')):
                    timing,pred=warm_and_measure(roots[variant],'v5',key,lines)
                    both[variant]=pred
                    rows.append(dict(task=key,repeat=rep,variant=variant,**timing))
                np.testing.assert_array_equal(both['plain'],both['cache'])
    dst=WORK/'v5/cold-start.json'
    dst.write_text(json.dumps(dict(sizes={k:p.stat().st_size for k,p in files.items()},runs=rows),indent=2))
    for key in 'ABC':
        print(key,[median(r['cold_process_ms'] for r in rows if r['task']==key and r['variant']==k) for k in files])

if __name__=='__main__':
    main()
