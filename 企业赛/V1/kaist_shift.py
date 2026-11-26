"""Chronological vibration-bin coverage on a native KAIST run; no constructed RUL labels."""
from datetime import datetime
from pathlib import Path
import io
import json
import zipfile
import numpy as np


def main():
    root=Path(__file__).resolve().parents[2]
    folder=root/"Supp_Datasets/B/02_KAIST_Run_to_Failure"
    summary=[]
    with zipfile.ZipFile(next(folder.glob("*.zip"))) as archive:
        files=sorted(n for n in archive.namelist() if n.startswith("LogFile_") and n.endswith(".csv"))
        for name in files:
            stamp=datetime.strptime(Path(name).stem,"LogFile_%Y-%m-%d-%H-%M-%S")
            with archive.open(name) as stream:
                block=np.loadtxt(io.BytesIO(b"".join(stream.readline() for _ in range(4096))),
                                 delimiter=",",usecols=(0,1),ndmin=2)
            if not np.isfinite(block).all():
                raise ValueError(name)
            amplitude=np.sqrt(np.mean((block-block.mean(0))**2,axis=0)).max()
            summary.append((stamp.isoformat(),float(amplitude),len(block)))
    split=len(summary)//2
    values=np.array([a for _,a,_ in summary])
    cuts=np.quantile(values[:split],[.25,.5,.75])
    bins=np.searchsorted(cuts,values)
    result=dict(records=len(values),sample_count=sum(s for _,_,s in summary),
                first=summary[0][0],last=summary[-1][0],train_cuts=cuts.tolist(),
                early_counts=np.bincount(bins[:split],minlength=4).tolist(),
                late_counts=np.bincount(bins[split:],minlength=4).tolist(),
                early_median=float(np.median(values[:split])),late_median=float(np.median(values[split:])))
    out=root/"企业赛/work/bearing/kaist-shift.json"
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,indent=2))
    print(result)


if __name__=="__main__":
    main()
