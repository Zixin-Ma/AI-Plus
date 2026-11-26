"""Alternate versions on one CPU thread; include public train loop and streamed output serialization."""
from pathlib import Path
import os
import subprocess
import sys

here=Path(__file__).resolve().parent
env=dict(os.environ,OPENBLAS_NUM_THREADS="1",OMP_NUM_THREADS="1",BASELINE_CPU_THREADS="1")
if __name__=="__main__":
    for repeat in range(3):
        versions=("campus","enterprise") if repeat%2==0 else ("enterprise","campus")
        for key in "ABC":
            for version in versions:
                output=here.parent/"work"/"resources"/str(repeat)
                subprocess.run([sys.executable,"-B",str(here/"benchmark.py"),key,version,"0","--output",str(output),"--profile"],env=env,check=True)
