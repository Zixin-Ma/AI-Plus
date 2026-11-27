"""Check deployment rules at full public release size with one audit domain hidden."""
import argparse
from pathlib import Path
import contest_submission as harness
import opening_study as opening

def run(task,route,label):
    opening.OUT=opening.HERE.parent/"work/v6/opening-controls"/label/route
    harness.EXPOSURE={}
    harness.REPLAY_SAMPLING={}
    if route in ("confidence","dense_guard"):
        harness.EXPOSURE[task]=("confidence",.6)
    if route in ("dense","dense_guard"):
        harness.WEIGHTS[task]=tuple(dict.fromkeys((*harness.WEIGHTS[task],.95,.70,.50,.25,.05)))
    if route=="residual":
        harness.REPLAY_SAMPLING[task]=route
    opening.run(task,"v6")


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("task",choices=list("ABC"))
    p.add_argument("route",choices=["base","confidence","dense","dense_guard","residual"])
    p.add_argument("--label",default="final")
    args=p.parse_args()
    run(args.task,args.route,args.label)
