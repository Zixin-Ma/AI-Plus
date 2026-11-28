"""Resume a single frozen trial. Poll read-only endpoints across transient outages."""
import argparse
from datetime import datetime, timezone
import json
import time
import platform_trial as trial

def now():return datetime.now(timezone.utc).isoformat(timespec="seconds")

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--poll-seconds",type=int,default=60)
    a=p.parse_args()
    if a.poll_seconds<20:raise ValueError("Use at least twenty seconds between reads")
    session=trial.client()
    deadline=time.monotonic()+6*3600
    while time.monotonic()<deadline:
        try:
            trial.status(session)
        except Exception as exc:
            print(now(),type(exc).__name__,"during read-only status",flush=True)
            time.sleep(a.poll_seconds)
            continue
        if (trial.OUT/"run.json").exists():
            status=trial.load("run-status.json")["status"]
            if status in ("failed","cancelled","quarantined"):raise RuntimeError(status)
            if status=="completed" and (trial.OUT/"report.json").exists():
                report=trial.load("report.json")
                if report.get("report",{}).get("usage_settled") and report.get("run",{}).get("status")=="completed":
                    print(now(),"Completed and settled",flush=True)
                    return
        elif (trial.OUT/"image-status.json").exists() and (trial.OUT/"document.json").exists():
            state=trial.load("image-status.json")["status"]
            if state in ("failed","rejected"):raise RuntimeError(state)
            if state=="ready":
                # start() checks server-side submissions and trials before posting.
                with (trial.OUT/"operation.lock").open("a") as lock:
                    trial.fcntl.flock(lock,trial.fcntl.LOCK_EX)
                    trial.start(session)
        time.sleep(a.poll_seconds)
    raise TimeoutError("Trial exceeded the platform deadline")
if __name__=="__main__":
    main()
