"""Read-only monitor for a trial whose run receipt already exists."""
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
    if not (trial.OUT/"run.json").exists():
        raise SystemExit("No existing run receipt to monitor")
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
                if (report.get("report") or {}).get("usage_settled") and report.get("run",{}).get("status")=="completed":
                    print(now(),"Completed and settled",flush=True)
                    return
        time.sleep(a.poll_seconds)
    raise TimeoutError("Trial exceeded the platform deadline")
if __name__=="__main__":
    main()
