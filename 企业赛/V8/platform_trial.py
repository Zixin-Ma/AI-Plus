"""Upload a frozen release and start exactly one trial, with durable receipts.

Run prepare, then status until the image is ready, then start. Read-only status
also archives reports. Credentials are read from server user state.
"""
import argparse
import base64
import fcntl
import hashlib
import json
from pathlib import Path
import tomllib

import requests
import qualify

BASE = "https://rsi-index.115.zone/api/v1"
OUT = qualify.OUT / "platform-trial"
SESSION = Path("/opt/data/private/mzx/.bjtu-remote-access/state/rsi-v5-session.json")


def write(name, data):
    target = OUT / name
    tmp = target.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2))
    tmp.replace(target)


def client():
    session = requests.Session()
    session.headers["Authorization"] = "Bearer " + json.loads(SESSION.read_text())["token"]
    return session


def call(session, method, endpoint, **options):
    response = session.request(method, BASE + endpoint, timeout=(20, 900), **options)
    if not response.ok:
        raise RuntimeError(f"{method} {endpoint} returned {response.status_code}: {response.text[:500]}")
    return response.json()


def load(name):
    return json.loads((OUT / name).read_text())


def prepare(session):
    assert qualify.manifest() == load_release("freeze.json")
    image = qualify.OUT / "delivery/harness.tar"
    digest = hashlib.file_digest(image.open("rb"), "sha256").hexdigest()
    assert digest == load_release("artifact-audit.json")["archive_sha256"]
    if not (OUT / "image.json").exists():
        with image.open("rb") as stream:
            record = call(session, "POST", "/image-archives", data=stream,
                          headers={"Content-Type": "application/x-tar"})
        write("image.json", record)
    if not (OUT / "document.json").exists():
        doc = qualify.OUT / "delivery/企业赛V8技术报告.pdf"
        record = call(session, "POST", "/documents",
                      json={"name": doc.name, "content_base64": base64.b64encode(doc.read_bytes()).decode()})
        write("document.json", record)
    print("Uploaded receipts", load("image.json")["id"], digest, flush=True)


def load_release(name):
    return json.loads((qualify.OUT / name).read_text())


def status(session):
    if (OUT / "image.json").exists():
        archive_id = load("image.json")["id"]
        images = call(session, "GET", "/image-archives")["items"]
        image = next(x for x in images if x["id"] == archive_id)
        write("image-status.json", image)
        print("Image", image["status"], image.get("last_error", ""), flush=True)
    rules = call(session, "GET", "/rules")
    write("rules-latest.json", rules)
    if (OUT / "run.json").exists():
        run_id = load("run.json")["id"]
        runs = call(session, "GET", "/runs")["items"]
        run = next(x for x in runs if x["id"] == run_id)
        write("run-status.json", run)
        print("Run", run_id, run["status"], flush=True)
        report = session.get(BASE + "/runs/" + run_id + "/report", timeout=40)
        if report.ok:
            write("report.json", report.json())
        if run["status"] in ("failed", "quarantined", "cancelled"):
            print(json.dumps(run, ensure_ascii=False), flush=True)


def start(session):
    assert qualify.manifest() == load_release("freeze.json")
    if (OUT / "run.json").exists():
        print("Existing trial", load("run.json")["id"])
        return
    status(session)
    assert load("image-status.json")["status"] == "ready"
    rules = load("rules-latest.json")
    assert rules["trial_enabled"] and not rules["trial_blockers"]
    if not (OUT / "submission.json").exists():
        key = tomllib.loads((Path.home() / ".pytrio/config.toml").read_text())["api_key"]
        payload = dict(model="Qwen/Qwen3.8-27B", model_source="base", base_model_confirmed=True,
                       api_key=key, image_archive_id=load("image.json")["id"],
                       document_id=load("document.json")["id"],
                       training_url="https://swanlab.cn/@Zixin-Ma/ai-plus-campus-rsi",
                       source_url="https://github.com/Zixin-Ma/AI-Plus/tree/main/企业赛/V8")
        submission = call(session, "POST", "/submissions", json=payload)
        # The API returns public metadata only. Do not persist credentials.
        submission.pop("api_key", None)
        write("submission.json", submission)
    submission_id = load("submission.json")["id"]
    validated = call(session, "POST", f"/submissions/{submission_id}/validate")
    write("validation.json", validated)
    # Verify server state and avoid creating another run after an ambiguous response.
    submissions = call(session, "GET", "/submissions")["items"]
    assert next(x for x in submissions if x["id"] == submission_id)["status"] == "valid"
    matching = [x for x in call(session, "GET", "/runs")["items"]
                if x["submission_id"] == submission_id and x["mode"] == "trial"]
    if matching:
        assert len(matching) == 1
        run = matching[0]
    else:
        write("trial-request.json", dict(submission_id=submission_id,
                                       image_sha256=load_release("artifact-audit.json")["archive_sha256"]))
        run = call(session, "POST", f"/submissions/{submission_id}/trial",
                   json=dict(evaluation_config_id=rules["rules"]["evaluation_config_id"],
                             execution="full_training", note="企业赛 V8 冻结验证；对比 V5 最近正式结果，重点检查最低阶段及 C 的 NASA 尾部代价。"))
    assert run["mode"] == "trial" and run["submission_id"] == submission_id
    write("run.json", run)
    print("Trial", run["id"], run["status"], flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["prepare", "status", "start"])
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "operation.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        globals()[args.action](client())


if __name__ == "__main__":
    main()
