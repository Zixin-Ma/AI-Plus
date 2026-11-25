"""Repeatable public validation with either a deterministic control or bounded live Qwen calls."""
import argparse
import hashlib
import json
import sys
import time
import tomllib
from pathlib import Path

from prepare import ROOT, WORK
sys.path.insert(0, str(ROOT / "app"))
from contest_submission import MODEL, decision, improvement, prepare, prompt
from industrial import Model, dataset, load_head


def rows(path):
    with path.open() as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def consult(messages):
    import requests
    if len(json.dumps(messages, ensure_ascii=False).encode()) > 3500:
        raise ValueError("Development input budget exceeded")
    key = tomllib.loads((Path.home() / ".pytrio/config.toml").read_text())["api_key"]
    response = requests.post("https://pytrio.com/api/openai/v1/chat/completions",
        headers={"Authorization": "Bearer " + key},
        json=dict(model=MODEL, messages=messages, max_tokens=160, temperature=0, enable_thinking=False),
        timeout=(12, 90), allow_redirects=False)
    response.raise_for_status()
    body = response.json()
    return body["choices"][0]["message"]["content"], body.get("usage", {}), body["id"]


def run(key, mode, rounds, out):
    if mode == "live" and rounds > 3:
        raise ValueError("development live limit is three calls per dataset")
    start = WORK / key
    parent = start
    model = Model.load(start / "checkpoint.npz")
    train = dataset(rows(start / "public/train.jsonl"), model)
    valid = dataset(rows(start / "public/validation.jsonl"), model)
    history = []
    for i in range(1, rounds + 1):
        folder = out / key / f"round-{i:02}"
        folder.mkdir(parents=True)
        t = time.monotonic()
        model = Model.load(parent / "checkpoint.npz")
        head = load_head(parent / "residual.npz" if (parent / "residual.npz").exists() else None)
        method, scores, config = prepare(key, i, model, head, train, valid, folder)
        parent_id = hashlib.sha256((parent / "checkpoint.npz").read_bytes()).hexdigest()
        messages = prompt(key, i, config, scores[method], history)
        if mode == "live":
            answer, usage, receipt = consult(messages)
            accepted, why = decision(answer, key, scores["parent"], scores[method])
        else:
            accepted = improvement(key, scores["parent"], scores[method]) > 1e-8
            why = "CPU control selects strictly better validation score."
            answer, usage, receipt = "", {}, None
        if accepted:
            parent = folder
        record = dict(id=f"{mode}-{key}-{i}", dataset=key, round=i, mode=mode, accepted=accepted,
                      parent_sha256=parent_id, reason=why, selected=str(parent.relative_to(WORK)),
                      selected_score=scores[method] if accepted else scores["parent"], elapsed_s=time.monotonic()-t,
                      config=config, messages=messages if mode == "live" else [], response=answer,
                      usage=usage, provider_id=receipt)
        history.append(record)
        (folder / "record.json").write_text(json.dumps(record, ensure_ascii=False, indent=2))
        print(record["id"],method,round(scores["parent"],5),round(record["selected_score"],5),flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["control", "live"])
    parser.add_argument("--keys", default="ABC")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--name", required=True)
    args = parser.parse_args()
    for key in args.keys:
        run(key, args.mode, args.rounds, WORK / args.name)
