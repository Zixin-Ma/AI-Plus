"""Bounded messages for an experiment planner and an artifact reviewer."""
import json

import numpy as np


TOOLS = ("refit", "continue", "adam")
STRENGTHS = ("full", "half")
METRIC = {"A": "macro-F1/weakest class recall (higher)",
          "B": "RMSE/worst bearing RMSE in minutes (lower)",
          "C": "RMSE/worst subset RMSE in cycles (lower)"}
METHOD = {"A": "drift-aware gas classifier", "B": "vibration-gated lifetime correction",
          "C": "operating-regime lifetime correction"}


def _number(value):
    return isinstance(value, (int, float, np.integer, np.floating)) and np.isfinite(value)


def _rounded(value):
    return float(f"{value:.5g}")


def _sample(x, limit=1024):
    if len(x) <= limit:
        return x
    return x[np.linspace(0, len(x) - 1, limit, dtype=int)]


def _history(key, records):
    stats = {}
    choices = {}
    last = {}
    for record in records:
        config = record.get("configuration") or record.get("config") or {}
        if not isinstance(config, dict):
            continue
        plan = config.get("plan") or {}
        tool = plan.get("tool") or config.get("method")
        if tool == "continued":
            tool = "continue"
        if tool not in TOOLS:
            continue
        scores = config.get("scores") or {}
        parent = scores.get("parent") or config.get("parent") or {}
        candidate = scores.get("candidate") or (config.get("trials") or {}).get(tool) or config.get("candidate") or {}
        if not isinstance(parent, dict) or not isinstance(candidate, dict):
            continue
        p, c = parent.get("primary"), candidate.get("primary")
        if not (_number(p) and _number(c)):
            continue
        signed = (c-p) if key == "A" else (p-c)
        entry = stats.setdefault(tool, [0, 0])
        entry[0] += int(signed > 1e-8)
        entry[1] += 1
        last = {"tool": tool, "strength": plan.get("strength", "full"),
                "gain": _rounded(signed)}
        deployed = config.get("deployed")
        if isinstance(deployed, bool):
            deployed = "candidate" if deployed else "parent"
        if deployed in ("candidate", "history", "parent"):
            counts = choices.setdefault(tool, [0, 0])
            counts[0] += int(deployed == "candidate")
            counts[1] += 1
            last["deployed"] = deployed
        if _number(candidate.get("worst")) and _number(parent.get("worst")):
            w = candidate["worst"] - parent["worst"]
            last["worst_gain"] = _rounded(w if key == "A" else -w)
    return {"rounds": len(records), "positive_primary_over_attempts": stats,
            "candidate_deployed_over_reported": choices, "last": last}


def diagnose(key, model, train, valid, history, parent_metrics):
    """Train-scaled public drift and prior measured actions. No validation labels enter fitting."""
    x, xv = _sample(train[0]), _sample(valid[0])
    train_mean = np.mean(x, axis=0)
    std = np.std(x, axis=0)
    scale = np.maximum(std, 1e-6 * np.maximum(1., abs(train_mean)))
    shift = (np.mean(xv, axis=0)-train_mean) / scale
    ranked = np.argsort(-abs(shift), kind="stable")[:2]
    names = model.meta["paths"]
    top = {"/".join(str(step[1]) for step in json.loads(names[i])): _rounded(shift[i])
           for i in ranked}
    parent = {k: _rounded(parent_metrics[k]) for k in ("primary", "worst")}
    if "worst_group" in parent_metrics:
        parent["worst_group"] = str(parent_metrics["worst_group"])[:72]
    return {"drift_train_sd": top,
            "parent": parent,
            "history": _history(key, history)}


def _pack(payload):
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def plan_request(key, index, diagnosis, has_head):
    # These qualitative priors come from engine/bearing-disjoint and blocked
    # training folds. No fitted external weights or row predictions are bundled.
    evidence = {"A":"class-covariance head improved 2/3 held-out blocks; helps weak classes",
                "B":"vibration-gated head lowered worst leave-one-bearing-out RMSE",
                "C":"condition-normalized residual lowered RMSE in 5/5 held-out engine folds"}
    actions = "refit=Adam+new tested head (costly); adam=Adam only (cheap)"
    if has_head:
        actions += "; continue=Adam+existing head (cheap)"
    probe = ("Both positive disable_head gains support trying adam (removes the harmful head). "
             if "disable_head_probe" in diagnosis else "")
    content = (f"{key}{index}: {METRIC[key]}. {actions}. Training-fold evidence: {evidence[key]}. "
               "Use evidence for cold start; after negligible refit gains consider cheaper continuation. "
               "Respond ONE line, e.g. refit|half|correct sensor bias. Middle field is literally full or half. Give your own short drift hypothesis. "
               + probe + _pack(diagnosis))
    return [{"role": "user", "content": content}]


def _parts(text, fields):
    if not isinstance(text, str):
        return []
    raw = text.strip()
    if raw.startswith("{"):
        try:
            obj = json.loads(raw)
        except ValueError:
            return []
        if not isinstance(obj, dict) or set(obj) != set(fields):
            return []
        parts = [obj[k] for k in fields]
    else:
        lines = [line.strip() for line in raw.splitlines() if line.strip()]
        if not lines:
            return []
        # Qwen often adds a second sentence on the next line. It is evidence
        # for the same decision. Reject subsequent directives or revisions.
        remainder = " ".join(lines[1:])
        revision = ("correction", "instead choose", "actually choose", "actually select")
        if "|" in remainder or any(word in remainder.lower() for word in revision):
            return []
        parts = lines[0].split("|")
        if len(parts) == len(fields) and remainder:
            parts[-1] += " " + remainder
    if len(parts) != len(fields) or not all(isinstance(v, str) for v in parts):
        return []
    parts = [v.strip() for v in parts]
    if not all(parts) or any("\n" in v or "\r" in v for v in parts):
        return []
    return parts


def read_plan(text, has_head):
    parts = _parts(text, ("tool", "strength", "hypothesis"))
    if parts:
        tool, strength, hypothesis = parts
        if tool in TOOLS and strength in STRENGTHS and (tool != "continue" or has_head):
            return {"tool": tool, "strength": strength, "hypothesis": hypothesis[:160],
                    "fallback": False}
    return {"tool": "continue" if has_head else "refit", "strength": "half",
            "hypothesis": "Try a half-rate backbone step after an invalid planner reply.",
            "fallback": True}


def final_request(key, plan, scores):
    parent = scores["parent"]
    direction = 1 if key == "A" else -1
    gains = {name: {field+"_gain": _rounded(direction * (entry[field] - parent[field]))
                    for field in ("primary", "worst")}
             for name, entry in scores.items() if name in ("candidate", "history")}
    labels = ", ".join(["parent", *gains])
    content = (f"{key}. Tested {plan['tool']} at {plan['strength']} LR. "
               f"Choose one of {labels}. Each gain is relative to parent; positive helps. "
               "Prefer improved overall with stable weakest group; tie keeps parent. "
               "Answer ONE line such as parent|tie or candidate|improved. Give the measured reason in <8 words. " + _pack(gains))
    return [{"role": "user", "content": content}]


def read_final(text):
    parts = _parts(text, ("choice", "reason"))
    if parts and parts[0] in ("candidate", "parent", "history"):
        return parts[0], parts[1][:160]
    return "parent", "Invalid review reply; retained parent."
