"""Bounded messages for an experiment planner and an artifact reviewer."""
import json

import numpy as np


TOOLS = ("refit", "continue", "adam")
STRENGTHS = ("full", "half")
METRIC = {"A": "macro-F1/weakest class recall (higher)",
          "B": "RMSE/worst vibration-bin RMSE in minutes (lower)",
          "C": "RMSE/worst operating-regime RMSE in cycles (lower)"}
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
    parent["floor_vs_static"] = _rounded(parent_metrics.get("floor", 0.))
    return {"drift_train_sd": top,
            "parent": parent,
            "history": _history(key, history)}


def _pack(payload):
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def plan_request(key, index, diagnosis, has_head):
    evidence = {"A": "gas amplitude and transient response", "B": "vibration and long-life retention",
                "C": "sensor wear under changing operating conditions"}
    tools = "refit=fit correction, adam=backbone only" + (", continue=keep correction" if has_head else "")
    prior = "Test refit on cold start unless drift evidence favors Adam. " if not has_head else ""
    text = (f"{key}{index} {METRIC[key]}. {tools}. Diagnose {evidence[key]}. "
            f"{prior}Return one line like refit|half|test sensor shift. Choose an action and give your reason.")
    return [{"role": "user", "content": text + _pack(diagnosis)}]


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


def final_request(key,plan,scores):
    parent=scores["parent"]
    direction=1 if key=="A" else -1
    options=["parent"]
    rows=[["parent",0,0,0]]
    for name,entry in scores.items():
        if name!="parent" and entry["allowed"]:
            options.append(name)
            rows.append([name,_rounded(100*direction*(entry["primary"]-parent["primary"])/max(1e-9,parent["primary"])),
                         _rounded(direction*(entry["worst"]-parent["worst"])),_rounded(100*(entry["floor"]-parent["floor"]))])
    if len(options)==1:
        text=f"{key}. Choose one of parent. Parent is approved and available. Return exactly: parent|retain validated model"
    else:
        text=(f"{key}. Choose one of {', '.join(options)}. All listed models passed. "
              f"Rows [option,meanGain%,weakGain,floorGainPt], higher favors promotion: {_pack(rows)}. "
              "Reply option|short reason.")
    return [{"role":"user","content":text}]

def read_final(text):
    parts = _parts(text, ("choice", "reason"))
    if not parts and isinstance(text,str) and len(text.strip().splitlines()) == 1:
        fields = [s.strip() for s in text.strip().split("|")]
        if len(fields)==3 and fields[0] in ("parent","history","candidate") and fields[1] in ("tie","improved","declined") and fields[2]:
            parts = [fields[0], fields[1]+" "+fields[2]]
    if parts and parts[0] in ("candidate", "parent", "history"):
        return parts[0], parts[1][:160]
    return "parent", "Invalid review reply; retained parent."
