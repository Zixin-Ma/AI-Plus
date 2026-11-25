"""CPU model continuation, public group diagnostics, and bounded host-model decisions."""
from functools import lru_cache
from pathlib import Path
from tempfile import mkdtemp, TemporaryDirectory
import importlib
import os
import shutil

import numpy as np

from industrial import Model, fields, load_head, metric, vector

MODEL = "Qwen/Qwen3.8-27B"
TASKS = ("classification", "regression")
MODULE = {"A": "gas", "B": "bearing", "C": "engine"}
METRIC = {"A": "macro_f1", "B": "rmse", "C": "rmse"}
PREDICT = ["python3", "/app/harness.py", "predict"]


@lru_cache(maxsize=4)
def extractor(schema):
    if schema == tuple(("features", i) for i in range(128)):
        return lambda s: s["features"]
    if schema == tuple(("sensors", i) for i in range(21)) + tuple(("settings", i) for i in range(3)):
        return lambda s: s["sensors"] + s["settings"]
    return lambda s: vector(s, schema)


def projection(rows, model, key, audit=False):
    read = extractor(fields(tuple(model.meta["paths"])))
    classes = {v: i for i, v in enumerate(model.meta["classes"])}
    x, y, groups = [], [], []
    for row in rows:
        x.append(read(row["inputs"]))
        label = row["answer"]
        y.append(classes[label] if classes else float(label))
        if audit:
            meta = row.get("meta") or {}
            groups.append(label if classes else meta.get("bearing" if key == "B" else "source_subset", "all"))
    x, y = np.asarray(x, dtype=float), np.asarray(y)
    if not len(y) or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Invalid public projection")
    return x, y, np.asarray(groups)


def values(key, model, head, x):
    if head:
        pred = importlib.import_module(MODULE[key]).predict(model, head, x)
    else:
        pred = model.predict(x)
    result = np.asarray(pred).reshape(len(x), -1)
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite prediction")
    return result if key == "A" else np.maximum(result, 0)


def measures(key, y, pred, groups):
    main = metric(key, y, pred)
    if key == "A":
        guess = pred.argmax(axis=1)
        quality = {str(groups[y == k][0]): float(np.mean(guess[y == k] == k)) for k in np.unique(y)}
        worst_group = min(quality, key=quality.get)
    else:
        sqerr = (pred[:, 0] - y) ** 2
        quality = {str(g): float(np.sqrt(sqerr[groups == g].mean())) for g in np.unique(groups)}
        worst_group = max(quality, key=quality.get)
    return {"primary": main, "worst": quality[worst_group], "worst_group": worst_group}


def gain(key, before, after):
    return (after - before) if key == "A" else (before - after)


def eligible(key, parent, candidate):
    # Ignore floating-point ties. Tolerate one percentage point of worst-class
    # recall, or 2% worst-device/fleet RMSE, to avoid chasing sample noise.
    allowance = 0.01 if key == "A" else 0.02 * max(1, parent["worst"])
    return (gain(key, parent["primary"], candidate["primary"]) > 1e-8 and
            gain(key, parent["worst"], candidate["worst"]) >= -allowance)


def continue_mlp(key, index, model, x, y, strength):
    old_steps = model.meta["steps"]
    # Bounded backbone update retains the checkpoint, Adam moments and scalers.
    limit = {"A": 2048, "B": 768, "C": 1536}[key]
    if len(y) > limit:
        idx = np.random.default_rng(2026 + index).choice(len(y), limit, replace=False)
        xb, yb = x[idx], y[idx]
    else:
        xb, yb = x, y
    model.meta["learning_rate"] = {"A": .001, "B": .0008, "C": .0006}[key]
    if strength == "half":
        model.meta["learning_rate"] *= .5
    model.fit(xb, yb, 1)
    return old_steps, len(yb)


def train(rnd):
    if rnd.model_name != MODEL:
        raise ValueError("Requires Qwen3.8-27B")
    history = rnd.history()
    key = rnd.dataset_key
    model = Model.load(rnd.parent.path("checkpoint.npz"))
    head = load_head(rnd.parent.path("head.npz") if "head.npz" in rnd.parent.declared_paths() else None)
    training = projection(rnd.data.training_rows(), model, key)
    validation = projection(rnd.data.validation_rows(), model, key, True)
    from policy import diagnose, plan_request, read_plan, final_request, read_final
    xv, yv, groups = validation
    def evaluate(base, correction):
        return measures(key, yv, values(key, base, correction, xv), groups)
    parent = evaluate(model, head)
    diagnosis = diagnose(key, model, training, validation, history, parent)
    if key == "B" and head:
        # Isolate whether a changed vibration regime has broken the prior correction.
        bare = evaluate(model, {})
        diagnosis["disable_head_probe"] = {
            "RMSE_gain": round(parent["primary"] - bare["primary"], 3),
            "worst_bearing_gain": round(parent["worst"] - bare["worst"], 3)}
    options = dict(temperature=0, enable_thinking=False)
    planner = rnd.model(plan_request(key, rnd.round_index, diagnosis, bool(head)),
                        max_tokens=min(rnd.max_output_tokens, 56), **options)
    plan = read_plan(planner.text, bool(head))
    name = plan["tool"]
    x, y, _ = training
    steps, sampled = continue_mlp(key, rnd.round_index, model, x, y, plan["strength"])
    if name == "refit":
        selected = importlib.import_module(MODULE[key]).fit(model, x, y, rnd.round_index)
    elif name == "continue":
        selected = head
    else:
        selected = {}
    scores = {"parent": parent, "candidate": evaluate(model, selected)}
    info = dict(method=name, plan=plan, diagnosis=diagnosis, planner_text=planner.text[:280],
                old_steps=steps, steps=model.meta["steps"], learning_rate=model.meta["learning_rate"],
                train_rows=len(y), adam_rows=sampled, validation_rows=len(yv), scores=scores)
    change = f"Executed {name} with {plan['strength']} learning rate. Adam steps {steps} to {model.meta['steps']}."
    desc = dict(hypothesis=plan["hypothesis"], change_summary=change, configuration=info,
                seed=model.meta["seed"])
    folder = Path(mkdtemp(prefix="candidate-", dir=os.environ.get("CONTEST_WORK", "/workspace")))
    model.save(folder / "checkpoint.npz")
    files = ["checkpoint.npz"]
    if selected:
        np.savez_compressed(folder / "head.npz", **selected)
        files.append("head.npz")
    code = ["contest_submission.py", "policy.py", "baseline_model.py", "industrial.py", MODULE[key] + ".py"]
    for f in code:
        shutil.copyfile(importlib.import_module(f[:-3]).__file__, folder / f)
    reply = rnd.submit_candidate(folder, checkpoint_paths=files, code_paths=code, predict_argv=PREDICT, **desc)
    score = scores["candidate"]
    measured = ((reply.get("summary") or {}).get("metrics") or {}).get(METRIC[key])
    if isinstance(measured, (int, float)) and np.isfinite(measured):
        score["primary"] = float(measured)
        info["measurement_source"] = "host"
    else:
        info["measurement_source"] = "public_recomputation"
    choices = {"candidate": reply["artifact_id"], "parent": rnd.parent.artifact_id}
    # Only revisit one previously deployed, host-declared artifact. Score it on
    # today's validation projection before showing it to the decision model.
    if not eligible(key, parent, score):
        allowed = {a["artifact_id"] for a in rnd.artifacts}
        for rec in reversed(history[-2:]):
            old_id = rec.get("selected_artifact_id")
            if old_id in allowed and old_id != choices["parent"] and rec.get("configuration",{}).get("plan"):
                with TemporaryDirectory(dir=folder.parent) as tmp:
                    dest = Path(tmp) / "restored"
                    rnd.restore_artifact(old_id, dest)
                    historical, aux, _ = artifact(dest)
                    scores["history"] = evaluate(historical, aux)
                    choices["history"] = old_id
                break
    receipt = rnd.model(final_request(key, plan, scores), max_tokens=min(rnd.max_output_tokens, 32), **options)
    requested, why = read_final(receipt.text)
    deployed = requested if requested in scores and eligible(key, parent, scores[requested]) else "parent"
    info.update(requested=requested, deployed=deployed, measured_ids=choices)
    reason = (f"{deployed}; {METRIC[key]} {parent['primary']:.5g} to "
              f"{scores[deployed]['primary']:.5g}. Qwen: {why}")
    rnd.finish(selected_artifact_id=choices[deployed],
               reason=reason, history_ids=[r["id"] for r in history], model_call_ids=list(rnd.call_ids), **desc)


@lru_cache(maxsize=1)
def artifact(folder):
    model = Model.load(folder / "checkpoint.npz")
    path = folder / "head.npz"
    return model, load_head(path if path.exists() else None), extractor(fields(tuple(model.meta["paths"])))


def predict(sample, model_dir):
    model, head, read = artifact(Path(model_dir))
    if sample.task != model.meta["task"]:
        raise ValueError("Task does not match checkpoint")
    key = sample.dataset_key or {128:"A", 34:"B", 24:"C"}[len(model.mean)]
    x = np.asarray([read(sample.inputs)], dtype=float)
    v = values(key, model, head, x)[0]
    if key != "A":
        return {"value": float(v[0])}
    probs = np.exp(v - v.max())
    probs /= probs.sum()
    i = int(np.argmax(probs))
    if np.count_nonzero(probs == probs[i]) > 1:
        probs[i] = np.nextafter(probs[i], 1.)
    return [{"label": c, "score": float(p)} for c, p in zip(model.meta["classes"], probs)]
