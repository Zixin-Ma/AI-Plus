"""CPU model continuation, public group diagnostics, and bounded host-model decisions."""
from functools import lru_cache
from pathlib import Path
from tempfile import mkdtemp, TemporaryDirectory
import importlib
import copy
import os
import shutil

import numpy as np

import guard
from industrial import Model, fields, load_head, metric, vector

MODEL = "Qwen/Qwen3.8-27B"
TASKS = ("classification", "regression")
MODULE = {"A": "gas", "B": "bearing", "C": "engine"}
METRIC = {"A": "macro_f1", "B": "rmse", "C": "rmse"}
RECALL_RESCUE = True
WEIGHTS = {"A": (1.,.65,.35,.15), "B": (1.,.65,.35,.15), "C": (1.,.85)}
OFFSET = {"B": .5, "C": 0.}
MEMORY = {"B": .18, "C": .10}
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
    if head and "life_offset" in head and key != "A":
        result = np.maximum(0, result + head["life_offset"])
    if head and "expert_weight" in head:
        weight = float(head["expert_weight"])
        result = weight * result + (1-weight) * guard.reference(key, head, x)
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
    if "floor" in parent and "floor" in candidate:
        main = gain(key, parent["primary"], candidate["primary"])
        scale = 1. if key == "A" else max(1., parent["primary"])
        floor = candidate["floor"] - parent["floor"]
        # Trade small group losses against a measured mean gain. Cap the
        # absolute drop, even when the overall gain is large.
        budget = min(.14, .025 + 1.4 * max(0., main / scale))
        if key == "A" and RECALL_RESCUE:
            # A previously missed class can recover at the cost of recall in
            # another class; account for the weakest absolute class as well.
            budget = max(budget, .65 * max(0., candidate["worst"] - parent["worst"]))
        if candidate.get("memory_loss", 0.) > MEMORY.get(key, 1.):
            return False
        return ((main > 1e-8 and floor >= -budget)
                or (floor > .04 and main >= -.004 * scale))
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
    groups = guard.strata(key, model, training[0], xv, yv)
    anchor = copy.copy(model)
    anchor.params = {k: v.copy() for k,v in model.params.items()}
    if "reference.npz" in rnd.parent.declared_paths():
        with np.load(rnd.parent.path("reference.npz"), allow_pickle=False) as archive:
            anchor.params = {k: archive[k] for k in model.params}
    if "expert_weight" in head:
        head = guard.freeze(guard.strip_mix(head), anchor, float(head["expert_weight"]))
    static_pred = values(key, anchor, {}, xv)
    if key == "A":
        static_pred = np.log(np.maximum(static_pred, 1e-300))
    panel = guard.Panel(key, yv, groups)
    static_groups = panel.score(static_pred)
    def measure(pred):
        by_group = panel.score(pred)
        worst = (min if key=="A" else max)(by_group,key=lambda g: by_group[g][0])
        metrics = dict(primary=metric(key,yv,pred),worst=by_group[worst][0],worst_group=worst)
        metrics.update(floor=guard.relative_floor(key, by_group, static_groups), groups=by_group)
        return metrics
    def evaluate(base, correction):
        return measure(values(key, base, correction, xv))
    # A training-only retention probe spans target quartiles. Its labels
    # are used exclusively for measuring the effect of an update.
    if key != "A":
        ids = np.linspace(0,len(training[1])-1,min(768,len(training[1])),dtype=int)
        probe_x, probe_y = training[0][ids], training[1][ids]
        bins = np.searchsorted(np.unique(np.quantile(probe_y,[.25,.50,.75])),probe_y)
        def errors(pred):
            sq = (pred[:,0]-probe_y)**2
            return np.array([np.sqrt(sq[bins==g].mean()) for g in np.unique(bins)])
        probe_base = values(key,anchor,{},probe_x)
        parent_errors = errors(values(key,model,head,probe_x))
        def memory_loss(pred):
            return float(np.max(errors(pred)-parent_errors)/max(1.,anchor.meta["target_scale"]))
    parent = evaluate(model, head)
    diagnosis = diagnose(key, model, training, validation, history, parent)
    if key == "B" and head:
        # Isolate whether a changed vibration regime has broken the prior correction.
        bare = evaluate(model, {})
        diagnosis["disable_head_probe"] = {
            "RMSE_gain": round(parent["primary"] - bare["primary"], 3),
            "worst_vibration_bin_gain": round(parent["worst"] - bare["worst"], 3)}
    options = dict(temperature=0, enable_thinking=False)
    # A new release or a changed public projection calls for a new diagnosis.
    # A stable projection reuses the last tested plan at half strength; the
    # reviewer still receives the actual trial outcomes before deployment.
    last = (history[-1].get("configuration") or {}) if history else {}
    old_size = last.get("train_rows", 0)
    old_shift = last.get("diagnosis", {}).get("drift_train_sd", {})
    shift = diagnosis["drift_train_sd"]
    drift_delta = max((abs(v-old_shift.get(k,0.)) for k,v in shift.items()), default=0.)
    replan = (key == "C" or not old_size or not head or rnd.round_index in (4,7,10)
              or abs(len(training[1])-old_size) > .15*old_size or drift_delta > .65)
    if replan:
        planner = rnd.model(plan_request(key, rnd.round_index, diagnosis, bool(head)),
                            max_tokens=min(rnd.max_output_tokens, 56), **options)
        planner_text = planner.text
        plan = read_plan(planner_text, bool(head))
    else:
        planner_text = ""
        plan = dict(tool="continue", strength="half", hypothesis="Stable release; retain fitted correction and test a small update.", fallback=False)
    name = plan["tool"]
    x, y, _ = training
    steps, sampled = continue_mlp(key, rnd.round_index, model, x, y, plan["strength"])
    if name == "refit":
        selected = importlib.import_module(MODULE[key]).fit(model, x, y, rnd.round_index)
    elif name == "continue":
        selected = head
    else:
        selected = {}
    variants = {}
    if selected:
        # Public validation selects from a bounded grid of fixed corrections. Only released
        # training pairs fitted their weights and the signal partitions.
        selected = guard.strip_mix(selected)
        selected.pop("life_offset", None)
        full = values(key, model, selected, xv)
        offsets = [0.]
        if key != "A":
            probe_full = values(key, model, selected, probe_x)
            if OFFSET[key] > 0.:
                sample = np.linspace(0,len(y)-1, min(512,len(y)),dtype=int)
                error = y[sample] - values(key,model,selected,x[sample])[:,0]
                step = OFFSET[key] * np.median(np.abs(error))
                offsets.extend([-step, step])
        for offset in offsets:
            shifted = full if offset == 0. else np.maximum(0., full + offset)
            for weight in WEIGHTS[key]:
                metrics = measure(weight * shifted + (1-weight) * static_pred)
                if key != "A":
                    replay = weight*np.maximum(0,probe_full+offset) + (1-weight)*probe_base
                    metrics["memory_loss"] = memory_loss(replay)
                variants[weight,offset] = metrics
        scale = 1. if key == "A" else max(1., parent["primary"])
        def priority(choice):
            score = variants[choice]
            overall = gain(key, parent["primary"], score["primary"]) / scale
            weakest = .28 * (score["worst"] - parent["worst"]) if key == "A" and RECALL_RESCUE else 0.
            return eligible(key, parent, score), overall + weakest + .45 * min(0., score["floor"])
        weight, offset = max(variants, key=priority)
        if offset:
            selected["life_offset"] = np.asarray(offset)
        selected = guard.freeze(selected, anchor, weight)
        candidate_score = variants[weight,offset]
    else:
        weight = 1.
        candidate_score = evaluate(model, {})
        if key != "A":
            candidate_score["memory_loss"] = memory_loss(values(key,model,{},probe_x))
    scores = {"parent": parent, "candidate": candidate_score}
    info = dict(method=name, plan=plan, diagnosis=diagnosis, planner_text=planner_text[:280], replanned=replan,
                old_steps=steps, steps=model.meta["steps"], learning_rate=model.meta["learning_rate"],
                expert_weight=weight, life_offset=float(selected.get("life_offset", 0.)), static_groups=static_groups, train_rows=len(y), adam_rows=sampled, validation_rows=len(yv), scores=scores)
    change = f"Executed {name} with {plan['strength']} learning rate. Adam steps {steps} to {model.meta['steps']}."
    desc = dict(hypothesis=plan["hypothesis"], change_summary=change, configuration=info,
                seed=model.meta["seed"])
    folder = Path(mkdtemp(prefix="candidate-", dir=os.environ.get("CONTEST_WORK", "/workspace")))
    model.save(folder / "checkpoint.npz")
    np.savez_compressed(folder / "reference.npz", **anchor.params)
    files = ["checkpoint.npz", "reference.npz"]
    if selected:
        np.savez_compressed(folder / "head.npz", **{k:v for k,v in selected.items() if not k.startswith("ref_")})
        files.append("head.npz")
    code = ["contest_submission.py", "policy.py", "baseline_model.py", "industrial.py", "guard.py", MODULE[key] + ".py"]
    if key == "A":
        code.append("prototype.py")
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
                    if key != "A":
                        scores["history"]["memory_loss"] = memory_loss(values(key,historical,aux,probe_x))
                    choices["history"] = old_id
                break
    for name, entry in scores.items():
        entry["allowed"] = name == "parent" or eligible(key,parent,entry)
    review = final_request(key, plan, scores)
    if not replan:
        review[0]["content"] += f" Previous decision: {last.get('deployed','unknown')}. Public rows: {len(y)}."
    receipt = rnd.model(review, max_tokens=min(rnd.max_output_tokens, 32), **options)
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
    head = load_head(path if path.exists() else None)
    if "expert_weight" in head and "ref_w1" not in head:
        anchor = copy.copy(model)
        with np.load(folder / "reference.npz", allow_pickle=False) as archive:
            anchor.params = {k:archive[k] for k in model.params}
        head = guard.freeze(guard.strip_mix(head), anchor, float(head["expert_weight"]))
    return model, head, extractor(fields(tuple(model.meta["paths"])))


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
