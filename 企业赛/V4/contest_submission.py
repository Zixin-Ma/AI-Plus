"""CPU model continuation, public group diagnostics, and bounded host-model decisions."""
from functools import lru_cache
from hashlib import blake2b
from operator import itemgetter
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
CONTEXT_TASKS = {"A", "C"}
ANCHOR_TASKS = {"B"}
CONTEXT_RADIUS = .18
SWITCH_SE = 0.
BLOCK_WEIGHT = .75
ENGINE_PANEL = None
ENGINE_FLOOR_WEIGHT = .6
ENGINE_OFFSET_MODE = "taper"
REUSE_ENGINE_PLAN = True
CLASS_DROP_LIMIT = None  # Optional development ablation.
WEIGHTS = {"A": (1.,.65,.35,.15), "B": (1.,.65,.35,.15), "C": (1.,.85)}
OFFSET = {"B": .5, "C": .15}
MEMORY = {"B": .18, "C": .10}
PREDICT = ["python3", "/app/harness.py", "predict"]


@lru_cache(maxsize=4)
def extractor(schema):
    if schema == tuple(("features", i) for i in range(128)):
        return lambda s: s["features"]
    if schema == tuple(("sensors", i) for i in range(21)) + tuple(("settings", i) for i in range(3)):
        return lambda s: s["sensors"] + s["settings"]
    if len(schema) == 34 and all(len(p) == 3 and p[0] == "features" for p in schema):
        axes = list(dict.fromkeys(p[1] for p in schema))
        if len(axes) == 2:
            names = tuple(p[2] for p in schema[:17])
            if schema == tuple(("features", axis, name) for axis in axes for name in names):
                get = itemgetter(*names)
                def bearing_features(row):
                    features = row["features"]
                    return get(features[axes[0]]) + get(features[axes[1]])
                return bearing_features
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
    if head and float(head.get("expert_weight", 1.)) == 0.:
        reference = guard.reference(key, head, x)
        if not np.isfinite(reference).all():
            raise ValueError("Nonfinite reference prediction")
        return reference
    if head and "backbone_only" not in head:
        pred = importlib.import_module(MODULE[key]).predict(model, head, x)
    else:
        pred = model.predict(x)
        if head and key == "A":
            pred = np.log(np.maximum(pred, 1e-300))
    result = np.asarray(pred).reshape(len(x), -1)
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite prediction")
    if head and "life_offset" in head and key != "A":
        result = guard.shift_life(result, float(head["life_offset"]), head.get("life_ramp"))
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
        if key == "A" and main < candidate.get("switch_margin", 0.) and candidate.get("switch_margin", 0.) > 0.:
            return False
        scale = 1. if key == "A" else max(1., parent["primary"])
        floor = candidate["floor"] - parent["floor"]
        # Trade small group losses against a measured mean gain. Cap the
        # absolute drop, even when the overall gain is large.
        budget = min(.14, .025 + 1.4 * max(0., main / scale))
        if key == "A" and CLASS_DROP_LIMIT is not None:
            # Pair each currently visible class with itself as the weakest group can change.
            for label, (current, count) in candidate.get("groups", {}).items():
                previous = parent["groups"][label][0]
                if (previous - current) * count / (count + 8.) > CLASS_DROP_LIMIT + max(0., main):
                    return False
        if key == "A" and RECALL_RESCUE:
            # A previously missed class can recover at the cost of recall in
            # another class; account for the weakest absolute class as well.
            budget = max(budget, .65 * max(0., candidate["worst"] - parent["worst"]))
        if candidate.get("release_movement", 0.) > .15:
            return False
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


def release_signature(x, y):
    digest = blake2b(digest_size=16)
    for array in (x, y):
        digest.update(repr((array.shape, array.dtype.str)).encode())
        digest.update(memoryview(np.ascontiguousarray(array)).cast("B"))
    return digest.hexdigest()


def release_movement(key, changed, has_head, pred, parent_pred, bins, scale):
    if key != "B" or not changed or not has_head:
        return 0.
    diff = (pred - parent_pred).ravel()
    shifts = [abs(float(np.mean(diff[bins == g]))) for g in np.unique(bins)]
    return max(shifts) / max(1., scale)


def f1_influence(y, scores):
    # Delta-method influence for the empirical macro-F1 ratio in each present class.
    pred = scores.argmax(1)
    classes = scores.shape[1]
    denom = np.bincount(y, minlength=classes) + np.bincount(pred, minlength=classes)
    hits = pred == y
    tp = np.bincount(y[hits], minlength=classes)
    factor = len(y) / (np.maximum(denom, 1) * max(1, np.count_nonzero(denom)))
    penalty = factor * 2. * tp / np.maximum(denom, 1)
    return 2. * hits * factor[y] - penalty[y] - penalty[pred]


def context_profile(model, x):
    # The immutable checkpoint scaler makes moments comparable across rounds.
    ids = np.linspace(0, len(x)-1, min(512, len(x)), dtype=int)
    z = np.clip((x[ids] - model.mean) / model.scale, -5., 5.)
    center = z.mean(0)
    return np.round(np.r_[center, np.abs(z-center).mean(0)], 3)


def prior_contexts(history, current, radius):
    matches = []
    for age, record in enumerate(reversed(history[-6:]), 1):
        cfg = record.get("configuration") or {}
        old = cfg.get("context_profile")
        if not isinstance(old, list) or len(old) != len(current):
            continue
        distance = float(np.sqrt(np.mean((np.asarray(old)-current)**2)))
        if distance <= radius:
            matches.append((distance, age, record, cfg))
    return sorted(matches, key=lambda entry: entry[:2])


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
    if key == "B" and BLOCK_WEIGHT:
        blocks = np.zeros(len(yv), dtype=int)
        for group in np.unique(groups):
            ids = np.flatnonzero(groups==group)
            blocks[ids] = np.arange(len(ids))*3//len(ids)
        counts = np.bincount(blocks)
        def block_errors(pred):
            sums = np.bincount(blocks, weights=(pred[:,0]-yv)**2)
            return np.sqrt(sums[counts>0]/counts[counts>0])
        static_error = block_errors(static_pred)
    if key == "C" and ENGINE_PANEL:
        if ENGINE_PANEL == "observed_life":
            cuts = np.quantile(training[1], [.33, .67])
            bands = np.searchsorted(np.unique(cuts), yv)
        else:
            # Visible static predictions define a stage surrogate without engine ID or true age.
            center = static_pred[:,0]
            bands = np.searchsorted(np.unique(np.quantile(center,[.33,.67])),center)
        band_panel = guard.Panel(key,yv,bands.astype(str))
        band_reference = band_panel.score(static_pred)
    def measure(pred):
        by_group = panel.score(pred)
        worst = (min if key=="A" else max)(by_group,key=lambda g: by_group[g][0])
        metrics = dict(primary=metric(key,yv,pred),worst=by_group[worst][0],worst_group=worst)
        if key == "C" and ENGINE_PANEL:
            band_scores = band_panel.score(pred)
            metrics["stage_floor"] = guard.relative_floor(key, band_scores, band_reference)
        if key == "B" and BLOCK_WEIGHT:
            metrics["block_floor"] = float(np.min((static_error - block_errors(pred)) / np.maximum(1.,static_error)))
        metrics.update(floor=guard.relative_floor(key, by_group, static_groups), groups=by_group)
        return metrics
    def evaluate(base, correction):
        return measure(values(key, base, correction, xv))
    last = (history[-1].get("configuration") or {}) if history else {}
    old_size = last.get("train_rows", 0)
    signature = release_signature(*training[:2])
    previous_signature = last.get("training_signature")
    same_rows = old_size == len(training[1]) and (previous_signature is None or signature == previous_signature)
    changed = bool(old_size) and not same_rows
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
        parent_pred = probe_base if not head and "reference.npz" not in rnd.parent.declared_paths() else values(key, model, head, probe_x)
        parent_errors = errors(parent_pred)
        def retention(pred):
            scale = max(1., anchor.meta["target_scale"])
            return dict(memory_loss=float(np.max(errors(pred)-parent_errors)/scale),
                        release_movement=release_movement(key, changed, bool(head), pred, parent_pred, bins, scale))
    parent_values = static_pred if not head and "reference.npz" not in rnd.parent.declared_paths() else values(key,model,head,xv)
    parent = measure(parent_values)
    if key == "A" and head and SWITCH_SE:
        parent_influence = f1_influence(yv, parent_values)
        parent_family = str(head.get("selected_family", "curve" if "curve_coeff" in head else "v1"))
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
    old_shift = last.get("diagnosis", {}).get("drift_train_sd", {})
    shift = diagnosis["drift_train_sd"]
    drift_delta = max((abs(v-old_shift.get(k,0.)) for k,v in shift.items()), default=0.)
    prior = last.get("plan") or {}
    reuse_engine = (REUSE_ENGINE_PLAN and key == "C" and prior.get("tool") == "refit"
                    and not prior.get("fallback") and last.get("deployed") == "candidate")
    replan = ((key == "C" and not reuse_engine) or not old_size or not head or rnd.round_index in (4,7,10)
              or abs(len(training[1])-old_size) > .15*old_size or drift_delta > .65)
    context = context_profile(model, xv) if key in CONTEXT_TASKS else np.empty(0)
    related = prior_contexts(history, context, CONTEXT_RADIUS) if context.size else []
    recalled = None
    if replan:
        for distance, age, record, cfg in related:
            oldplan = cfg.get("plan") or {}
            if (cfg.get("replanned") and cfg.get("deployed") == "candidate" and
                    not oldplan.get("fallback", True) and oldplan.get("tool") in ("adam","refit","continue")
                    and oldplan.get("strength") in ("half","full") and (head or oldplan["tool"] != "continue")):
                recalled = oldplan, record["id"], distance
                break
    if recalled:
        oldplan, origin, distance = recalled
        replan = False
        planner_text = ""
        plan = dict(tool=oldplan["tool"], strength=oldplan["strength"], fallback=False,
                    hypothesis=f"Repeat the approved {oldplan['tool']} on a recurring input distribution.",
                    source_round=origin, context_distance=round(distance, 4))
    elif replan:
        planner = rnd.model(plan_request(key, rnd.round_index, diagnosis, bool(head)),
                            max_tokens=min(rnd.max_output_tokens, 56), **options)
        planner_text = planner.text
        plan = read_plan(planner_text, bool(head))
    else:
        planner_text = ""
        if reuse_engine:
            plan = dict(tool="refit", strength=prior["strength"], hypothesis="Repeat the last approved engine refit on the stable release.", fallback=False)
        else:
            plan = dict(tool="continue", strength="half", hypothesis="Stable release; retain fitted correction and test a small update.", fallback=False)
    name = plan["tool"]
    x, y, _ = training
    steps, sampled = continue_mlp(key, rnd.round_index, model, x, y, plan["strength"])
    chosen_family = "backbone"
    candidate_heads = {}
    if name == "refit":
        expert = importlib.import_module(MODULE[key])
        if key == "A":
            candidate_heads = expert.fit_candidates(model, x, y, rnd.round_index)
        elif key == "C":
            options_c = dict(previous=head, stable_release=same_rows)
            if hasattr(expert, "fit_candidates"):
                candidate_heads = expert.fit_candidates(model, x, y, rnd.round_index, **options_c)
            else:
                candidate_heads = {"engine": expert.fit(model, x, y, rnd.round_index, **options_c)}
        elif hasattr(expert, "fit_candidates"):
            candidate_heads = expert.fit_candidates(model, x, y, rnd.round_index)
        else:
            candidate_heads = {"bearing": expert.fit(model, x, y, rnd.round_index)}
    elif name == "continue" and head:
        candidate_heads = {"parent_head": head}
    if key in ANCHOR_TASKS and not candidate_heads:
        candidate_heads = {"backbone": {"backbone_only": np.asarray(1)}}
    selected = {}
    variants = {}
    ramps = {}
    if candidate_heads:
        for family, fitted in candidate_heads.items():
            fitted = guard.strip_mix(fitted)
            fitted.pop("life_offset", None)
            fitted.pop("life_ramp", None)
            candidate_heads[family] = fitted
            full = values(key, model, fitted, xv)
            offsets = [0.]
            if key != "A":
                probe_full = values(key, model, fitted, probe_x)
                if OFFSET[key] > 0. and "backbone_only" not in fitted:
                    sample = np.linspace(0,len(y)-1,min(512,len(y)),dtype=int)
                    training_prediction = values(key,model,fitted,x[sample])[:,0]
                    residual = y[sample] - training_prediction
                    if key == "C" and ENGINE_OFFSET_MODE == "taper":
                        ramps[family] = np.quantile(training_prediction,[.20,.65])
                    step = OFFSET[key] * np.median(np.abs(residual))
                    if key != "C" or ENGINE_OFFSET_MODE == "symmetric":
                        offsets.extend([-step, step])
                    elif ENGINE_OFFSET_MODE == "up":
                        offsets.append(step)
                    elif parent["floor"] < 0.:
                        # Release only a training-sized correction when underestimation
                        # is consistent across the visible static-prediction bands.
                        cuts = np.unique(np.quantile(static_pred[:,0],[.33,.67]))
                        band = np.searchsorted(cuts,static_pred[:,0])
                        bias = yv-full[:,0]
                        if all(np.mean(bias[band==i]) > step for i in np.unique(band)):
                            offsets.append(step)
            for offset in offsets:
                shifted = full if offset == 0. else guard.shift_life(full,offset,ramps.get(family))
                weights = WEIGHTS[key] + ((0.,) if key in ANCHOR_TASKS else ())
                for weight in weights:
                    predictions = weight * shifted + (1-weight) * static_pred
                    metrics = measure(predictions)
                    if key == "A" and head and SWITCH_SE and family not in ("parent_head",parent_family):
                        diff = f1_influence(yv,predictions) - parent_influence
                        metrics["switch_margin"] = float(SWITCH_SE * np.std(diff) / np.sqrt(max(1,len(diff)-1)))
                    if key != "A":
                        replay = weight*guard.shift_life(probe_full,offset,ramps.get(family)) + (1-weight)*probe_base
                        metrics.update(retention(replay))
                    variants[family,weight,offset] = metrics
        scale = 1. if key == "A" else max(1., parent["primary"])
        def priority(choice):
            score = variants[choice]
            overall = gain(key, parent["primary"], score["primary"]) / scale
            weakest = .28 * (score["worst"] - parent["worst"]) if key == "A" and RECALL_RESCUE else 0.
            return eligible(key, parent, score), overall + weakest + .45 * min(0., score["floor"]) + BLOCK_WEIGHT * min(0.,score.get("block_floor",0.)) + ENGINE_FLOOR_WEIGHT * min(0.,score.get("stage_floor",0.))
        chosen_family, weight, offset = max(variants, key=priority)
        selected = candidate_heads[chosen_family]
        if weight == 0.:
            # A zero mixture needs only the frozen reference and can skip expert inference.
            selected = {"backbone_only": np.asarray(1)}
            chosen_family_display = "static"
        else:
            chosen_family_display = chosen_family
        if key == "A" and chosen_family != "parent_head" and SWITCH_SE:
            selected["selected_family"] = np.asarray(chosen_family)
        if offset and weight:
            selected["life_offset"] = np.asarray(offset)
            if chosen_family in ramps:
                selected["life_ramp"] = ramps[chosen_family]
        selected = guard.freeze(selected, anchor, weight)
        candidate_score = variants[chosen_family,weight,offset]
        chosen_family = chosen_family_display
    else:
        weight = 1.
        candidate_score = evaluate(model, {})
        if key != "A":
            candidate_score.update(retention(values(key,model,{},probe_x)))
    scores = {"parent": parent, "candidate": candidate_score}
    info = dict(method=name, context_profile=context.tolist(), training_signature=signature, same_training=same_rows, head_family=chosen_family, plan=plan, diagnosis=diagnosis, planner_text=planner_text[:280], replanned=replan,
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
    code = ["contest_submission.py", "policy.py", "baseline_model.py", "industrial.py", "guard.py", "serving.py", MODULE[key] + ".py"]
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
        ordered = [entry[2] for entry in related] + list(reversed(history[-2:]))
        for rec in ordered:
            old_id = rec.get("selected_artifact_id")
            if old_id in allowed and old_id != choices["parent"] and rec.get("configuration",{}).get("plan"):
                with TemporaryDirectory(dir=folder.parent) as tmp:
                    dest = Path(tmp) / "restored"
                    rnd.restore_artifact(old_id, dest)
                    historical, aux, _ = artifact(dest)
                    scores["history"] = evaluate(historical, aux)
                    if key != "A":
                        scores["history"].update(retention(values(key,historical,aux,probe_x)))
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


@lru_cache(maxsize=2)
def prepared_artifact(model_dir):
    from serving import prepare
    model, head, read = artifact(Path(model_dir))
    key = {128:"A", 34:"B", 24:"C"}[len(model.mean)]
    return key, model.meta["task"], read, prepare(key, model, head)


def predict(sample, model_dir):
    key, task, read, infer = prepared_artifact(model_dir)
    if sample.task != task or (sample.dataset_key and sample.dataset_key != key):
        raise ValueError("Sample does not match checkpoint")
    return infer(read(sample.inputs))
