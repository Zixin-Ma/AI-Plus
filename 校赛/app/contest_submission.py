"""Workspace v2 train/predict entry points for all three industrial tasks."""
import json
import os
import re
import shutil
from functools import lru_cache
from pathlib import Path
from tempfile import mkdtemp

from industrial import Model, dataset, drift, experiments, load_head, output
import numpy as np

MODEL = "Qwen/Qwen3.8-27B"
TASKS = ("classification", "regression")
PREDICT = ["python3", "/app/harness.py", "predict"]
METRIC = {"A": "macro_f1", "B": "rmse", "C": "rmse"}


def improvement(key, before, after):
    return (after - before) * (1 if key == "A" else -1)


def prepare(key, index, model, head, train, valid, folder):
    shifts = drift(train[0], valid[0], model)
    old_steps = model.meta["steps"]
    scores, heads = experiments(key, model, head, train, valid, index)
    best = max(heads, key=lambda n: improvement(key, scores["parent"], scores[n]))
    model.save(folder / "checkpoint.npz")
    if heads[best]:
        np.savez(folder / "residual.npz", **heads[best])
    config = dict(method=best, parent_steps=old_steps, steps=model.meta["steps"],
                  learning_rate=model.meta["learning_rate"], train_rows=len(train[1]),
                  validation_rows=len(valid[1]), shift=shifts, metric=METRIC[key],
                  public_ablation=scores)
    return best, scores, config


def prompt(key, index, config, measured, history):
    history = [{k: r[k] for k in ("id", "reason", "selected_artifact_id") if k in r}
               for r in history[-2:]]
    before = config["public_ablation"]["parent"]
    gain = improvement(key, before, measured)
    brief = dict(task=key, round=index, parent_score=round(before, 5), candidate_score=round(measured, 5),
                 improvement_positive_is_better=round(gain, 5), performance="improved" if gain > 1e-8 else "no improvement",
                 trials={k: round(v, 5) for k,v in config["public_ablation"].items()},
                 candidate=config["method"], metric=METRIC[key], shift=config["shift"],
                 history=str(history)[:480])
    return [{"role":"user","content":
             "Review this CPU training experiment. A maximizes macro F1, B/C minimize RMSE. "
             "Choose whether to deploy the measured candidate or retain the current parent. "
             "Use the supplied signed improvement. Deploy a positive gain unless history gives a concrete reason to decline. Reply exactly JSON "
             '{"choice":"candidate" or "parent","reason":"one short sentence"}. '
             + json.dumps(brief, ensure_ascii=False, separators=(",", ":"))}]


def decision(text, key, before, measured):
    try:
        obj = json.loads(re.search(r"\{[^{}]*\}", text).group())
        choice, why = obj["choice"], str(obj["reason"])[:400]
        if choice not in ("candidate", "parent") or not why.strip():
            raise ValueError
    except (ValueError, AttributeError, KeyError, TypeError):
        return False, "Unparseable model decision; retained the parent."
    if improvement(key, before, measured) <= 1e-8:
        return False, f"Validation guard retained parent. Model requested {choice}: {why}"
    return choice == "candidate", why


def train(rnd):
    if rnd.model_name != MODEL:
        raise ValueError("Bind the requested Qwen/Qwen3.8-27B base model")
    key = rnd.dataset_key
    history = rnd.history()
    model = Model.load(rnd.parent.path("checkpoint.npz"))
    head = load_head(rnd.parent.path("residual.npz") if "residual.npz" in rnd.parent.declared_paths() else None)
    train_set = dataset(rnd.data.training_rows(), model)
    val_set = dataset(rnd.data.validation_rows(), model)
    folder = Path(mkdtemp(prefix="candidate-", dir=os.environ.get("CONTEST_WORK", "/workspace")))
    method, scores, config = prepare(key, rnd.round_index, model, head, train_set, val_set, folder)
    code = ["industrial.py", "baseline_model.py", "contest_submission.py"]
    for n in code:
        shutil.copyfile(Path(__file__).with_name(n), folder / n)
    hypothesis = "A regularized residual on train-only transformed signals can improve the continued MLP under shift."
    change = f"Continued Adam from {config['parent_steps']} to {config['steps']} steps; selected {method} from public ablations."
    description = dict(hypothesis=hypothesis, change_summary=change, configuration=config, seed=model.meta["seed"])
    ckpts = ["checkpoint.npz"] + (["residual.npz"] if (folder/"residual.npz").exists() else [])
    reply = rnd.submit_candidate(folder, checkpoint_paths=ckpts, code_paths=code, predict_argv=PREDICT, **description)
    metrics = (reply.get("summary") or {}).get("metrics") or {}
    measured = metrics.get(METRIC[key])
    # Unknown host metric schema: the host must still approve the upload, and the
    # model sees that only the independently recomputed public score was available.
    host_metric = isinstance(measured, (float, int)) and np.isfinite(measured)
    if not host_metric:
        measured = scores[method]
    config["measurement_source"] = "host" if host_metric else "recomputed_public_validation"
    receipt = rnd.model(prompt(key, rnd.round_index, config, measured, history),
                        temperature=0, enable_thinking=False, max_tokens=min(rnd.max_output_tokens, 160))
    accepted, why = decision(receipt.text, key, scores["parent"], measured)
    rnd.finish(selected_artifact_id=reply["artifact_id"] if accepted else rnd.parent.artifact_id,
               reason=why, history_ids=[r["id"] for r in history], model_call_ids=list(rnd.call_ids),
               **description)


@lru_cache(maxsize=1)
def load_artifact(folder):
    base = Model.load(folder / "checkpoint.npz")
    return base, load_head(folder / "residual.npz" if (folder / "residual.npz").is_file() else None)


def predict(sample, model_dir):
    model, head = load_artifact(Path(model_dir))
    if model.meta["task"] != sample.task:
        raise ValueError("Task does not match checkpoint")
    key = sample.dataset_key or {128:"A", 34:"B", 24:"C"}[len(model.mean)]
    return output(model, head, key, sample.inputs)
