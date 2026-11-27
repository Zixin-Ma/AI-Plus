"""Compile immutable per-artifact prediction operations. No input or output rows are cached."""
import importlib
import math

import numpy as np


def reference_predictor(key, head):
    mean, scale = head["ref_mean"], head["ref_scale"]
    w1, b1, w2, b2 = (head["ref_" + name] for name in ("w1","b1","w2","b2"))
    unit, offset = float(head["ref_unit"]), float(head["ref_offset"])

    def predict(row):
        z = (row-mean)/scale
        np.maximum(z, -10., out=z)
        np.minimum(z, 10., out=z)
        raw = np.tanh(z @ w1 + b1) @ w2 + b2
        if key == "A":
            raw -= raw.max()
            return raw - np.log(np.exp(raw).sum())
        return max(0., float(raw[0])*unit + offset)

    return predict


def prepare(key, model, head, use_expert_hook=True):
    weight = float(head.get("expert_weight", 1.))
    reference = reference_predictor(key,head) if weight != 1. else None
    log_output = bool(head)
    bounded = bool(head) and key != "A"
    offset = float(head.get("life_offset",0.))
    width = len(model.mean)
    labels = tuple(model.meta["classes"])
    if weight == 0.:
        raw = reference
    else:
        expert = None
        if head and "backbone_only" not in head:
            module = importlib.import_module({"A":"gas","B":"bearing","C":"engine"}[key])
            hook = getattr(module, "prepare_inference", None)
            expert = hook(model,head) if use_expert_hook and hook else None
            if expert is None:
                if key == "A":
                    expert = lambda row: module.predict(model,head,row[None,:])[0]
                else:
                    expert = lambda row: float(np.asarray(module.predict(model,head,row[None,:])).item())
        elif key == "A" and log_output:
            expert = lambda row: np.log(np.maximum(model.predict_one(row),1e-300))
        else:
            expert = model.predict_one
        if offset and bounded:
            base_expert = expert
            if "life_ramp" in head:
                low, high = (float(x) for x in head["life_ramp"])
                span = max(high-low, 1e-9)
                def expert(row):
                    raw_value = base_expert(row)
                    return max(0.,raw_value+offset*min(1.,max(0.,(raw_value-low)/span)))
            else:
                expert = lambda row: max(0.,base_expert(row)+offset)
        if weight == 1.:
            raw = expert
        else:
            raw = lambda row: weight*expert(row)+(1.-weight)*reference(row)

    def infer(features):
        row = np.asarray(features, dtype=float)
        if row.shape != (width,) or not np.isfinite(row).all():
            raise ValueError("Invalid prediction row")
        pred = raw(row)
        if key != "A":
            value = float(pred)
            if not math.isfinite(value):
                raise ValueError("Nonfinite predicted lifetime")
            return {"value":max(0.,value)}
        pred = np.asarray(pred)
        probs = np.exp(pred-pred.max()) if log_output else pred.copy()
        total = float(probs.sum())
        if not math.isfinite(total) or total <= 0.:
            raise ValueError("Invalid class probabilities")
        probs /= total
        best = int(probs.argmax())
        if np.count_nonzero(probs == probs[best]) > 1:
            probs[best] = np.nextafter(probs[best],1.)
        return [{"label":name,"score":score} for name,score in zip(labels,probs.tolist())]

    return infer
