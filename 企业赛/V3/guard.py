"""Compare each deployable candidate with the same frozen checkpoint on visible regimes."""
import json
import numpy as np

BEARING_PANEL = "amplitude"


def strata(key, model, train, valid, labels):
    if key == "A":
        return np.asarray(labels).astype(str)
    if key == "B":
        cols = [i for i, p in enumerate(model.meta["paths"]) if json.loads(p)[-1][1] == "rms"]
        cuts = np.unique(np.quantile(train[:, cols].max(1), [.25, .50, .75]))
        bins = np.searchsorted(cuts, valid[:, cols].max(1))
        if BEARING_PANEL != "amplitude":
            from bearing import order_evidence
            peaks = [i for i,p in enumerate(model.meta["paths"]) if json.loads(p)[-1][1] == "dominant_freq"]
            if len(peaks) == 2:
                scores = order_evidence(valid, peaks)
                groups = scores.argmax(1)
                if BEARING_PANEL == "order":
                    return np.asarray([f"order-{j}" for j in groups])
                # Median-split amplitude keeps per-cell support higher than quartiles.
                return np.asarray([f"order-{j}-amp-{int(b>=2)}" for j,b in zip(groups,bins)])
        return np.asarray([f"vibration-{g}" for g in bins])
    settings, counts = np.unique(np.round(train[:, [21, 23]]), axis=0, return_counts=True)
    centers = settings[np.argsort(-counts)[:8]]
    dist = ((valid[:, None, [21, 23]] - centers) ** 2).sum(2)
    ids = dist.argmin(1)
    return np.asarray([f"settings-{a:g}/{b:g}" for a,b in centers])[ids]


def profile(key, y, pred, groups):
    losses = (pred.argmax(1) == y) if key == "A" else (pred[:, 0] - y) ** 2
    result = {}
    for group in np.unique(groups):
        mask = groups == group
        val = losses[mask].mean()
        result[str(group)] = (float(val if key == "A" else np.sqrt(val)), int(mask.sum()))
    return result


class Panel:
    """Compile the grouping once for all candidates on this public projection."""
    def __init__(self, key, y, groups):
        self.key, self.y = key, y
        names, self.ids = np.unique(groups, return_inverse=True)
        self.names = names.tolist()
        self.counts = np.bincount(self.ids)

    def score(self, pred):
        if self.key == "A":
            per_row = pred.argmax(1) == self.y
        else:
            per_row = (pred[:, 0] - self.y)**2
        means = np.bincount(self.ids, weights=per_row, minlength=len(self.counts))/self.counts
        scores = means if self.key == "A" else np.sqrt(means)
        return {name: (float(v), int(n)) for name,v,n in zip(self.names,scores,self.counts)}


def relative_floor(key, scores, static):
    gains = []
    for group, (score, n) in scores.items():
        base = static[group][0]
        # Add four pseudo-observations at the static performance for tiny slices.
        delta = score-base if key == "A" else (base-score) / max(1., base)
        gains.append(delta * n / (n + 4))
    return float(min(gains))


def freeze(head, anchor, weight):
    """Keep just inference weights; optimizer state stays in the parent checkpoint."""
    if weight == 1.:
        return head
    state = dict(head, expert_weight=np.asarray(weight), ref_mean=anchor.mean,
                 ref_scale=anchor.scale, ref_offset=np.asarray(anchor.meta["target_mean"]),
                 ref_unit=np.asarray(anchor.meta["target_scale"]))
    state.update({"ref_"+k:v for k,v in anchor.params.items()})
    return state


def reference(key, head, x):
    z = np.clip((x-head["ref_mean"]) / head["ref_scale"], -10., 10.)
    out = np.tanh(z @ head["ref_w1"] + head["ref_b1"]) @ head["ref_w2"] + head["ref_b2"]
    if key == "A":
        out -= out.max(axis=1, keepdims=True)
        return out - np.log(np.exp(out).sum(1, keepdims=True))
    return np.maximum(0., out * head["ref_unit"] + head["ref_offset"])


def strip_mix(head):
    return {k:v for k,v in head.items() if not k.startswith("ref_") and k != "expert_weight"}
