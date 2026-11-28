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
    rounded = np.round(train[:, [21, 23]])
    first, a = np.unique(rounded[:, 0], return_inverse=True)
    second, b = np.unique(rounded[:, 1], return_inverse=True)
    cells = len(first)*len(second)
    if cells <= min(65536, 2*len(train)):
        # Occupied indices have the same lexicographic order as unique(axis=0).
        histogram = np.bincount(a*len(second)+b, minlength=cells)
        used = np.flatnonzero(histogram)
        counts = histogram[used]
        settings = np.column_stack((first[used//len(second)], second[used%len(second)]))
    else:
        settings, counts = np.unique(rounded, axis=0, return_counts=True)
    settings[settings == 0.] = 0.  # A rounded signed zero denotes the same operating point.
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
        return self.summarize(per_row)

    def summarize(self,per_row):
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


def shift_life(values, delta, ramp=None):
    if not delta:
        return np.maximum(values, 0.)
    if ramp is not None:
        lo, hi = ramp
        fraction = np.clip((values-lo)/max(float(hi-lo), 1e-9), 0., 1.)
        return np.maximum(0., values + delta*fraction)
    return np.maximum(0., values + delta)


class ExposurePanel:
    """Measure excess validation loss within static-confidence or predicted-life bands."""
    def __init__(self, key, y, groups, reference, axis):
        self.key = key
        self.y = y
        if key == "A":
            leaders = np.partition(reference, -2, axis=1)[:, -2:]
            level = leaders[:, 1]-leaders[:, 0]
        else:
            level = reference[:, 0]
        _, group_ids = np.unique(groups, return_inverse=True)
        bins = np.empty(len(y), dtype=int)
        for group in np.unique(group_ids):
            ids = np.flatnonzero(group_ids == group)
            if axis == "confidence":
                ids = ids[np.argsort(level[ids], kind="stable")]
            bins[ids] = 3 * group + 3*np.arange(len(ids)) // len(ids)
        self.bins = bins
        self.counts = np.bincount(bins)
        self.active = self.counts >= 4
        self.base = self.loss(reference)
        self.scale = 1. if key == "A" else max(1., self.base.mean())

    def loss(self, prediction):
        if self.key == "A":
            return np.asarray(prediction.argmax(axis=1) != self.y, dtype=float)
        return (prediction[:, 0] - self.y)**2

    def downside(self, prediction):
        if not self.active.any():
            return 0.
        loss = self.loss(prediction)
        diff = self.base-loss
        gain = np.bincount(self.bins, weights=diff) / ((self.counts+12)*self.scale)
        # Average the weakest quarter, shrinking cells with little validation support.
        ordered = np.sort(gain[self.active])
        return float(min(0., ordered[:max(1,(len(ordered)+3)//4)].mean()))

class PairedRisk:
    """Excess error relative to the deployed parent on feature-only validation slices."""
    def __init__(self,key,train,x,y,parent,mode):
        self.key,self.y=key,y
        self.base=self.loss(parent)
        # Marginal location and scale shifts identify features on which old proportions
        # may no longer be representative. The axes use no audit samples or labels.
        t=train[np.linspace(0,len(train)-1,min(768,len(train)),dtype=int)]
        t=np.sign(t)*np.log1p(np.abs(t))
        z=np.sign(x)*np.log1p(np.abs(x))
        center=t.mean(0)
        scale=np.maximum(t.std(0), .05)
        z=np.clip((z-center)/scale,-8,8)
        shift=np.abs(z.mean(0))+.35*np.abs(np.log(np.maximum(.1,z.std(0))))
        features=[]
        for j in np.argsort(-shift,kind="stable"):
            v=z[:,j]
            if np.ptp(v)<1e-6:
                continue
            if any(abs(np.corrcoef(v,b)[0,1])>.9 for b in features):
                continue
            features.append(v)
            if len(features)==3:
                break
        self.groups=[]
        for v in features:
            edges=np.unique(np.quantile(v,[.25,.5,.75]))
            groups=np.searchsorted(edges,v)
            count=np.bincount(groups,minlength=4)
            self.groups.append((groups,count))
        self.mode=mode
        if key=="A":
            self.scale=1.
        else:
            self.scale=max(1.,self.base.mean())
    def loss(self,pred):
        if self.key=="A":
            return (pred.argmax(1)!=self.y).astype(float)
        return (pred[:,0]-self.y)**2

    def score(self,pred):
        excess=(self.base-self.loss(pred))/self.scale
        estimates=[]
        for ids,counts in self.groups:
            totals=np.bincount(ids,weights=excess,minlength=len(counts))
            ok=counts>=5
            if not np.any(ok):
                continue
            # Weakly supported cells shrink toward no change from the parent.
            rates=totals[ok]/(counts[ok]+12.)
            if self.mode=="relative":
                support=np.bincount(ids,weights=self.base/self.scale,minlength=len(counts))[ok]
                rates=totals[ok]/(12.+support)
            if self.mode=="se":
                seconds=np.bincount(ids,weights=excess**2,minlength=len(counts))[ok]
                se=np.sqrt(np.maximum(0.,seconds/counts[ok]-(totals[ok]/counts[ok])**2)/counts[ok])
                rates=rates-.5*se
            if self.mode=="penalty":
                estimates.append(min(0.,float(rates.min())))
            else:
                estimates.append(float(rates.min()))
        return float(np.mean(estimates)) if estimates else 0.
