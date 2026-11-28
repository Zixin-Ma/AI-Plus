"""Matched experiments on training-trajectory effects in bearing residual calibration."""
import copy
from concurrent.futures import ThreadPoolExecutor
from importlib.util import module_from_spec, spec_from_file_location
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
BASE = HERE.parent / "V6"
OUT = HERE.parent / "work/v7/bearing"
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(BASE))


def module(name, path):
    spec = spec_from_file_location(name, path)
    mod = module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


v6 = module("frozen_b6", BASE / "bearing.py")
MODES = {
    "balanced": (0.55, 1.0, 0.0),
    "short": (0.9, 1.0, 0.0),
    "within": (0.55, 0.2, 0.0),
    "withinall": (1.0, 0.2, 0.0),
    "jack": (0.55, 0.45, 1.0),
    "pooled": (0.0, 1.0, 0.0),
}


def calibration(model, h, x, y, mode):
    runs = v6._training_runs(y)
    if not runs:
        return h
    if mode.startswith(("crossridge", "crossweak", "steady", "discrete", "safeshape", "shared")):
        name, amount = mode.split("-")
        end = np.mean([y[r.stop - 1] <= max(1e-8, 1.1 * np.median(np.abs(np.diff(y[r])))) for r in runs])
        if end >= 0.65:
            return calibration(model, h, x, y, "masssafe-0.90")
        route = {
            "crossridge": "shrinkrff",
            "crossweak": "weakrff",
            "steady": "fullridge",
            "discrete": "within",
            "safeshape": "short",
            "shared": "jack",
        }[name]
        if name in ("crossridge", "crossweak", "steady"):
            return calibration(model, h, x, y, route + "-" + amount)
        k = float(amount)
        n = calibration(model, h, x, y, route)
        return {**h, "coef": h["coef"] + k * (n["coef"] - h["coef"])}
    if mode.startswith(
        (
            "certify",
            "certprior",
            "certmix",
            "certblend",
            "certorder",
            "certwhiten",
            "certv6",
            "certweak",
            "certunit",
            "certlite",
            "certorig",
        )
    ):
        name, q = mode.split("-")
        end = np.mean([y[r.stop - 1] <= max(1e-8, 1.1 * np.median(np.abs(np.diff(y[r])))) for r in runs])
        if end >= 0.65:
            if name == "certprior":
                return calibration(model, h, x, y, "majority-" + q)
            return calibration(model, h, x, y, "masssafe-" + q)
        if name == "certv6":
            return h
        route = "weakrff-1" if name == "certweak" else "shrinkrff-1"
        head = calibration(model, h, x, y, route)
        if name in ("certmix", "certblend", "certwhiten", "certunit", "certlite", "certorig"):
            maybe = calibration(model, h, x, y, "majority-" + q)
            if maybe is h:
                return head
            b = {
                "certmix": 0.85,
                "certblend": 0.5,
                "certwhiten": 0.65,
                "certunit": 0.35,
                "certlite": 0.15,
                "certorig": 0.05,
            }[name]
            head.update(
                order_means=h["order_means"] + b * (maybe["order_means"] - h["order_means"]),
                prior=h["prior"] + b * (maybe["prior"] - h["prior"]),
            )
        return head
    if mode.startswith(
        (
            "votefast",
            "votebase",
            "votepair",
            "votebal",
            "votefull",
            "votesev",
            "voteunit",
            "votepool",
            "voteprefix",
        )
    ):
        name, arg = mode.split("-")
        quant = float(arg)
        target = calibration(model, h, x, y, "masssafe-0.90")
        shift = (target["order_means"] - h["order_means"]) * (0.5 / 0.6) + (target["prior"] - h["prior"]) * (
            1 - 0.5 / 0.6
        )
        if np.max(abs(shift)) < 1e-7:
            return h
        old = v6.predict(model, h, x)
        delta = v6.predict(model, target, x) - old
        base = model.predict(x)
        ev = v6.order_evidence(x, h["order_cols"])
        pr = np.exp(7 * (ev - ev.max(1, keepdims=True)))
        pr /= pr.sum(1, keepdims=True)
        ratio = v6._band_ratio(x, h["curve_bands"])
        aa = base if name in ("votebase", "votefull", "votepool", "voteprefix") else pr[:, 2]
        bb = ratio if name not in ("votepair", "voteunit") else base
        cuts = np.quantile(aa, [0.25, 0.5, 0.75])
        knots = np.quantile(bb, [0.25, 0.5, 0.75])
        grid = 4 * np.searchsorted(cuts, aa) + np.searchsorted(knots, bb)
        ratios = []
        ss = []
        uu = []
        for r in runs:
            extra = delta[r]
            res = y[r] - old[r]
            num = np.bincount(grid[r], weights=res * extra, minlength=16).reshape(4, 4)
            den = np.bincount(grid[r], weights=extra ** 2, minlength=16).reshape(4, 4)
            num = v6._adjacent_sum(num)
            den = v6._adjacent_sum(den)
            ridge = max(1.0, np.mean(extra ** 2)) * 4
            ratios.append(np.clip(num / (den + ridge), 0.0, 1.0))
            ss.append(num / (r.stop - r.start))
            uu.append(den / (r.stop - r.start))
        frac = np.quantile(ratios, quant, axis=0)
        if name == "votepool":
            frac = np.clip(np.sum(ss, axis=0) / (np.sum(uu, axis=0) + 1.0), 0, 1)
        if (
            name == "votefull"
            and np.mean([y[r.stop - 1] < 1.1 * np.median(abs(np.diff(y[r]))) for r in runs]) >= 0.65
        ):
            return target
        return {
            **h,
            "vote_shift": shift,
            "vote_cuts": cuts,
            "vote_knots": knots,
            "vote_rate": frac,
            "vote_mode": np.array(name),
        }
    if mode.startswith(("phase", "late", "bound", "blendphase")):
        name, arg = mode.split("-")
        target = calibration(model, h, x, y, "masssafe-0.90")
        shift = (target["order_means"] - h["order_means"]) * (0.5 / 0.6) + (target["prior"] - h["prior"]) * (
            1 - 0.5 / 0.6
        )
        if np.max(abs(shift)) < 1e-7:
            return h
        ids = np.concatenate(
            [np.linspace(r.start, r.stop - 1, min(180, r.stop - r.start), dtype=int) for r in runs]
        )
        t = np.concatenate([np.linspace(1.0, 0.0, min(180, r.stop - r.start)) for r in runs])
        z = np.clip((np.sign(x[ids]) * np.log1p(abs(x[ids])) - h["center"]) / h["scale"], -6, 6)
        if "phy" in name:
            keep = np.zeros(z.shape[1], bool)
            keep[h["curve_bands"]] = True
            keep[h["rms_columns"]] = True
            z[:, ~keep] = 0.0
        Z = np.column_stack((np.ones(len(z)), z))
        gram = Z.T @ Z
        gram.flat[:: len(gram) + 1] += 0.035 * len(Z)
        coef = np.linalg.solve(gram, Z.T @ t)
        pred = Z @ coef
        uq = np.quantile(pred, [0.25, 0.75])
        return {
            **h,
            "phase_shift": shift,
            "phase_coef": coef,
            "phase_cuts": uq,
            "phase_rate": np.array(float(arg)),
            "phase_mode": np.array(name),
            "phase_amplitude": np.array(np.std(y)),
        }
    if mode.startswith(("supported", "masssafe", "majority", "signedmix")):
        g, level = mode.split("-")
        q = float(level)
        route = ("fullintegral-" if g == "masssafe" else "envelope-") + level
        candidate = calibration(model, h, x, y, route)
        if candidate is h:
            return h
        raw = v6.order_evidence(x, h["order_cols"])
        pr = np.exp(7.0 * (raw - raw.max(1, keepdims=True)))
        pr /= pr.sum(1, keepdims=True)
        high = np.asarray([pr[r, 2].mean() for r in runs])
        # Estimate regime support using source devices; an ambiguous peak has posterior one third.
        confidence = float(np.clip((np.max(high) - 1.0 / 3) / (2.0 / 3 - 1.0 / 3), 0, 1))
        if g in ("majority", "masssafe"):
            confidence = float(np.max(high) >= 0.5)
        if g == "signedmix":
            life = np.array([np.mean(y[r]) for r in runs])
            cov = np.corrcoef(np.argsort(np.argsort(life)), np.argsort(np.argsort(high)))[0, 1]
            confidence *= min(1.0, max(0.0, 2 * cov))
        return {
            **h,
            "order_means": h["order_means"] + confidence * (candidate["order_means"] - h["order_means"]),
            "prior": h["prior"] + confidence * (candidate["prior"] - h["prior"]),
        }
    if mode.startswith(
        ("envelope", "rankenvelope", "fullintegral", "minimax", "blendenergy", "stageunsafe", "uncensored")
    ):
        name, val = mode.split("-")
        q = float(val)
        near = np.mean(y <= 0.02 * np.quantile(y, 0.9))
        if near < 0.025:
            return h
        complete = np.mean([y[r.stop - 1] <= max(1e-8, 1.1 * np.median(np.abs(np.diff(y[r])))) for r in runs])
        if name in ("fullintegral", "minimax") and complete >= 0.65:
            return calibration(model, h, x, y, "integral-x")
        mode0 = "liferank-1" if name == "rankenvelope" else "geolife-1"
        target = calibration(model, h, x, y, mode0)
        means = target["order_means"].copy()
        means[:2] = np.minimum(h["order_means"][:2], means[:2])
        means[2] = np.quantile(y, q)
        if name == "blendenergy":
            g = np.clip((near - 0.025) / 0.015, 0, 1)
            means = h["order_means"] + g * (means - h["order_means"])
            target["prior"] = h["prior"] + g * (target["prior"] - h["prior"])
        return {**target, "order_means": means}
    if mode.startswith(
        (
            "fullridge",
            "centeredridge",
            "endridge",
            "faridge",
            "augmented",
            "shrinkrff",
            "weakrff",
            "pooledridge",
            "robustridge",
            "capprior",
        )
    ):
        tag, blend = mode.split("-")
        blend = float(blend)
        # Retain the common map; refine regression sufficient statistics over the release.
        ids = np.linspace(0, len(y) - 1, min(len(y), 8192), dtype=int)
        if tag in ("faridge", "augmented", "endridge"):
            ids = np.concatenate(
                [np.linspace(r.start, r.stop - 1, min(256, r.stop - r.start), dtype=int) for r in runs]
            )
        X = v6._design(x[ids], h)
        t = y[ids] - model.predict(x[ids])
        wt = np.ones(len(ids))
        if tag == "robustridge":
            e = np.abs(X @ h["coef"] - t)
            wt = np.minimum(1.0, np.quantile(e, 0.70) / np.maximum(e, 1.0))
        G = (X.T * wt) @ X
        G.flat[:: len(G) + 1] += 0.008 * wt.sum()
        if tag in ("shrinkrff", "weakrff"):
            start = 1 + x.shape[1]
            G.flat[start * (len(G) + 1) :: len(G) + 1] += (0.04 if tag == "shrinkrff" else 0.12) * wt.sum()
        b = X.T @ (wt * t)
        coef = np.linalg.solve(G, b)
        if tag in ("endridge", "augmented", "pooledridge"):
            complete = np.mean(
                [y[r.stop - 1] <= max(1e-8, 1.1 * np.median(abs(np.diff(y[r])))) for r in runs]
            )
            if complete >= 0.65:
                return calibration(model, h, x, y, "liferank-1")
        return {**h, "coef": (1 - blend) * h["coef"] + blend * coef}
    if mode.startswith(
        (
            "complete-",
            "protect-",
            "ramp-",
            "hybrid-",
            "pooledfast-",
            "transfer-",
            "energy-",
            "integral-",
            "defer-",
            "partial-",
            "rankfull-",
            "jackfull-",
            "guarded-",
        )
    ):
        kind = mode.split("-")[0]
        complete = np.mean([y[r.stop - 1] <= max(1e-8, 1.1 * np.median(np.abs(np.diff(y[r])))) for r in runs])
        hnew = calibration(model, h, x, y, ("liferank-1" if kind in ("rankfull", "partial") else "geolife-1"))
        if kind in ("complete", "rankfull", "jackfull", "hybrid", "defer", "partial", "ramp"):
            if complete < 0.65:
                if kind == "hybrid":
                    return calibration(model, h, x, y, "withinall")
                if kind == "jackfull":
                    return calibration(model, h, x, y, "jack")
                if kind in ("ramp", "partial"):
                    b = 0.20 if kind == "partial" else complete
                    return {
                        **h,
                        "prior": h["prior"] + b * (hnew["prior"] - h["prior"]),
                        "order_means": h["order_means"] + b * (hnew["order_means"] - h["order_means"]),
                    }
                return h
            return hnew
        if kind == "guarded":
            if "safe_bias" not in h:
                return h
            return hnew
        ord = v6.order_evidence(x, h["order_cols"])
        pr = np.exp(7 * (ord - ord.max(1, keepdims=True)))
        pr /= pr.sum(1, keepdims=True)
        orig = h["order_means"]
        pred = hnew["order_means"].copy()
        if kind == "pooledfast":
            pred[2] = max(orig[2], pred[2])
            hnew["prior"] = h["prior"]
        elif kind in ("integral", "transfer"):
            mass = pr.mean(0)
            missing = max(0.0, (orig - pred) @ mass + 0.2 * (h["prior"] - hnew["prior"]))
            pred[2] += min(
                (1 if kind == "integral" else 0.5) * missing / max(0.05, mass[2]), np.quantile(y, 0.9) * 0.65
            )
        elif kind == "energy":
            pred[:2] = np.minimum(orig[:2], pred[:2])
            pred[2] = np.quantile(y, 0.95)
        else:
            pred[[0, 2]] = orig[[0, 2]]
            hnew["prior"] = h["prior"]
        return {**hnew, "order_means": pred}
    if mode.startswith(("lifecycle", "liferank", "trackfit", "lifestage", "stagefit", "geolife")):
        # Use exactly one vote per source trajectory to estimate nominal remaining time.
        # Source window counts disproportionately represent long-lived devices.
        tag, blend = mode.split("-")
        blend = float(blend)
        theta = np.array([np.mean(np.sqrt(np.maximum(y[r], 0.0))) for r in runs])
        order = v6.order_evidence(x, h["order_cols"])
        w = np.exp(7.0 * (order - order.max(1, keepdims=True)))
        w /= w.sum(1, keepdims=True)
        prob = np.array([w[r].mean(0) for r in runs])
        if tag in ("trackfit", "stagefit"):
            mu = np.median(theta)
            # Regression of a trajectory-level response on average order probabilities.
            gram = prob.T @ prob + 0.12 * np.eye(3)
            pri = np.linalg.solve(gram, prob.T @ theta + 0.12 * mu)
            pri = np.sort(np.clip(pri, theta.min(), theta.max()) ** 2)
        elif tag == "geolife":
            pri = np.quantile(theta, [0.1, 0.40, 0.90]) ** 2
        elif tag in ("liferank", "lifestage"):
            pri = np.array([np.mean(b) ** 2 for b in np.array_split(np.sort(theta), 3)])
        else:
            wt = prob ** 3
            wt /= np.maximum(wt.sum(1, keepdims=True), 1e-10)
            prior = np.average(theta, weights=np.ones(len(runs)))
            pri = np.sort(((wt.T @ theta + 0.5 * prior) / (wt.sum(0) + 0.5)) ** 2)
        h = dict(
            h,
            order_means=(1 - blend) * h["order_means"] + blend * pri,
            prior=np.asarray((1.0 - blend) * h["prior"] + blend * (theta.mean() ** 2)),
        )
        if tag in ("lifestage", "stagefit"):
            q = np.sort([np.quantile(y[r], 0.25) for r in runs])
            med = np.sort([np.median(y[r]) for r in runs])
            h["curve_short"] = (1.0 - blend) * h["curve_short"] + blend * np.array(
                [b.mean() for b in np.array_split(q, 3)]
            )
            h["curve_spans"] = (1.0 - blend) * h["curve_spans"] + 2 * blend * np.array(
                [b.mean() for b in np.array_split(med, 3)]
            )
        return h
    ex, between, jack = MODES[mode]
    ids = np.random.default_rng(2027).choice(len(y), min(1536, len(y)), replace=False)
    gid = np.full(len(y), -1, int)
    for j, r in enumerate(runs):
        gid[r] = j
    keep = gid[ids] >= 0
    ids = ids[keep]
    design = v6._design(x[ids], h)
    residual = y[ids] - model.predict(x[ids])
    g = gid[ids]
    n = np.bincount(g, minlength=len(runs))
    mass = np.maximum(n, 1.0) ** (-ex)
    mass /= np.mean(mass[g])
    wt = mass[g]
    gram = (design.T * wt) @ design
    rhs = design.T @ (wt * residual)
    if between < 1.0 or jack:
        # Separate within-device slopes from offsets unique to each observed lifetime.
        means = np.stack(
            [design[g == j].mean(0) if n[j] else np.zeros(design.shape[1]) for j in range(len(n))]
        )
        targets = np.bincount(g, weights=residual, minlength=len(n)) / np.maximum(n, 1)
        device_mass = mass * n
        center = np.average(means, axis=0, weights=device_mass)
        offsets = means - center
        gram -= (1.0 - between) * (offsets.T * device_mass) @ offsets
        rhs -= (1.0 - between) * offsets.T @ (device_mass * targets)
    gram.flat[:: len(gram) + 1] += len(design) * 0.008
    coef = np.linalg.solve(gram, rhs)
    if jack:
        # Source-validation directional consensus of independent trajectory gradients.
        diff = design @ (coef - h["coef"])
        base = design @ h["coef"] - residual
        num = np.bincount(g, weights=-base * diff, minlength=len(n))
        den = np.bincount(g, weights=diff * diff, minlength=len(n))
        rate = np.quantile(np.clip(num / np.maximum(1.0, den), 0.0, 1.0), 0.35)
        coef = h["coef"] + rate * (coef - h["coef"])
    return {**h, "coef": coef}


def gated_legacy(impl, model, h, x, base):
    if "phase_shift" not in h and "vote_shift" not in h:
        return v6._legacy_prediction(model, h, x, base)
    est = impl.corrected(model, h, x)
    low, high = h["amplitude_bounds"]
    intensity = impl._amplitude(x, h["rms_columns"])
    weight = 0.60 * (1.0 - np.clip((intensity - low) / max(high - low, 1e-9), 0, 1))
    pr = impl.order_evidence(x, h["order_cols"])
    pr = np.exp(7.0 * (pr - pr.max(1, keepdims=True)))
    pr /= pr.sum(1, keepdims=True)
    prior = (1 - 0.5 / 0.6) * h["prior"] + (0.5 / 0.6) * (pr @ h["order_means"])
    if "vote_shift" in h:
        name = str(h["vote_mode"])
        aa = base if name in ("votebase", "votefull", "votepool", "voteprefix") else pr[:, 2]
        bb = impl._band_ratio(x, h["curve_bands"]) if name not in ("votepair", "voteunit") else base
        delta = (pr @ h["vote_shift"]) * h["vote_rate"][
            np.searchsorted(h["vote_cuts"], aa - 1e-10), np.searchsorted(h["vote_knots"], bb - 1e-10)
        ]
    else:
        z = np.clip((np.sign(x) * np.log1p(np.abs(x)) - h["center"]) / h["scale"], -6, 6)
        phase = h["phase_coef"][0] + z @ h["phase_coef"][1:]
        a, b = h["phase_cuts"]
        t = np.clip((phase - a) / max(1e-4, b - a), 0.0, 1.0)
        delta = pr @ h["phase_shift"]
        k = float(h["phase_rate"])
        gate = t ** k
        name = str(h["phase_mode"])
        if name.startswith("phaseboth"):
            delta *= np.where(delta > 0, gate, (1 - t) ** k)
        elif name.startswith("phaseall"):
            delta *= gate
        elif name.startswith("late"):
            delta *= np.where(delta > 0, gate, 1.0 + 0.5 * (1 - t))
        else:
            delta = np.minimum(0.0, delta) + gate * np.maximum(0.0, delta)
        if name.startswith("bound"):
            delta = pr @ h["phase_shift"]
            # Source-scale allowance is tapered from estimated early to late life.
            if name == "boundv6":
                delta = delta * 0
            limit = k * float(h["phase_amplitude"]) * (1.0 - t) ** 2
            delta -= limit
    cur = (1 - weight) * est + weight * (prior + delta)
    if "tail_cut" in h:
        mask = (base > h["tail_cut"]) & (cur < base)
        if mask.any():
            hi = np.clip((base[mask] - h["tail_cut"]) / h["tail_range"], 0, 1)
            cur[mask] += h["tail_coverage"] * hi * impl.familiarity(x[mask], h) * (base[mask] - cur[mask])
    return np.maximum(0.0, cur)


def install(route):
    if route == "frozen":
        return module("production_b7", HERE / "bearing.py")
    impl = module("ablation_b", BASE / "bearing.py")
    if route == "v6":
        return impl

    def fit(m, x, y, i):
        heads = v6.fit_candidates(m, x, y, i)
        if route.startswith("slew-"):
            kind, val = route.split("-")
            blend = float(val)
            runs = v6._training_runs(y)
            complete = bool(
                runs
                and np.mean([y[r.stop - 1] <= max(1e-8, 1.1 * np.median(abs(np.diff(y[r])))) for r in runs])
                >= 0.65
            )
            prev = m.meta.get("bearing_prior_mass", 1.0 if i == 1 and complete else 0.0)
            mass = prev + blend * (float(complete) - prev)
            m.meta["bearing_prior_mass"] = mass
            for tag, h in heads.items():
                raw = calibration(m, h, x, y, "shrinkrff-1") if not complete else h
                if complete:
                    other = calibration(m, h, x, y, "masssafe-0.90")
                    if mass < 0.95:
                        raw = calibration(m, h, x, y, "shrinkrff-1")
                    raw = {
                        **raw,
                        "order_means": h["order_means"] + mass * (other["order_means"] - h["order_means"]),
                        "prior": h["prior"] + mass * (other["prior"] - h["prior"]),
                    }
                heads[tag] = raw
            return heads
        if route.startswith("choice"):
            original = dict(heads)
            for tag, h in original.items():
                heads[tag] = calibration(m, h, x, y, "certweak-0.90")
                if np.mean(y <= 0.02 * np.quantile(y, 0.9)) >= 0.025 and len(original) == 1:
                    proposal = calibration(m, h, x, y, route.split("choice-")[1] + "-0.90")
                    heads[tag + "_exposure"] = proposal
            return heads
        for tag, h in heads.items():
            heads[tag] = calibration(m, h, x, y, route)
        return heads

    impl.fit_candidates = fit
    impl._legacy_prediction = lambda m, h, x, b: gated_legacy(impl, m, h, x, b)
    return impl


def one(route, mode, i):
    if i not in range(20):
        raise ValueError("Reserved stream")
    import benchmark, opening_study

    sys.modules["bearing"] = install(route)
    dest = OUT / route
    dest.mkdir(exist_ok=True)
    if mode == "opening":
        opening_study.OUT = dest
        opening_study.run("B", "v6")
    else:
        benchmark.OUT = dest
        benchmark.run("B", "v6", i, mode)


def folds(routes):
    from industrial import Model
    import contest_submission as cs

    helper = module("b3", HERE.parent / "V3/bearing_v3_study.py")
    start = Model.load(helper.loader.DATA / "checkpoint.npz")
    result = {}

    def evaluate(m, x, y, z, t):
        head = v6.fit(m, x, y, 1)
        scores = {}
        for r in routes:
            impl = install(r)
            clone = copy.deepcopy(m)
            if r in ("v6", "frozen"):
                proposal = impl.fit(clone, x, y, 1)
            else:
                proposal = calibration(clone, head, x, y, r)
            scores[r] = float(np.mean((impl.predict(clone, proposal, z) - t) ** 2) ** 0.5)
        return scores

    for name, (x, y, group) in [
        ("XJTU", helper.loader.load("train", start)),
        ("FEMTO", helper.external(start)),
    ]:
        folds = {}
        for device in np.unique(group):
            train = group != device
            m = Model("regression", start.meta["paths"], [], x[train], y[train], hidden=24)
            m.fit(x[train], y[train], 15)
            cs.continue_mlp("B", 1, m, x[train], y[train], "half")
            folds[str(device)] = evaluate(m, x[train], y[train], x[~train], y[~train])
        result[name] = folds
    x, y, _ = helper.loader.load("train", start)
    z, t, _ = helper.loader.load("validation", start)
    cs.continue_mlp("B", 1, start, x, y, "half")
    result["public"] = evaluate(start, x, y, z, t)
    ext = module("b4", HERE.parent / "V4/bearing_v4_study.py")
    ext.OUT = OUT
    z, t = ext.kaist_samples(start, helper)
    result["KAIST"] = evaluate(start, x, y, z, t)
    (OUT / ("external-" + "-".join(routes) + ".json")).write_text(json.dumps(result, indent=2))


def sweep(routes, modes, orders, workers):
    specs = [(r, m, i) for r in routes for m in modes for i in (orders[:1] if m == "opening" else orders)]

    def job(spec):
        r, m, i = spec
        dest = OUT / r / ("v6-B.json" if m == "opening" else f"{m}-v6-B-{i}.json")
        if dest.exists():
            return
        with (OUT / f"{r}-{m}-{i}.log").open("w") as log:
            subprocess.run(
                [sys.executable, "-B", __file__, "one", r, m, str(i)], stdout=log, stderr=log, check=True
            )

    with ThreadPoolExecutor(workers) as pool:
        list(pool.map(job, specs))


def known_mask(code):
    """Refit the frozen candidate on each Cartesian product of the three public windows."""
    import hashlib
    import benchmark
    import tempfile

    original = benchmark.partitions
    grid = HERE.parent / "work/v6/bearing-grid"
    archive = {r["audit_mask"]: r for r in json.loads((grid / "design.json").read_text())["grid"]}
    out = OUT / "grid"
    out.mkdir(exist_ok=True)

    def partition(rows, key, split):
        release, phase, audit = original(rows, key, 0)
        if len(rows) == 1679:
            audit[:] = False
            devices = np.array([r["meta"]["bearing"] for r in rows])
            for j, name in enumerate(np.unique(devices)):
                ids = np.flatnonzero(devices == name)
                ids = ids[np.argsort([rows[i]["meta"]["snapshot"] for i in ids])]
                for k, block in enumerate(np.array_split(ids, 4)):
                    audit[np.array_split(block, 3)[(code // (3 ** j) + k) % 3]] = True
            ref = archive[hashlib.sha256(audit.tobytes()).hexdigest()]
            partition.ref = ref
        return release, phase, audit

    benchmark.partitions = partition
    sys.modules["bearing"] = install("frozen")
    with tempfile.TemporaryDirectory(dir=out) as tmp:
        benchmark.OUT = Path(tmp)
        benchmark.run("B", "v6", 0, "schedule")
        actual = json.loads((Path(tmp) / "schedule-v6-B-0.json").read_text())
        ref = partition.ref
        control = json.loads((grid / f"schedule-v6-B-{ref['split']}.json").read_text())
        for a, b in zip(control["rounds"], actual["rounds"]):
            assert all(a[k] == b[k] for k in ("stage", "audit", "selection", "static"))
        (out / f"mask-{code:02}.json").write_text(json.dumps({"mask": ref, "frozen": actual, "v6": control}))


def serve(routes):
    """Real candidate selection, full-validation gain and parsed singleton requests."""
    import benchmark, serving_study as svc, contest_submission as agent

    svc.ASSETS = OUT / "artifacts"
    svc.RESULT = OUT / "serving"
    svc.WORK = OUT
    static = agent.Model.load(benchmark.WORK / "B" / "checkpoint.npz")
    valid = svc.rows("B", "validation")
    xv, yv, _ = agent.projection(valid, static, "B")
    measured = {}
    for r in routes:
        sys.modules["bearing"] = install(r)
        svc.make_artifact("B", r)
        dest = svc.ASSETS / r / "B"
        m, h, _ = agent.artifact(dest)
        pred = agent.values("B", m, h, xv).reshape(-1)
        baseline = agent.values("B", static, {}, xv).reshape(-1)
        measured[r] = dict(
            selected=json.loads((dest / "record.json").read_text()),
            count=len(yv),
            rmse=float(np.mean((yv - pred) ** 2) ** 0.5),
            baseline_rmse=float(np.mean((yv - baseline) ** 2) ** 0.5),
            mae=float(np.abs(yv - pred).mean()),
            checkpoint_bytes=sum(f.stat().st_size for f in dest.glob("*.npz")),
        )
        for i in range(7):
            svc.measure("B", r, "v6", i)
    (OUT / "opening_full.json").write_text(json.dumps(measured, indent=2))


def verify():
    """Cross-check the frozen implementation against its independent experimental form."""
    import contest_submission as cs
    import time
    from industrial import Model

    helper = module("b3", HERE.parent / "V3/bearing_v3_study.py")
    start = Model.load(helper.loader.DATA / "checkpoint.npz")
    x, y, groups = helper.loader.load("train", start)
    z, t, valid_groups = helper.loader.load("validation", start)
    cs.continue_mlp("B", 1, start, x, y, "half")
    new = install("frozen")
    experiment = install("slew-0.3")
    full = np.arange(len(x))
    censored = np.sort(
        np.concatenate(
            [
                g[: max(1, int(0.85 * len(g)))]
                for g in (np.flatnonzero(groups == i) for i in np.unique(groups))
            ]
        )
    )
    old_head = v6.fit(start, x, y, 1)
    report = {}
    legacy = v6.predict(start, old_head, z)
    report["old_batch_abs"] = float(np.max(abs(legacy - new.predict(start, old_head, z))))
    old_row = new.prepare_inference(start, old_head)
    report["old_scalar_abs"] = float(np.max(abs(legacy - np.array([old_row(r) for r in z]))))
    for name, releases in [
        ("full", [full]),
        ("changing", [censored, full, full, censored]),
        ("shuffled", [np.random.default_rng(412).permutation(full)]),
    ]:
        m1 = copy.deepcopy(start)
        m2 = copy.deepcopy(start)
        output = []
        for i, ids in enumerate(releases, 1):
            h1 = experiment.fit_candidates(m1, x[ids], y[ids], i)
            h2 = new.fit_candidates(m2, x[ids], y[ids], i)
            assert set(h1) == set(h2)
            for k in h1:
                a, b = h1[k], h2[k]
                assert set(a) == set(b)
                orig = experiment.predict(m1, a, z)
                actual = new.predict(m2, b, z)
                row = new.prepare_inference(m2, b)
                pred = np.array([row(r) for r in z])
                output.append(
                    dict(
                        index=i,
                        head=k,
                        rows=len(ids),
                        serialized_array_bytes=sum(v.nbytes for v in b.values()),
                        max_coef=float(np.max(abs(a["coef"] - b["coef"]))),
                        abs_prediction=float(np.max(abs(actual - orig))),
                        scalar_abs=float(np.max(abs(actual - pred))),
                        mass_research=m1.meta.get("bearing_prior_mass"),
                        mass_frozen=m2.meta.get("bearing_prior_mass"),
                    )
                )
                assert output[-1]["abs_prediction"] < 1e-8 and output[-1]["scalar_abs"] < 1e-6
        report[name] = output
    samples = z[np.linspace(0, len(z) - 1, 400, dtype=int)]
    times = {}
    cost = {}
    for name, impl in [("v6", v6), ("frozen", new)]:
        rates = []
        fits = []
        checkpoints = []
        for k in range(7):
            clone = copy.deepcopy(start)
            t0 = time.process_time_ns()
            head = impl.fit(clone, x, y, 1)
            fits.append((time.process_time_ns() - t0) / 1e9)
            row = impl.prepare_inference(clone, head)
            for r in samples:
                row(r)
            t0 = time.process_time_ns()
            for r in samples:
                row(r)
            rates.append((time.process_time_ns() - t0) / 1000 / len(samples))
        times[name] = rates
        cost[name] = fits
    report.update(scalar_cpu_us=times, fit_cpu_s=cost)
    (OUT / "verification.json").write_text(json.dumps(report, indent=2))


def summarize():
    """Aggregate paired scores without treating repeated public windows as independent units."""
    from statistics import mean, median

    def aggregate(records):
        stages = range(len(records[0]["stage_means"]))
        return {
            **{
                key: mean(row[key] for row in records)
                for key in ("stage_mean", "stage_floor", "worst", "delay", "forget")
            },
            "stages": [mean(r["stage_means"][k] for r in records) for k in stages],
            "cpu_per_round": mean(round_["cpu"] for record in records for round_ in record["rounds"]),
            "calls_per_round": mean(round_["calls"] for record in records for round_ in record["rounds"]),
            "payload_bytes": mean(
                round_["checkpoint_bytes"] for record in records for round_ in record["rounds"]
            ),
        }

    summary = {}
    routes = sorted(d for d in OUT.iterdir() if d.is_dir() and any(d.glob("schedule-*.json")))
    sections = [("pilot", [0, 5, 8, 10]), ("development", list(range(14))), ("held", list(range(14, 20)))]
    for label, orders in sections:
        data = {}
        for folder in routes:
            if label == "held" and folder.name not in {"v6", "frozen"}:
                continue
            modes = {}
            for mode in ("schedule", "trace", "drift", "official", "fresh"):
                files = [folder / f"{mode}-v6-B-{i}.json" for i in orders]
                if all(f.is_file() for f in files):
                    modes[mode] = aggregate([json.loads(f.read_text()) for f in files])
            if modes:
                data[folder.name] = modes
        summary[label] = {"orders": orders, "routes": data}
    rows = [json.loads(f.read_text()) for f in sorted((OUT / "grid").glob("mask-*.json"))]
    assert len(rows) == 27 and len({r["mask"]["audit_mask"] for r in rows}) == 27
    for name, group in [
        ("grid27", rows),
        ("old_unseen18", [r for r in rows if not r["mask"]["seen_before"]]),
    ]:
        scores = {r: aggregate([row[r] for row in group]) for r in ("v6", "frozen")}
        for k in ("stage_floor", "stage_mean", "worst", "delay", "forget"):
            pairs = [(r["v6"][k], r["frozen"][k]) for r in group]
            scores[k] = {
                "up": sum(f > v + 1e-9 for v, f in pairs),
                "down": sum(f < v - 1e-9 for v, f in pairs),
                "positive": [sum(x[i] > 0 for x in pairs) for i in (0, 1)],
            }
        summary[name] = scores
    summary["opening_excluded"] = {
        r: json.loads((OUT / r / "v6-B.json").read_text()) for r in ("v6", "frozen")
    }
    summary["opening_full"] = json.loads((OUT / "opening_full.json").read_text())
    summary["external"] = json.loads((OUT / "external-v6-frozen.json").read_text())
    summary["verification"] = json.loads((OUT / "verification.json").read_text())
    summary["serving"] = {
        r: {
            k: median(
                json.loads(f.read_text())["compiled"][k] for f in (OUT / "serving").glob(f"B-{r}-v6-*.json")
            )
            for k in ("p50_us", "p95_us", "mean_us")
        }
        for r in ("v6", "frozen")
    }
    summary["freeze"] = json.loads((OUT / "choice.json").read_text())
    (OUT / "bearing-results.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    a, *args = sys.argv[1:]
    if a == "one":
        one(args[0], args[1], int(args[2]))
    elif a == "mask":
        known_mask(int(args[0]))
    elif a == "grid":
        with ThreadPoolExecutor(int(args[0])) as pool:
            list(
                pool.map(
                    lambda i: subprocess.run([sys.executable, "-B", __file__, "mask", str(i)], check=True),
                    range(27),
                )
            )
    elif a == "serve":
        serve(args[0].split(","))
    elif a == "summary":
        summarize()
    elif a == "verify":
        verify()
    elif a == "folds":
        folds(args[0].split(","))
    else:
        sweep(args[0].split(","), args[1].split(","), list(map(int, args[2].split(","))), int(args[3]))
