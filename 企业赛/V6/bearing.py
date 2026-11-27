"""Remaining-life calibration using operating orders, band-energy ranks and impact support."""
from functools import lru_cache
import json
import math

import numpy as np


@lru_cache(maxsize=8)
def _feature_map(rows, dimensions):
    """Recreate the fixed NumPy PCG64 map used by both train and predict."""
    rng = np.random.default_rng(2027)
    rng.choice(rows, min(1536, rows), replace=False)
    return (rng.normal(0., 1.6 / np.sqrt(dimensions), (dimensions, 64)),
            rng.uniform(-np.pi, np.pi, 64))


def feature_map(state):
    if "projection" in state:
        return state["projection"], state["phase"]
    return _feature_map(int(state["map_rows"]), len(state["center"]))


def _design(x, state):
    log_features = np.sign(x) * np.log1p(abs(x))
    normalized = np.clip((log_features - state["center"]) / state["scale"], -6, 6)
    projection, phase = feature_map(state)
    waves = np.cos(normalized @ projection + phase)
    return np.column_stack((np.ones(len(x)), normalized, waves))


def _amplitude(x, rms_columns):
    peak_channel = np.max(x[:, rms_columns], axis=1)
    return np.log(np.maximum(peak_channel, 1e-8))


def fit(model, x, y, index):
    """Fit once on released training pairs after the caller continues the parent MLP."""
    rng = np.random.default_rng(2027)
    subset = rng.choice(len(y), min(1536, len(y)), replace=False)
    sample = x[subset]
    log_sample = np.sign(sample) * np.log1p(abs(sample))
    low, high = np.percentile(log_sample, [25, 75], axis=0)
    rms_columns = np.array([i for i, path in enumerate(model.meta["paths"])
                            if json.loads(path)[-1][1] == "rms"], dtype=np.int64)
    if len(rms_columns) != 2:
        raise ValueError("B projection requires horizontal and vertical RMS")
    amplitude = _amplitude(x, rms_columns)
    state = {
        "center": np.median(log_sample, axis=0),
        "scale": np.maximum(high - low, np.maximum(.05 * log_sample.std(axis=0), 1e-6)),
        "map_rows": np.asarray(len(y), dtype=np.int64),
        "rms_columns": rms_columns,
        "amplitude_bounds": np.quantile(amplitude, [.75, .99]),
        "prior": np.asarray(np.mean(np.sqrt(np.maximum(y, 0))) ** 2)
    }
    design = _design(sample, state)
    gram = design.T @ design
    gram.flat[::len(gram) + 1] += len(sample) * .008
    residual = y[subset] - model.predict(sample)
    state["coef"] = np.linalg.solve(gram, design.T @ residual)
    freq_cols = np.array([i for i, path in enumerate(model.meta["paths"])
                          if json.loads(path)[-1][1] == "dominant_freq"],dtype=int)
    if len(freq_cols) == 2:
        evidence = order_evidence(x, freq_cols)
        state.update(order_prior(x, y, freq_cols, evidence))
        base_predictions = None
        # Calibrating a directional tail correction requires source support
        # near both ends of life. Early-only releases keep the original prior.
        late_fraction=np.mean(y <= .02*np.quantile(y,.9))
        coverage=min(1.,late_fraction/.025)
        if coverage>0:
            base_predictions = model.predict(x)
            state.update(tail_state(x, y, base_predictions, state, .60))
            state["tail_threshold"] *= .70
            state["tail_coverage"]=np.asarray(coverage)
        # For the three specified XJTU regimes the bearing-load life formula
        # (P**-3 / rpm) increases with shaft speed as radial load decreases.
        state["order_means"] = np.sort(state["order_means"])
        _lifetime_curve(model, x, y, state, evidence, base_predictions)
        if "curve_bands" in state:
            _calibrate_release(model, state, x, y, late_fraction, base_predictions)
    return state


def fit_candidates(model, x, y, index):
    """On a mature first release compare the cross-run calibration with its parent head."""
    head = fit(model, x, y, index)
    if index == 1 and "safe_bias" in head and np.mean(y <= .02*np.quantile(y, .9)) >= .05:
        original = {k: v for k, v in head.items() if k not in ("safe_bias", "safe_cuts", "safe_knots")}
        return {"bearing": original, "consensus": head}
    return {"bearing": head}


def corrected(model, state, x):
    return np.maximum(0, model.predict(x) + .85 * (_design(x, state) @ state["coef"]))


def _predict_one(model, head, x, base):
    # The host streams one row per request. Skip building a full design matrix for a scalar.
    row = x[0]
    z = np.clip((np.sign(row) * np.log1p(abs(row)) - head["center"]) / head["scale"], -6, 6)
    projection, phase = feature_map(head)
    waves = np.cos(z @ projection + phase)
    coef = head["coef"]
    cut = len(row) + 1
    residual = coef[0] + z @ coef[1:cut] + waves @ coef[cut:]
    base=base[0]
    estimate = max(0., base + .85 * residual)
    a, b = head["rms_columns"]
    intensity = math.log(max(row[a], row[b], 1e-8))
    low, high = head["amplitude_bounds"]
    weight = .60 * (1 - min(1., max(0., (intensity-low) / max(high-low, 1e-9))))
    prediction=(1-weight)*estimate+weight*head["prior"]
    if "order_cols" in head:
        a,b=head["order_cols"]
        w0,w1,w2=_pair_weights(float(row[a]),float(row[b]))
        m0,m1,m2=head["order_means"]
        prior=w0*m0+w1*m1+w2*m2
        prediction += (.5/.60)*weight*(prior-head["prior"])
    if "tail_cut" in head and base>head["tail_cut"] and prediction<base:
        high=min(1.,(base-head["tail_cut"])/head["tail_range"])
        distance=np.min(np.mean((z-head["tail_center"])**2*head["tail_prec"],axis=1))
        low,upper=head["tail_threshold"]
        familiar=1-min(1.,max(0.,(distance-low)/max(upper-low,1e-9)))
        prediction+=head["tail_coverage"]*high*familiar*(base-prediction)
    return np.array([max(0.,prediction)])


def _legacy_prediction(model, head, x, base):
    if len(x) == 1:
        return _predict_one(model, head, x, base)
    estimate=np.maximum(0.,base+.85*(_design(x,head)@head["coef"]))
    low, high = head["amplitude_bounds"]
    intensity = _amplitude(x, head["rms_columns"])
    # Extreme acceleration should retain the signal-driven prediction even on unusual bearings.
    prior_weight = .60 * (1 - np.clip((intensity - low) / max(high - low, 1e-9), 0, 1))
    prediction = (1-prior_weight)*estimate + prior_weight*head["prior"]
    if "order_cols" in head:
        prediction += (.5/.60)*prior_weight*(order_predict(x,head)-head["prior"])
    if "tail_cut" in head:
        mask=(base>head["tail_cut"]) & (prediction<base)
        if mask.any():
            high=np.clip((base[mask]-head["tail_cut"])/head["tail_range"],0,1)
            prediction[mask]+=head["tail_coverage"]*high*familiarity(x[mask],head)*(base[mask]-prediction[mask])
    return np.maximum(0.,prediction)


def order_evidence(x, columns):
    """Compatibility of spectral peaks with shaft and rolling-element orders."""
    peaks=x[:, columns]
    speeds=np.array([35.,37.5,40.])
    ratio=7.92/34.55  # LDK UER204 ball and pitch diameters in mm
    orders=np.array([1.,4*(1-ratio),4*(1+ratio),.5*(1-ratio**2)/ratio])
    nominal=speeds[:,None]*orders
    harmonic=np.rint(peaks[:,:,None,None]/nominal)
    error=peaks[:,:,None,None]-harmonic*nominal
    width=np.maximum(1.7, .0025*peaks[:,:,None,None])
    agreement=np.exp(-.5*(error/width)**2-.04*harmonic)*(harmonic>=1)
    return agreement.max(3).mean(1)


def order_prior(x, y, columns, evidence=None):
    if evidence is None:
        evidence=order_evidence(x, columns)
    votes=np.exp(7.*(evidence-evidence.max(1,keepdims=True)))
    probabilities=votes/votes.sum(1,keepdims=True)
    roots=np.sqrt(np.maximum(y,0.))
    winner=evidence.argmax(1)
    hard=np.array([(roots[winner==j].sum()+20*roots.mean())/(np.sum(winner==j)+20) for j in range(3)])
    coverage=min(1.,np.mean(y <= .02*np.quantile(y,.9))/.025)
    penalty=.015*len(y)
    gram=probabilities.T@probabilities+penalty*np.eye(3)
    means=np.maximum(0.,np.linalg.solve(gram,probabilities.T@roots+penalty*roots.mean()))
    return {"order_means":coverage*means**2+(1-coverage)*hard**2, "order_cols":np.asarray(columns)}


def order_predict(x, state):
    evidence=order_evidence(x,state["order_cols"])
    votes=np.exp(7.*(evidence-evidence.max(1,keepdims=True)))
    return votes @ state["order_means"] / votes.sum(1)


@lru_cache(maxsize=2048)
def _single_evidence(peak):
    return order_evidence(np.array([[peak]]),[0])[0]


@lru_cache(maxsize=4096)
def _pair_weights(first, second):
    weights=np.exp(3.5*(_single_evidence(first)+_single_evidence(second)))
    return tuple(weights/weights.sum())


def tail_state(x, y, pred, state, quantile=.65):
    """Compact support estimate for long-life corrections using high predicted-life data."""
    start=np.quantile(pred,quantile)
    mask=pred>=start
    points=x[mask]
    if len(points)>1536:
        points=points[np.random.default_rng(107).choice(len(points),1536,replace=False)]
    z=np.clip((np.sign(points)*np.log1p(abs(points))-state["center"])/state["scale"],-6,6)
    rng=np.random.default_rng(110)
    centers=z[rng.choice(len(z),min(8,len(z)),replace=False)].copy()
    for _ in range(6):
        ids=((z[:,None]-centers)**2).sum(2).argmin(1)
        for i in range(len(centers)):
            sample=z[ids==i]
            if len(sample):centers[i]=sample.mean(0)
    variances=[]
    for i in range(len(centers)):
        sample=z[ids==i]
        var=sample.var(0) if len(sample)>=4 else z.var(0)
        variances.append(np.clip(var+.05*z.var(0)+.008,.008,None))
    precision=1/np.stack(variances)
    delta=((z[:,None]-centers)**2*precision).mean(2).min(1)
    return dict(tail_cut=np.asarray(start),tail_range=np.asarray(max(1.,y.std()*.40)),
                tail_center=centers,tail_prec=precision,
                tail_threshold=np.quantile(delta,[.90,.99]))


def familiarity(x, state):
    z=np.clip((np.sign(x)*np.log1p(abs(x))-state["center"])/state["scale"],-6,6)
    d=((z[:,None]-state["tail_center"])**2*state["tail_prec"]).mean(2).min(1)
    low,high=state["tail_threshold"]
    return 1.-np.clip((d-low)/max(high-low,1e-9),0,1)

def _lifetime_curve(model, x, y, state, evidence, base_predictions):
    """Learn low-band energy ranks separately for the three mechanical conditions."""
    paths = [json.loads(path)[-1][1] for path in model.meta["paths"]]
    bands = np.array([i for feature in ("band1_energy_0_2k", "band3_energy_6k_12k")
                      for i, name in enumerate(paths) if name == feature], dtype=int)
    if len(bands) != 4:
        return
    weights = np.exp(7. * (evidence - evidence.max(1, keepdims=True)))
    weights /= weights.sum(1, keepdims=True)
    weights += 1e-6
    levels = np.linspace(.025, .975, 13)
    log_bands = np.log(np.maximum(x[:, bands], 1e-12))
    low = log_bands[:, :2].mean(1)
    low_order = np.argsort(low)
    life_order = np.argsort(y)
    cuts, short, spans = [], [], []
    for condition in weights.T:
        mass = np.cumsum(condition[low_order])
        cuts.append(np.interp(levels * mass[-1], mass, low[low_order]))
        life_mass = np.cumsum(condition[life_order])
        q25, q50 = np.interp(np.array([.25, .5])*life_mass[-1], life_mass, y[life_order])
        short.append(q25)
        spans.append(2. * q50)
    # The stronger population adjustment needs enough labelled end-of-life examples.
    near_failure = int(np.count_nonzero(y <= .02 * np.quantile(y, .9)))
    mature = near_failure >= .05*len(y)
    if not mature and near_failure >= max(12, .025*len(y)):
        # On a sparse release, several distinct observed terminal signatures
        # supply endpoint evidence that a short, censored prefix cannot provide.
        terminal = x[y == 0., :][:, bands]
        mature = len(terminal) >= 3 and len(np.unique(terminal, axis=0)) >= 3
    state.update(curve_bands=bands, curve_cuts=np.asarray(cuts), curve_short=np.sort(short),
                 curve_spans=np.sort(spans), curve_mature=np.asarray(mature),
                 curve_impact=np.quantile(log_bands[:, 2:].mean(1), [.20,.75]))
    if mature:
        state["curve_base_cut"] = np.asarray(np.median(base_predictions))
        state["curve_base_width"] = np.asarray(max(1., .5 * y.std()))


_CURVE_SURVIVAL = np.linspace(.975, .025, 13)


def _rank_adjust(old, base, ranked, short, slow, fast, impact, state):
    diff = ranked - old
    if not state["curve_mature"]:
        return old + fast**2*diff + .4*slow**2*np.minimum(0., short-old)
    # Source life ranks apply to low-impact observations. Strong impulses
    # can indicate acute damage as well as a different vibration scale.
    lower, upper = state["curve_impact"]
    gate = 1. - np.clip((impact-lower) / max(upper-lower, 1e-8), 0., 1.)
    high = np.clip((base-state["curve_base_cut"]) / state["curve_base_width"], 0., 1.)
    delta = fast*np.maximum(diff, 0.) + (1.-high)*np.minimum(diff, 0.)
    delta += .5*(1.-fast)*(1.-high)*np.minimum(0., short-old)
    return old + .7*gate*delta


def _legacy_curve(model, state, x, base):
    old = _legacy_prediction(model, state, x, base)
    if "curve_bands" not in state:
        return old
    if len(x) == 1:
        row = x[0]
        a,b,c,d = state["curve_bands"]
        low = .5*(math.log(max(row[a],1e-12)) + math.log(max(row[b],1e-12)))
        high = .5*(math.log(max(row[c],1e-12)) + math.log(max(row[d],1e-12)))
        f0,f1 = state["order_cols"]
        probs = _pair_weights(float(row[f0]),float(row[f1]))
        predicted = 0.
        for weight, cuts, span in zip(probs,state["curve_cuts"],state["curve_spans"]):
            predicted += weight*span*np.interp(low, cuts, _CURVE_SURVIVAL)
        short = sum(prob*value for prob,value in zip(probs,state["curve_short"]))
        previous = old[0]
        diff = predicted - previous
        if not state["curve_mature"]:
            return np.asarray([previous + probs[2]**2*diff + .4*probs[0]**2*min(0., short-previous)])
        lower,upper = state["curve_impact"]
        coverage = 1. - min(1.,max(0.,(high-lower)/max(upper-lower, 1e-8)))
        base_strength = min(1.,max(0.,(base[0]-state["curve_base_cut"])/state["curve_base_width"]))
        drop = (1.-base_strength)*(min(diff,0.)+.5*(1.-probs[2])*min(0.,short-previous))
        return np.asarray([previous + .7*coverage*(probs[2]*max(diff,0.)+drop)])
    log_bands = np.log(np.maximum(x[:,state["curve_bands"]],1e-12))
    low, high = log_bands[:,:2].mean(1),log_bands[:,2:].mean(1)
    evidence = order_evidence(x,state["order_cols"])
    weights = np.exp(7.*(evidence-evidence.max(1,keepdims=True)))
    weights /= weights.sum(1,keepdims=True)
    ranks = np.stack([np.interp(low,cuts,_CURVE_SURVIVAL) for cuts in state["curve_cuts"]],axis=1)
    predictions = (weights*ranks) @ state["curve_spans"]
    return _rank_adjust(old,base,predictions,weights@state["curve_short"],
                        weights[:,0],weights[:,2],high,state)

def _training_runs(y):
    """Check for ordered, decreasing sequences before assigning one vote to each."""
    delta = np.diff(y)
    if len(y) < 32 or np.mean(delta >= 0.) > .14:
        return []
    boundaries = np.r_[0, np.flatnonzero(delta > 0.) + 1, len(y)]
    runs = [slice(a, b) for a, b in zip(boundaries[:-1], boundaries[1:]) if b-a >= 10]
    if not 3 <= len(runs) <= 80 or sum(run.stop-run.start for run in runs) < .85*len(y):
        return []
    return runs


def _band_ratio(x, columns):
    low, high = np.split(np.log(np.maximum(x[:, columns], 1e-12)), 2, axis=1)
    return (high-low).mean(1)


def _adjacent_sum(grid):
    smoothed = np.pad(grid, ((1, 1), (0, 0)))
    first = smoothed[1:-1] + .5*(smoothed[:-2] + smoothed[2:])
    smoothed = np.pad(first, ((0, 0), (1, 1)))
    return smoothed[:, 1:-1] + .5*(smoothed[:, :-2] + smoothed[:, 2:])


def _calibrate_release(model, state, x, y, near_failure, base):
    """Calibrate how much the released trajectories support the population head."""
    if near_failure < .025:
        state["prefix_fraction"] = np.asarray(.10 + .90*near_failure/.025)
        return
    strength = .75*(1.-np.clip((near_failure-.04)/.025, 0., 1.))
    runs = _training_runs(y)
    if not runs or strength <= 0.:
        return
    if base is None:
        base = model.predict(x)
    prediction = _legacy_curve(model, state, x, base)
    residual = y-prediction
    fractions = [.2, .4, .6, .8]
    cuts = np.quantile(base, fractions)
    ratio = _band_ratio(x, state["curve_bands"])
    knots = np.quantile(ratio, fractions)
    cells = 5*np.searchsorted(cuts, base) + np.searchsorted(knots, ratio)
    votes = []
    for run in runs:
        num = np.bincount(cells[run], weights=residual[run], minlength=25).reshape(5, 5)
        den = np.bincount(cells[run], minlength=25).reshape(5, 5)
        votes.append(_adjacent_sum(num) / (_adjacent_sum(den) + 8.))
    votes = np.asarray(votes)
    correction = np.median(votes, axis=0)*np.abs(np.sign(votes).mean(0))**2
    cap = .2*y.std()
    state.update(safe_cuts=cuts, safe_knots=knots,
                 safe_bias=strength*np.clip(correction, -cap, cap))


def predict(model, state, x):
    base = model.predict(x)
    old = _legacy_curve(model, state, x, base)
    if "prefix_fraction" in state:
        return np.maximum(0., base + state["prefix_fraction"]*(old-base))
    if "safe_bias" in state:
        i = np.searchsorted(state["safe_cuts"], base - 1e-10)
        j = np.searchsorted(state["safe_knots"], _band_ratio(x, state["curve_bands"]) - 1e-10)
        direction = np.sign(base-old)
        movement = np.minimum(abs(base-old), np.maximum(0., direction*state["safe_bias"][i,j]))
        return np.maximum(0., old+direction*movement)
    return old


# Four characteristic frequencies for each of the three shaft speeds.
_BALL_RATIO = 7.92 / 34.55
_NOMINAL_ORDERS = tuple(
    tuple(speed*order for order in (1., 4.*(1.-_BALL_RATIO), 4.*(1.+_BALL_RATIO),
                                    .5*(1.-_BALL_RATIO**2)/_BALL_RATIO))
    for speed in (35., 37.5, 40.))


@lru_cache(maxsize=2048)
def _scalar_order_scores(peak):
    width = max(1.7, .0025*peak)
    answers = []
    for family in _NOMINAL_ORDERS:
        best = 0.
        for nominal in family:
            harmonic = round(peak/nominal)
            if harmonic >= 1:
                error = (peak-harmonic*nominal)/width
                best = max(best, math.exp(-.5*error*error-.04*harmonic))
        answers.append(best)
    return answers


@lru_cache(maxsize=4096)
def _scalar_order_weights(first, second):
    first_votes = _scalar_order_scores(first)
    second_votes = _scalar_order_scores(second)
    votes = tuple(math.exp(3.5*(a+b)) for a,b in zip(first_votes,second_votes))
    total = sum(votes)
    return votes[0]/total, votes[1]/total, votes[2]/total


def _life_interpolator(cuts, span):
    """Compile the equally spaced percentiles into an interpolation over measured energy."""
    from bisect import bisect_right
    points = tuple(map(float,cuts))
    values = tuple(float(span*p) for p in _CURVE_SURVIVAL)
    slopes = tuple((right-left)/(end-start) if end>start else 0.
                   for left,right,start,end in zip(values,values[1:],points,points[1:]))
    def at(value):
        if value < points[0]:
            return values[0]
        if value >= points[-1]:
            return values[-1]
        idx = bisect_right(points,value)-1
        return values[idx]+(value-points[idx])*slopes[idx]
    return at


def prepare_inference(model, head):
    """Compile one immutable artifact for scalar requests; mixing stays with the controller."""
    if not head or "coef" not in head:
        return None
    predict_base = getattr(model,"predict_one",None)
    if predict_base is None:
        def predict_base(row):
            return float(model.predict(row[None])[0])
    projection, phase = feature_map(head)
    center, scale = head["center"], head["scale"]
    coef = head["coef"]
    cut = len(center)+1
    intercept = float(coef[0])
    linear,wave = coef[1:cut],coef[cut:]
    horizontal,vertical = map(int,head["rms_columns"])
    low_amp,high_amp = map(float,head["amplitude_bounds"])
    amp_span = max(high_amp-low_amp,1e-9)
    prior = float(head["prior"])
    scoped = "order_cols" in head
    if scoped:
        fa,fb = map(int,head["order_cols"])
        m0,m1,m2 = map(float,head["order_means"])
    support = "tail_cut" in head
    if support:
        cut_life,spread = float(head["tail_cut"]),float(head["tail_range"])
        centers,precision = head["tail_center"],head["tail_prec"]
        tl,tu = map(float,head["tail_threshold"])
        t_width = max(tu-tl,1e-9)
        coverage = float(head["tail_coverage"])
    has_curve = "curve_bands" in head
    if has_curve:
        band_a,band_b,band_c,band_d = map(int,head["curve_bands"])
        at0,at1,at2 = (_life_interpolator(cut,span)
                       for cut,span in zip(head["curve_cuts"],head["curve_spans"]))
        s0,s1,s2 = map(float,head["curve_short"])
        mature = bool(head["curve_mature"])
        if mature:
            impact_low,impact_high = map(float,head["curve_impact"])
            impact_span = max(impact_high-impact_low,1e-8)
            strong_cut = float(head["curve_base_cut"])
            strong_width = float(head["curve_base_width"])

    fraction = float(head.get("prefix_fraction", 1.))
    lookup = "safe_bias" in head
    if lookup:
        from bisect import bisect_left
        life_knots = tuple(head["safe_cuts"])
        ratio_knots = tuple(head["safe_knots"])
        biases = head["safe_bias"]

    def final_value(value, base, row):
        if fraction < 1.:
            return float(max(0., base + fraction*(value-base)))
        if lookup:
            ratio = .5*((math.log(max(row[band_c],1e-12))-math.log(max(row[band_a],1e-12)))
                        +(math.log(max(row[band_d],1e-12))-math.log(max(row[band_b],1e-12))))
            delta = biases[bisect_left(life_knots,base-1e-10),bisect_left(ratio_knots,ratio-1e-10)]
            movement = min(0., max(base-value,delta)) if base<value else max(0.,min(base-value,delta))
            return float(max(0., value+movement))
        return float(value)

    def predict_row(row):
        base = predict_base(row)
        z = (np.copysign(np.log1p(abs(row)),row)-center)/scale
        np.maximum(z,-6.,out=z)
        np.minimum(z,6.,out=z)
        residual = intercept + z@linear + np.cos(z@projection+phase)@wave
        estimate = max(0.,base+.85*residual)
        intensity = math.log(max(row[horizontal],row[vertical],1e-8))
        weight = .60*(1.-min(1.,max(0.,(intensity-low_amp)/amp_span)))
        current = (1.-weight)*estimate + weight*prior
        if scoped:
            slow,medium,fast = _scalar_order_weights(float(row[fa]),float(row[fb]))
            current += (.5/.60)*weight*(slow*m0+medium*m1+fast*m2-prior)
        if support and base>cut_life and current<base:
            diff = z-centers
            distance = (diff*diff*precision).sum(1).min()/len(z)
            familiar = 1.-min(1.,max(0.,(distance-tl)/t_width))
            current += coverage*min(1.,(base-cut_life)/spread)*familiar*(base-current)
        current = max(0.,current)
        if not has_curve:
            return final_value(current,base,row)
        if mature:
            impact = .5*(math.log(max(row[band_c],1e-12))+math.log(max(row[band_d],1e-12)))
            if impact-impact_low>=impact_span:
                return final_value(current,base,row)
            gate = 1.-min(1.,max(0.,(impact-impact_low)/impact_span))
        intensity = .5*(math.log(max(row[band_a],1e-12))+math.log(max(row[band_b],1e-12)))
        ranked = slow*at0(intensity)+medium*at1(intensity)+fast*at2(intensity)
        short = slow*s0+medium*s1+fast*s2
        change = ranked-current
        if not mature:
            return final_value(current + fast**2*change + .4*slow**2*min(0.,short-current),base,row)
        trusted = min(1.,max(0.,(base-strong_cut)/strong_width))
        drop = (1.-trusted)*(min(0.,change)+.5*(1.-fast)*min(0.,short-current))
        return final_value(current + .7*gate*(fast*max(change,0.)+drop),base,row)
    return predict_row
