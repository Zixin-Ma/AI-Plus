"""Bounded, condition-aware neural residual on the continued C-MAPSS checkpoint."""
import numpy as np


def _regimes(x, centers):
    # These two sensor settings distinguish the six documented operating regimes.
    settings = x[:, [-3, -1]]
    return np.sum((settings[:, None] - centers) ** 2, axis=2).argmin(axis=1)


def _training_residual(raw, target, aligned=True):
    """Blend unconstrained and deployed residuals with bounded signed tail weights."""
    error = raw - target
    # The serving head clips the correction to 65 cycles and halves extensions.
    # A small outside-clip slope keeps a saturated training prediction movable.
    residual = error
    if aligned:
        scale = np.where(raw > 0., .50, 1.)
        clipped = np.clip(raw, -65/55, 65/55)
        slope = scale * np.where(abs(raw) < 65/55, 1., .08)
        residual = .40*error + .60*(scale*clipped-target)*slope
    weights = np.exp(np.clip(np.where(residual > 0., residual/.85, -residual/1.11), 0., 3.2))
    positive = residual > 0.
    for mask in (positive, ~positive):
        if np.any(mask):
            weights[mask] /= weights[mask].mean()
    return residual*(1+.15*(weights-1))


def _curvature_features(sensors, axes):
    projections = np.clip(sensors @ axes, -3., 3.)
    return .5*(projections**2-1.)


def fit_neural(model, x, y, limit=24576, epochs=20, width=48, ordinal_weight=.7, stable=False):
    rng = np.random.default_rng(145)
    if len(x) > limit:
        ids = rng.choice(len(x), limit, replace=False)
        x, y = x[ids], y[ids]
    centers, counts = np.unique(np.round(x[:, [-3, -1]]), axis=0, return_counts=True)
    centers = centers[np.argsort(-counts)[:8]]
    groups = _regimes(x, centers)
    means = np.stack([x[groups == k, :-3].mean(0) for k in range(len(centers))])
    scales = np.stack([x[groups == k, :-3].std(0) for k in range(len(centers))])
    inv_scale = 1 / np.maximum(scales, 1e-4)
    sensors = np.clip((x[:, :-3] - means[groups]) * inv_scale[groups], -6, 6)
    # Low-variance sensor contrasts and the main common-mode directions encode
    # different degradation patterns. All axes are learned from released rows.
    sensor_count = sensors.shape[1]
    cov = sensors.T @ sensors / len(sensors)
    eigenvalues, axes = np.linalg.eigh(cov)
    active = np.flatnonzero(eigenvalues > .035)
    curve_axis = None
    if len(active) >= 6:
        chosen = np.r_[active[:3], active[-3:]]
        curve_axis = axes[:, chosen] / np.sqrt(eigenvalues[chosen] + .10)
        sensors = np.column_stack((sensors, _curvature_features(sensors, curve_axis)))
    base = model.predict(x).reshape(-1)
    design = np.column_stack((sensors, (base - 100) / 60, np.eye(len(centers))[groups])).astype(np.float32)
    target = ((y - base) / 55).astype(np.float32)
    thresholds = np.quantile(y, [.20, .45, .70])
    near_threshold = (y[:, None] < thresholds).astype(np.float32)

    # The initial predictor and RNG state match the original head. Curvature
    # channels start at zero and are activated by gradients from released rows.
    weights = rng.normal(0, .22, (sensor_count + 1 + len(centers), width)).astype(np.float32)
    if curve_axis is not None:
        zeros = np.zeros((curve_axis.shape[1], width), dtype=np.float32)
        weights = np.vstack((weights[:sensor_count], zeros, weights[sensor_count:]))
    bias = np.zeros(width, dtype=np.float32)
    # Column zero predicts the standardised residual; other columns supervise
    # rank at three training-only lifetime thresholds for the optional stable-pool ordinal readout.
    output = np.zeros((width, 4), dtype=np.float32)
    output[:, 0] = rng.normal(0, .03, width)
    intercept = np.zeros(4, dtype=np.float32)
    params = [weights, bias, output, intercept]
    first = [np.zeros_like(a) for a in params]
    second = [np.zeros_like(a) for a in params]
    step = 0
    for epoch in range(epochs):
        order = rng.permutation(len(y))
        rate = .003 / (1 + epoch / 10)
        for idx in np.array_split(order, max(1, len(y) // 256)):
            batch = design[idx]
            hidden = np.tanh(batch @ weights + bias)
            delta = hidden @ output + intercept
            delta[:, 0] = _training_residual(delta[:, 0], target[idx], not stable) * (2 / len(idx))
            logits = np.clip(delta[:, 1:], -30., 30.)
            delta[:, 1:] = (1 / (1 + np.exp(-logits)) - near_threshold[idx]) * (ordinal_weight / (3*len(idx)))
            back = (delta @ output.T) * (1 - hidden*hidden)
            gradients = (batch.T @ back, back.sum(0), hidden.T @ delta, delta.sum(0))
            step += 1
            for param, momentum, variance, grad in zip(params, first, second, gradients):
                momentum *= .9
                momentum += .1 * grad
                variance *= .999
                variance += .001 * grad**2
                param -= rate * momentum / (1 - .9**step) / (np.sqrt(variance / (1 - .999**step)) + 1e-8)
    # Keep deployment arithmetic in the same precision as existing checkpoints.
    weights = weights.astype(float)
    bias = bias.astype(float)
    head = dict(centers=centers, means=means, inv_scale=inv_scale,
                w_sensor=weights[:-len(centers)-1].copy(), w_base=weights[-len(centers)-1] / 60,
                bias=bias + weights[-len(centers):] - weights[-len(centers)-1] * (100 / 60),
                output=55 * output[:, 0].astype(float), intercept=55 * intercept[:1].astype(float))
    if curve_axis is not None:
        head["curve_axis"] = curve_axis
        head["curve_weight"] = head["w_sensor"][sensor_count:].copy()
        head["w_sensor"] = head["w_sensor"][:sensor_count].copy()
    if stable:
        head.update(ord_out=output[:,1:].astype(float), ord_bias=intercept[1:].astype(float))
    return head


def fit(model, x, y, index, previous=None, stable_release=False):
    """Shorter updates for new releases; revisit the full fit on a stable pool."""
    stable = bool(stable_release and previous)
    if stable:
        state = fit_neural(model, x, y, epochs=20, width=48, ordinal_weight=.2, stable=True)
    else:
        state = fit_neural(model, x, y, epochs=12, width=64)
    idx = np.linspace(0,len(x)-1,min(len(x),2048),dtype=int)
    estimate = predict_neural(model,state,x[idx],65.)
    lo,hi = np.quantile(estimate,[.35,.65])
    # Least squares shrinks extremes toward the mean. Correct the tails mildly
    # when released training lifetimes vary more than the model predictions.
    amp = np.clip(np.std(y)/max(1.,np.std(estimate)) - 1., .15,.7)
    state.update(health_lo=np.array(lo),health_hi=np.array(hi),health_amp=np.array(amp))
    state["extension_fraction"] = np.asarray(.50)
    if stable:
        state = _calibrate_ordinal(model, state, x, y)
        if "ord_coef" not in state:
            state.pop("ord_out")
            state.pop("ord_bias")
    return _calibrate_output(model, state, x, y, stable)


def _residual(head, z, base, group):
    value = z @ head["w_sensor"]
    if "curve_axis" in head:
        value = value + _curvature_features(z, head["curve_axis"]) @ head["curve_weight"]
    hidden = np.tanh(value + base * head["w_base"] + head["bias"][group])
    return hidden @ head["output"] + head["intercept"][0]


def predict_neural(model, head, x, correction_limit=None, base=None):
    if base is None:
        base = model.predict(x).reshape(-1)
    if len(x) == 1:
        centers = head["centers"]
        group = np.argmin((x[0, -3] - centers[:, 0]) ** 2 + (x[0, -1] - centers[:, 1]) ** 2)
        z = np.clip((x[0, :-3] - head["means"][group]) * head["inv_scale"][group], -6, 6)
        correction = _residual(head, z, base[0], group)
        if correction_limit is not None:
            correction = min(correction_limit, max(-correction_limit, correction))
        return np.maximum(0, base + correction)
    group = _regimes(x, head["centers"])
    z = np.clip((x[:, :-3] - head["means"][group]) * head["inv_scale"][group], -6, 6)
    correction = _residual(head, z, base[:, None], group)
    if correction_limit is not None:
        correction = np.clip(correction, -correction_limit, correction_limit)
    return np.maximum(0, base + correction)


def _predict_prior(model, head, x):
    base = model.predict(x).reshape(-1)
    p = predict_neural(model, head, x, correction_limit=65., base=base)
    if "health_amp" not in head:
        return p
    # Life extensions may delay maintenance. Preserve the full downward correction
    # and apply a fixed fraction to extensions beyond the continued backbone.
    fraction = float(head.get("extension_fraction", 1.))
    if len(x) == 1:
        value = float(p[0])
        high = float(head["health_hi"])
        low = float(head["health_lo"])
        tail = max(0., value-high) + min(0., value-low)
        correction = min(25., max(-25., .2*float(head["health_amp"])*tail))
        value = max(0., value+correction)
        p[0] = value - (1.-fraction)*max(0., value-base[0])
        return np.maximum(0., p)
    tails = np.maximum(0., p-head["health_hi"]) + np.minimum(0., p-head["health_lo"])
    p = np.maximum(0., p+np.clip(.2*head["health_amp"]*tails, -25., 25.))
    return np.maximum(0., p-(1.-fraction)*np.maximum(0., p-base))


def _ordinal_hidden(model, head, x):
    base = model.predict(x).reshape(-1)
    group = _regimes(x, head["centers"])
    z = np.clip((x[:,:-3]-head["means"][group])*head["inv_scale"][group], -6., 6.)
    value = z@head["w_sensor"]
    if "curve_axis" in head:
        value = value + _curvature_features(z, head["curve_axis"])@head["curve_weight"]
    hidden = np.tanh(value+base[:,None]*head["w_base"]+head["bias"][group])
    return group, hidden


def _ordinal_basis(head, prior, group, hidden):
    q = 1/(1+np.exp(-np.clip(hidden@head["ord_out"]+head["ord_bias"], -30., 30.)))
    age_proxy = np.clip((prior-100)/60, -2., 3.)
    return np.column_stack((np.ones(len(prior)), age_proxy, q, q*q, q*age_proxy[:,None],
                            np.eye(len(head["centers"]))[group]))


def predict(model, head, x):
    prior = _predict_prior(model, head, x)
    if "ord_coef" in head:
        group, hidden = _ordinal_hidden(model, head, x)
        delta = _ordinal_basis(head, prior, group, hidden)@head["ord_coef"]
        prior = np.maximum(0., prior+np.clip(delta, -10., 10.))
    return prior


def _reserve_rows(n, separate_ordinal=False):
    if n < 24576 + 512:
        return np.empty(0, dtype=int)
    rng = np.random.default_rng(145)
    free = np.ones(n, dtype=bool)
    free[rng.choice(n, 24576, replace=False)] = False
    pool = np.flatnonzero(free)
    ordinal = rng.choice(pool, min(len(pool), 16384), replace=False)
    if not separate_ordinal:
        return ordinal
    free[ordinal] = False
    pool = np.flatnonzero(free)
    if len(pool) < 512:
        return np.empty(0, dtype=int)
    return np.random.default_rng(621).choice(pool, min(16384, len(pool)), replace=False)


def _calibrate_output(model, head, x, y, stable):
    """Fit a readout update on reserved rows, then check lifetime bands and both cost tails."""
    ids = _reserve_rows(len(y), separate_ordinal=stable)
    if not len(ids):
        return head
    xx, target = x[ids], y[ids]
    error = predict(model, head, xx) - target
    base = model.predict(xx).reshape(-1)
    _, hidden = _ordinal_hidden(model, head, xx)
    raw = hidden @ head["output"] + head["intercept"][0]
    value = np.maximum(0., base + np.clip(raw, -65., 65.))
    tail = np.maximum(0., value-head["health_hi"]) + np.minimum(0., value-head["health_lo"])
    amp = .2 * float(head["health_amp"])
    pre = np.maximum(0., value + np.clip(amp*tail, -25., 25.))
    slope = ((np.abs(raw) < 65.) & (value > 0.)).astype(float)
    slope *= 1 + amp*((tail != 0.) & (np.abs(amp*tail) < 25.))
    slope *= np.where(pre > base, float(head["extension_fraction"]), 1.)
    jac = np.column_stack((hidden, np.ones(len(ids)))) * slope[:,None]
    bands = np.searchsorted(np.quantile(target, [.2, .4, .6, .8]), target)
    fit_rows = np.arange(len(ids)) % 2 == 0
    check = ~fit_rows

    def band_ratios(err, rows):
        groups = bands[rows]
        loss = np.bincount(groups, weights=err[rows]**2, minlength=5)
        reference = np.bincount(groups, weights=(base[rows]-target[rows])**2, minlength=5)
        return loss / (100. + reference)

    ratios = band_ratios(error, fit_rows)
    importance = np.exp(np.clip(2.5*(ratios - ratios.mean()), -1.5, 1.5))
    importance /= importance.mean()
    weights = (.5 + .5*importance[bands])[fit_rows]
    j, e = jac[fit_rows], error[fit_rows]
    curvature = j.T @ (j*weights[:,None])/len(j) + np.eye(j.shape[1])*.06
    delta = np.linalg.solve(curvature, -j.T @ (weights*e)/len(j))
    normals = []
    for sign, scale in ((1., 10.), (-1., 13.)):
        exponent = sign*e/scale
        for trim in (.975, .998):
            v = sign*np.where(exponent > 0., np.exp(np.minimum(exponent, np.quantile(exponent, trim))), 0.)
            normal = v @ j / max(1., np.abs(v).sum())
            dual = np.linalg.solve(curvature, normal)
            normals.append((normal, dual, max(1e-10, normal @ dual)))
    for _ in range(12):
        for normal, dual, denom in normals:
            delta -= max(0., normal @ delta)*dual/denom
    delta /= max(1., np.sqrt(np.mean((j @ delta)**2))/6.)

    def tails(err):
        return np.array([np.mean(np.expm1(np.clip(sign*err[check]/scale, 0., 16.)))
                         for sign, scale in ((1., 10.), (-1., 13.))])

    ceiling = tails(error)*1.025
    max_band = band_ratios(error, check).max() + .002
    max_error = np.mean(error[check]**2)*1.0005
    for step in (.8, .45, .2):
        updated = dict(head, output=head["output"] + step*delta[:-1],
                       intercept=head["intercept"] + step*delta[-1])
        trial = predict(model, updated, xx) - target
        if (np.mean(trial[check]**2) <= max_error and band_ratios(trial, check).max() <= max_band
                and np.all(tails(trial) <= ceiling)):
            return updated
    return head


def _calibrate_ordinal(model, head, x, y):
    ids = _reserve_rows(len(y))
    if not len(ids):
        return head
    xx = x[ids]
    prior = predict(model, head, xx)
    group, hidden = _ordinal_hidden(model, head, xx)
    design = _ordinal_basis(head, prior, group, hidden)
    error = prior-y[ids]
    covariance = design.T@design/len(ids) + np.eye(design.shape[1])*.015
    coef = np.linalg.solve(covariance, -design.T@error/len(ids))
    # Project the RMSE descent direction against both NASA tails. Upper clipping
    # at two quantiles prevents any single engine from determining the direction.
    directions = []
    for sign, scale in ((1., 10.), (-1., 13.)):
        exponent = sign*error/scale
        for quantile in (.97, .997):
            trimmed = np.minimum(exponent, np.quantile(exponent, quantile))
            weights = sign*np.where(exponent>0, np.exp(trimmed), 0.)
            normal = weights@design/max(1., np.abs(weights).sum())
            dual = np.linalg.solve(covariance, normal)
            directions.append((normal, dual, max(1e-9, normal@dual)))
    for _ in range(12):
        for normal, dual, denom in directions:
            coef -= max(0., float(normal@coef))/denom*dual
    for fraction in (.7, .4, .2):
        delta = np.clip(design@(coef*fraction), -10., 10.)
        if all(np.sum(np.expm1(np.clip(sign*(error+delta)/scale, 0., 22.))) <=
               np.sum(np.expm1(np.clip(sign*error/scale, 0., 22.)))*(1+1e-7)
               for sign, scale in ((1., 10.), (-1., 13.))):
            head["ord_coef"] = coef*fraction
            break
    return head


def prepare_inference(model, head):
    """Prepare an equivalent fixed head for streaming single-sample inference."""
    if "w_sensor" not in head:
        return None
    centers = head["centers"]
    center_a, center_b = centers[:, 0], centers[:, 1]
    means, inv_scale = head["means"], head["inv_scale"]
    w_sensor, w_base = head["w_sensor"], head["w_base"]
    bias, output, intercept = head["bias"], head["output"], float(head["intercept"][0])
    predict_base = model.predict_one
    low = float(head.get("health_lo", 0.))
    high = float(head.get("health_hi", 0.))
    amplitude = .2*float(head.get("health_amp", 0.))
    fraction = float(head.get("extension_fraction", 1.))

    axes = head.get("curve_axis")
    curve_weight = head.get("curve_weight")
    ordinal = head.get("ord_coef")
    if ordinal is not None:
        ord_out, ord_bias = head["ord_out"], head["ord_bias"]

    def one(row):
        base = predict_base(row)
        group = np.argmin((row[-3]-center_a)**2 + (row[-1]-center_b)**2)
        sensors = (row[:-3]-means[group])*inv_scale[group]
        np.maximum(sensors, -6., out=sensors)
        np.minimum(sensors, 6., out=sensors)
        value = sensors @ w_sensor
        if axes is not None:
            value = value + _curvature_features(sensors, axes) @ curve_weight
        h = np.tanh(value + base*w_base + bias[group])
        residual = float(h @ output + intercept)
        value = max(0., base+min(65., max(-65., residual)))
        tail = max(0., value-high) + min(0., value-low)
        value = max(0., value+min(25., max(-25., amplitude*tail)))
        prior = max(0., value-(1.-fraction)*max(0., value-base))
        if ordinal is None:
            return prior
        q = 1/(1+np.exp(-np.clip(h@ord_out+ord_bias, -30., 30.)))
        t = min(3., max(-2., (prior-100)/60))
        delta = ordinal[0]+t*ordinal[1]+q@(ordinal[2:5]+q*ordinal[5:8]+t*ordinal[8:11])+ordinal[11+group]
        return max(0., prior+min(10., max(-10., delta)))
    return one
