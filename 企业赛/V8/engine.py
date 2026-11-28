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


def _released_targets(y, index, previous):
    """Infer complete cycle runs only from labels already released for training."""
    default_levels = (.20, .45, .70)
    inherited = bool(previous.get("trajectory_support", False)) if previous else False
    if len(y) < 512:
        return y, default_levels, False, inherited
    steps = np.diff(y)
    cycles = np.isclose(steps, -1., atol=1e-6)
    if cycles.mean() < .93:
        # N-CMAPSS supplies many sensor readings at one cycle label.
        plateaus = np.mean(np.isclose(steps, 0., atol=1e-6)) >= .50
        return y, (.15, .50, .85) if plateaus else default_levels, plateaus, inherited
    starts = np.r_[0, 1 + np.flatnonzero(~cycles)]
    ends = np.r_[starts[1:], len(y)]
    lengths = ends - starts
    valid = lengths >= 16
    if valid.sum() < 8 or lengths[valid].sum() < .90 * len(y):
        return y, default_levels, False, inherited
    # Endpoint coverage estimates how much of a released trajectory was observed.
    remaining = np.median((y[ends - 1] / np.maximum(1., y[starts]))[valid])
    supported = (.005 <= remaining < .16) if index == 1 else inherited
    if remaining < .005:
        return y, default_levels, True, supported
    weight = .25 * float(np.clip((.10 - remaining) / .05, 0., 1.)) if supported else 0.
    if weight == 0.:
        return y, default_levels, False, supported
    unit = np.repeat(np.arange(len(starts)), lengths)
    lifespan = y[starts] + 1.
    # Weakly pool early-life labels across runs and taper to zero at failure.
    shift = (np.median(lifespan[valid]) - lifespan[unit]) * np.clip(
        y / np.maximum(1., lifespan[unit] - 1.), 0., 1.)
    return y + weight * np.clip(shift, -90., 90.) * valid[unit], default_levels, False, supported


def _use_iterate_average(y, index, stable):
    """Average late optimizer iterates when this release includes near-failure examples."""
    if stable or len(y) < 512:
        return False
    cycles = np.isclose(np.diff(y), -1., atol=1e-6)
    if cycles.mean() < .93:
        return False
    starts = np.r_[0, 1 + np.flatnonzero(~cycles)]
    ends = np.r_[starts[1:], len(y)]
    valid = (ends - starts) >= 16
    if valid.sum() < 8:
        return False
    remaining = np.median((y[ends-1] / np.maximum(1., y[starts]))[valid])
    return .005 <= remaining < .16 or (index == 1 and remaining < .005)


def fit_neural(model, x, y, limit=24576, epochs=20, width=48, ordinal_weight=.7, stable=False, levels=(.20, .45, .70), soft=False, average=False):
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
    thresholds = np.quantile(y, levels)
    if soft:
        bandwidth = max(2., .10 * float(np.quantile(y, .75) - np.quantile(y, .25)))
        logits = np.clip((thresholds - y[:, None]) / bandwidth, -30., 30.)
        near_threshold = (1 / (1 + np.exp(-logits))).astype(np.float32)
    else:
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
    averaged = [np.zeros_like(param) for param in params] if average else []
    snapshots = 0
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
        if average and epoch >= epochs * .50:
            for snapshot, param in zip(averaged, params):
                snapshot += param
            snapshots += 1
    if snapshots:
        for param, snapshot in zip(params, averaged):
            param[:] = snapshot / snapshots
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
    target, levels, soft, supported = _released_targets(y, index, previous)
    if stable:
        state = fit_neural(model, x, target, epochs=20, width=48, ordinal_weight=.2,
                           stable=True, levels=levels, soft=soft)
    else:
        state = fit_neural(model, x, target, epochs=12, width=64, levels=levels, soft=soft,
                           average=_use_iterate_average(y, index, stable))
    state["trajectory_support"] = np.asarray(supported, dtype=np.int64)
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
    state = _calibrate_output(model, state, x, y, stable)
    return _refine_hidden(model, state, x, y) if stable else state


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


def _refine_hidden(m,head,x,y):
    """Jointly calibrate four groups of hidden input weights on separate released rows."""
    ids=_reserve_rows(len(y),separate_ordinal=True)
    if len(ids)<256:return head
    xx,yy=x[ids],y[ids]
    g,hidden=_ordinal_hidden(m,head,xx)
    z=np.clip((xx[:,:-3]-head["means"][g])*head["inv_scale"][g],-6,6)
    base=m.predict(xx).reshape(-1)
    design=np.column_stack((z,(base-100.)/60,np.eye(len(head["centers"]))[g]))
    output=head["output"]
    moves=np.zeros((len(output),4))
    for j,neurons in enumerate(np.array_split(np.argsort(output),4)):
        moves[neurons,j]=output[neurons]/max(1.,np.linalg.norm(output[neurons]))
    raw=hidden@output+head["intercept"][0]
    slope=(np.abs(raw)<65.)*np.where(raw>0.,float(head["extension_fraction"]),1.)
    sens=((1.-hidden**2)@(moves*output[:,None]))*slope[:,None]
    jac=(sens[:,:,None]*design[:,None,:]).reshape(len(yy),-1)
    error=predict(m,head,xx)-yy
    train=np.arange(len(ids))%2==0
    check=~train
    j,e=jac[train],error[train]
    scale=np.maximum(np.sqrt(np.mean(j*j,axis=0)),.25)
    j=j/scale
    gram=j.T@j/len(j)+np.eye(j.shape[1])*.38
    step=np.linalg.solve(gram,-j.T@e/len(j))
    # First-order projections preserve both extreme positive and extreme negative tails.
    con=[]
    for sign,denom in ((1.,10.),(-1.,13.)):
        ex=sign*e/denom
        for q in (.98,.998):
            weight=sign*np.where(ex>0,np.exp(np.minimum(ex,np.quantile(ex,q))),0.)
            normal=weight@j/max(1.,np.abs(weight).sum())
            dual=np.linalg.solve(gram,normal)
            con.append((normal,dual,max(1e-10,normal@dual)))
    for _ in range(12):
        for normal,dual,d in con:step-=max(0.,normal@step)*dual/d
    step/=max(1.,np.sqrt(np.mean((j@step)**2))/6.)
    update=(step/scale).reshape(4,-1).T@moves.T
    n=z.shape[1]
    def risks(err):
        groups=np.searchsorted(np.quantile(yy[check],[.25,.5,.75]),yy[check])
        checkerr=err[check]
        return np.r_[[np.mean(checkerr[groups==i]**2) for i in np.unique(groups)],
                     [np.mean(np.expm1(np.minimum(20.,np.maximum(0.,sgn*checkerr/den))))
                      for sgn,den in ((1.,10.),(-1.,13.))]]
    safe=risks(error)
    for fraction in (.6,.3,.15):
        d=update*fraction
        trial=dict(head,w_sensor=head["w_sensor"]+d[:n],
                   w_base=head["w_base"]+d[n]/60.,bias=head["bias"]+d[n+1:]-d[n]*100./60.)
        e=predict(m,trial,xx)-yy
        risk=risks(e)
        if (np.mean(e[check]**2)<np.mean(error[check]**2) and
            np.mean(abs(e[check]))<=np.mean(abs(error[check]))*1.001 and
            np.all(risk[:-2]<=safe[:-2]*1.018) and np.all(risk[-2:]<=safe[-2:]*1.015)):
            return trial
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
    low = float(head.get("health_lo", 0.))
    high = float(head.get("health_hi", 0.))
    amplitude = .2*float(head.get("health_amp", 0.))
    fraction = float(head.get("extension_fraction", 1.))

    axes = head.get("curve_axis")
    curve_weight = head.get("curve_weight")
    ordinal = head.get("ord_coef")
    if ordinal is not None:
        ord_out, ord_bias = head["ord_out"], head["ord_bias"]

    # Work from one contiguous projection per input vector. The nonlinear
    # curvature channels still see exactly the normalized and clipped sensors.
    projected = np.ascontiguousarray(np.column_stack((w_sensor,axes))) if axes is not None else w_sensor
    width = w_sensor.shape[1]
    sensor_count = len(means[0])
    integer_centers = np.array_equal(centers, np.rint(centers))
    lookup = {(int(a),int(b)):i for i,(a,b) in enumerate(centers)} if integer_centers else {}
    combined = np.ascontiguousarray(np.column_stack((output,ord_out))) if ordinal is not None else None
    reciprocal = 1/model.scale
    bw,bb,bo,bi = (model.params[k] for k in ("w1","b1","w2","b2"))
    target_scale, target_mean = model.meta["target_scale"],model.meta["target_mean"]

    def one(row):
        z=(row-model.mean)*reciprocal
        np.maximum(z,-10.,out=z)
        np.minimum(z,10.,out=z)
        bh=np.tanh(z@bw+bb)
        base=max(0.,float(bh@bo[:,0]+bi[0])*target_scale+target_mean)
        a,b=float(row[-3]),float(row[-1])
        ia,ib=round(a),round(b)
        group=lookup.get((ia,ib))
        # A matched integer lattice point is nearest within its open cell.
        # Boundary or unseen cells use the original exact tie-breaking order.
        if group is None or abs(a-ia)>=.5 or abs(b-ib)>=.5:
            group=np.argmin((a-center_a)**2+(b-center_b)**2)
        sensors=(row[:sensor_count]-means[group])*inv_scale[group]
        np.maximum(sensors,-6.,out=sensors)
        np.minimum(sensors,6.,out=sensors)
        proj=sensors@projected
        value=proj[:width]
        if axes is not None:
            curve=proj[width:]
            np.maximum(curve,-3.,out=curve)
            np.minimum(curve,3.,out=curve)
            value+=((curve*curve-1.)*.5)@curve_weight
        h=np.tanh(value+base*w_base+bias[group])
        if combined is None:
            residual=float(h@output+intercept)
        else:
            logits=h@combined
            residual=float(logits[0]+intercept)
        value=max(0.,base+min(65.,max(-65.,residual)))
        tail=max(0.,value-high)+min(0.,value-low)
        value=max(0.,value+min(25.,max(-25.,amplitude*tail)))
        prior=max(0.,value-(1.-fraction)*max(0.,value-base))
        if ordinal is None:
            return prior
        logits=logits[1:]+ord_bias
        np.minimum(logits,30.,out=logits)
        np.maximum(logits,-30.,out=logits)
        q=1/(1+np.exp(-logits))
        t=min(3.,max(-2.,(prior-100)/60))
        delta=ordinal[0]+t*ordinal[1]+q@(ordinal[2:5]+q*ordinal[5:8]+t*ordinal[8:11])+ordinal[11+group]
        return max(0.,prior+min(10.,max(-10.,delta)))
    return one
