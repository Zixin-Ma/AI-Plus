"""Bounded, condition-aware neural residual on the continued C-MAPSS checkpoint."""
import numpy as np


def _regimes(x, centers):
    # These two sensor settings distinguish the six documented operating regimes.
    settings = x[:, [-3, -1]]
    return np.sum((settings[:, None] - centers) ** 2, axis=2).argmin(axis=1)


def fit_neural(model, x, y, limit=24576, epochs=20, width=48, ordinal_weight=.7):
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
    base = model.predict(x).reshape(-1)
    design = np.column_stack((sensors, (base - 100) / 60, np.eye(len(centers))[groups])).astype(np.float32)
    target = ((y - base) / 55).astype(np.float32)
    thresholds = np.quantile(y, [.20, .45, .70])
    near_threshold = (y[:, None] < thresholds).astype(np.float32)

    weights = rng.normal(0, .22, (design.shape[1], width)).astype(np.float32)
    bias = np.zeros(width, dtype=np.float32)
    # Column zero predicts the standardised residual; other columns supervise
    # rank at three training-only lifetime thresholds and are discarded at serving.
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
            delta[:, 0] = (delta[:, 0] - target[idx]) * (2 / len(idx))
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
    return dict(centers=centers, means=means, inv_scale=inv_scale,
                w_sensor=weights[:-len(centers)-1].copy(), w_base=weights[-len(centers)-1] / 60,
                bias=bias + weights[-len(centers):] - weights[-len(centers)-1] * (100 / 60),
                output=55 * output[:, 0].astype(float), intercept=55 * intercept[:1].astype(float))


def fit(model, x, y, index, previous=None, stable_release=False):
    """Shorter updates for new releases; revisit the full fit on a stable pool."""
    if stable_release and previous:
        state = fit_neural(model, x, y, epochs=20, width=48, ordinal_weight=.2)
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
    return state


def _residual(head, z, base, group):
    hidden = np.tanh(z @ head["w_sensor"] + base * head["w_base"] + head["bias"][group])
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


def predict(model, head, x):
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

    def one(row):
        base = predict_base(row)
        group = np.argmin((row[-3]-center_a)**2 + (row[-1]-center_b)**2)
        sensors = (row[:-3]-means[group])*inv_scale[group]
        np.maximum(sensors, -6., out=sensors)
        np.minimum(sensors, 6., out=sensors)
        h = np.tanh(sensors @ w_sensor + base*w_base + bias[group])
        residual = float(h @ output + intercept)
        value = max(0., base+min(65., max(-65., residual)))
        tail = max(0., value-high) + min(0., value-low)
        value = max(0., value+min(25., max(-25., amplitude*tail)))
        return max(0., value-(1.-fraction)*max(0., value-base))
    return one
