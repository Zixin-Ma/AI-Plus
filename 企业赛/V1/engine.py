"""Bounded, condition-aware neural residual on the continued C-MAPSS checkpoint."""
import numpy as np


def _regimes(x, centers):
    # These two sensor settings distinguish the six documented operating regimes.
    settings = x[:, [-3, -1]]
    return np.sum((settings[:, None] - centers) ** 2, axis=2).argmin(axis=1)


def fit_neural(model, x, y, limit=24576, epochs=20, width=48):
    rng = np.random.default_rng(145)
    if len(x) > limit:
        sample = rng.choice(len(x), limit, replace=False)
        x, y = x[sample], y[sample]
    centers, counts = np.unique(np.round(x[:, [-3, -1]]), axis=0, return_counts=True)
    centers = centers[np.argsort(-counts)[:8]]
    groups = _regimes(x, centers)
    means = np.stack([x[groups == k, :-3].mean(0) for k in range(len(centers))])
    scales = np.stack([x[groups == k, :-3].std(0) for k in range(len(centers))])
    inv_scale = 1 / np.maximum(scales, 1e-4)
    sensors = np.clip((x[:, :-3] - means[groups]) * inv_scale[groups], -6, 6)
    base = model.predict(x).reshape(-1)
    design = np.column_stack((sensors, (base - 100) / 60, np.eye(len(centers))[groups]))
    target = (y - base) / 55

    weights = rng.normal(0, .22, (design.shape[1], width))
    bias = np.zeros(width)
    output = rng.normal(0, .03, width)
    intercept = np.zeros(1)
    params = [weights, bias, output, intercept]
    first = [np.zeros_like(p) for p in params]
    second = [np.zeros_like(p) for p in params]
    step = 0
    for epoch in range(epochs):
        order = rng.permutation(len(y))
        for idx in np.array_split(order, max(1, len(y) // 256)):
            h = np.tanh(design[idx] @ weights + bias)
            delta = (h @ output + intercept - target[idx]) * (2 / len(idx))
            back = delta[:, None] * output * (1 - h * h)
            gradients = [design[idx].T @ back, back.sum(0), h.T @ delta, np.array([delta.sum()])]
            step += 1
            for j, grad in enumerate(gradients):
                first[j] = .9 * first[j] + .1 * grad
                second[j] = .999 * second[j] + .001 * grad ** 2
                rate = .003 / (1 + epoch / 10)
                params[j] -= (rate * first[j] / (1 - .9 ** step) /
                              (np.sqrt(second[j] / (1 - .999 ** step)) + 1e-8))
    # Fold the regime intercept and output scaling once for fast single-row serving.
    return dict(centers=centers, means=means, inv_scale=inv_scale,
                w_sensor=weights[:-len(centers)-1].copy(), w_base=weights[-len(centers)-1] / 60,
                bias=bias + weights[-len(centers):] - weights[-len(centers)-1] * (100 / 60),
                output=55 * output, intercept=55 * intercept)


def fit(model, x, y, index):
    state = fit_neural(model,x,y)
    idx = np.linspace(0,len(x)-1,min(len(x),2048),dtype=int)
    estimate = predict_neural(model,state,x[idx],65.)
    lo,hi = np.quantile(estimate,[.35,.65])
    # Least squares shrinks extremes toward the mean. Correct the tails mildly
    # when released training lifetimes vary more than the model predictions.
    amp = np.clip(np.std(y)/max(1.,np.std(estimate)) - 1., .15,.7)
    state.update(health_lo=np.array(lo),health_hi=np.array(hi),health_amp=np.array(amp))
    return state


def _residual(head, z, base, group):
    hidden = np.tanh(z @ head["w_sensor"] + base * head["w_base"] + head["bias"][group])
    return hidden @ head["output"] + head["intercept"][0]


def predict_neural(model, head, x, correction_limit=None):
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
    p = predict_neural(model,head,x,correction_limit=65.)
    if "health_amp" not in head:
        return p
    if len(x) == 1:
        value = float(p[0])
        high = float(head["health_hi"])
        low = float(head["health_lo"])
        tail = max(0.,value-high) + min(0.,value-low)
        correction = min(25., max(-25., .2*float(head["health_amp"])*tail))
        p[0] = max(0.,value+correction)
        return p
    tails = np.maximum(0.,p-head["health_hi"]) + np.minimum(0.,p-head["health_lo"])
    return np.maximum(0.,p+np.clip(.2*head["health_amp"]*tails,-25.,25.))
