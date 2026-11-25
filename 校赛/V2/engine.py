"""Bounded, condition-aware neural residual on the continued C-MAPSS checkpoint."""
import numpy as np


def _regimes(x, centers):
    # These two sensor settings distinguish the six documented operating regimes.
    settings = x[:, [21, 23]]
    return np.sum((settings[:, None] - centers) ** 2, axis=2).argmin(axis=1)


def fit_neural(model, x, y, limit=24576, epochs=20, width=48):
    rng = np.random.default_rng(145)
    if len(x) > limit:
        sample = rng.choice(len(x), limit, replace=False)
        x, y = x[sample], y[sample]
    centers, counts = np.unique(np.round(x[:, [21, 23]]), axis=0, return_counts=True)
    centers = centers[np.argsort(-counts)[:8]]
    groups = _regimes(x, centers)
    means = np.stack([x[groups == k, :21].mean(0) for k in range(len(centers))])
    scales = np.stack([x[groups == k, :21].std(0) for k in range(len(centers))])
    inv_scale = 1 / np.maximum(scales, 1e-4)
    sensors = np.clip((x[:, :21] - means[groups]) * inv_scale[groups], -6, 6)
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
                w_sensor=weights[:21].copy(), w_base=weights[21] / 60,
                bias=bias + weights[22:] - weights[21] * (100 / 60),
                output=55 * output, intercept=55 * intercept)


def fit(model, x, y, index):
    return fit_neural(model, x, y)


def _residual(head, z, base, group):
    hidden = np.tanh(z @ head["w_sensor"] + base * head["w_base"] + head["bias"][group])
    return hidden @ head["output"] + head["intercept"][0]


def predict_neural(model, head, x, correction_limit=None):
    base = model.predict(x).reshape(-1)
    if len(x) == 1:
        centers = head["centers"]
        group = np.argmin((x[0, 21] - centers[:, 0]) ** 2 + (x[0, 23] - centers[:, 1]) ** 2)
        z = np.clip((x[0, :21] - head["means"][group]) * head["inv_scale"][group], -6, 6)
        correction = _residual(head, z, base[0], group)
        if correction_limit is not None:
            correction = min(correction_limit, max(-correction_limit, correction))
        return np.maximum(0, base + correction)
    group = _regimes(x, head["centers"])
    z = np.clip((x[:, :21] - head["means"][group]) * head["inv_scale"][group], -6, 6)
    correction = _residual(head, z, base[:, None], group)
    if correction_limit is not None:
        correction = np.clip(correction, -correction_limit, correction_limit)
    return np.maximum(0, base + correction)


def predict(model, head, x):
    return predict_neural(model, head, x, correction_limit=65.)
