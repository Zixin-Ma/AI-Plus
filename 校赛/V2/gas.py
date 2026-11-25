"""Gas-array classification with a shared covariance and a relative-response expert."""
import numpy as np


def signatures(x):
    # Each of the 16 sensors supplies eight response summaries. Averaging
    # adjacent channels before the logarithm suppresses single-channel noise.
    paired = x.reshape(-1, 8, 2, 8).mean(axis=2)
    magnitude = np.log(np.maximum(np.abs(paired), 1e-6))
    kinetics = np.concatenate((
        magnitude[:, :, :2],
        magnitude[:, :, 2:] - magnitude[:, :, :1],
        np.diff(magnitude[:, :, 2:5], axis=2),
        np.diff(magnitude[:, :, 5:8], axis=2),
    ), axis=2)
    relative = kinetics - kinetics.mean(axis=1, keepdims=True)
    return kinetics.reshape(len(x), -1), relative.reshape(len(x), -1)


def _discriminant(features, y, classes, rank, class_weight, ridge):
    center = features.mean(axis=0)
    scale = np.maximum(features.std(axis=0), 0.025)
    scaled = np.clip((features - center) / scale, -8, 8)
    variance, axes = np.linalg.eigh(scaled.T @ scaled / len(y))
    projection = axes[:, -rank:] / np.sqrt(np.maximum(variance[-rank:], 0.05))
    reduced = scaled @ projection
    means, covariances = [], []
    for label in range(classes):
        group = reduced[y == label]
        mean = group.mean(axis=0)
        residual = group - mean
        means.append(mean)
        covariances.append(residual.T @ residual / max(len(group) - 1, 1))
    means = np.stack(means)
    pooled = np.mean(covariances, axis=0)
    regularizer = ridge * np.eye(rank)
    state = dict(center=center, scale=scale)
    if class_weight == 0:
        # Compile shared Mahalanobis distances to one affine predictor. The
        # remaining quadratic term has the same value for every class.
        covariance = (1 - ridge) * pooled + regularizer
        weights = np.linalg.solve(covariance, means.T)
        state.update(weights=projection @ weights, bias=-0.5 * (means * weights.T).sum(1))
    else:
        cov = (1 - class_weight) * pooled + class_weight * np.stack(covariances)
        cov = (1 - ridge) * cov + regularizer
        state.update(projection=projection, means=means, precision=np.linalg.inv(cov),
                     logdet=np.linalg.slogdet(cov)[1])
    return state


def fit(model, x, y, index):
    classes = len(model.meta["classes"])
    if np.bincount(y.astype(int), minlength=classes).min() < 2:
        return {"mlp_only": np.array(1)}
    absolute, relative = signatures(x)
    pooled = _discriminant(absolute, y, classes, 48, 0.0, 0.005)
    shape = _discriminant(relative, y, classes, 24, 0.8, 0.005)
    return {**{"absolute_" + k: v for k, v in pooled.items()},
            **{"relative_" + k: v for k, v in shape.items()}}


def components(model, head, x):
    base = np.log(np.maximum(model.predict(x), 1e-12))
    if "mlp_only" in head:
        return base, np.zeros_like(base), np.zeros_like(base)
    absolute, relative = signatures(x)
    absolute = np.clip((absolute - head["absolute_center"]) / head["absolute_scale"], -8, 8)
    a = (absolute @ head["absolute_weights"] + head["absolute_bias"]) * 0.25
    relative = np.clip((relative - head["relative_center"]) / head["relative_scale"], -8, 8)
    difference = (relative @ head["relative_projection"])[:, None, :] - head["relative_means"]
    distance = np.einsum("ncd,cde,nce->nc", difference, head["relative_precision"], difference)
    b = -0.5 * (distance + head["relative_logdet"])
    # Limit an individual expert's log odds in a drifted response.
    return base, np.clip(a-a.max(1, keepdims=True), -10, 0), np.clip(b-b.max(1, keepdims=True), -10, 0)


def predict(model, head, x):
    base, absolute, relative = components(model, head, x)
    return base + absolute + relative
