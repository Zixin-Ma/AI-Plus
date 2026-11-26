"""Gas response experts: compact kernel ridge and class-conditional affine subspaces."""
import numpy as np


def _normalize(x, center, scale):
    return np.clip((x - center) / scale, -8, 8)


def _kernel(x, centers, center_norm):
    squared = np.mean(x * x, axis=1)[:, None] + center_norm - (2 / x.shape[1]) * (x @ centers.T)
    return np.exp(-.65 * np.maximum(0., squared))


def fit(features, log_readings, y, count):
    state = dict(kernel_center=features.mean(0), kernel_scale=np.maximum(features.std(0), .025),
                 sub_center=log_readings.mean(0), sub_scale=np.maximum(log_readings.std(0), .025))
    x = _normalize(features, state["kernel_center"], state["kernel_scale"])
    z = _normalize(log_readings, state["sub_center"], state["sub_scale"])
    centers, means, axes = [], [], []
    for cls in range(count):
        ids = np.flatnonzero(y == cls)
        # Every stored landmark is the mean of a block of released training features.
        for segment in np.array_split(ids, min(16, max(1, len(ids)//4))):
            centers.append(x[segment].mean(0))
        group = z[ids]
        mean = group.mean(0)
        basis = np.linalg.svd(group-mean, full_matrices=False)[2][:3]
        padded = np.zeros((3,z.shape[1]))
        padded[:len(basis)] = basis
        means.append(mean)
        axes.append(padded)
    centers = np.stack(centers)
    center_norm = np.mean(centers**2, axis=1)
    design = _kernel(x, centers, center_norm)
    regularizer = .001*len(y)*_kernel(centers,centers,center_norm) + .0001*np.eye(len(centers))
    weights = 1 / np.maximum(1., np.bincount(y, minlength=count)[y] / 40)
    transpose = design.T * weights
    coef = np.linalg.solve(transpose @ design + regularizer, transpose @ np.eye(count)[y])
    state.update(kernel_landmarks=centers,kernel_norm=center_norm,kernel_coef=coef,
                 sub_means=np.stack(means),sub_axes=np.stack(axes))
    return state


def scores(features, log_readings, state):
    x = _normalize(features,state["kernel_center"],state["kernel_scale"])
    kernel_scores = _kernel(x,state["kernel_landmarks"],state["kernel_norm"]) @ state["kernel_coef"]
    z = _normalize(log_readings,state["sub_center"],state["sub_scale"])
    residual = z[:,None,:] - state["sub_means"]
    aligned = np.einsum("ncd,ckd->nck",residual,state["sub_axes"])
    distance = np.maximum(0., (residual**2).sum(2) - .94*(aligned**2).sum(2)) / z.shape[1]
    return 20 * kernel_scores, -6 * distance
