"""Conditional response curves and low-rank class geometry for a CPU gas array."""
import numpy as np


def _normalize(x, center, scale):
    return np.clip((x-center)/scale, -8., 8.)


def fit_curve(features, raw, amplitude, y, classes):
    state = dict(curve_center=features.mean(0), curve_scale=np.maximum(features.std(0),.03),
                 log_center=raw.mean(0), log_scale=np.maximum(raw.std(0),.025),
                 amp_center=np.asarray(amplitude.mean()), amp_scale=np.asarray(max(amplitude.std(),.15)))
    z = _normalize(features,state["curve_center"],state["curve_scale"])
    v = _normalize(raw,state["log_center"],state["log_scale"])
    q = np.clip((amplitude-state["amp_center"])/state["amp_scale"],-3.,3.)
    design = np.column_stack((np.ones(len(q)),q,q*q))
    coeff, residuals, covariances = [], [], []
    means, axes = [], []
    for cls in range(classes):
        mask = y==cls
        f = design[mask]
        co = np.linalg.solve(f.T@f + np.diag([.01,1.,2.]),f.T@z[mask])
        err = z[mask]-f@co
        coeff.append(co)
        residuals.append(err)
        covariances.append(err.T@err / max(1,len(err)-1))
        group = v[mask]
        mu = group.mean(0)
        basis = np.linalg.svd(group-mu,full_matrices=False)[2][:3]
        padded = np.zeros((3,v.shape[1]))
        padded[:len(basis)] = basis
        means.append(mu)
        axes.append(padded)
    e,u = np.linalg.eigh(np.mean(covariances,axis=0))
    proj = u[:,-24:]/np.sqrt(e[-24:]+.12)
    local, gain = [], []
    for err in residuals:
        h = err@proj
        ev,vec = np.linalg.eigh(h.T@h / max(1,len(h)-1))
        local.append(vec[:,-4:].T)
        gain.append(ev[-4:]/(1+ev[-4:]))
    state.update(curve_proj=proj, curve_coeff=np.stack(coeff)@proj,
                 curve_axes=np.stack(local), curve_gain=np.stack(gain),
                 log_means=np.stack(means), log_axes=np.stack(axes))
    return state


def score_curve(features, raw, amplitude, state):
    z = _normalize(features,state["curve_center"],state["curve_scale"]) @ state["curve_proj"]
    v = _normalize(raw,state["log_center"],state["log_scale"])
    q = np.clip((amplitude-state["amp_center"])/state["amp_scale"],-3.,3.)
    c = state["curve_coeff"]
    residual = z[:,None,:] - (c[:,0]+q[:,None,None]*(c[:,1]+q[:,None,None]*c[:,2]))
    alignment = np.einsum("ncd,ckd->nck",residual,state["curve_axes"])
    dist = (residual**2).sum(2) - (state["curve_gain"]*alignment**2).sum(2)
    score = np.clip(-.3*(dist-dist.min(1,keepdims=True)),-12.,0.)
    raw_res = v[:,None,:]-state["log_means"]
    aligned = np.einsum("ncd,ckd->nck",raw_res,state["log_axes"])
    d = np.maximum(0.,(raw_res**2).sum(2)-.94*(aligned**2).sum(2))
    return score - 3*d/v.shape[1]


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
