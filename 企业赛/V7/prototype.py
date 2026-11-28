"""Conditional response curves and low-rank class geometry for a CPU gas array."""
import numpy as np


def _normalize(x, center, scale):
    return np.clip((x-center)/scale, -8., 8.)


def fit_curve(features, raw, amplitude, y, classes, raw_state=None, pooled_tail=False):
    state = dict(curve_center=features.mean(0), curve_scale=np.maximum(features.std(0),.03),
                 log_center=raw.mean(0), log_scale=np.maximum(raw.std(0),.025),
                 amp_center=np.asarray(amplitude.mean()), amp_scale=np.asarray(max(amplitude.std(),.15)))
    z = _normalize(features,state["curve_center"],state["curve_scale"])
    v = _normalize(raw,state["log_center"],state["log_scale"])
    q = np.clip((amplitude-state["amp_center"])/state["amp_scale"],-3.,3.)
    design = np.column_stack((np.ones(len(q)),q,q*q))
    if pooled_tail:
        # Estimate one dose derivative from within-gas variation. Removing the gas
        # means keeps class offsets from contaminating the shared direction.
        centered_q=np.empty_like(q)
        centered_z=np.empty_like(z)
        for label in range(classes):
            ids=y==label
            centered_q[ids]=q[ids]-q[ids].mean()
            centered_z[ids]=z[ids]-z[ids].mean(0)
        shared=centered_q@centered_z/(centered_q@centered_q+1.)
    coeff, residuals, covariances = [], [], []
    means, axes = [], []
    for cls in range(classes):
        mask = y==cls
        f = design[mask]
        if pooled_tail:
            center=q[mask].mean()
            t=q[mask]-center
            local=np.column_stack((np.ones(len(t)),t,t*t))
            penalty=np.diag([.01,9.,10.])
            target=np.zeros((3,z.shape[1]))
            target[1]=shared
            c=np.linalg.solve(local.T@local+penalty,local.T@z[mask]+penalty@target)
            # Store global polynomial coefficients for the unmodified serving ABI.
            co=np.stack((c[0]-center*c[1]+center*center*c[2],c[1]-2*center*c[2],c[2]))
        else:
            co = np.linalg.solve(f.T@f + np.diag([.01,1.,2.]),f.T@z[mask])
        err = z[mask]-f@co
        coeff.append(co)
        residuals.append(err)
        covariances.append(err.T@err / max(1,len(err)-1))
        if raw_state is None:
            group = v[mask]
            mu = group.mean(0)
            basis = np.linalg.svd(group-mu,full_matrices=False)[2][:3]
            padded = np.zeros((3,v.shape[1]))
            padded[:len(basis)] = basis
            means.append(mu)
            axes.append(padded)
    if raw_state is not None:
        means, axes = raw_state["sub_means"], raw_state["sub_axes"]
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
    magnitude=features[:,::12].mean(1)
    for cls in range(count):
        ids = np.flatnonzero(y == cls)
        # Every stored landmark is the mean of a block of released training features.
        for segment in np.array_split(ids[np.argsort(magnitude[ids],kind="stable")], min(16, max(1, len(ids)//4))):
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


def curve_distance(features, amplitude, state, extrapolate=False):
    z = _normalize(features, state["curve_center"], state["curve_scale"]) @ state["curve_proj"]
    q = np.clip((amplitude-state["amp_center"]) / state["amp_scale"], -3., 3.)
    coeff = state["curve_coeff"]
    if extrapolate:
        # A class curve is quadratic on the central 90% of observed amplitudes.
        # Beyond this interval its boundary tangent runs for at most half a source SD.
        lo, hi = state["amplitude_bounds"].T
        anchor = np.clip(q[:, None], lo, hi)[:, :, None]
        extension = np.clip(q[:, None, None]-anchor, -.5, .5)
        mean = (coeff[:, 0] + anchor*(coeff[:, 1] + anchor*coeff[:, 2])
                + extension*(coeff[:, 1] + 2*anchor*coeff[:, 2]))
    else:
        t = q[:, None, None]
        mean = coeff[:, 0] + t*(coeff[:, 1] + t*coeff[:, 2])
    res = z[:, None, :] - mean
    alignment = np.einsum("ncd,ckd->nck", res, state["curve_axes"])
    return np.maximum(0., (res*res).sum(2) - (state["curve_gain"]*alignment**2).sum(2))


def raw_distance(raw, center, scale, means, axes):
    res = _normalize(raw, center, scale)[:, None, :] - means
    aligned = np.einsum("ncd,ckd->nck", res, axes)
    return np.maximum(0., (res*res).sum(2)-.94*(aligned*aligned).sum(2)) / raw.shape[1]


def curve_log_score(distance):
    return np.clip(-.3*(distance-distance.min(1, keepdims=True)), -12., 0.)


def combine_gas_views(kinetic, raw, fraction):
    """Downweight the kinetic correction when class-centered evidence disagrees."""
    k = kinetic-kinetic.mean(axis=-1, keepdims=True)
    r = raw-raw.mean(axis=-1, keepdims=True)
    alignment = (k*r).sum(axis=-1, keepdims=True)
    length = np.sqrt((k*k).sum(axis=-1, keepdims=True)*(r*r).sum(axis=-1, keepdims=True))
    agreement = np.maximum(0., alignment/np.maximum(length,1e-8))
    return (1.-fraction+fraction*agreement)*kinetic+raw


def robust_score(features, raw, amplitude, state):
    distance = curve_distance(features, amplitude, state, extrapolate=True)
    dimension = state["curve_proj"].shape[1]
    kinetic = curve_log_score(dimension*np.log1p(distance/dimension))
    raw_d = raw_distance(raw, state["log_center"], state["log_scale"],
                         state["log_means"], state["log_axes"])
    raw_log = -3*np.log1p(raw_d)
    if "agreement_fraction" in state:
        return combine_gas_views(kinetic,raw_log,float(state["agreement_fraction"]))
    return kinetic + raw_log


def response_shape(readings):
    signal = readings.reshape(-1,16,8)
    # Normalized steady response, then six transient rates per unit steady excursion.
    steady = signal[:,:,:1]
    return np.concatenate((signal[:,:,1:2], signal[:,:,2:]/np.maximum(abs(steady),1.)),axis=2)


def invariant_features(z,location,scale):
    adjusted = np.arcsinh((z-location)/scale)
    mean = adjusted.mean(1)
    return np.concatenate(((adjusted-mean[:,None,:]).reshape(len(z),-1),mean),axis=1)


def fit_physical(z, y, classes, groups=None, amplitude=None):
    location = np.median(z,axis=0)
    scale = np.maximum(np.median(abs(z-location),axis=0)*1.4826,1e-8)
    f = invariant_features(z,location,scale)
    mid = f.mean(0)
    std = np.maximum(f.std(0),.1)
    scaled=(f-mid)/std
    means=np.stack([scaled[y==k].mean(0) for k in range(classes)])
    within=np.mean([scaled[y==k].var(0) for k in range(classes)],axis=0)
    relevance=np.clip((means.var(0)+.1)/(within+.1),.10,10.)**.25
    # The last block carries response common to all sensors, which can shift
    # together with dose or background. Keep a bounded contribution.
    relevance[-z.shape[-1]:]=.25
    relevance/=np.sqrt(np.mean(relevance**2))
    std=std/relevance
    f = np.clip((f-mid)/std,-8.,8.)
    width = max(float(np.mean((f-f.mean(0))**2)),1e-5)
    nodes = []
    for cls in range(classes):
        ids = np.flatnonzero(y==cls)
        if amplitude is not None:
            # Cover the observed response range of each gas, independently of
            # the order in which its labeled source records were released.
            ids = ids[np.argsort(amplitude[ids],kind="stable")]
        elif groups is not None:
            ids = ids[np.argsort(groups[ids],kind="stable")]
        # At most six equally populated strata keep serving cost fixed.
        for block in np.array_split(ids,min(6,max(1,len(ids)//3))):
            nodes.append(f[block].mean(0))
    nodes = np.stack(nodes)
    norm = (nodes*nodes).mean(1)
    def kernel(a):
        sq = (a*a).mean(1)[:,None]+norm-(2./a.shape[1])*(a@nodes.T)
        return np.exp(-.85*np.maximum(sq,0.)/width)
    k = kernel(f)
    wt = len(y)/(classes*np.bincount(y,minlength=classes)[y])
    coef = np.linalg.solve((k.T*wt)@k+.003*len(y)*kernel(nodes)+.001*np.eye(len(nodes)),
                           (k.T*wt)@np.eye(classes)[y])
    return dict(phys_loc=location,phys_scale=scale,phys_mid=mid,phys_std=std,
                phys_nodes=nodes,phys_norm=norm,phys_width=np.asarray(width),phys_coef=coef)


def physical_vote(guide):
    signal = guide-guide.max(axis=-1,keepdims=True)
    second = np.partition(signal,-2,axis=-1)[...,-2,None]
    return np.clip(8*signal,-6.,0.)*np.clip(-second/.4,.25,1.)


def physical_score(x,head):
    f = invariant_features(response_shape(x),head["phys_loc"],head["phys_scale"])
    v = np.clip((f-head["phys_mid"])/head["phys_std"],-8.,8.)
    sq = (v*v).mean(1)[:,None]+head["phys_norm"]-(2./v.shape[1])*(v@head["phys_nodes"].T)
    guide = np.exp(-.85*np.maximum(0.,sq)/head["phys_width"])@head["phys_coef"]
    return physical_vote(guide)
