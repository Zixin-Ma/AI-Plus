"""Gas-array classification with a shared covariance and a relative-response expert."""
import numpy as np
import prototype


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


def fit_legacy(model, x, y, index, derived=None, raw=None):
    classes = len(model.meta["classes"])
    if np.bincount(y.astype(int), minlength=classes).min() < 2:
        return {"mlp_only": np.array(1)}
    absolute, relative = signatures(x) if derived is None else derived
    if raw is None:
        raw = np.log(np.maximum(np.abs(x), .02))
    pooled = _discriminant(absolute, y, classes, 48, 0.0, 0.005)
    shape = _discriminant(relative, y, classes, 24, 0.8, 0.005)
    return {**prototype.fit(absolute, raw, y, classes), **{"absolute_" + k: v for k, v in pooled.items()},
            **{"relative_" + k: v for k, v in shape.items()}}


def components(model, head, x, signatures_pair=None):
    base = np.log(np.maximum(model.predict(x), 1e-12))
    if "mlp_only" in head:
        return base, np.zeros_like(base), np.zeros_like(base)
    absolute, relative = signatures_pair if signatures_pair is not None else signatures(x)
    absolute = np.clip((absolute - head["absolute_center"]) / head["absolute_scale"], -8, 8)
    a = (absolute @ head["absolute_weights"] + head["absolute_bias"]) * 0.25
    relative = np.clip((relative - head["relative_center"]) / head["relative_scale"], -8, 8)
    difference = (relative @ head["relative_projection"])[:, None, :] - head["relative_means"]
    distance = np.einsum("ncd,cde,nce->nc", difference, head["relative_precision"], difference)
    b = -0.5 * (distance + head["relative_logdet"])
    # Limit an individual expert's log odds in a drifted response.
    return base, np.clip(a-a.max(1, keepdims=True), -10, 0), np.clip(b-b.max(1, keepdims=True), -10, 0)


def predict_legacy(model, head, x):
    if "kernel_landmarks" not in head:
        base, absolute, relative = components(model, head, x)
        return base + absolute + relative
    derived = signatures(x)
    base, absolute, relative = components(model, head, x, derived)
    kernel, subspace = prototype.scores(derived[0], np.log(np.maximum(np.abs(x), .02)), head)
    return base + .4 * (absolute + relative) + .5 * (kernel + subspace)

def curve_inputs(x):
    # The mean log magnitude of the 16 steady-response channels is observable
    # per row. Its calibration uses released training features only.
    paired = x.reshape(-1,8,2,8).mean(2)
    log = np.log(np.maximum(abs(paired),1e-6))
    features = np.concatenate((log[:,:,:2],log[:,:,2:]-log[:,:,:1],
                               np.diff(log[:,:,2:5],axis=2),np.diff(log[:,:,5:8],axis=2)),axis=2)
    amplitude = np.log(np.maximum(abs(x[:,::8]),1.)).mean(1)
    return features.reshape(len(x),-1),np.log(np.maximum(abs(x),.02)),amplitude


def fit(model, x, y, index):
    classes=len(model.meta["classes"])
    if np.bincount(y.astype(int),minlength=classes).min()<2:
        return {"mlp_only":np.asarray(1)}
    return prototype.fit_curve(*curve_inputs(x),y,classes)


def _response_amplitude(x):
    return np.log(np.maximum(abs(x[:, ::8]), 1.)).mean(1)


def _temper_backbone(log_prob, head):
    # The uncertainty of a rare class count grows approximately as 1/sqrt(n).
    # In a small release, keep the physical evidence intact and temper the
    # centered backbone log odds. A sufficiently supported head has no extra key.
    if "backbone_shrink" in head:
        return log_prob - float(head["backbone_shrink"])*(log_prob-log_prob.mean(axis=-1,keepdims=True))
    return log_prob


def predict(model, head, x):
    if "amplitude_bounds" in head:
        return (_temper_backbone(np.log(np.maximum(model.predict(x),1e-12)),head)
                + prototype.robust_score(*curve_inputs(x), head))
    if "sensor_evidence_cap" in head:
        derived = signatures(x)
        raw = np.log(np.maximum(np.abs(x), .02))
        base, absolute, relative = components(model, head, x, derived)
        kernel, subspace = prototype.scores(derived[0], raw, head)
        primary = base + .4*(absolute+relative) + .5*(kernel+subspace)
        d = prototype.curve_distance(derived[0], _response_amplitude(x), head)
        # The backbone already contributes to primary. Only the additional sensor
        # evidence votes here. Both heads share exactly the same raw-subspace term.
        sensor = prototype.curve_log_score(d) + .5*subspace
        upper = sensor.max(1, keepdims=True)
        result = primary + np.clip(sensor-upper, -float(head["sensor_evidence_cap"]), 0.)
        if "phys_nodes" in head:
            result += prototype.physical_score(x,head)
        return result
    if "curve_coeff" in head:
        return np.log(np.maximum(model.predict(x),1e-12)) + prototype.score_curve(*curve_inputs(x),head)
    return predict_legacy(model,head,x)


def fit_candidates(model, x, y, index):
    classes = len(model.meta["classes"])
    if np.bincount(y.astype(int), minlength=classes).min() < 2:
        return {"backbone": {"mlp_only": np.asarray(1)}}
    absolute, relative = signatures(x)
    raw = np.log(np.maximum(np.abs(x), .02))
    amplitude = _response_amplitude(x)
    legacy = fit_legacy(model, x, y, index, (absolute, relative), raw)
    curve = prototype.fit_curve(absolute, raw, amplitude, y, classes, raw_state=legacy)
    q = np.clip((amplitude-curve["amp_center"])/curve["amp_scale"], -3., 3.)
    bounds = np.stack([np.quantile(q[y == i], [.05,.95]) for i in range(classes)])
    legacy.update({k:v for k,v in curve.items() if k.startswith(("curve_", "amp_"))})
    legacy["sensor_evidence_cap"] = np.asarray(4.)
    curve["amplitude_bounds"] = bounds
    curve["agreement_fraction"] = np.asarray(.30)
    smallest = np.bincount(y.astype(int), minlength=classes).min()
    shrink = .5 * np.clip(4./np.sqrt(smallest) - 1., 0., 1.)
    if shrink > 0.:
        curve["backbone_shrink"] = np.asarray(shrink, dtype=np.float32)
    legacy.update(prototype.fit_physical(prototype.response_shape(x),y,classes,amplitude=amplitude))
    candidates = {"tail": curve, "independent": legacy}
    # Training and covariance estimation use float64. Archive only the fitted
    # heads in float32; the parent backbone keeps its original optimizer state.
    return {family: {name: value.astype(np.float32) if value.dtype == np.float64 else value
                     for name, value in head.items()} for family, head in candidates.items()}

def prepare_inference(model, head):
    """Compile immutable, in-memory factors for a single contest-array measurement."""
    is_tail = "amplitude_bounds" in head
    if not is_tail and "sensor_evidence_cap" not in head:
        return None
    state = {key: np.asarray(value, dtype=np.float64) for key, value in head.items()
             if not key.startswith("ref_")}
    prefix = "log_" if is_tail else "sub_"
    means, axes = state[prefix+"means"], state[prefix+"axes"]
    norm = (means*means).sum(1)
    axial_center = np.einsum("ckd,cd->ck", axes, means)
    flat_axes = axes.reshape(-1, means.shape[1])
    nclass = len(means)
    scale = {key: 1./value for key,value in state.items() if key.endswith("_scale")}
    coeff = state["curve_coeff"]
    single = getattr(model, "predict_one", None)
    if single is None:
        single = lambda row: model.predict(row[None])[0]

    if not is_tail:
        precision = state["relative_precision"]
        mu = state["relative_means"]
        # Expand (t-mu)' P (t-mu) into quadratic, linear and constant terms.
        rel_quad = precision.reshape(nclass, -1)
        rel_lin = -np.einsum("cij,cj->ci",precision+precision.transpose(0,2,1),mu)
        rel_offset = np.einsum("ci,cij,cj->c",mu,precision,mu) + state["relative_logdet"]
        same_scale = np.array_equal(state["kernel_scale"],state["absolute_scale"]) and np.array_equal(
            state["kernel_center"],state["absolute_center"])
        same_curve = np.array_equal(state["kernel_scale"],state["curve_scale"]) and np.array_equal(
            state["kernel_center"],state["curve_center"])
    else:
        lo,hi = state["amplitude_bounds"].T

    def normalize(v,key):
        return np.clip((v-state[key+"_center"])*scale[key+"_scale"], -8., 8.)

    def predict_one(row):
        row = np.asarray(row, dtype=np.float64)
        source = row.reshape(8,2,8)
        logs = np.log(np.maximum(abs((source[:,0]+source[:,1])*.5),1e-6))
        kinetic = np.empty((8,12),dtype=np.float64)
        kinetic[:,:2] = logs[:,:2]
        kinetic[:,2:8] = logs[:,2:] - logs[:,:1]
        kinetic[:,8:10] = logs[:,3:5]-logs[:,2:4]
        kinetic[:,10:12] = logs[:,6:8]-logs[:,5:7]
        f = kinetic.ravel()
        raw = np.log(np.maximum(abs(row),.02))
        v = normalize(raw,prefix[:-1])
        alignment = (flat_axes@v).reshape(nclass,-1)-axial_center
        raw_d = np.maximum(0.,v@v-2*(means@v)+norm-.94*np.einsum("ij,ij->i",alignment,alignment))/len(v)
        q = np.clip((np.log(np.maximum(abs(row[::8]),1.)).mean()-state["amp_center"])*scale["amp_scale"],-3.,3.)
        if is_tail:
            z = normalize(f,"curve")@state["curve_proj"]
            t = np.clip(q,lo,hi)[:,None]
            displacement = np.clip(q-t,-.5,.5)
            center = coeff[:,0]+t*(coeff[:,1]+t*coeff[:,2])+displacement*(coeff[:,1]+2*t*coeff[:,2])
        else:
            ka = normalize(f,"kernel")
            z = (ka if same_curve else normalize(f,"curve"))@state["curve_proj"]
            center = coeff[:,0]+q*(coeff[:,1]+q*coeff[:,2])
        rr = z-center
        stretch = np.einsum("ckd,cd->ck",state["curve_axes"],rr)
        d = np.maximum(0.,np.einsum("cd,cd->c",rr,rr)-np.einsum("ck,ck,ck->c",state["curve_gain"],stretch,stretch))
        base = np.log(np.maximum(single(row),1e-12))
        if is_tail:
            base = _temper_backbone(base,state)
            d = z.size*np.log1p(d/z.size)
            kinetic_log=np.clip(-.3*(d-d.min()),-12.,0.)
            raw_log=-3*np.log1p(raw_d)
            if "agreement_fraction" in state:
                return base+prototype.combine_gas_views(kinetic_log,raw_log,float(state["agreement_fraction"]))
            return base+kinetic_log+raw_log
        absolute = ((ka if same_scale else normalize(f,"absolute"))@state["absolute_weights"] + state["absolute_bias"])*.25
        relative = (kinetic-kinetic.mean(0)).ravel()
        t = normalize(relative,"relative")@state["relative_projection"]
        rq = -.5*(rel_quad@np.multiply.outer(t,t).ravel() + rel_lin@t + rel_offset)
        sq = ka@ka/ka.size + state["kernel_norm"]-(2./ka.size)*(state["kernel_landmarks"]@ka)
        kernel = np.exp(-.65*np.maximum(0.,sq))@state["kernel_coef"]
        score = base + .4*(np.clip(absolute-absolute.max(),-10.,0.)+np.clip(rq-rq.max(),-10.,0.))+10*kernel-3*raw_d
        evidence = np.clip(-.3*(d-d.min()),-12.,0.)-3*raw_d
        score += np.clip(evidence-evidence.max(),-float(state["sensor_evidence_cap"]),0.)
        if "phys_nodes" in state:
            measured = row.reshape(16,8)
            signal = np.empty((16,7))
            signal[:,0] = measured[:,1]
            signal[:,1:] = measured[:,2:]/np.maximum(abs(measured[:,:1]),1.)
            a = np.arcsinh((signal-state["phys_loc"])*scale["phys_scale"])
            channel_mean = a.sum(0)/len(a)
            f = np.concatenate(((a-channel_mean).ravel(),channel_mean))
            z = np.clip((f-state["phys_mid"])/state["phys_std"],-8.,8.)
            sq = (z@z)/z.size+state["phys_norm"]-(2./z.size)*(state["phys_nodes"]@z)
            guide = np.exp(-.85*np.maximum(0.,sq)/state["phys_width"])@state["phys_coef"]
            score += prototype.physical_vote(guide)
        return score
    return predict_one
