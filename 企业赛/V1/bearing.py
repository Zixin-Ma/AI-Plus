"""Remaining-life calibration with a vibration-dependent population prior."""
from functools import lru_cache
import json
import math

import numpy as np


def _design(x, state):
    log_features = np.sign(x) * np.log1p(abs(x))
    normalized = np.clip((log_features - state["center"]) / state["scale"], -6, 6)
    waves = np.cos(normalized @ state["projection"] + state["phase"])
    return np.column_stack((np.ones(len(x)), normalized, waves))


def _amplitude(x, rms_columns):
    peak_channel = np.max(x[:, rms_columns], axis=1)
    return np.log(np.maximum(peak_channel, 1e-8))


def fit(model, x, y, index):
    """Fit once on released training pairs after the caller continues the parent MLP."""
    rng = np.random.default_rng(2027)
    subset = rng.choice(len(y), min(1536, len(y)), replace=False)
    sample = x[subset]
    log_sample = np.sign(sample) * np.log1p(abs(sample))
    low, high = np.percentile(log_sample, [25, 75], axis=0)
    rms_columns = np.array([i for i, path in enumerate(model.meta["paths"])
                            if json.loads(path)[-1][1] == "rms"], dtype=np.int64)
    if len(rms_columns) != 2:
        raise ValueError("B projection requires horizontal and vertical RMS")
    amplitude = _amplitude(x, rms_columns)
    state = {
        "center": np.median(log_sample, axis=0),
        "scale": np.maximum(high - low, np.maximum(.05 * log_sample.std(axis=0), 1e-6)),
        "projection": rng.normal(0, 1.6 / np.sqrt(x.shape[1]), (x.shape[1], 64)),
        "phase": rng.uniform(-np.pi, np.pi, 64),
        "rms_columns": rms_columns,
        "amplitude_bounds": np.quantile(amplitude, [.75, .99]),
        "prior": np.asarray(np.mean(np.sqrt(np.maximum(y, 0))) ** 2)
    }
    design = _design(sample, state)
    gram = design.T @ design
    gram.flat[::len(gram) + 1] += len(sample) * .008
    residual = y[subset] - model.predict(sample)
    state["coef"] = np.linalg.solve(gram, design.T @ residual)
    freq_cols = np.array([i for i, path in enumerate(model.meta["paths"])
                          if json.loads(path)[-1][1] == "dominant_freq"],dtype=int)
    if len(freq_cols) == 2:
        state.update(order_prior(x, y, freq_cols))
        # For the three specified XJTU regimes the bearing-load life formula
        # (P**-3 / rpm) increases with shaft speed as radial load decreases.
        state["order_means"] = np.sort(state["order_means"])
    return state


def corrected(model, state, x):
    return np.maximum(0, model.predict(x) + .85 * (_design(x, state) @ state["coef"]))


def _predict_one(model, head, x):
    # The host streams one row per request. Skip building a full design matrix for a scalar.
    row = x[0]
    z = np.clip((np.sign(row) * np.log1p(abs(row)) - head["center"]) / head["scale"], -6, 6)
    waves = np.cos(z @ head["projection"] + head["phase"])
    coef = head["coef"]
    cut = len(row) + 1
    residual = coef[0] + z @ coef[1:cut] + waves @ coef[cut:]
    estimate = max(0., model.predict(x)[0] + .85 * residual)
    a, b = head["rms_columns"]
    intensity = math.log(max(row[a], row[b], 1e-8))
    low, high = head["amplitude_bounds"]
    weight = .60 * (1 - min(1., max(0., (intensity-low) / max(high-low, 1e-9))))
    prediction=(1-weight)*estimate+weight*head["prior"]
    if "order_cols" in head:
        a,b=head["order_cols"]
        w0,w1,w2=_pair_weights(float(row[a]),float(row[b]))
        m0,m1,m2=head["order_means"]
        prior=w0*m0+w1*m1+w2*m2
        prediction += (.5/.60)*weight*(prior-head["prior"])
    return np.array([max(0.,prediction)])


def predict(model, head, x):
    if len(x) == 1:
        return _predict_one(model, head, x)
    estimate = corrected(model, head, x)
    low, high = head["amplitude_bounds"]
    intensity = _amplitude(x, head["rms_columns"])
    # Extreme acceleration should retain the signal-driven prediction even on unusual bearings.
    prior_weight = .60 * (1 - np.clip((intensity - low) / max(high - low, 1e-9), 0, 1))
    prediction = (1-prior_weight)*estimate + prior_weight*head["prior"]
    if "order_cols" in head:
        prediction += (.5/.60)*prior_weight*(order_predict(x,head)-head["prior"])
    return np.maximum(0.,prediction)


def order_evidence(x, columns):
    """Compatibility of spectral peaks with shaft and rolling-element orders."""
    peaks=x[:, columns]
    speeds=np.array([35.,37.5,40.])
    ratio=7.92/34.55  # LDK UER204 ball and pitch diameters in mm
    orders=np.array([1.,4*(1-ratio),4*(1+ratio),.5*(1-ratio**2)/ratio])
    nominal=speeds[:,None]*orders
    harmonic=np.rint(peaks[:,:,None,None]/nominal)
    error=peaks[:,:,None,None]-harmonic*nominal
    width=np.maximum(1.7, .0025*peaks[:,:,None,None])
    agreement=np.exp(-.5*(error/width)**2-.04*harmonic)*(harmonic>=1)
    return agreement.max(3).mean(1)


def order_prior(x, y, columns):
    evidence=order_evidence(x, columns)
    group=evidence.argmax(1)
    mu=np.sqrt(np.maximum(y,0)).mean()
    means=np.array([(np.sqrt(np.maximum(y[group==i],0)).sum()+20.*mu)/
                    (np.sum(group==i)+20) for i in range(evidence.shape[1])])
    # Relative confidence vanishes when multiple speeds explain a peak.
    return {"order_means":means**2, "order_cols":np.asarray(columns)}


def order_predict(x, state):
    evidence=order_evidence(x,state["order_cols"])
    votes=np.exp(7.*(evidence-evidence.max(1,keepdims=True)))
    return votes @ state["order_means"] / votes.sum(1)


@lru_cache(maxsize=2048)
def _single_evidence(peak):
    return order_evidence(np.array([[peak]]),[0])[0]


@lru_cache(maxsize=4096)
def _pair_weights(first, second):
    weights=np.exp(3.5*(_single_evidence(first)+_single_evidence(second)))
    return tuple(weights/weights.sum())
