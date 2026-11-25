"""One-row RUL prediction from condition-local sensor interactions and a continued MLP."""
import numpy as np


def operating_features(x):
    return x[:, [21, 23]]


def partition(x, centers):
    return np.sum((operating_features(x)[:, None] - centers) ** 2, axis=2).argmin(1)


def design(x, raw_prediction, center, scale, anchors, method):
    z = np.clip((x[:, :21] - center) / scale, -5, 5)
    p = raw_prediction[:, None] / 100
    features = [np.ones((len(x), 1)), p, z]
    if method != "linear":
        # Sensor interactions distinguish the two simulated failure mechanisms.
        upper = np.triu_indices(z.shape[1])
        features.append(z[:, upper[0]] * z[:, upper[1]])
    if method == "kernel":
        d2 = np.maximum(np.sum(z * z, axis=1)[:, None] + np.sum(anchors * anchors, axis=1)
                        - 2 * (z @ anchors.T), 0)
        features.append(np.exp(-d2 / 24))
    return np.column_stack(features)


def train_head(model, x, y, method="kernel", sample_limit=16384, seed=3817):
    rng = np.random.default_rng(seed)
    if len(x) > sample_limit:
        sample = rng.choice(len(x), sample_limit, replace=False)
        x, y = x[sample], y[sample]
    centers, counts = np.unique(np.round(operating_features(x)), axis=0, return_counts=True)
    centers = centers[np.argsort(-counts)[:8]]
    groups = partition(x, centers)
    base = model.predict(x).reshape(-1)
    means, scales, anchors, coefs = [], [], [], []
    for group in range(len(centers)):
        subset = groups == group
        sensors = x[subset, :21]
        mean = sensors.mean(0)
        # Treat numerical dust in constant channels as zero.
        sd = sensors.std(0)
        scale = np.where(sd > np.maximum(1e-6, np.abs(mean) * 1e-8), sd, 1.)
        pts = sensors[rng.choice(len(sensors), 96, replace=len(sensors) < 96)]
        knots = np.clip((pts - mean) / scale, -5, 5)
        phi = design(x[subset], base[subset], mean, scale, knots, method)
        gram = phi.T @ phi
        gram.flat[::len(gram) + 1] += .0007 * len(phi) + .15
        coef = np.linalg.solve(gram, phi.T @ (y[subset] - base[subset]))
        means.append(mean)
        scales.append(scale)
        anchors.append(knots)
        coefs.append(coef)
    return dict(centers=centers, means=np.stack(means), scales=np.stack(scales),
                anchors=np.stack(anchors), coefs=np.stack(coefs), method=np.array(method))


def predict(model, head, x):
    result = model.predict(x).reshape(-1)
    groups = partition(x, head["centers"])
    for group in np.unique(groups):
        subset = groups == group
        phi = design(x[subset], result[subset], head["means"][group], head["scales"][group],
                     head["anchors"][group], str(head["method"]))
        result[subset] += phi @ head["coefs"][group]
    return np.maximum(result, 0)


def fit_density(model, x, y, sample_limit=24576, shrinkage=.30):
    """Estimate a separate regularized sensor distribution for each operating regime and life bin."""
    rng=np.random.default_rng(663)
    if len(x)>sample_limit:
        use=rng.choice(len(x),sample_limit,replace=False)
        x,y=x[use],y[use]
    centers,counts=np.unique(np.round(operating_features(x)),axis=0,return_counts=True)
    centers=centers[np.argsort(-counts)[:8]]
    groups=partition(x,centers)
    sensors=x[:, [1,2,3,6,7,8,10,11,12,13,14,16,19,20]]
    edges=np.percentile(y,[10,25,40,55,70,85,94])
    bins=np.digitize(y,edges)
    arrays={key:[] for key in ("center","scale","precision","expectation","offset","mu")}
    for group in range(len(centers)):
        sub=groups==group
        xs=sensors[sub]
        center=xs.mean(0)
        scale=np.maximum(xs.std(0),1e-5)
        z=np.clip((xs-center)/scale,-8,8)
        basecov=z.T@z/max(len(z),1)
        precision,mu,offset,expectation=[],[],[],[]
        for b in range(len(edges)+1):
            match=bins[sub]==b
            n=int(np.sum(match))
            obs=z[match]
            delta=obs.mean(0) if n else np.zeros(z.shape[1])
            deviations=obs-delta
            covariance=(1-shrinkage)*(deviations.T@deviations)/(n+1)+shrinkage*basecov+.075*np.eye(len(delta))
            precision.append(np.linalg.inv(covariance))
            offset.append(np.log((n+.01)/(len(z)+.08))-.5*np.linalg.slogdet(covariance)[1])
            expectation.append(y[sub][match].mean() if n else np.mean(y))
            mu.append(delta)
        for key,value in zip(arrays,(center,scale,precision,expectation,offset,mu)):
            arrays[key].append(value)
    return {**{key:np.array(v) for key,v in arrays.items()},"centers":centers}


def predict_density(model, head, x):
    sensors=x[:, [1,2,3,6,7,8,10,11,12,13,14,16,19,20]]
    groups=partition(x,head['centers'])
    result=np.empty(len(x))
    for g in np.unique(groups):
        sub=groups==g
        z=np.clip((sensors[sub]-head['center'][g])/head['scale'][g],-8,8)
        logprob=np.empty((len(z),len(head['offset'][g])))
        for b,(mu,prec) in enumerate(zip(head['mu'][g],head['precision'][g])):
            delta=z-mu
            logprob[:,b]=head['offset'][g,b]-.5*np.sum((delta@prec)*delta,axis=1)
        prob=np.exp(logprob-logprob.max(axis=1,keepdims=True))
        prob/=prob.sum(axis=1,keepdims=True)
        result[sub]=prob@head['expectation'][g]
    return result


CHANNELS = np.array([1, 2, 3, 6, 7, 8, 10, 11, 12, 13, 14, 16, 19, 20])


def normalized_design(x, raw, head):
    group = partition(x, head["centers"])
    z = np.clip((x[:, CHANNELS] - head["mu"][group]) / head["scale"][group], -6, 6)
    onehot = np.eye(len(head["centers"]))[group]
    base = raw[:, None] / 100
    h = np.column_stack((z, base))
    hidden = np.maximum(h @ head["matrix"] + head["bias"], 0)
    return np.column_stack((np.ones(len(x)), h, onehot,
                            (onehot[:, :, None] * z[:, None, :]).reshape(len(x), -1),
                            hidden))


def fit_pool(model, x, y, limit=32768, width=160, penalty=.001):
    rng = np.random.default_rng(3817)
    if len(x) > limit:
        use = rng.choice(len(x), limit, replace=False)
        x, y = x[use], y[use]
    centers, counts = np.unique(np.round(operating_features(x)), axis=0, return_counts=True)
    centers = centers[np.argsort(-counts)[:8]]
    group = partition(x, centers)
    mus = np.array([x[group == k][:, CHANNELS].mean(0) for k in range(len(centers))])
    scales = np.array([x[group == k][:, CHANNELS].std(0) for k in range(len(centers))])
    base = model.predict(x).reshape(-1)
    head = dict(centers=centers, mu=mus, scale=np.maximum(scales, 1e-5),
                matrix=rng.normal(0, 1 / np.sqrt(len(CHANNELS) + 1), (len(CHANNELS) + 1, width)),
                bias=rng.uniform(-1.5, 1.5, width))
    design = normalized_design(x, base, head)
    gram = design.T @ design
    gram.flat[::len(gram) + 1] += penalty * len(y)
    head["coef"] = np.linalg.solve(gram, design.T @ (y - base))
    return head


def predict_pool(model, head, x):
    base = model.predict(x).reshape(-1)
    return np.maximum(0, base + normalized_design(x, base, head) @ head["coef"])


def tree_inputs(x, raw, head):
    groups = partition(x, head["centers"])
    z = (x[:, CHANNELS] - head["mu"][groups]) / head["scale"][groups]
    return np.column_stack((z, raw, groups, x[:, [8, 13]]))


def fit_trees(model, x, y, limit=16384, ntrees=96):
    rng = np.random.default_rng(3817)
    if len(x) > limit:
        keep = rng.choice(len(x), limit, replace=False)
        x, y = x[keep], y[keep]
    centers, counts = np.unique(np.round(operating_features(x)), axis=0, return_counts=True)
    centers = centers[np.argsort(-counts)[:8]]
    gr = partition(x, centers)
    mus = np.array([x[gr == k][:, CHANNELS].mean(0) for k in range(len(centers))])
    scales = np.array([x[gr == k][:, CHANNELS].std(0) for k in range(len(centers))])
    head = dict(centers=centers, mu=mus, scale=np.maximum(scales, 1e-5))
    base = model.predict(x).reshape(-1)
    xx = tree_inputs(x, base, head)
    edges = np.quantile(xx, np.arange(1,32)/32, axis=0).T
    bins = np.array([np.searchsorted(t, v) for t, v in zip(edges, xx.T)])
    leaves, dimensions, thresholds = [], [], []
    error = y - base
    for tree in range(ntrees):
        node = np.zeros(len(y), dtype=np.int32)
        feat, cut = [], []
        for depth in range(4):
            n = 2**depth
            offset = node * 32
            best = -np.inf
            best_idx, best_cut = 0, 0
            for i, col in enumerate(bins):
                code = offset + col
                size = np.bincount(code, minlength=n*32).reshape(n,32).cumsum(axis=1)
                summ = np.bincount(code, weights=error, minlength=n*32).reshape(n,32).cumsum(axis=1)
                rightsize = size[:, -1:] - size
                rightsum = summ[:, -1:] - summ
                gain = ((summ**2)/(size+8) + (rightsum**2)/(rightsize+8)).sum(axis=0)
                j = int(np.argmax(gain[:-1]))
                if gain[j] > best:
                    best, best_idx, best_cut = gain[j], i, j
            feat.append(best_idx)
            cut.append(edges[best_idx,best_cut])
            node = 2 * node + (bins[best_idx] > best_cut)
        values = .17 * np.bincount(node,weights=error,minlength=16)/(np.bincount(node,minlength=16)+8)
        leaves.append(values)
        error = error - values[node]
        dimensions.append(feat)
        thresholds.append(cut)
    head.update(dims=np.array(dimensions), thresholds=np.array(thresholds), leaves=np.array(leaves))
    return head


def predict_trees(model, head, x):
    base = model.predict(x).reshape(-1)
    xx = tree_inputs(x, base, head)
    dims, cuts, leaves = head["dims"], head["thresholds"], head["leaves"]
    result = np.empty(len(x))
    for start in range(0,len(x),512):
        chunk = xx[start:start+512]
        nodes = np.zeros((len(chunk),len(dims)), dtype=np.int32)
        for depth in range(dims.shape[1]):
            nodes = 2 * nodes + (chunk[:,dims[:,depth]] > cuts[:,depth])
        result[start:start+len(chunk)] = base[start:start+len(chunk)] + leaves[np.arange(len(dims)),nodes].sum(1)
    return np.maximum(result, 0)


def shared_basis(x, head):
    group=partition(x,head['centers'])
    z=np.clip((x[:,:21]-head['mu'][group])/head['scale'][group],-6,6)
    settings=(x[:,21:]-head['settings_mean'])/head['settings_sd']
    norm=np.column_stack((z,settings))
    feats=[np.ones((len(x),1)), norm,
           np.cos(norm@head['proj']+head['phase'])]
    # A supervised direction is fitted separately for each operating regime.
    if 'directions' in head:
        axis=np.einsum('ni,nij->nj',z,head['directions'][group])
        t=(axis-head['axis_center'])/head['axis_scale']
        t=np.clip(t,-6,6)
        for knot in (-1.5,-.5,.5,1.5):
            feats.append(np.maximum(t-knot,0))
        feats.extend([t,t**2,t[:,:1]*t[:,1:]])
    return np.column_stack(feats)


def fit_shared(model,x,y,limit=24576,width=96,supervised=True):
    rng=np.random.default_rng(3817)
    if len(x)>limit:
        idx=rng.choice(len(x),limit,replace=False)
        x,y=x[idx],y[idx]
    centers,counts=np.unique(np.round(operating_features(x)),axis=0,return_counts=True)
    centers=centers[np.argsort(-counts)[:8]]
    g=partition(x,centers)
    means=np.array([x[g==k,:21].mean(0) for k in range(len(centers))])
    scales=np.maximum(np.array([x[g==k,:21].std(0) for k in range(len(centers))]),1e-4)
    head=dict(centers=centers,mu=means,scale=scales,settings_mean=x[:,21:].mean(0),
        settings_sd=np.maximum(x[:,21:].std(0),1e-6),proj=rng.normal(0,1.2/np.sqrt(24),(24,width)),
        phase=rng.uniform(-np.pi,np.pi,width))
    z=np.clip((x[:,:21]-means[g])/scales[g],-6,6)
    if supervised:
        thresholds=np.quantile(y,[.15,.4,.65,.85])
        levels=np.column_stack((y/100,y[:,None]<thresholds))
        directions=[]
        for k in range(len(centers)):
            fields=z[g==k]
            gram=fields.T@fields
            gram.flat[::len(gram)+1]+=.01*len(fields)
            directions.append(np.linalg.solve(gram,fields.T@levels[g==k]))
        head['directions']=np.stack(directions)
        a=np.einsum('ni,nij->nj',z,head['directions'][g])
        head['axis_center']=a.mean(0)
        head['axis_scale']=np.maximum(a.std(0),1e-4)
    base=model.predict(x).reshape(-1)
    phi=shared_basis(x,head)
    gram=phi.T@phi
    gram.flat[::len(gram)+1]+=.001*len(x)
    head['coef']=np.linalg.solve(gram,phi.T@(y-base))
    return head


def predict_shared(model,head,x):
    return np.maximum(0,model.predict(x).reshape(-1)+shared_basis(x,head)@head['coef'])


