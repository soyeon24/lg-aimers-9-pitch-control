"""Neural-net preprocessing and inference -- numpy only.

The net is trained with torch (work/train_nn.py) but nothing torch-shaped ever
reaches the evaluation server: only the weight matrices and the standardisation
constants go into the artifact, and inference is three matmuls.

GELU is the tanh approximation on both sides -- torch is asked for
`nn.GELU(approximate="tanh")` during training, and `_gelu` below is the same
formula -- so the numpy path reproduces the torch path exactly rather than
approximately.
"""
import numpy as np

BASE_STATE_LEVELS = 8
SQRT_2_OVER_PI = 0.7978845608028654


def fit_prep(X):
    """Median fill values, missing-indicator columns, and standardisation."""
    A = X.to_numpy(dtype="float32")
    ind = np.where(np.isnan(A).mean(axis=0) > 1e-3)[0].astype("int64")
    med = np.nanmedian(A, axis=0).astype("float32")
    med = np.where(np.isnan(med), 0.0, med).astype("float32")
    B = _fill(A, med, ind)
    mu = B.mean(axis=0).astype("float32")
    sd = B.std(axis=0)
    sd = np.where(sd < 1e-6, 1.0, sd).astype("float32")
    return dict(med=med, ind=ind, mu=mu, sd=sd,
                bs=int(X.columns.get_loc("base_state")))


def _fill(A, med, ind):
    miss = np.isnan(A)
    return np.hstack([np.where(miss, med, A), miss[:, ind].astype("float32")])


def prep(X, p):
    A = X.to_numpy(dtype="float32")
    bs = A[:, p["bs"]]
    A = _fill(A, p["med"], p["ind"])
    A = np.clip((A - p["mu"]) / p["sd"], -6.0, 6.0)
    oh = np.zeros((len(A), BASE_STATE_LEVELS), dtype="float32")
    oh[np.arange(len(A)),
       np.nan_to_num(bs, nan=0.0).astype(int).clip(0, BASE_STATE_LEVELS - 1)] = 1.0
    return np.hstack([A, oh]).astype("float32")


def _gelu(h):
    return 0.5 * h * (1.0 + np.tanh(SQRT_2_OVER_PI * (h + 0.044715 * h ** 3)))


def forward(A, w, chunk=200000):
    """Probabilities from one exported net; chunked to bound peak memory."""
    out = np.empty(len(A), dtype="float64")
    for i in range(0, len(A), chunk):
        h = A[i:i + chunk]
        for k, (W, b) in enumerate(zip(w["W"], w["b"])):
            h = h @ W + b
            if k < len(w["W"]) - 1:
                h = _gelu(h)
        out[i:i + chunk] = 1.0 / (1.0 + np.exp(-h.squeeze(-1).astype("float64")))
    return out


def nn_predict(X, nn_art):
    """Seed-averaged probabilities for the whole net ensemble."""
    A = prep(X, nn_art["prep"])
    return np.mean([forward(A, w) for w in nn_art["nets"]], axis=0)


def load_weights(path):
    """Read a train_nn.py export back into the shape nn_predict expects."""
    z = np.load(path, allow_pickle=False)
    nets = []
    for k in range(int(z["n_nets"])):
        layers = int(z[f"n{k}_layers"])
        nets.append({"W": [z[f"n{k}_W{i}"] for i in range(layers)],
                     "b": [z[f"n{k}_b{i}"] for i in range(layers)]})
    prep = dict(med=z["prep_med"], ind=z["prep_ind"], mu=z["prep_mu"],
                sd=z["prep_sd"], bs=int(z["prep_bs"]))
    return {"nets": nets, "prep": prep, "columns": [str(c) for c in z["columns"]]}
