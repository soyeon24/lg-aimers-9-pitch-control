"""Shared pieces for the neural-net experiments.

The net is weak on its own (769 against the tree blend's 870) but blending it in
at w=0.3 is worth +20, because it is wrong in different places: it interpolates
smoothly where trees step, and every unit sees all 107 features at once instead
of one axis-aligned path.

Only the weight matrices and the standardisation constants ever reach the
artifact, so the submission needs numpy and nothing else -- `forward_numpy`
below is the exact inference path that would ship.
"""
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

BASE_STATE_LEVELS = 8
torch.set_num_threads(max(1, (os.cpu_count() or 4) - 2))


# ------------------------------------------------------------------ preprocess
def fit_prep(X):
    A = X.to_numpy(dtype="float32")
    ind = np.where(np.isnan(A).mean(axis=0) > 1e-3)[0]
    med = np.nanmedian(A, axis=0).astype("float32")
    med = np.where(np.isnan(med), 0.0, med).astype("float32")
    B = _fill(A, med, ind)
    mu = B.mean(axis=0).astype("float32")
    sd = B.std(axis=0)
    sd = np.where(sd < 1e-6, 1.0, sd).astype("float32")
    return dict(cols=list(X.columns), med=med, ind=ind, mu=mu, sd=sd,
                bs=X.columns.get_loc("base_state"))


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


# ------------------------------------------------------------------------ net
class MLP(nn.Module):
    def __init__(self, d, hidden=(256, 128), dropout=0.0):
        super().__init__()
        layers = []
        prev = d
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.GELU(approximate="tanh")]
            if dropout:
                layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_nn(Atr, ytr, Aevals, seed, hidden=(256, 128), epochs=6, bs=8192,
             lr=2e-3, wd=1e-4, dropout=0.0, verbose=False):
    torch.manual_seed(seed)
    model = MLP(Atr.shape[1], hidden, dropout)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    xt, yt = torch.from_numpy(Atr), torch.from_numpy(ytr.astype("float32"))
    n = len(xt)
    steps = epochs * (n // bs + 1)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps,
                                                pct_start=0.2)
    lossf = nn.BCEWithLogitsLoss()
    g = torch.Generator().manual_seed(seed)
    done = 0
    t0 = time.time()
    for ep in range(epochs):
        perm = torch.randperm(n, generator=g)
        tot = 0.0
        for i in range(0, n, bs):
            j = perm[i:i + bs]
            opt.zero_grad()
            loss = lossf(model(xt[j]), yt[j])
            loss.backward()
            opt.step()
            if done < steps - 1:
                sched.step()
                done += 1
            tot += loss.item() * len(j)
        if verbose:
            print(f"      ep{ep} {tot / n:.5f}", flush=True)
    model.eval()
    outs = []
    with torch.no_grad():
        for A in Aevals:
            outs.append(torch.sigmoid(model(torch.from_numpy(A))).numpy().astype("float64"))
    return model, outs, time.time() - t0


def export(model):
    """Weights as plain numpy, for an artifact that needs no torch at inference."""
    ws, bsz = [], []
    for m in model.net:
        if isinstance(m, nn.Linear):
            ws.append(m.weight.detach().numpy().T.astype("float32").copy())
            bsz.append(m.bias.detach().numpy().astype("float32").copy())
    return dict(W=ws, b=bsz)


def forward_numpy(A, w):
    """GELU MLP forward pass; must match MLP.forward exactly."""
    h = A
    for i, (W, b) in enumerate(zip(w["W"], w["b"])):
        h = h @ W + b
        if i < len(w["W"]) - 1:
            # tanh-approximated GELU on both sides, so the numpy path is exact
            # and the artifact needs nothing beyond numpy
            h = 0.5 * h * (1.0 + np.tanh(0.7978845608028654
                                         * (h + 0.044715 * h ** 3)))
    z = h.squeeze(-1)
    return 1.0 / (1.0 + np.exp(-z))
