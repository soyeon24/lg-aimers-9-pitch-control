"""A neural net as a second opinion.

Blending leaf15 with leaf31 was worth +9 on the holdout purely from disagreement
between two tree shapes. An MLP disagrees far more: it interpolates smoothly
where trees step, and it mixes all 107 features in every unit instead of down a
path of axis-aligned splits. If diversity is what is left to harvest, this is
where the most of it is.

Nothing about the submission depends on torch. The net is fitted here, and only
its weight matrices and the standardisation constants would go into the
artifact -- inference is three numpy matmuls.
"""
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import harness
import lab_v6
from common import TARGET, comp_score, load, logit, sigmoid

VAL = 2024
TREE_SEEDS = (0, 1, 2, 3)
NN_SEEDS = (0, 1, 2, 3)
EPS = 1e-6
BASE_STATE_LEVELS = 8

torch.set_num_threads(os.cpu_count() - 2)


# ------------------------------------------------------------------ preprocess
def fit_prep(X):
    """Constants for the numpy-only forward pass: medians, means, scales."""
    A = X.to_numpy(dtype="float32")
    nan_rate = np.isnan(A).mean(axis=0)
    ind = np.where(nan_rate > 1e-3)[0]
    med = np.nanmedian(A, axis=0).astype("float32")
    med = np.where(np.isnan(med), 0.0, med).astype("float32")
    B = _apply_fill(A, med, ind)
    mu = B.mean(axis=0)
    sd = B.std(axis=0)
    sd = np.where(sd < 1e-6, 1.0, sd).astype("float32")
    return dict(cols=list(X.columns), med=med, ind=ind, mu=mu, sd=sd,
                bs=X.columns.get_loc("base_state"))


def _apply_fill(A, med, ind):
    miss = np.isnan(A)
    A = np.where(miss, med, A)
    return np.hstack([A, miss[:, ind].astype("float32")])


def prep(X, p):
    A = X.to_numpy(dtype="float32")
    bs = A[:, p["bs"]]
    A = _apply_fill(A, p["med"], p["ind"])
    A = np.clip((A - p["mu"]) / p["sd"], -6.0, 6.0)
    oh = np.zeros((len(A), BASE_STATE_LEVELS), dtype="float32")
    idx = np.nan_to_num(bs, nan=0.0).astype(int).clip(0, BASE_STATE_LEVELS - 1)
    oh[np.arange(len(A)), idx] = 1.0
    return np.hstack([A, oh]).astype("float32")


class MLP(nn.Module):
    def __init__(self, d, h1=256, h2=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, h1), nn.GELU(),
            nn.Linear(h1, h2), nn.GELU(),
            nn.Linear(h2, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_nn(Atr, ytr, Ava, seed, epochs=6, bs=8192, lr=2e-3):
    torch.manual_seed(seed)
    model = MLP(Atr.shape[1])
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    xt = torch.from_numpy(Atr)
    yt = torch.from_numpy(ytr.astype("float32"))
    n = len(xt)
    steps = epochs * (n // bs + 1)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps,
                                                pct_start=0.2)
    lossf = nn.BCEWithLogitsLoss()
    g = torch.Generator().manual_seed(seed)
    done = 0
    for ep in range(epochs):
        t = time.time()
        perm = torch.randperm(n, generator=g)
        tot = 0.0
        for i in range(0, n, bs):
            j = perm[i:i + bs]
            opt.zero_grad()
            out = model(xt[j])
            loss = lossf(out, yt[j])
            loss.backward()
            opt.step()
            if done < steps - 1:
                sched.step()
                done += 1
            tot += loss.item() * len(j)
        print(f"    seed {seed} epoch {ep} logloss {tot / n:.5f} "
              f"({time.time() - t:.0f}s)", flush=True)
    model.eval()
    with torch.no_grad():
        pv = torch.sigmoid(model(torch.from_numpy(Ava))).numpy()
    return model, pv.astype("float64")


def main():
    df = load()
    tr, va = df[df.season < VAL], df[df.season == VAL]
    ytr, yva = tr[TARGET].to_numpy(), va[TARGET].to_numpy()
    ctx = lab_v6.ctx_fn(tr)
    Xtr = lab_v6.base(tr, ctx)
    Xva = lab_v6.base(va, ctx)[list(Xtr.columns)]
    gt = va.game_type.to_numpy()

    def oracle(p, y):
        p = np.clip(p, EPS, 1 - EPS)
        return comp_score(sigmoid(logit(p) + logit(y.mean()) - logit(p.mean())), y)

    def report(p, tag):
        print(f"  {tag:34s} all {oracle(p, yva):7.1f}   "
              f"R {oracle(p[gt == 'R'], yva[gt == 'R']):7.1f}   "
              f"F {oracle(p[gt == 'F'], yva[gt == 'F']):7.1f}", flush=True)

    t = time.time()
    p15 = harness.pmean(harness.fit_models(Xtr, ytr, TREE_SEEDS), Xva)
    p31 = harness.pmean(harness.fit_models(
        Xtr, ytr, TREE_SEEDS,
        dict(max_leaf_nodes=31, l2_regularization=1.0, min_samples_leaf=2000)), Xva)
    ptree = 0.6 * p15 + 0.4 * p31
    print(f"[trees {time.time() - t:.0f}s]", flush=True)
    report(p15, "leaf15")
    report(ptree, "tree blend (0.6/0.4)")

    p = fit_prep(Xtr)
    Atr, Ava = prep(Xtr, p), prep(Xva, p)
    print(f"nn input {Atr.shape}", flush=True)
    pns = []
    for sd in NN_SEEDS:
        _, pn = train_nn(Atr, ytr, Ava, sd)
        pns.append(pn)
        report(pn, f"nn seed {sd}")
    pnn = np.mean(pns, axis=0)
    report(pnn, f"nn x{len(NN_SEEDS)}")

    print("\n--- tree blend + nn ---", flush=True)
    for w in (0.1, 0.2, 0.3, 0.4, 0.5):
        report((1 - w) * ptree + w * pnn, f"w_nn={w}")
    print("\n--- in logit space ---", flush=True)
    for w in (0.2, 0.3, 0.4):
        report(sigmoid((1 - w) * logit(ptree) + w * logit(pnn)), f"logit w_nn={w}")
    print("\ndone", flush=True)


if __name__ == "__main__":
    main()
