"""Diversity on the tree side.

Adding more net-shaped members did nothing -- linear, deeper, squared-error and
entity-embedding variants all landed at or below the v5 stack. So the remaining
disagreement, if any, has to come from trees that are built differently rather
than merely tuned differently: column subsampling, a different optimisation
path, or a different slice of history.

The recency model is the interesting one. Season sample weighting was rejected
twice as a *replacement* for the full fit, but a model trained only on the last
few seasons is a different proposition as a blend *member*: weaker on its own,
yet fitted to the regime the evaluation season actually lives in.
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import harness
import lab_v6
import nnlib
from common import TARGET, comp_score, load, logit, sigmoid

VAL = 2024
SEEDS = (0, 1)
EPS = 1e-6
CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache",
                     f"trees_{VAL}.npz")

df = load()
tr, va = df[df.season < VAL], df[df.season == VAL]
ytr, yva = tr[TARGET].to_numpy(), va[TARGET].to_numpy()
ctx = lab_v6.ctx_fn(tr)
Xtr = lab_v6.base(tr, ctx)
cols = list(Xtr.columns)
Xva = lab_v6.base(va, ctx)[cols]
gt = va.game_type.to_numpy()

z = np.load(CACHE)
p15, p31 = z["p15"], z["p31"]
prep = nnlib.fit_prep(Xtr)
Atr, Ava = nnlib.prep(Xtr, prep), nnlib.prep(Xva, prep)
pnn = np.mean([nnlib.train_nn(Atr, ytr, [Ava], sd, hidden=(256, 128), epochs=3)[1][0]
               for sd in (0, 1)], axis=0)
base = 0.65 * (0.65 * p15 + 0.35 * p31) + 0.35 * pnn


def oracle(p, mask=None):
    y = yva if mask is None else yva[mask]
    p = np.clip(p if mask is None else p[mask], EPS, 1 - EPS)
    return comp_score(sigmoid(logit(p) + logit(y.mean()) - logit(p.mean())), y)


print(f"v5 stack                            {oracle(base):8.1f}   "
      f"R {oracle(base, gt == 'R'):7.1f}  F {oracle(base, gt == 'F'):7.1f}\n",
      flush=True)


def fit(params=None, since=None, drop_season=False):
    t = time.time()
    X, Y = Xtr, ytr
    Xv = Xva
    if drop_season:
        keep = [c for c in cols if c != "season"]
        X, Xv = Xtr[keep], Xva[keep]
    if since is not None:
        m = tr.season.to_numpy() >= since
        X, Y = X[m], ytr[m]
    ms = harness.fit_models(X, Y, SEEDS, params)
    return harness.pmean(ms, Xv), time.time() - t


CANDIDATES = [
    ("max_features=0.6", dict(params=dict(max_features=0.6))),
    ("max_features=0.3", dict(params=dict(max_features=0.3))),
    ("lr=0.02 iter=1200", dict(params=dict(learning_rate=0.02, max_iter=1200))),
    ("leaf63 msl500 l2=0.1", dict(params=dict(max_leaf_nodes=63, min_samples_leaf=500,
                                              l2_regularization=0.1))),
    ("recent only (>=2022)", dict(since=2022)),
    ("recent only (>=2021)", dict(since=2021)),
    ("no `season` column", dict(drop_season=True)),
]

for tag, kw in CANDIDATES:
    p, el = fit(**kw)
    line = [f"w={w}:{oracle((1 - w) * base + w * p):7.1f}" for w in (0.1, 0.2, 0.3)]
    print(f"  {tag:24s} alone {oracle(p):7.1f}  ({el:.0f}s)", flush=True)
    print(f"    on top of v5 stack: " + "  ".join(line), flush=True)

print("\ndone", flush=True)
