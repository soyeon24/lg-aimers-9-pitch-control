"""Capacity wants to be different for R and F.

At four seeds, leaf15 scores R 858.8 / F 534.7 while leaf31 (l2=1, msl=2000)
scores R 839.6 / F 593.4. F is only 12% of rows but a 60-point swing there is
worth about +7 overall, and neither setting is good at both. So: blend them,
switch between them by game_type, or train a dedicated F model -- and while the
fits are in memory, also check season sample weighting, which was rejected long
ago on the broken 2023 holdout.
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import harness
import lab_v6
from common import TARGET, comp_score, load, logit, sigmoid

VAL = 2024
SEEDS = (0, 1, 2, 3)
P15 = {}
P31 = dict(max_leaf_nodes=31, l2_regularization=1.0, min_samples_leaf=2000)

df = load()
tr = df[df.season < VAL]
va = df[df.season == VAL]
ytr, yva = tr[TARGET].to_numpy(), va[TARGET].to_numpy()
ctx = lab_v6.ctx_fn(tr)
Xtr = lab_v6.base(tr, ctx)
cols = list(Xtr.columns)
Xva = lab_v6.base(va, ctx)[cols]
gt = va.game_type.to_numpy()
prev = tr[tr.season == VAL - 1]
Xprev = lab_v6.base(prev, ctx)[cols]
prev_rate = float(prev[TARGET].mean())
true_rate = float(yva.mean())
print(f"X={Xtr.shape} true={true_rate:.4f}", flush=True)


def oracle(p, y):
    return comp_score(sigmoid(logit(p) + logit(y.mean()) - logit(p.mean())), y)


def report(p, tag):
    print(f"  {tag:38s} all {oracle(p, yva):7.1f}   R {oracle(p[gt=='R'], yva[gt=='R']):7.1f}"
          f"   F {oracle(p[gt=='F'], yva[gt=='F']):7.1f}", flush=True)


def fit(params, w=None, tag=""):
    t = time.time()
    ms = harness.fit_models(Xtr, ytr, SEEDS, params) if w is None else _fitw(params, w)
    pv = harness.pmean(ms, Xva)
    print(f"  [{tag} {time.time()-t:.0f}s]", flush=True)
    return pv


def _fitw(params, w):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from common import CAT_COLS, PARAMS
    p = dict(PARAMS)
    p.update(params)
    ci = [Xtr.columns.get_loc(c) for c in CAT_COLS if c in Xtr.columns]
    out = []
    for sd in SEEDS:
        m = HistGradientBoostingClassifier(categorical_features=ci, random_state=sd, **p)
        m.fit(Xtr, ytr, sample_weight=w)
        out.append(m)
    return out


p15 = fit(P15, tag="leaf15")
report(p15, "leaf15")
p31 = fit(P31, tag="leaf31")
report(p31, "leaf31 l2=1 msl2000")

print("\n--- blends ---", flush=True)
for w in (0.2, 0.3, 0.4, 0.5, 0.7):
    report((1 - w) * p15 + w * p31, f"blend w={w}")

print("\n--- switch by game_type ---", flush=True)
sw = np.where(gt == "F", p31, p15)
report(sw, "leaf15 on R, leaf31 on F")
for w in (0.5,):
    report(np.where(gt == "F", (1 - w) * p15 + w * p31, p15),
           f"leaf15 on R, blend w={w} on F")

print("\n--- season sample weighting ---", flush=True)
seasons = tr.season.to_numpy()
for hl in (3.0, 5.0):
    w = 0.5 ** ((VAL - 1 - seasons) / hl)
    report(fit(P15, w=w, tag=f"halflife {hl}"), f"leaf15, season half-life {hl}")

print("\nTOTAL done", flush=True)
