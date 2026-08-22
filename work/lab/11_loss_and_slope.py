"""Are we optimising the right loss, and is the model over-confident?

The competition metric is Brier -- plain squared error on a 0/1 target -- but
HistGradientBoostingClassifier fits log loss. Those disagree: log loss pays a lot
of attention to getting extreme probabilities right, which is exactly where this
problem has no signal. A squared-error regressor on the 0/1 label optimises the
scoring rule directly, so it is worth a straight comparison.

Second question: with v4's much wider prediction spread, is the model now
over-confident? If so, squashing the logits toward the mean (slope a < 1) buys
Brier back. That was tested on v3 features and came out flat at a = 1; the
feature set has changed enough to be worth re-asking.
"""
import os
import sys
import time

import numpy as np
from sklearn.ensemble import (HistGradientBoostingClassifier,
                              HistGradientBoostingRegressor)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import lab_v6
from common import CAT_COLS, PARAMS, TARGET, comp_score, load, logit, sigmoid

VAL = 2024
SEEDS = (0, 1, 2, 3)
EPS = 1e-6

df = load()
tr, va = df[df.season < VAL], df[df.season == VAL]
ytr, yva = tr[TARGET].to_numpy(), va[TARGET].to_numpy()
ctx = lab_v6.ctx_fn(tr)
Xtr = lab_v6.base(tr, ctx)
cols = list(Xtr.columns)
Xva = lab_v6.base(va, ctx)[cols]
gt = va.game_type.to_numpy()
ci = [Xtr.columns.get_loc(c) for c in CAT_COLS]
print(f"X={Xtr.shape}  true={yva.mean():.4f}", flush=True)


def oracle(p, y):
    p = np.clip(p, EPS, 1 - EPS)
    return comp_score(sigmoid(logit(p) + logit(y.mean()) - logit(p.mean())), y)


def report(p, tag):
    print(f"  {tag:34s} all {oracle(p, yva):7.1f}   "
          f"R {oracle(p[gt == 'R'], yva[gt == 'R']):7.1f}   "
          f"F {oracle(p[gt == 'F'], yva[gt == 'F']):7.1f}", flush=True)


def fit(cls, **extra):
    t = time.time()
    p = dict(PARAMS)
    p.update(extra)
    ps = []
    for sd in SEEDS:
        m = cls(categorical_features=ci, random_state=sd, **p)
        m.fit(Xtr, ytr)
        ps.append(m.predict_proba(Xva)[:, 1] if hasattr(m, "predict_proba")
                  else m.predict(Xva))
    print(f"  [{cls.__name__} {time.time() - t:.0f}s]", flush=True)
    return np.clip(np.mean(ps, axis=0), EPS, 1 - EPS)


pc = fit(HistGradientBoostingClassifier)
report(pc, "log loss (current)")

pr = fit(HistGradientBoostingRegressor, loss="squared_error")
report(pr, "squared error on 0/1")

print("\n--- blend ---", flush=True)
for w in (0.3, 0.5, 0.7):
    report((1 - w) * pc + w * pr, f"blend w_regressor={w}")

print("\n--- logit slope on the classifier ---", flush=True)
zc = logit(pc) - logit(pc.mean())
for a in (0.85, 0.9, 0.95, 1.0, 1.05, 1.1):
    report(sigmoid(a * zc + logit(pc.mean())), f"a={a}")

print("\n--- logit slope on the blend (w=0.5) ---", flush=True)
pb = 0.5 * pc + 0.5 * pr
zb = logit(pb) - logit(pb.mean())
for a in (0.9, 0.95, 1.0, 1.05):
    report(sigmoid(a * zb + logit(pb.mean())), f"a={a}")

print("\ndone", flush=True)
