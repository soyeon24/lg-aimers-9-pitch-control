"""Axes that have not been touched at all yet.

Everything rejected so far was a *variation* of something already in the blend:
another MLP shape, another tree hyperparameter, another slice of history. Three
things here are structurally different:

bagging      HistGBM has no subsample parameter, so every seed sees all the
             rows and differs only in tie-breaking. Bootstrapping the rows by
             hand gives the kind of independence that makes random forests work,
             and it is the one tree-diversity axis never tried.
forest       ExtraTrees is a different algorithm, not a different setting:
             averaging deep independent trees instead of boosting shallow
             dependent ones. Weak alone on thin signal, but wrong elsewhere.
residual     stacking in sequence rather than in parallel -- fit a small GBM to
             what the current blend still gets wrong, instead of averaging
             another opinion into it.
interactions leaf-15 trees are depth-limited, so a product the tree would need
             four splits to approximate is cheap to just hand it.
"""
import os
import sys
import time

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import harness
import lab_v6
import nnlib
from common import CAT_COLS, PARAMS, TARGET, comp_score, load, logit, sigmoid

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

z = np.load(CACHE)
prep = nnlib.fit_prep(Xtr)
Atr, Ava = nnlib.prep(Xtr, prep), nnlib.prep(Xva, prep)
pnn = np.mean([nnlib.train_nn(Atr, ytr, [Ava], sd, hidden=(256, 128), epochs=3)[1][0]
               for sd in (0, 1)], axis=0)
base = 0.65 * (0.65 * z["p15"] + 0.35 * z["p31"]) + 0.35 * pnn


def oracle(p):
    q = np.clip(p, EPS, 1 - EPS)
    return comp_score(sigmoid(logit(q) + logit(yva.mean()) - logit(q.mean())), yva)


def report(p, tag, weights=(0.1, 0.2, 0.3)):
    print(f"  {tag:32s} alone {oracle(p):7.1f}", flush=True)
    print("    on top of base: " + "  ".join(
        f"w={w}:{oracle((1 - w) * base + w * p):7.1f}" for w in weights), flush=True)


print(f"base {oracle(base):7.1f}\n", flush=True)

# ---------------------------------------------------------------- bagging
print("--- row-bootstrap bagging ---", flush=True)
t = time.time()
ci = [cols.index(c) for c in CAT_COLS]
ps = []
rng = np.random.default_rng(0)
for k in range(4):
    idx = rng.integers(0, len(Xtr), len(Xtr))
    m = HistGradientBoostingClassifier(categorical_features=ci, random_state=k, **PARAMS)
    m.fit(Xtr.iloc[idx], ytr[idx])
    ps.append(m.predict_proba(Xva)[:, 1])
report(np.mean(ps, axis=0), f"bagged leaf15 x4 ({time.time() - t:.0f}s)")

# ---------------------------------------------------------------- forest
print("\n--- extremely randomised trees ---", flush=True)
Ftr = np.nan_to_num(Xtr.to_numpy(dtype="float32"), nan=-999.0)
Fva = np.nan_to_num(Xva.to_numpy(dtype="float32"), nan=-999.0)
for tag, kw in [("depth12 msl2000 n80", dict(max_depth=12, min_samples_leaf=2000,
                                             n_estimators=80)),
                ("depth20 msl5000 n80", dict(max_depth=20, min_samples_leaf=5000,
                                             n_estimators=80))]:
    t = time.time()
    et = ExtraTreesClassifier(n_jobs=-1, random_state=0, max_features=0.5, **kw)
    et.fit(Ftr, ytr)
    report(et.predict_proba(Fva)[:, 1], f"extratrees {tag} ({time.time() - t:.0f}s)")

# ---------------------------------------------------------------- residual
print("\n--- sequential stack: GBM on the blend's residual ---", flush=True)
t = time.time()
ptr = 0.65 * (0.65 * harness.pmean(harness.fit_models(Xtr, ytr, (0,)), Xtr)
              + 0.35 * harness.pmean(harness.fit_models(
                  Xtr, ytr, (0,), dict(max_leaf_nodes=31, l2_regularization=1.0,
                                       min_samples_leaf=2000)), Xtr)) \
    + 0.35 * nnlib.train_nn(Atr, ytr, [Atr], 0, hidden=(256, 128), epochs=3)[1][0]
print(f"  [in-sample blend rebuilt {time.time() - t:.0f}s]", flush=True)
from sklearn.ensemble import HistGradientBoostingRegressor
t = time.time()
resid = ytr - ptr
rm = HistGradientBoostingRegressor(categorical_features=ci, random_state=0,
                                   max_iter=200, learning_rate=0.03,
                                   max_leaf_nodes=15, l2_regularization=10.0,
                                   min_samples_leaf=10000, early_stopping=False)
rm.fit(Xtr, resid)
corr = rm.predict(Xva)
print(f"  [residual model {time.time() - t:.0f}s]", flush=True)
for a in (0.25, 0.5, 1.0):
    print(f"    base + {a} * residual: {oracle(np.clip(base + a * corr, EPS, 1 - EPS)):7.1f}",
          flush=True)

print("\ndone", flush=True)
