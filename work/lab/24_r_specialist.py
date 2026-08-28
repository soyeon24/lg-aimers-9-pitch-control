"""If an F specialist works, why not an R specialist?

The F model was worth +5.6 on the holdout and +13.3 on the leaderboard, on 12%
of the rows. The argument was that a model fitted on F alone never has to
compromise with the other 88%. That argument is symmetric: the shared model also
spends capacity separating the old and new F regimes and fitting F leaves, none
of which serves the R rows that carry 88% of the score.

I assumed R was already covered because it dominates the training set. That was
an assumption, not a measurement. This measures it.

Also settles the F specialist ensemble for v7: several combinations landed in
898.1-898.7 against v6's 897.9, and the consistency across combinations -- not
one lucky config -- is what makes it worth taking.
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
Xva = lab_v6.base(va, ctx)[list(Xtr.columns)]
isF = va.game_type.to_numpy() == "F"
allmask = np.ones(len(yva), bool)

z = np.load(CACHE)
prep = nnlib.fit_prep(Xtr)
Atr, Ava = nnlib.prep(Xtr, prep), nnlib.prep(Xva, prep)
pnn = np.mean([nnlib.train_nn(Atr, ytr, [Ava], sd, hidden=(256, 128), epochs=3)[1][0]
               for sd in (0, 1)], axis=0)
base = 0.65 * (0.65 * z["p15"] + 0.35 * z["p31"]) + 0.35 * pnn


def oracle(p, mask):
    y = yva[mask]
    q = np.clip(p[mask], EPS, 1 - EPS)
    return comp_score(sigmoid(logit(q) + logit(y.mean()) - logit(q.mean())), y)


trF = tr.game_type.to_numpy() == "F"

# v6's specialist, and the ensemble that beat it, so v7 has a like-for-like base
print("building the F side first", flush=True)
fp = {}
for tag, params in [("leaf31", dict(max_leaf_nodes=31, min_samples_leaf=1000,
                                    l2_regularization=1.0)),
                    ("leaf15", dict(min_samples_leaf=2000))]:
    fp[tag] = harness.pmean(harness.fit_models(Xtr[trF], ytr[trF], SEEDS, params), Xva)


def withF(p, w):
    q = base.copy()
    q[isF] = (1 - w) * base[isF] + w * p[isF]
    return q


v6 = withF(fp["leaf31"], 0.25)
fens = 0.5 * (fp["leaf31"] + fp["leaf15"])
v7f = withF(fens, 0.30)
print(f"  base (no specialist)   all {oracle(base, allmask):7.1f}  F {oracle(base, isF):7.1f}")
print(f"  v6 (leaf31 w=0.25)     all {oracle(v6, allmask):7.1f}  F {oracle(v6, isF):7.1f}")
print(f"  F ensemble w=0.30      all {oracle(v7f, allmask):7.1f}  F {oracle(v7f, isF):7.1f}\n",
      flush=True)

print("--- R specialist (fitted on R rows only) ---", flush=True)
trR = ~trF
isR = ~isF
for tag, params in [("leaf15 (same as tight)", {}),
                    ("leaf31 msl2000 l2=1", dict(max_leaf_nodes=31,
                                                 min_samples_leaf=2000,
                                                 l2_regularization=1.0)),
                    ("leaf15 msl5000", dict(min_samples_leaf=5000))]:
    t = time.time()
    p = harness.pmean(harness.fit_models(Xtr[trR], ytr[trR], SEEDS, params), Xva)
    print(f"  {tag:24s} alone(R) {oracle(p, isR):7.1f}  ({time.time() - t:.0f}s)",
          flush=True)
    line = []
    for w in (0.15, 0.25, 0.35, 0.5):
        q = v7f.copy()
        q[isR] = (1 - w) * v7f[isR] + w * p[isR]
        line.append(f"w={w}: all {oracle(q, allmask):7.1f} R {oracle(q, isR):7.1f}")
    print("    " + "   ".join(line), flush=True)

print("\ndone", flush=True)
