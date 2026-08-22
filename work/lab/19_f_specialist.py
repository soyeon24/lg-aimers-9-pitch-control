"""The F subset is the last big gap: 583.6 against R's 890.2.

F is 12% of rows, so closing half that gap is worth roughly +20 overall. The
loose trees already help there (600 vs 552 for the tight ones), which says F
wants finer splits than the shared model can afford to spend. A model fitted on
F rows alone can spend all of them there -- it only has ~135k rows to learn
from, but it never has to compromise with the 88% of the data that wants
something else.

Blended in for F rows only; R rows keep the v5 stack untouched.
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
isF = gt == "F"

z = np.load(CACHE)
prep = nnlib.fit_prep(Xtr)
Atr, Ava = nnlib.prep(Xtr, prep), nnlib.prep(Xva, prep)
pnn = np.mean([nnlib.train_nn(Atr, ytr, [Ava], sd, hidden=(256, 128), epochs=3)[1][0]
               for sd in SEEDS], axis=0)
base = 0.65 * (0.65 * z["p15"] + 0.35 * z["p31"]) + 0.35 * pnn


def oracle(p, mask):
    y = yva[mask]
    q = np.clip(p[mask], EPS, 1 - EPS)
    return comp_score(sigmoid(logit(q) + logit(y.mean()) - logit(q.mean())), y)


print(f"v5 stack   all {oracle(base, np.ones(len(yva), bool)):7.1f}  "
      f"R {oracle(base, ~isF):7.1f}  F {oracle(base, isF):7.1f}\n", flush=True)

mF = tr.game_type.to_numpy() == "F"
XF, yF = Xtr[mF], ytr[mF]
print(f"F training rows {len(XF)}", flush=True)

VARIANTS = [
    ("leaf15 msl2000", dict(min_samples_leaf=2000)),
    ("leaf31 msl1000 l2=1", dict(max_leaf_nodes=31, min_samples_leaf=1000,
                                 l2_regularization=1.0)),
    ("leaf15 msl500 iter200", dict(min_samples_leaf=500, max_iter=200)),
]
for tag, params in VARIANTS:
    t = time.time()
    p = harness.pmean(harness.fit_models(XF, yF, SEEDS, params), Xva)
    print(f"  F-specialist {tag:22s} alone(F) {oracle(p, isF):7.1f}  "
          f"({time.time() - t:.0f}s)", flush=True)
    line = []
    for w in (0.2, 0.3, 0.5, 0.7):
        q = base.copy()
        q[isF] = (1 - w) * base[isF] + w * p[isF]
        line.append(f"w={w}: F {oracle(q, isF):6.1f} all {oracle(q, np.ones(len(yva), bool)):7.1f}")
    print("    " + "   ".join(line), flush=True)

print("\ndone", flush=True)
