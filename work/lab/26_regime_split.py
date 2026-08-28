"""Two new feature families, triaged on the 2024 holdout.

Everything the v7 round rejected was another *opinion* about the same features:
another tree setting, another net shape, another slice of history, and now
(experiment 25) another algorithm family entirely -- bagging, ExtraTrees and
residual stacking all landed below the 892.4 base. The ensemble is saturated.
What has not been touched is the feature table.

regime   `asof_pitcher_success_rate` pools pitches measured under two different
         rules -- F changed definition in 2023 -- and the mix differs wildly by
         pitcher (median F share .19, quartiles .01 / .89). feats7 splits the
         career carry table by regime so the tree sees a clean rate, a
         regime-matched prior, and how contaminated the pooled one is.
platoon  per-pitcher and per-batter handedness splits. `same_hand` carries the
         league-average version; the individual deviation is missing.
count    per-pitcher behaviour when ahead versus behind in the count.

Judged on the oracle (calibration-free) score so the offset cannot mask or
manufacture a difference. Trees only, 2 seeds, tight+loose at the production
0.65/0.35 -- the net is a fixed 0.35 on top of exactly this and would only add
noise to a feature comparison.
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import feats7
import feats8
import harness
import lab_v6
from common import TARGET, comp_score, load, logit, sigmoid

SEEDS = (0, 1)
EPS = 1e-6
LOOSE = dict(max_leaf_nodes=31, l2_regularization=1.0, min_samples_leaf=2000)
VAL = int(sys.argv[1]) if len(sys.argv) > 1 else 2024

df = load()


def oracle(p, y):
    q = np.clip(p, EPS, 1 - EPS)
    return comp_score(sigmoid(logit(q) + logit(y.mean()) - logit(q.mean())), y)


def evaluate(V, build, ctx_fn, tag):
    tr, va = df[df.season < V], df[df.season == V]
    ytr, yva = tr[TARGET].to_numpy(), va[TARGET].to_numpy()
    isF = va.game_type.to_numpy() == "F"
    t = time.time()
    ctx = ctx_fn(tr)
    Xtr = build(tr, ctx)
    cols = list(Xtr.columns)
    Xva = build(va, ctx)[cols]
    p15 = harness.pmean(harness.fit_models(Xtr, ytr, SEEDS), Xva)
    p31 = harness.pmean(harness.fit_models(Xtr, ytr, SEEDS, LOOSE), Xva)
    p = 0.65 * p15 + 0.35 * p31
    print(f"  {tag:26s} f={len(cols):3d} | all {oracle(p, yva):7.1f} "
          f"R {oracle(p[~isF], yva[~isF]):7.1f} F {oracle(p[isF], yva[isF]):7.1f} "
          f"| tight {oracle(p15, yva):7.1f} loose {oracle(p31, yva):7.1f} "
          f"({time.time() - t:.0f}s)", flush=True)
    np.save(os.path.join("cache", f"f7_{tag.replace(' ', '_')}_{V}.npy"), p)
    return p


CONFIGS = [
    ("baseline v6", lab_v6.base, lab_v6.ctx_fn),
    ("regime", feats7.make(), feats7.fit),
    ("regime+platoon", feats8.make(platoon=True, count=False), feats8.fit),
    ("regime+platoon+count", feats8.make(platoon=True, count=True), feats8.fit),
]

print(f"===== holdout {VAL} =====", flush=True)
for tag, build, ctx_fn in CONFIGS:
    evaluate(VAL, build, ctx_fn, tag)

print("\ndone", flush=True)
