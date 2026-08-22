"""Does the net blend hold up on a season other than 2024?

This is the check that v2 skipped: a +20 that only exists on one holdout is the
exact shape of measurement noise. 2023 cannot be used at all (the F regime break
makes even the oracle score 0), so 2021 and 2022 are the available second
opinions -- smaller training sets and a different F regime, so absolute numbers
will not match 2024, only the direction matters.
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

SEASONS = (2021, 2022, 2024)
TREE_SEEDS = (0, 1, 2, 3)
NN_SEEDS = (0, 1, 2, 3)
# e3 and e6 tie on 2024 (blend 892.9 vs 893.2) but differ a lot standalone
# (835 vs 777), so carry both here and prefer whichever holds up across seasons.
NN_VARIANTS = [("e3", dict(hidden=(256, 128), epochs=3, lr=2e-3)),
               ("e6", dict(hidden=(256, 128), epochs=6, lr=2e-3))]
EPS = 1e-6

df = load()
for V in SEASONS:
    t = time.time()
    tr, va = df[df.season < V], df[df.season == V]
    ytr, yva = tr[TARGET].to_numpy(), va[TARGET].to_numpy()
    ctx = lab_v6.ctx_fn(tr)
    Xtr = lab_v6.base(tr, ctx)
    Xva = lab_v6.base(va, ctx)[list(Xtr.columns)]
    gt = va.game_type.to_numpy()

    def oracle(p):
        p = np.clip(p, EPS, 1 - EPS)
        return comp_score(sigmoid(logit(p) + logit(yva.mean()) - logit(p.mean())), yva)

    p15 = harness.pmean(harness.fit_models(Xtr, ytr, TREE_SEEDS), Xva)
    p31 = harness.pmean(harness.fit_models(
        Xtr, ytr, TREE_SEEDS,
        dict(max_leaf_nodes=31, l2_regularization=1.0, min_samples_leaf=2000)), Xva)
    ptree = 0.6 * p15 + 0.4 * p31

    prep = nnlib.fit_prep(Xtr)
    Atr, Ava = nnlib.prep(Xtr, prep), nnlib.prep(Xva, prep)
    print(f"\n=== V{V} ({time.time() - t:.0f}s, train {len(tr)}) ===", flush=True)
    print(f"  leaf15 only        {oracle(p15):8.1f}", flush=True)
    print(f"  tree blend         {oracle(ptree):8.1f}", flush=True)
    for name, kw in NN_VARIANTS:
        pnn = np.mean([nnlib.train_nn(Atr, ytr, [Ava], sd, **kw)[1][0]
                       for sd in NN_SEEDS], axis=0)
        print(f"  nn {name} alone        {oracle(pnn):8.1f}", flush=True)
        for w in (0.2, 0.3, 0.4):
            p = np.clip((1 - w) * ptree + w * pnn, EPS, 1 - EPS)
            pr, yr = p[gt == "R"], yva[gt == "R"]
            r = comp_score(sigmoid(logit(pr) + logit(yr.mean()) - logit(pr.mean())), yr)
            print(f"    tree+nn {name} w={w}   {oracle(p):8.1f}   R {r:8.1f}",
                  flush=True)
print("\ndone", flush=True)
