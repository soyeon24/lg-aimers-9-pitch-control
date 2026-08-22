"""Tune the net, with the tree predictions cached so each run costs ~30s.

Refitting the trees takes 12 minutes and they never change here, so they are
fitted once and stored. What matters is not the net's solo score but how much it
moves the blend -- a weaker, more different net can beat a stronger, more
tree-like one.
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
TREE_SEEDS = (0, 1, 2, 3)
CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache",
                     f"trees_{VAL}.npz")
EPS = 1e-6

df = load()
tr, va = df[df.season < VAL], df[df.season == VAL]
ytr, yva = tr[TARGET].to_numpy(), va[TARGET].to_numpy()
ctx = lab_v6.ctx_fn(tr)
Xtr = lab_v6.base(tr, ctx)
Xva = lab_v6.base(va, ctx)[list(Xtr.columns)]
gt = va.game_type.to_numpy()

if os.path.exists(CACHE):
    z = np.load(CACHE)
    p15, p31 = z["p15"], z["p31"]
    print("[trees from cache]", flush=True)
else:
    t = time.time()
    p15 = harness.pmean(harness.fit_models(Xtr, ytr, TREE_SEEDS), Xva)
    p31 = harness.pmean(harness.fit_models(
        Xtr, ytr, TREE_SEEDS,
        dict(max_leaf_nodes=31, l2_regularization=1.0, min_samples_leaf=2000)), Xva)
    np.savez_compressed(CACHE, p15=p15, p31=p31)
    print(f"[trees fitted {time.time() - t:.0f}s -> {CACHE}]", flush=True)

ptree = 0.6 * p15 + 0.4 * p31


def oracle(p, y):
    p = np.clip(p, EPS, 1 - EPS)
    return comp_score(sigmoid(logit(p) + logit(y.mean()) - logit(p.mean())), y)


def report(p, tag):
    print(f"  {tag:40s} all {oracle(p, yva):7.1f}   "
          f"R {oracle(p[gt == 'R'], yva[gt == 'R']):7.1f}   "
          f"F {oracle(p[gt == 'F'], yva[gt == 'F']):7.1f}", flush=True)


report(ptree, "tree blend 0.6/0.4")

prep = nnlib.fit_prep(Xtr)
Atr, Ava = nnlib.prep(Xtr, prep), nnlib.prep(Xva, prep)
print(f"nn input {Atr.shape}\n", flush=True)

# Longer training destroys it: at 15 epochs the net scores a flat 0, because the
# signal is so thin (BSS around 0.01) that it memorises noise almost immediately.
# So the grid runs the other way -- fewer epochs, smaller nets, more decay.
CONFIGS = [
    ("h256-128 e6   wd1e-4", dict(hidden=(256, 128), epochs=6)),
    ("h256-128 e3   wd1e-4", dict(hidden=(256, 128), epochs=3)),
    ("h256-128 e4   wd1e-4", dict(hidden=(256, 128), epochs=4)),
    ("h256-128 e8   wd1e-4", dict(hidden=(256, 128), epochs=8)),
    ("h128-64  e6   wd1e-4", dict(hidden=(128, 64), epochs=6)),
    ("h128-64  e10  wd1e-4", dict(hidden=(128, 64), epochs=10)),
    ("h64-32   e10  wd1e-4", dict(hidden=(64, 32), epochs=10)),
    ("h256-128 e8   wd1e-2", dict(hidden=(256, 128), epochs=8, wd=1e-2)),
    ("h256-128 e10  do0.3", dict(hidden=(256, 128), epochs=10, dropout=0.3)),
    ("h512-256 e4   wd1e-4", dict(hidden=(512, 256), epochs=4)),
    ("h256-128 e6   lr5e-4", dict(hidden=(256, 128), epochs=6, lr=5e-4)),
]

best = None
for tag, kw in CONFIGS:
    ps, tt = [], 0.0
    for sd in (0, 1):
        _, (pn,), el = nnlib.train_nn(Atr, ytr, [Ava], sd, **kw)
        ps.append(pn)
        tt += el
    pnn = np.mean(ps, axis=0)
    print(f"[{tag}  {tt:.0f}s]", flush=True)
    report(pnn, f"  nn alone")
    line = []
    for w in (0.2, 0.3, 0.4):
        s = oracle((1 - w) * ptree + w * pnn, yva)
        line.append(f"w={w}:{s:7.1f}")
        if best is None or s > best[0]:
            best = (s, tag, w)
    print(f"  blend  " + "   ".join(line), flush=True)

print(f"\nbest: {best[1]}  w_nn={best[2]}  -> {best[0]:.1f}", flush=True)
print("done", flush=True)
