"""Fit the blend weights instead of eyeballing them -- and per game_type.

The three members disagree in a structured way: the loose trees are much better
on F and much worse on R. One global weight vector has to compromise; letting R
and F rows carry their own weights costs nothing at inference (game_type is on
every row) and is still a train-time constant.

Weights are fitted on one season and then *checked* on another. Four free
parameters on 250k rows is not much fitting, but a weight vector that only works
on 2024 is exactly the trap this project has fallen into before.
"""
import itertools
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import harness
import lab_v6
import nnlib
from common import TARGET, comp_score, load, logit, sigmoid

SEASONS = (2022, 2024)
TREE_SEEDS = (0, 1, 2, 3)
NN_SEEDS = (0, 1, 2, 3)
EPS = 1e-6
HERE = os.path.dirname(os.path.abspath(__file__))

df = load()
data = {}
for V in SEASONS:
    cache = os.path.join(HERE, "cache", f"members3_{V}.npz")
    if os.path.exists(cache):
        z = np.load(cache)
        data[V] = {k: z[k] for k in z.files}
        print(f"[V{V} from cache]", flush=True)
        continue
    t = time.time()
    tr, va = df[df.season < V], df[df.season == V]
    ytr, yva = tr[TARGET].to_numpy(), va[TARGET].to_numpy()
    ctx = lab_v6.ctx_fn(tr)
    Xtr = lab_v6.base(tr, ctx)
    Xva = lab_v6.base(va, ctx)[list(Xtr.columns)]
    p15 = harness.pmean(harness.fit_models(Xtr, ytr, TREE_SEEDS), Xva)
    p31 = harness.pmean(harness.fit_models(
        Xtr, ytr, TREE_SEEDS,
        dict(max_leaf_nodes=31, l2_regularization=1.0, min_samples_leaf=2000)), Xva)
    prep = nnlib.fit_prep(Xtr)
    Atr, Ava = nnlib.prep(Xtr, prep), nnlib.prep(Xva, prep)
    pnn = np.mean([nnlib.train_nn(Atr, ytr, [Ava], sd, hidden=(256, 128),
                                  epochs=3)[1][0] for sd in NN_SEEDS], axis=0)
    d = dict(p15=p15, p31=p31, pnn=pnn, y=yva,
             R=(va.game_type.to_numpy() == "R").astype("int8"))
    np.savez_compressed(cache, **d)
    data[V] = d
    print(f"[V{V} fitted {time.time() - t:.0f}s]", flush=True)


def oracle(p, y):
    p = np.clip(p, EPS, 1 - EPS)
    return comp_score(sigmoid(logit(p) + logit(y.mean()) - logit(p.mean())), y)


def mix(d, w):
    return w[0] * d["p15"] + w[1] * d["p31"] + w[2] * d["pnn"]


GRID = [w for w in itertools.product(np.arange(0.0, 1.01, 0.05), repeat=3)
        if abs(sum(w) - 1.0) < 1e-9]
print(f"\n{len(GRID)} weight vectors on the simplex", flush=True)

V5 = (0.65 * 0.65, 0.65 * 0.35, 0.35)
print("\n=== v5 weights ===")
for V in SEASONS:
    print(f"  V{V}  {oracle(mix(data[V], V5), data[V]['y']):8.1f}")

print("\n=== global weights fitted per season ===")
best = {}
for V in SEASONS:
    d = data[V]
    sc = [(oracle(mix(d, w), d["y"]), w) for w in GRID]
    sc.sort(reverse=True)
    best[V] = sc[0]
    print(f"  V{V}  best {sc[0][0]:8.1f} at "
          f"({sc[0][1][0]:.2f},{sc[0][1][1]:.2f},{sc[0][1][2]:.2f})   "
          f"top-5 " + " ".join(f"{s:.0f}" for s, _ in sc[:5]))
for V in SEASONS:
    other = [s for s in SEASONS if s != V]
    for O in other:
        d = data[O]
        print(f"  weights fitted on {V} applied to {O}: "
              f"{oracle(mix(d, best[V][1]), d['y']):8.1f}")

print("\n=== per game_type weights ===")
for V in SEASONS:
    d = data[V]
    out = {}
    for g, m in (("R", d["R"] == 1), ("F", d["R"] == 0)):
        sub = {k: d[k][m] for k in ("p15", "p31", "pnn")}
        sc = [(oracle(mix(sub, w), d["y"][m]), w) for w in GRID]
        sc.sort(reverse=True)
        out[g] = sc[0]
        print(f"  V{V} {g}  best {sc[0][0]:8.1f} at "
              f"({sc[0][1][0]:.2f},{sc[0][1][1]:.2f},{sc[0][1][2]:.2f})"
              f"   v5 weights {oracle(mix(sub, V5), d['y'][m]):8.1f}")
    for O in SEASONS:
        e = data[O]
        p = np.where(e["R"] == 1, mix(e, out["R"][1]), mix(e, out["F"][1]))
        tag = "self" if O == V else "cross"
        print(f"    -> applied to {O} ({tag}): {oracle(p, e['y']):8.1f}")

print("\ndone", flush=True)
