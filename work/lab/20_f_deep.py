"""Push the F specialist further.

v6's F model was worth +5.6 on the holdout and +13.3 on the leaderboard. That
gap is probably not luck: the 2024 holdout trains its F model on a single
post-break season (2023), while the submitted model gets two (2023 + 2024). So
the holdout systematically understates anything F-specific, and the real thing
should do better than what we measure here.

Questions:
  * which slice of history should the specialist see -- all of it, or only the
    post-break regime that 2025 actually belongs to?
  * how much weight, and does averaging several specialists beat picking one?
  * does an F-only net add the same kind of diversity there as it did globally?

F fits are cheap (12-30s on ~130k rows), so this sweeps widely.
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
SEEDS = (0, 1, 2, 3)
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


def mixF(p, w):
    q = base.copy()
    q[isF] = (1 - w) * base[isF] + w * p[isF]
    return q


print(f"v5 stack (no specialist)   all {oracle(base, allmask):7.1f}  "
      f"F {oracle(base, isF):7.1f}\n", flush=True)

trF = tr.game_type.to_numpy() == "F"
season = tr.season.to_numpy()

SLICES = [
    ("all seasons", trF),
    ("2023+ only (post-break)", trF & (season >= 2023)),
    ("2022+ only", trF & (season >= 2022)),
    ("2021+ only", trF & (season >= 2021)),
]
PARAMS = [
    ("leaf31 msl1000 l2=1", dict(max_leaf_nodes=31, min_samples_leaf=1000,
                                 l2_regularization=1.0)),
    ("leaf15 msl2000", dict(min_samples_leaf=2000)),
]

preds = {}
for sname, m in SLICES:
    for pname, params in PARAMS:
        t = time.time()
        p = harness.pmean(harness.fit_models(Xtr[m], ytr[m], SEEDS, params), Xva)
        preds[(sname, pname)] = p
        line = [f"w={w}:{oracle(mixF(p, w), allmask):7.1f}/F{oracle(mixF(p, w), isF):6.1f}"
                for w in (0.2, 0.35, 0.5)]
        print(f"  {sname:24s} {pname:20s} n={int(m.sum()):6d} "
              f"alone(F) {oracle(p, isF):6.1f} ({time.time() - t:.0f}s)", flush=True)
        print(f"    " + "   ".join(line), flush=True)

print("\n--- F-only net ---", flush=True)
t = time.time()
mF = trF
pnnF = np.mean([nnlib.train_nn(Atr[mF], ytr[mF], [Ava], sd, hidden=(128, 64),
                               epochs=6)[1][0] for sd in SEEDS], axis=0)
preds[("all seasons", "net128-64 e6")] = pnnF
print(f"  F-only net  alone(F) {oracle(pnnF, isF):6.1f} ({time.time() - t:.0f}s)",
      flush=True)
print("    " + "   ".join(
    f"w={w}:{oracle(mixF(pnnF, w), allmask):7.1f}/F{oracle(mixF(pnnF, w), isF):6.1f}"
    for w in (0.2, 0.35, 0.5)), flush=True)

print("\n--- averaging specialists ---", flush=True)
COMBOS = [
    ("all+2023+ (leaf31)", [("all seasons", "leaf31 msl1000 l2=1"),
                            ("2023+ only (post-break)", "leaf31 msl1000 l2=1")]),
    ("leaf31+leaf15 (all)", [("all seasons", "leaf31 msl1000 l2=1"),
                             ("all seasons", "leaf15 msl2000")]),
    ("trees+net (all)", [("all seasons", "leaf31 msl1000 l2=1"),
                         ("all seasons", "net128-64 e6")]),
    ("everything", list(preds)),
]
for tag, keys in COMBOS:
    p = np.mean([preds[k] for k in keys], axis=0)
    line = [f"w={w}:{oracle(mixF(p, w), allmask):7.1f}/F{oracle(mixF(p, w), isF):6.1f}"
            for w in (0.2, 0.35, 0.5, 0.65)]
    print(f"  {tag:24s} alone(F) {oracle(p, isF):6.1f}", flush=True)
    print("    " + "   ".join(line), flush=True)

print("\ndone", flush=True)
