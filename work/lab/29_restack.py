"""Re-settle the whole stack on the two-view tree base.

NN weight 0.35 and F-specialist weight 0.25 were both fitted against a tree
blend that scored 869.8. The two-view blend scores 884.0, so both numbers are
now too large: a weaker member deserves less room once the thing it is being
mixed into got better. Carrying them over unchanged would give back part of the
gain, which is the standard way a good feature result turns into a bad
submission.

Tree predictions come from experiment 28's cache, so the only new fits here are
the nets and the F specialists -- and both are measured on each view, because
whichever view suits the trees need not suit them.
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import feats8
import harness
import lab_v6
import nnlib
from common import TARGET, comp_score, load, logit, sigmoid
from sklearn.ensemble import HistGradientBoostingClassifier

VAL = int(sys.argv[1]) if len(sys.argv) > 1 else 2024
SEEDS = (0, 1)
NSEED = int(sys.argv[2]) if len(sys.argv) > 2 else 4
EPS = 1e-6

df = load()
tr, va = df[df.season < VAL], df[df.season == VAL]
ytr, yva = tr[TARGET].to_numpy(), va[TARGET].to_numpy()
isF = va.game_type.to_numpy() == "F"
trF = tr.game_type.to_numpy() == "F"


def oracle(p, m=None):
    m = slice(None) if m is None else m
    y = yva[m]
    q = np.clip(p[m], EPS, 1 - EPS)
    return comp_score(sigmoid(logit(q) + logit(y.mean()) - logit(q.mean())), y)


VIEWS = {"A": (lab_v6.base, lab_v6.ctx_fn, "A_base"),
         "B": (feats8.make(platoon=True, count=True), feats8.fit, "B_full")}

tree, nn, fspec = {}, {}, {}
for v, (build, ctx_fn, cache_tag) in VIEWS.items():
    t = time.time()
    ctx = ctx_fn(tr)
    Xtr = build(tr, ctx)
    cols = list(Xtr.columns)
    Xva = build(va, ctx)[cols]
    cache = os.path.join("cache", f"view_{cache_tag}_{VAL}_s{NSEED}.npz")
    if os.path.exists(cache):
        z = np.load(cache)
    else:
        z = {"p15": harness.pmean(harness.fit_models(Xtr, ytr, SEEDS), Xva),
             "p31": harness.pmean(harness.fit_models(
                 Xtr, ytr, SEEDS, dict(max_leaf_nodes=31, l2_regularization=1.0,
                                       min_samples_leaf=2000)), Xva)}
        np.savez(cache, **z)
    tree[v] = 0.65 * z["p15"] + 0.35 * z["p31"]
    prep = nnlib.fit_prep(Xtr)
    Atr, Ava = nnlib.prep(Xtr, prep), nnlib.prep(Xva, prep)
    nn[v] = np.mean([nnlib.train_nn(Atr, ytr, [Ava], sd, hidden=(256, 128),
                                    epochs=3)[1][0] for sd in SEEDS], axis=0)
    cat_idx = [cols.index(c) for c in harness.CAT_COLS if c in cols]
    for ftag, fp in [("leaf31", dict(max_leaf_nodes=31, l2_regularization=1.0,
                                     min_samples_leaf=1000)),
                     ("leaf15", dict(min_samples_leaf=2000))]:
        ps = []
        for sd in SEEDS:
            m = HistGradientBoostingClassifier(categorical_features=cat_idx,
                                               random_state=sd,
                                               **dict(harness.PARAMS, **fp))
            m.fit(Xtr[trF], ytr[trF])
            ps.append(m.predict_proba(Xva)[:, 1])
        fspec[(v, ftag)] = np.mean(ps, axis=0)
    print(f"[view {v} f={len(cols)} {time.time() - t:.0f}s] tree {oracle(tree[v]):7.1f} "
          f"nn {oracle(nn[v]):7.1f} "
          + "  ".join(f"fspec-{k[1]}(F) {oracle(p, isF):7.1f}"
                      for k, p in fspec.items() if k[0] == v), flush=True)

print("\n--- tree mix A/B ---", flush=True)
for wb in (0.4, 0.5, 0.6):
    T = (1 - wb) * tree["A"] + wb * tree["B"]
    print(f"  wB={wb:.1f}  all {oracle(T):7.1f} R {oracle(T, ~isF):7.1f} "
          f"F {oracle(T, isF):7.1f}", flush=True)

T = 0.5 * (tree["A"] + tree["B"])

print("\n--- net weight and net view ---", flush=True)
NETS = {"A": nn["A"], "B": nn["B"], "avg": 0.5 * (nn["A"] + nn["B"])}
best = (None, -1)
for tag, p in NETS.items():
    line = []
    for w in (0.0, 0.10, 0.15, 0.20, 0.25, 0.35):
        sc = oracle((1 - w) * T + w * p)
        line.append(f"w={w:.2f}:{sc:7.1f}")
        if sc > best[1]:
            best = ((tag, w), sc)
    print(f"  net {tag:3s} " + "  ".join(line), flush=True)
(ntag, nw), nsc = best
print(f"  -> best net {ntag} at w={nw:.2f} ({nsc:.1f})", flush=True)
S = (1 - nw) * T + nw * NETS[ntag]

print("\n--- F specialist on top ---", flush=True)
CANDS = {f"{v}-{k}": p for (v, k), p in fspec.items()}
CANDS["B-ens"] = 0.5 * (fspec[("B", "leaf31")] + fspec[("B", "leaf15")])
CANDS["A-ens"] = 0.5 * (fspec[("A", "leaf31")] + fspec[("A", "leaf15")])
for tag, p in CANDS.items():
    line = []
    for w in (0.0, 0.10, 0.15, 0.20, 0.25, 0.30):
        q = S.copy()
        q[isF] = (1 - w) * S[isF] + w * p[isF]
        line.append(f"w={w:.2f}:{oracle(q):7.1f}/{oracle(q, isF):6.1f}")
    print(f"  {tag:10s} " + "  ".join(line), flush=True)

np.savez(os.path.join("cache", f"restack_{VAL}.npz"),
         **{f"tree_{k}": v for k, v in tree.items()},
         **{f"nn_{k}": v for k, v in nn.items()},
         **{f"f_{a}_{b}": v for (a, b), v in fspec.items()})
print("\ndone", flush=True)
