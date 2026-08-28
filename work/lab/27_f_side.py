"""Re-settle the F side on whatever feature set experiment 26 leaves standing.

v6 mixes an F-only model into F rows at 0.25. Two things about that number are
soft. First, it was tuned on a base that no longer exists if the new feature
families land. Second, the 2024 holdout systematically *understates* the F
specialist: it trains on one season of new-regime F (26k rows) while the real
2025 model trains on two (56k). The holdout optimum is therefore a lower bound
on the real one, which is the documented reason the leaderboard transfer has
run at 111% and 237%.

That second point cannot be tested directly -- no holdout has two new-regime
seasons. What can be tested is the direction: if telling the specialist to care
more about new-regime rows helps even at 26k, it will help more at 56k. So the
sample-weight sweep here is the measurable proxy for "more new-regime data",
and `newonly` is its limit case.
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import harness
import nnlib
from common import TARGET, comp_score, load, logit, sigmoid
from sklearn.ensemble import HistGradientBoostingClassifier

VAL = 2024
SEEDS = (0, 1)
EPS = 1e-6
FEATS = sys.argv[1] if len(sys.argv) > 1 else "regime+platoon+count"

df = load()
tr, va = df[df.season < VAL], df[df.season == VAL]
ytr, yva = tr[TARGET].to_numpy(), va[TARGET].to_numpy()
isF = va.game_type.to_numpy() == "F"
trF = tr.game_type.to_numpy() == "F"
trFnew = trF & (tr.season.to_numpy() >= 2023)


def pick(tag):
    import feats7
    import feats8
    import lab_v6
    return {"baseline v6": (lab_v6.base, lab_v6.ctx_fn),
            "regime": (feats7.make(), feats7.fit),
            "regime+platoon": (feats8.make(platoon=True, count=False), feats8.fit),
            "regime+platoon+count": (feats8.make(platoon=True, count=True), feats8.fit),
            }[tag]


build, ctx_fn = pick(FEATS)
t = time.time()
ctx = ctx_fn(tr)
Xtr = build(tr, ctx)
cols = list(Xtr.columns)
Xva = build(va, ctx)[cols]
cat_idx = [cols.index(c) for c in harness.CAT_COLS if c in cols]
print(f"[features {FEATS} {Xtr.shape} {time.time() - t:.0f}s]", flush=True)

prep = nnlib.fit_prep(Xtr)
pnn = np.mean([nnlib.train_nn(nnlib.prep(Xtr, prep), ytr, [nnlib.prep(Xva, prep)], sd,
                              hidden=(256, 128), epochs=3)[1][0] for sd in (0, 1)], axis=0)
p15 = harness.pmean(harness.fit_models(Xtr, ytr, SEEDS), Xva)
p31 = harness.pmean(harness.fit_models(
    Xtr, ytr, SEEDS, dict(max_leaf_nodes=31, l2_regularization=1.0,
                          min_samples_leaf=2000)), Xva)
base = 0.65 * (0.65 * p15 + 0.35 * p31) + 0.35 * pnn


def oracle(p, mask=None):
    m = slice(None) if mask is None else mask
    y = yva[m]
    q = np.clip(p[m], EPS, 1 - EPS)
    return comp_score(sigmoid(logit(q) + logit(y.mean()) - logit(q.mean())), y)


print(f"base  all {oracle(base):7.1f}  R {oracle(base, ~isF):7.1f} "
      f"F {oracle(base, isF):7.1f}\n", flush=True)

F_CFG = [("leaf31 msl1000", dict(max_leaf_nodes=31, l2_regularization=1.0,
                                 min_samples_leaf=1000)),
         ("leaf15 msl2000", dict(min_samples_leaf=2000))]
NEW_W = [("plain", 1.0), ("new x3", 3.0), ("new x8", 8.0), ("newonly", None)]

preds = {}
for ftag, fparams in F_CFG:
    for wtag, wnew in NEW_W:
        t = time.time()
        if wnew is None:
            m, sw = trFnew, None
        else:
            m = trF
            sw = np.where(trFnew[trF], wnew, 1.0)
        ps = []
        for sd in SEEDS:
            mo = HistGradientBoostingClassifier(
                categorical_features=cat_idx, random_state=sd,
                **dict(harness.PARAMS, **fparams))
            mo.fit(Xtr[m], ytr[m], sample_weight=sw)
            ps.append(mo.predict_proba(Xva)[:, 1])
        preds[(ftag, wtag)] = np.mean(ps, axis=0)
        print(f"  {ftag} / {wtag:8s} ({int(m.sum())} rows, {time.time() - t:.0f}s)",
              flush=True)


def withF(p, w):
    q = base.copy()
    q[isF] = (1 - w) * base[isF] + w * p[isF]
    return q


print()
for key, p in preds.items():
    line = "  ".join(f"w={w:.2f}: {oracle(withF(p, w)):7.1f}/{oracle(withF(p, w), isF):7.1f}"
                     for w in (0.20, 0.30, 0.40, 0.50))
    print(f"  {key[0]:14s} {key[1]:8s} alone(F) {oracle(preds[key], isF):7.1f} | {line}",
          flush=True)

print("\n--- averaged over the two capacities ---", flush=True)
for wtag, _ in NEW_W:
    p = 0.5 * (preds[(F_CFG[0][0], wtag)] + preds[(F_CFG[1][0], wtag)])
    line = "  ".join(f"w={w:.2f}: {oracle(withF(p, w)):7.1f}/{oracle(withF(p, w), isF):7.1f}"
                     for w in (0.20, 0.30, 0.40, 0.50))
    print(f"  ens {wtag:8s} alone(F) {oracle(p, isF):7.1f} | {line}", flush=True)

np.savez(os.path.join("cache", f"f_side_{VAL}.npz"), base=base, p15=p15, p31=p31,
         pnn=pnn, **{f"{a}|{b}": v for (a, b), v in preds.items()})
print("\ndone", flush=True)
