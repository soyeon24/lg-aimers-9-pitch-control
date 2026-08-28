"""Feature-set diversity: the ensemble axis that was never tried.

Experiment 25 closed the algorithm axis -- bagging, ExtraTrees and residual
stacking all scored below the base, on top of the nets and tree capacities
already rejected in v5/v6. Every one of those varied the *model* while holding
the feature table fixed. Experiment 26 built a second feature table, and the
average of the two scores 883.8 against 869.8 for either view's own blend.

That is a diversity result, not an accuracy one: the regime view *alone* is
worse than baseline (868.2 vs 869.8) yet lifts the average by +5. Two views
disagree in different places, which is exactly the property the nets were added
for and the tree settings failed to provide.

This settles three things before any of it goes near the submission:
  seeds     the 2024 gap has to survive four seeds, not two (noise is +-4)
  season    and it has to show up on a second holdout, or it is one season's luck
  content   is the gain the whole feats8 view, or just the count split? The
            cheapest adoption that captures it is the one to take.
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

EPS = 1e-6
LOOSE = dict(max_leaf_nodes=31, l2_regularization=1.0, min_samples_leaf=2000)
VAL = int(sys.argv[1]) if len(sys.argv) > 1 else 2024
SEEDS = tuple(range(int(sys.argv[2]) if len(sys.argv) > 2 else 4))
ONLY = sys.argv[3].split(",") if len(sys.argv) > 3 else None
CACHE = "cache"

import fe  # noqa: E402

# A_base/B_full are the lab builders the views were prototyped in; Aprod/Bprod
# are production fe.py's two views of the same thing. They are kept side by side
# on purpose -- a silent mismatch between the lab and the shipped preprocessing
# is the most expensive mistake available in this contest, so the merge gets
# checked by rerunning the comparison through the shipped code.
VIEWS = {
    "A_base": (lab_v6.base, lab_v6.ctx_fn),
    "B_full": (feats8.make(platoon=True, count=True), feats8.fit),
    "C_count": (feats8.make(platoon=False, count=True, regime=False), feats8.fit),
    "D_regime": (feats7.make(), feats7.fit),
    "Aprod": (lambda d, ctx: fe.build_features(d, ctx, extras=False), fe.fit_context),
    "Bprod": (lambda d, ctx: fe.build_features(d, ctx, extras=True), fe.fit_context),
    # 2021 loses 30 points on `all` while staying flat on R, i.e. the damage is
    # confined to old-regime F rows -- a row type 2025 does not contain. The
    # share columns are the suspects: they overlap with what `f_old_regime`
    # already says about the group gap, and 2021 is the season where that gap is
    # enormous (.70 vs .51). Dropping them is the cheap test of that reading.
    "Bnoshare": (lambda d, ctx: fe.build_features(d, ctx, extras=True).drop(
        columns=[c for c in fe.EXTRA_COLS if c.endswith("_share")]), fe.fit_context),
}

if ONLY:
    VIEWS = {k: v for k, v in VIEWS.items() if k in ONLY}

df = load()
tr, va = df[df.season < VAL], df[df.season == VAL]
ytr, yva = tr[TARGET].to_numpy(), va[TARGET].to_numpy()
isF = va.game_type.to_numpy() == "F"


def oracle(p, m=None):
    m = slice(None) if m is None else m
    y = yva[m]
    q = np.clip(p[m], EPS, 1 - EPS)
    return comp_score(sigmoid(logit(q) + logit(y.mean()) - logit(q.mean())), y)


def view(tag):
    path = os.path.join(CACHE, f"view_{tag}_{VAL}_s{len(SEEDS)}.npz")
    if os.path.exists(path):
        z = np.load(path)
        return z["p15"], z["p31"]
    build, ctx_fn = VIEWS[tag]
    t = time.time()
    ctx = ctx_fn(tr)
    Xtr = build(tr, ctx)
    cols = list(Xtr.columns)
    Xva = build(va, ctx)[cols]
    p15 = harness.pmean(harness.fit_models(Xtr, ytr, SEEDS), Xva)
    p31 = harness.pmean(harness.fit_models(Xtr, ytr, SEEDS, LOOSE), Xva)
    np.savez(path, p15=p15, p31=p31)
    print(f"  [{tag} f={len(cols)} {time.time() - t:.0f}s]", flush=True)
    return p15, p31


print(f"===== holdout {VAL}, {len(SEEDS)} seeds =====", flush=True)
P = {}
for tag in VIEWS:
    p15, p31 = view(tag)
    P[tag] = 0.65 * p15 + 0.35 * p31
    print(f"  {tag:10s} alone   all {oracle(P[tag]):7.1f} R {oracle(P[tag], ~isF):7.1f} "
          f"F {oracle(P[tag], isF):7.1f}   (tight {oracle(p15):7.1f} "
          f"loose {oracle(p31):7.1f})", flush=True)

print("\n--- A averaged with each other view ---", flush=True)
for tag in [t for t in ("B_full", "C_count", "D_regime", "Bprod", "Bnoshare") if t in P]:
    for w in (0.3, 0.4, 0.5, 0.6):
        q = (1 - w) * P["Aprod" if tag.startswith("B") and tag != "B_full"
                          else "A_base"] + w * P[tag]
        print(f"  A + {w:.1f}*{tag:9s} all {oracle(q):7.1f} R {oracle(q, ~isF):7.1f} "
              f"F {oracle(q, isF):7.1f}", flush=True)

print("\n--- three views ---", flush=True)
for tags in [t for t in [("A_base", "B_full", "D_regime"),
                         ("A_base", "C_count", "D_regime"),
                         ("A_base", "B_full", "C_count")]
             if all(k in P for k in t)]:
    q = np.mean([P[t] for t in tags], axis=0)
    print(f"  {'+'.join(t[0] for t in tags):8s} all {oracle(q):7.1f} "
          f"R {oracle(q, ~isF):7.1f} F {oracle(q, isF):7.1f}", flush=True)

print("\ndone", flush=True)
