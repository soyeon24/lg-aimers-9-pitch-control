"""Can the league level ride in on a per-row feature instead of a forecast?

The season-rate forecast is the single most expensive constant in the pipeline:
backtests put its RMSE around 0.018, and the score charges 401,000 * error^2,
so roughly 130 points of unavoidable tax.

But the level is not really unobservable. The row-average of the reconstructed
season-to-date rate moves with the league rate at correlation 0.951 across
2020-2024 (career asof manages only 0.48). So the thing we are trying to
forecast is already sitting inside every evaluation row.

The catch is that we may only use it *per row*. Taking the mean of
std_success across evaluation rows would be a distribution statistic of the test
set, which the rules forbid outright. A per-row term

    z_i += gamma * (anchor_i - anchor_mean_of_last_training_season)

is a different thing: each row is transformed on its own, exactly like any other
feature, and the level tracking is an emergent property of the average. That is
allowed -- but it forces the same coefficient onto the cross-sectional spread,
which is mostly noise. Whether the trade pays is what this measures.

The `AGGREGATE (diagnostic only)` rows below DO use a test-set mean. They are
printed to size the ceiling and are not submittable.
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import harness
import lab_v6
from common import TARGET, comp_score, load, logit, sigmoid
from forecast import forecast_next_rate

SEASONS = (2021, 2022, 2023, 2024)
SEEDS = (0, 1)
GAMMAS = (0.0, 1.0, 2.0, 3.0, 5.0, 8.0)
SCALES = (1.0, 0.5, 0.0)
ANCHORS = ("std_success", "bat_std_success", "mix")
EPS = 1e-6

df = load()
rows = {}

for V in SEASONS:
    t = time.time()
    tr, va = df[df.season < V], df[df.season == V]
    ytr, yva = tr[TARGET].to_numpy(), va[TARGET].to_numpy()
    ctx = lab_v6.ctx_fn(tr)
    Xtr = lab_v6.base(tr, ctx)
    cols = list(Xtr.columns)
    prev = tr[tr.season == V - 1]
    Xva = lab_v6.base(va, ctx)[cols]
    Xprev = lab_v6.base(prev, ctx)[cols]
    ms = harness.fit_models(Xtr, ytr, SEEDS)
    ms += harness.fit_models(Xtr, ytr, SEEDS,
                             dict(max_leaf_nodes=31, l2_regularization=1.0,
                                  min_samples_leaf=2000))
    n = len(SEEDS)
    pv = 0.6 * harness.pmean(ms[:n], Xva) + 0.4 * harness.pmean(ms[n:], Xva)
    pp = 0.6 * harness.pmean(ms[:n], Xprev) + 0.4 * harness.pmean(ms[n:], Xprev)

    def anchor(X, name):
        if name == "mix":
            return 0.5 * (X["std_success"].to_numpy() + X["bat_std_success"].to_numpy())
        return X[name].to_numpy()

    rows[V] = dict(
        pv=pv, ref=float(pp.mean()), yva=yva,
        prev_rate=float(prev[TARGET].mean()),
        forecast=forecast_next_rate(tr),
        anch_va={a: np.nan_to_num(anchor(Xva, a),
                                  nan=ctx["rp"]["asof_pitcher_success_rate"])
                 for a in ANCHORS},
        anch_prev={a: float(np.nanmean(anchor(Xprev, a))) for a in ANCHORS},
    )
    print(f"[V{V} fitted {time.time() - t:.0f}s] true={yva.mean():.4f} "
          f"prev={rows[V]['prev_rate']:.4f} forecast={rows[V]['forecast']:.4f}",
          flush=True)
    for a in ANCHORS:
        d = rows[V]["anch_va"][a].mean() - rows[V]["anch_prev"][a]
        print(f"    anchor {a:16s} drift {d:+.4f} vs true "
              f"{yva.mean() - rows[V]['prev_rate']:+.4f}  "
              f"sd within season {rows[V]['anch_va'][a].std():.4f}", flush=True)


def score(V, gamma, s, a, aggregate=False):
    r = rows[V]
    z = logit(r["pv"])
    tg = r["prev_rate"] + s * (r["forecast"] - r["prev_rate"])
    z = z + logit(tg) - logit(r["ref"])
    if gamma:
        av = r["anch_va"][a]
        base = r["anch_prev"][a]
        z = z + gamma * ((av.mean() if aggregate else av) - base)
    return comp_score(sigmoid(z), r["yva"])


print("\n===== current scheme (gamma=0, s=1) =====", flush=True)
cur = {V: score(V, 0.0, 1.0, "std_success") for V in SEASONS}
print("  " + "  ".join(f"V{V}={cur[V]:7.1f}" for V in SEASONS)
      + f"   mean={np.mean(list(cur.values())):7.1f}  min={min(cur.values()):7.1f}")

for a in ANCHORS:
    print(f"\n===== anchor = {a} =====", flush=True)
    for s in SCALES:
        for g in GAMMAS:
            sc = [score(V, g, s, a) for V in SEASONS]
            print(f"  s={s:.1f} gamma={g:4.1f}  "
                  + "  ".join(f"{x:7.1f}" for x in sc)
                  + f"   mean={np.mean(sc):7.1f}  min={min(sc):7.1f}", flush=True)

print("\n===== AGGREGATE (diagnostic only, uses a test-set mean -- NOT submittable) =====")
for a in ANCHORS:
    for g in (1.0, 2.0, 3.0, 5.0, 8.0):
        sc = [score(V, g, 0.0, a, aggregate=True) for V in SEASONS]
        print(f"  {a:16s} gamma={g:4.1f}  " + "  ".join(f"{x:7.1f}" for x in sc)
              + f"   mean={np.mean(sc):7.1f}  min={min(sc):7.1f}", flush=True)

print("\n===== oracle mean (ceiling) =====")
orc = []
for V in SEASONS:
    r = rows[V]
    p = sigmoid(logit(r["pv"]) + logit(r["yva"].mean()) - logit(r["pv"].mean()))
    orc.append(comp_score(p, r["yva"]))
print("  " + "  ".join(f"V{V}={x:7.1f}" for V, x in zip(SEASONS, orc))
      + f"   mean={np.mean(orc):7.1f}")
print("\ndone", flush=True)
