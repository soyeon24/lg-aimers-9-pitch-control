"""Carry the league level on the row instead of forecasting it.

The season-rate forecast is the most expensive constant in the pipeline: its
backtest RMSE is about 0.018 and the score charges 401,000 * error^2, i.e. an
expected tax near 130 points -- larger than every resolution gain from v4 to v7
combined. 2024 happened to come out at 0.0018, which is luck, not a property.

But the level is already inside every evaluation row. The row-average of the
reconstructed season-to-date rate sits a near-constant -0.004 under the true
league rate in all six training seasons, so a bias constant learned from train
turns it into a level estimate with residual sigma near 0.0035 -- five times
tighter than the forecast.

Experiment 13 tried this as a post-hoc logit shift, z += gamma*(a_i - a_prev),
which forces one coefficient onto the cross-sectional spread as well, where the
anchor is mostly noise. This is the other formulation: put the anchor in as an
*offset* with its coefficient pinned to 1 and let the tree fit only what is left
over. The level then rides in per row and no forecast is needed at all. Every
row is still transformed on its own, so the row-independence rule holds.

Arms, all on the same feature build and the same seed:
  clf_none   production classifier with the forecast switched off (the tax)
  clf_fc     production shape: classifier + forecast offset at 0.85
  reg_plain  squared-error regressor on y, no offset (isolates the loss change)
  anchor     the anchor alone, as a prediction (level only, no resolution)
  reg_off    squared-error regressor on y - anchor, prediction = anchor + g
  reg_off_fc reg_off with the forecast offset still applied on top
"""
import os
import sys
import time

import numpy as np
from sklearn.ensemble import (HistGradientBoostingClassifier,
                              HistGradientBoostingRegressor)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import fe  # noqa: E402
from common import CAT_COLS, PARAMS, TARGET, comp_score, load, logit, sigmoid  # noqa: E402
from forecast import forecast_next_rate  # noqa: E402

EPS = 1e-6
SEASONS = tuple(int(s) for s in (sys.argv[1].split(",") if len(sys.argv) > 1
                                 else ["2024", "2022", "2021"]))
SEED = 0
K_MAIN = 100.0
K_SWEEP = (0.0, 30.0, 100.0, 200.0)


def prev_rate_table(tr):
    """(season, game_type) -> that group's league rate. A train-only constant."""
    g = tr.groupby(["season", "game_type"])[TARGET].mean()
    return {(int(s), t): float(v) for (s, t), v in g.items()}


def anchor(X, ctx, prt, k):
    """Season-to-date success rate, shrunk toward last season's group rate.

    num/den come from the row's own asof_* columns minus a per-(pitcher, season)
    constant learned from train -- the same reconstruction fe.py already does.
    """
    an = X["asof_pitcher_n"].to_numpy(dtype="float64")
    rate = X["asof_pitcher_success_rate"].to_numpy(dtype="float64")
    c = fe._lookup(ctx, "pit", X["pitcher_id"].to_numpy(), X["season"].to_numpy())
    num = np.nan_to_num(rate) * an - np.nan_to_num(c["c_success"])
    den = np.maximum(an - np.nan_to_num(c["c_n"]), 0.0)

    season = X["season"].to_numpy(dtype="int64")
    gt = X["game_type"].to_numpy()
    fallback = float(np.mean(list(prt.values())))
    prior = np.array([prt.get((int(s) - 1, t), prt.get((int(s), t), fallback))
                      for s, t in zip(season, gt)], dtype="float64")
    if k <= 0:
        return np.where(den > 0, num / np.maximum(den, 1.0), prior)
    return (num + k * prior) / (den + k)


def score_arm(name, p, y, true_rate, out):
    p = np.clip(p, EPS, 1 - EPS)
    s = comp_score(p, y)
    orc = comp_score(sigmoid(logit(p) + logit(true_rate) - logit(p.mean())), y)
    out.append((name, s, orc, float(p.mean())))
    print(f"  {name:11s} {s:8.1f}  oracle {orc:8.1f}  mean {p.mean():.4f} "
          f"(true {true_rate:.4f}, off by {p.mean() - true_rate:+.4f})", flush=True)


def run(val):
    df = load()
    tr = df[df.season < val]
    va = df[df.season == val]
    y = tr[TARGET].to_numpy(dtype="float64")
    yva = va[TARGET].to_numpy(dtype="float64")
    true_rate = float(yva.mean())

    t = time.time()
    ctx = fe.fit_context(tr)
    Xtr = fe.build_features(tr, ctx, extras=False)
    Xva = fe.build_features(va, ctx, extras=False)[list(Xtr.columns)]
    prt = prev_rate_table(tr)
    print(f"\n===== holdout {val} =====  train {len(tr)} rows, "
          f"{Xtr.shape[1]} features ({time.time() - t:.0f}s)", flush=True)

    # --- how well does the anchor track the level, before any model touches it
    seasons = sorted(int(s) for s in tr.season.unique())[1:]
    gt_tr = tr["game_type"].to_numpy()
    s_tr = tr["season"].to_numpy(dtype="int64")
    a_tr = anchor(Xtr, ctx, prt, K_MAIN)
    print(f"  anchor bias by season (mean anchor - true rate), K={K_MAIN:.0f}")
    bias = []
    for s in seasons:
        m = s_tr == s
        b = float(a_tr[m].mean() - y[m].mean())
        bias.append(b)
        print(f"    {s}  {a_tr[m].mean():.4f} vs {y[m].mean():.4f}   {b:+.4f}")
    b_hat = float(np.mean(bias))
    print(f"    learned bias constant  {b_hat:+.4f}  (spread {np.std(bias):.4f})")

    print(f"  K sweep on the holdout (bias-corrected anchor mean vs true "
          f"{true_rate:.4f}):")
    for k in K_SWEEP:
        atr_k = anchor(Xtr, ctx, prt, k)
        bk = float(np.mean([atr_k[s_tr == s].mean() - y[s_tr == s].mean()
                            for s in seasons]))
        av = anchor(Xva, ctx, prt, k) - bk
        print(f"    K={k:5.0f}  mean {av.mean():.4f}  "
              f"off by {av.mean() - true_rate:+.4f}")
    a_va = anchor(Xva, ctx, prt, K_MAIN)

    # --- what the production pipeline does instead
    prev = float(df[df.season == val - 1][TARGET].mean())
    fc = forecast_next_rate(tr)
    target = prev + 0.85 * (fc - prev)
    print(f"  forecast {fc:.4f} -> applied {target:.4f}  "
          f"(off by {target - true_rate:+.4f})", flush=True)

    ci = [Xtr.columns.get_loc(c) for c in CAT_COLS if c in Xtr.columns]
    prevm = s_tr == val - 1
    Xprev = Xtr[prevm]
    out = []

    t = time.time()
    clf = HistGradientBoostingClassifier(categorical_features=ci, random_state=SEED,
                                         **PARAMS).fit(Xtr, y)
    p_clf = clf.predict_proba(Xva)[:, 1]
    ref = float(clf.predict_proba(Xprev)[:, 1].mean())
    print(f"  [classifier {time.time() - t:.0f}s]", flush=True)
    score_arm("clf_none", p_clf, yva, true_rate, out)
    score_arm("clf_fc", sigmoid(logit(p_clf) + logit(target) - logit(ref)),
              yva, true_rate, out)

    t = time.time()
    reg = HistGradientBoostingRegressor(categorical_features=ci, random_state=SEED,
                                        loss="squared_error", **PARAMS).fit(Xtr, y)
    print(f"  [regressor {time.time() - t:.0f}s]", flush=True)
    score_arm("reg_plain", reg.predict(Xva), yva, true_rate, out)

    a_tr_c, a_va_c = a_tr - b_hat, a_va - b_hat
    score_arm("anchor", a_va_c, yva, true_rate, out)

    t = time.time()
    off = HistGradientBoostingRegressor(categorical_features=ci, random_state=SEED,
                                        loss="squared_error",
                                        **PARAMS).fit(Xtr, y - a_tr_c)
    p_off = a_va_c + off.predict(Xva)
    print(f"  [offset regressor {time.time() - t:.0f}s]", flush=True)
    score_arm("reg_off", p_off, yva, true_rate, out)
    refo = float((a_tr_c[prevm] + off.predict(Xprev)).mean())
    score_arm("reg_off_fc", sigmoid(logit(np.clip(p_off, EPS, 1 - EPS))
                                    + logit(target) - logit(refo)),
              yva, true_rate, out)
    del gt_tr
    return out


results = {v: run(v) for v in SEASONS}
print("\n===== summary (score / oracle) =====")
names = [n for n, _, _, _ in results[SEASONS[0]]]
print(f"{'arm':11s} " + "  ".join(f"{v:>17d}" for v in SEASONS))
for i, n in enumerate(names):
    print(f"{n:11s} " + "  ".join(f"{results[v][i][1]:8.1f}/{results[v][i][2]:8.1f}"
                                  for v in SEASONS))
