"""Holdout harness v2.

Why this replaces the old one
----------------------------
The 2023 holdout scores 0 -- even with oracle calibration -- because game_type F
collapsed from .709 to .473 between 2022 and 2023, uniformly across every team.
That is a one-off definition change, not a trend, and it makes any full-data
2023 score meaningless. Averaging it into a verdict (as work/evaluate.py did)
is what made earlier ideas look worthless.

So we score three ways per holdout season:
  all  -- what the leaderboard would see
  R    -- regular games only: no regime break in any season, the clean signal
  F    -- the broken subset, reported for diagnosis only

and we log the drift diagnostics needed to size the offset:
  beta = (mean prediction on season V) - (mean prediction on season V-1)
         divided by the true rate change, i.e. how much of the league drift the
         features carry on their own. The fixed offset should supply only the
         remaining (1 - beta) share, otherwise it double counts.
"""
import time

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sklearn.ensemble import HistGradientBoostingClassifier

from common import (CAT_COLS, PARAMS, TARGET, Ctx, comp_score, extrapolate,
                    load, logit, sigmoid)


def _group_forecast(tr, g):
    """Next-season rate for one game_type, using the same consensus rule."""
    import forecast as _f
    y = tr[tr.game_type == g].groupby("season")[TARGET].mean().to_numpy(dtype=float)
    if g == "F":
        post = tr[(tr.game_type == "F") & (tr.season >= _f_break())]
        yp = post.groupby("season")[TARGET].mean().to_numpy(dtype=float)
        if len(yp) >= 2:
            return float(np.mean([_f._consensus(yp), yp[-1] + _r_delta(tr)]))
        if len(yp) == 1:
            return float(yp[-1] + _r_delta(tr))
    return _f._consensus(y)


def _f_break():
    return 2023


def _r_delta(tr):
    import forecast as _f
    y = tr[tr.game_type == "R"].groupby("season")[TARGET].mean().to_numpy(dtype=float)
    return _f._consensus(y) - y[-1]


def target_rate_of(ctx):
    if isinstance(ctx, dict):
        return extrapolate(ctx["season_rates"])
    return ctx.target_rate


EXTRA_CAT = ["park_id"]


def fit_models(Xtr, ytr, seeds, params=None, cat_cols=None):
    p = dict(PARAMS)
    if params:
        p.update(params)
    cat_cols = CAT_COLS + EXTRA_CAT if cat_cols is None else cat_cols
    ci = [Xtr.columns.get_loc(c) for c in cat_cols if c in Xtr.columns]
    out = []
    for sd in seeds:
        m = HistGradientBoostingClassifier(categorical_features=ci, random_state=sd, **p)
        m.fit(Xtr, ytr)
        out.append(m)
    return out


def pmean(models, X):
    return np.mean([m.predict_proba(X)[:, 1] for m in models], axis=0)


def run(build_fn, val_season, seeds=(0,), params=None, offset_scale=1.0,
        train_filter=None, tag="", verbose=True, ctx_fn=Ctx):
    df = load()
    tr_full = df[df.season < val_season]
    va = df[df.season == val_season]
    ctx = ctx_fn(tr_full)
    tr = tr_full[train_filter(tr_full)] if train_filter is not None else tr_full
    ytr, yva = tr[TARGET].values, va[TARGET].values

    # reference rows always come from the *unfiltered* previous season, so the
    # offset anchor keeps matching the league rate it is compared against
    pv_rows = tr_full[tr_full.season == val_season - 1]

    t = time.time()
    Xtr = build_fn(tr, ctx)
    cols = list(Xtr.columns)
    Xva = build_fn(va, ctx)[cols]
    Xprev = build_fn(pv_rows, ctx)[cols]
    tfeat = time.time() - t

    t = time.time()
    models = fit_models(Xtr, ytr, seeds, params)
    tfit = time.time() - t

    pprev = pmean(models, Xprev)
    ref = float(pprev.mean())
    pv = pmean(models, Xva)

    prev_rate = float(df[df.season == val_season - 1][TARGET].mean())
    true_rate = float(yva.mean())
    true_drift = true_rate - prev_rate
    model_drift = float(pv.mean()) - ref
    beta = model_drift / true_drift if abs(true_drift) > 1e-4 else np.nan

    # offset in logit space, optionally shrunk because the model already moved
    d_hat = target_rate_of(ctx) - prev_rate
    target = prev_rate + offset_scale * d_hat
    adj = sigmoid(logit(pv) + logit(target) - logit(ref))
    orc = sigmoid(logit(pv) + logit(true_rate) - logit(pv.mean()))

    # same fit, different offset shrinkage -- free, no refit
    scales = {}
    for sc in (1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.0):
        tg = prev_rate + sc * d_hat
        scales[sc] = comp_score(sigmoid(logit(pv) + logit(tg) - logit(ref)), yva)

    rmask = va.game_type.values == "R"
    prevR = float(df[(df.season == val_season - 1) & (df.game_type == "R")][TARGET].mean())
    trueR = float(yva[rmask].mean())
    refR = float(pprev[pv_rows.game_type.values == "R"].mean())
    betaR = ((float(pv[rmask].mean()) - refR) / (trueR - prevR)
             if abs(trueR - prevR) > 1e-4 else np.nan)

    # per-group calibration: F and R have their own forecasts, so give each its
    # own constant instead of forcing one shift on both
    gt = va.game_type.values
    gtr = pv_rows.game_type.values
    zg = logit(pv).copy()
    for g in ("R", "F"):
        pr = float(df[(df.season == val_season - 1) & (df.game_type == g)][TARGET].mean())
        fc = _group_forecast(tr_full, g)
        tg = pr + offset_scale * (fc - pr)
        rf = float(pprev[gtr == g].mean())
        zg[gt == g] += logit(tg) - logit(rf)
    adj_group = sigmoid(zg)

    res = dict(season=val_season, nfeat=Xtr.shape[1], ntrain=len(tr),
               adj=comp_score(adj, yva), oracle=comp_score(orc, yva),
               adjR=comp_score(adj[gt == "R"], yva[gt == "R"]),
               oracleR=comp_score(orc[gt == "R"], yva[gt == "R"]),
               adjF=comp_score(adj[gt == "F"], yva[gt == "F"]),
               oracleF=comp_score(orc[gt == "F"], yva[gt == "F"]),
               true=true_rate, target=target, adj_mean=float(adj.mean()),
               ref=ref, raw_mean=float(pv.mean()),
               true_drift=true_drift, model_drift=model_drift, beta=beta,
               betaR=betaR, true_driftR=trueR - prevR,
               model_driftR=float(pv[rmask].mean()) - refR,
               scales=scales, t=tfeat + tfit,
               adj_group=comp_score(adj_group, yva),
               adj_groupR=comp_score(adj_group[gt == "R"], yva[gt == "R"]),
               adj_groupF=comp_score(adj_group[gt == "F"], yva[gt == "F"]))
    if verbose:
        print(f"  V{val_season} f={res['nfeat']:3d} | all {res['adj']:7.1f}/"
              f"{res['oracle']:7.1f}  R {res['adjR']:7.1f}/{res['oracleR']:7.1f}"
              f"  F {res['adjF']:8.1f}/{res['oracleF']:8.1f} | "
              f"beta={beta:+.2f} drift {true_drift:+.4f} vs model {model_drift:+.4f} "
              f"| adjmean={res['adj_mean']:.4f} true={true_rate:.4f} ({res['t']:.0f}s)",
              flush=True)
        print(f"        group-offset all {res['adj_group']:7.1f} "
              f"R {res['adj_groupR']:7.1f} F {res['adj_groupF']:7.1f}", flush=True)
        print("        offset_scale " + "  ".join(
            f"{k:.1f}:{v:7.1f}" for k, v in res["scales"].items())
            + f"   betaR={betaR:+.2f} (driftR {trueR - prevR:+.4f} vs "
              f"{float(pv[rmask].mean()) - refR:+.4f})", flush=True)
    return res


def bench(build_fn, tag, seasons=(2021, 2022, 2023, 2024), seeds=(0,), **kw):
    print(f"--- {tag} ---", flush=True)
    rs = [run(build_fn, v, seeds=seeds, **kw) for v in seasons]
    m = lambda k: np.mean([r[k] for r in rs])
    print(f"  MEAN group-offset all {m('adj_group'):7.1f} R {m('adj_groupR'):7.1f} "
          f"F {m('adj_groupF'):7.1f}", flush=True)
    print(f"  MEAN  all {m('adj'):7.1f}/{m('oracle'):7.1f}   "
          f"R {m('adjR'):7.1f}/{m('oracleR'):7.1f}   "
          f"beta={np.nanmean([r['beta'] for r in rs]):+.2f} "
          f"betaR={np.nanmean([r['betaR'] for r in rs]):+.2f}", flush=True)
    sc = {k: np.mean([r["scales"][k] for r in rs]) for k in rs[0]["scales"]}
    print("  MEAN offset_scale " + "  ".join(f"{k:.1f}:{v:7.1f}" for k, v in sc.items()),
          flush=True)
    # pooled slope of model drift on true drift, forced through the origin
    x = np.array([r["true_driftR"] for r in rs]); y = np.array([r["model_driftR"] for r in rs])
    print(f"  pooled betaR (R-only, through origin) = {(x @ y) / (x @ x):+.3f}", flush=True)
    return rs
