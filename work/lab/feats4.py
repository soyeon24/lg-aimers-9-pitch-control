"""v4 features: season-to-date reconstruction.

`asof_pitcher_n` is a career counter that runs continuously from 2019 into the
evaluation season (verified on the sample test.csv: a pitcher's asof_n there
picks up exactly where train.csv left off). So for any row we can recover what
the pitcher has done *this season so far*:

    season_n     = asof_n            - career_n_at_end_of_previous_season
    season_succ  = asof_rate*asof_n  - career_successes_at_end_of_prev_season

Both subtracted constants are per-pitcher numbers learned from train.csv, so
each evaluation row is still transformed on its own -- no test-row aggregation.

That matters because `asof_pitcher_success_rate` is career-cumulative and badly
stale (a 10-year veteran's number barely moves), while prev1/3/5-game rates are
far too noisy. Season-to-date sits in between and is the natural skill estimate.
"""
import numpy as np
import pandas as pd

from common import base_frame, add_v3

K_PIT = 200.0   # shrinkage strength for pitcher season-to-date rates
K_BAT = 300.0
K_MIX = 100.0


def _merge(X, tbl, key):
    return X.merge(tbl, on=[key, "season"], how="left")


def _rate(num, den, prior, k):
    """Shrunk rate; den may be 0 or negative from counter slack."""
    den = np.maximum(den, 0.0)
    return (num + k * prior) / (den + k)


def add_season_to_date(X, ctx, pitcher=True, batter=True, mix=True,
                       prev_season=True):
    if pitcher:
        X = _merge(X, ctx.pit, "pitcher_id")
        cn = X["c_n"].fillna(0.0).values
        an = X["asof_pitcher_n"].values.astype("float64")
        sn = np.maximum(an - cn, 0.0)
        X["career_prev_n"] = cn
        X["season_n"] = sn
        X["is_debut_season"] = (cn <= 0).astype("float64")
        for r in ["success", "reverse", "middle", "ball", "strike"]:
            cum = X[f"asof_pitcher_{r}_rate"].values * an
            base = X["c_" + r].fillna(0.0).values
            prior = ctx.rp[f"asof_pitcher_{r}_rate"]
            std = _rate(cum - base, sn, prior, K_PIT)
            car = np.where(cn > 0, base / np.maximum(cn, 1.0), np.nan)
            X[f"std_{r}"] = std
            X[f"career_{r}"] = car
            X[f"std_vs_career_{r}"] = std - np.where(np.isnan(car), prior, car)
        if prev_season:
            psn = X["s_n"].values
            for r in ["success", "middle", "ball", "strike"]:
                prior = ctx.rp[f"asof_pitcher_{r}_rate"]
                X[f"prevseason_{r}"] = _rate(X["s_" + r].values, psn, prior, K_PIT)
            X["prevseason_n"] = psn
        X = X.drop(columns=[c for c in X.columns
                            if c.startswith(("c_", "s_")) and not c.startswith("season")])

    if mix:
        X = _merge(X, ctx.mix, "pitcher_id")
        cn = X["c_n_mix"].fillna(0.0).values
        an = X["asof_pitcher_pitchmix_n"].values.astype("float64")
        sn = np.maximum(an - cn, 0.0)
        for r in ["fastball", "breaking", "offspeed"]:
            cum = X[f"asof_pitcher_{r}_rate"].values * an
            base = X["c_" + r + "_mix"].fillna(0.0).values
            prior = ctx.rp[f"asof_pitcher_{r}_rate"]
            X[f"std_{r}"] = _rate(cum - base, sn, prior, K_MIX)
        X = X.drop(columns=[c for c in X.columns if c.endswith("_mix")])

    if batter:
        X = _merge(X, ctx.bat, "batter_id")
        cn = X["c_n"].fillna(0.0).values
        an = X["asof_batter_n"].values.astype("float64")
        sn = np.maximum(an - cn, 0.0)
        X["bat_career_prev_n"] = cn
        X["bat_season_n"] = sn
        for r in ["success", "middle"]:
            cum = X[f"asof_batter_{r}_rate"].values * an
            base = X["c_" + r].fillna(0.0).values
            prior = ctx.rp[f"asof_batter_{r}_rate"]
            X[f"bat_std_{r}"] = _rate(cum - base, sn, prior, K_BAT)
            X[f"bat_career_{r}"] = np.where(cn > 0, base / np.maximum(cn, 1.0), np.nan)
        if prev_season:
            psn = X["s_n"].values
            for r in ["success", "middle"]:
                prior = ctx.rp[f"asof_batter_{r}_rate"]
                X[f"bat_prevseason_{r}"] = _rate(X["s_" + r].values, psn, prior, K_BAT)
        X = X.drop(columns=[c for c in X.columns
                            if c.startswith(("c_", "s_")) and not c.startswith("season")])
    return X


def build_v4(d, ctx):
    X = add_v3(base_frame(d), ctx.priors)
    return add_season_to_date(X, ctx)


def make_builder(**kw):
    def f(d, ctx):
        X = add_v3(base_frame(d), ctx.priors)
        return add_season_to_date(X, ctx, **kw)
    return f
