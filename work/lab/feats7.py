"""Career history split by measurement regime.

Why
---
`asof_pitcher_success_rate` pools every pitch a pitcher has ever thrown, and
those pitches were not all measured the same way. game_type F changed
definition in 2023 (.709 -> .473, uniformly across teams), so a pitcher whose
career is half old-regime F carries a career rate inflated by ~.20 for reasons
that have nothing to do with his control.

That is not a rare corner. In every season roughly half of the pitchers who
appear in R games also appear in F games, and the per-pitcher F share of career
pitches has median .19 with quartiles .01 and .89 -- it is spread across the
whole range, not concentrated at the ends. So the contamination differs wildly
from pitcher to pitcher and the tree cannot back it out of a single pooled rate.

`f_old_regime` already tells the tree which *rows* were measured under the old
rule. What it cannot say is how much of the *history feature* was.

What this adds
--------------
Career totals through the end of the previous season, split three ways --
R, new-regime F (>=2023), old-regime F (<2023) -- as per-(pitcher, season)
constants learned from train.csv, exactly like the existing carry tables. Every
evaluation row is still transformed on its own.

From them:
  career_clean        career rate with old-regime F pitches removed
  career_own/_cross   the rate measured under this row's own regime, and the
                      other one -- an F row's best prior is F history
  career_f_share      how much of the history is F at all
  career_fold_share   how much of it is the broken regime

Row counts are exact: asof_pitcher_n is a contiguous career counter, verified
to equal the row count for every (pitcher, season) pair in train.csv, so
counting rows per regime recovers the split totals with no slack.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import fe  # noqa: E402

TARGET = "control_success"
SEASON_MUL = fe.SEASON_MUL
F_BREAK = fe.F_BREAK_SEASON
REGIMES = ("R", "FN", "FO")
K = 200.0
K_BAT = 300.0


def _regime_code(gt, season):
    """0 = regular, 1 = F under the new rule, 2 = F under the old one."""
    return np.where(gt == "R", 0, np.where(season >= F_BREAK, 1, 2))


def _split_carry(tr, key):
    """(entity, season) -> career n/successes per regime at the end of season-1."""
    d = pd.DataFrame({"e": tr[key].to_numpy(dtype="int64"),
                      "s": tr["season"].to_numpy(dtype="int64"),
                      "r": _regime_code(tr["game_type"].to_numpy(),
                                        tr["season"].to_numpy()),
                      "y": tr[TARGET].to_numpy(dtype="float64")})
    g = d.groupby(["e", "s", "r"]).agg(n=("y", "size"), k=("y", "sum")).reset_index()
    ents = np.sort(d.e.unique())
    seasons = list(range(int(d.s.min()), int(d.s.max()) + 3))
    mi = pd.MultiIndex.from_product([ents, seasons], names=["e", "s"])

    out, names = [], []
    for ri, tag in enumerate(REGIMES):
        sub = (g[g.r == ri].set_index(["e", "s"])[["n", "k"]]
               .reindex(mi).fillna(0.0))
        # cumulative career, then shift so a row never sees its own season
        prev = sub.groupby(level=0).cumsum().groupby(level=0).shift(1)
        out += [prev["n"].to_numpy(), prev["k"].to_numpy()]
        names += [tag + "_n", tag + "_k"]

    keys = (mi.get_level_values(0).to_numpy(dtype="int64") * SEASON_MUL
            + mi.get_level_values(1).to_numpy(dtype="int64"))
    order = np.argsort(keys)
    return keys[order], np.column_stack(out)[order], names


def fit(tr):
    ctx = fe.fit_context(tr)
    for tag, key in (("rpit", "pitcher_id"), ("rbat", "batter_id")):
        k, v, names = _split_carry(tr, key)
        ctx[tag + "_keys"], ctx[tag + "_vals"], ctx[tag + "_cols"] = k, v, names
    ctx["prior_regime"] = {
        t: float(tr.loc[_regime_code(tr.game_type.to_numpy(),
                                     tr.season.to_numpy()) == i, TARGET].mean())
        for i, t in enumerate(REGIMES)}
    return ctx


def _shrunk(k, n, prior, strength):
    return (k + strength * prior) / (n + strength)


def _side(X, ctx, tag, idcol, prefix, strength, is_F):
    c = fe._lookup(ctx, tag, X[idcol].to_numpy(), X["season"].to_numpy())
    v = {n: np.nan_to_num(c[n]) for n in ctx[tag + "_cols"]}
    pr = ctx["prior_regime"]

    nR, nFN, nFO = v["R_n"], v["FN_n"], v["FO_n"]
    tot = nR + nFN + nFO
    clean_n, clean_k = nR + nFN, v["R_k"] + v["FN_k"]

    rateR = _shrunk(v["R_k"], nR, pr["R"], strength)
    rateFN = _shrunk(v["FN_k"], nFN, pr["FN"], strength)
    X[prefix + "career_clean"] = _shrunk(clean_k, clean_n, pr["R"], strength)
    X[prefix + "career_R"] = rateR
    X[prefix + "career_FN"] = rateFN
    X[prefix + "career_R_minus_FN"] = rateR - rateFN
    # the prior measured the way this row will be measured, and the other one
    X[prefix + "career_own"] = np.where(is_F, rateFN, rateR)
    X[prefix + "career_cross"] = np.where(is_F, rateR, rateFN)
    X[prefix + "career_own_n"] = np.log1p(np.where(is_F, nFN, nR))
    X[prefix + "career_f_share"] = np.where(tot > 0, (nFN + nFO) / np.maximum(tot, 1.0),
                                            np.nan)
    X[prefix + "career_fold_share"] = np.where(tot > 0, nFO / np.maximum(tot, 1.0),
                                               np.nan)
    return X


def make(pit=True, bat=True):
    def build(d, ctx):
        X = fe.build_features(d, ctx, extras=False)
        is_F = X["game_type"].to_numpy() == fe.CAT_MAPS["game_type"]["F"]
        if pit:
            X = _side(X, ctx, "rpit", "pitcher_id", "", K, is_F)
            # the season-to-date rate is measured under this row's regime, so
            # comparing it against the regime-matched prior is apples to apples
            X["std_vs_own"] = X["std_success"].to_numpy() - X["career_own"].to_numpy()
        if bat:
            X = _side(X, ctx, "rbat", "batter_id", "bat_", K_BAT, is_F)
        return X

    return build
