"""v6 candidates, on top of v4's season-to-date block.

park      the home team is derivable (top_bottom T means the away side bats, so
          the pitcher is home -- verified: score_diff_pitcher_team equals
          score_diff_home exactly on those rows). Home team id is the ballpark.
contrast  std_success and bat_std_success both sit on this season's league
          level, so their difference cancels the level out and leaves a pure
          matchup term -- useful precisely because the level is the part we
          cannot forecast.
multik    one shrinkage constant has to serve both a 40-pitch rookie and a
          2000-pitch workhorse; giving the tree a light and a heavy version
          lets it choose per row instead.
eb        shrink this season's rate toward the pitcher's own previous season
          rather than toward the league mean.
prevgames unpack prev1/3/5 game rates into disjoint windows (game 2-3, game 4-5).
"""
import numpy as np

from common import base_frame, add_v3
from feats4 import add_season_to_date

PARK_CODES = {t: i for i, t in enumerate(range(12, 27))}


def build(d, ctx, park=False, contrast=False, multik=False, eb=False,
          prevgames=False, **std_kw):
    X = add_v3(base_frame(d), ctx.priors)
    X = add_season_to_date(X, ctx, **std_kw)

    if park:
        home = np.where(X["top_bottom"].values == 1,
                        X["pitcher_team_id"].values, X["batter_team_id"].values)
        X["park_id"] = [PARK_CODES.get(int(t), np.nan) for t in home]
        X["park_id"] = X["park_id"].astype("float64")
        X["pitcher_home"] = (X["top_bottom"] == 1).astype("float64")

    if contrast or multik or eb:
        p = ctx.pit.set_index(["pitcher_id", "season"])
        key = list(zip(X["pitcher_id"].values, X["season"].values))
        cn = p["c_n"].reindex(key).values
        cs = p["c_success"].reindex(key).values
        sn_prev = p["s_n"].reindex(key).values
        ss_prev = p["s_success"].reindex(key).values
        cn = np.nan_to_num(cn)
        cs = np.nan_to_num(cs)
        an = X["asof_pitcher_n"].values.astype("float64")
        num = X["asof_pitcher_success_rate"].values * an - cs
        den = np.maximum(an - cn, 0.0)
        prior = ctx.rp["asof_pitcher_success_rate"]

        if multik:
            for k in (50.0, 600.0):
                X[f"std_success_k{int(k)}"] = (num + k * prior) / (den + k)
        if eb:
            pv = np.where(sn_prev > 0, ss_prev / np.maximum(sn_prev, 1.0), prior)
            pv = np.where(np.isnan(pv), prior, pv)
            for k in (200.0, 600.0):
                X[f"std_eb{int(k)}"] = (num + k * pv) / (den + k)
        if contrast:
            X["pit_minus_bat"] = X["std_success"] - X["bat_std_success"]
            X["pit_minus_bat_mid"] = X["std_middle"] - X["bat_std_middle"]

    if prevgames:
        p1 = X["asof_pitcher_prev1_game_success_rate"]
        p3 = X["asof_pitcher_prev3_game_success_rate"]
        p5 = X["asof_pitcher_prev5_game_success_rate"]
        X["games23_success"] = (3 * p3 - p1) / 2
        X["games45_success"] = (5 * p5 - 3 * p3) / 2
        m1 = X["asof_pitcher_prev1_game_middle_rate"]
        m3 = X["asof_pitcher_prev3_game_middle_rate"]
        X["games23_middle"] = (3 * m3 - m1) / 2
    return X


def make(**kw):
    def f(d, ctx):
        return build(d, ctx, **kw)
    return f
