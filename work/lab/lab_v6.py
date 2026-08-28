"""Candidate features layered on top of the production builder.

park      the home team is derivable: top_bottom T means the away side bats, so
          the pitcher's team is home (verified -- score_diff_pitcher_team equals
          score_diff_home on exactly those rows). Home team id is the ballpark.
contrast  std_success and bat_std_success both sit on this season's league level,
          so their difference cancels the level and leaves a pure matchup term.
multik    one shrinkage constant has to serve a 40-pitch rookie and a 2000-pitch
          workhorse; give the tree a light and a heavy version and let it pick.
eb        shrink toward the pitcher's own previous season instead of the league.
prevgames unpack prev1/3/5 game rates into disjoint windows (games 2-3, 4-5).
usage     pitches per season and per career -- starter vs reliever workload.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import fe  # noqa: E402

PARK_CODES = {t: i for i, t in enumerate(range(12, 27))}


def ctx_fn(tr):
    return fe.fit_context(tr)


def make(park=False, contrast=False, multik=False, eb=False, prevgames=False,
         usage=False):
    def build(d, ctx):
        X = fe.build_features(d, ctx, extras=False)
        rp = ctx["rp"]

        if park:
            home = np.where(X["top_bottom"].to_numpy() == 1,
                            X["pitcher_team_id"].to_numpy(),
                            X["batter_team_id"].to_numpy())
            X["park_id"] = np.array([PARK_CODES.get(int(t), np.nan) for t in home],
                                    dtype="float64")

        if multik or eb:
            c = fe._lookup(ctx, "pit", X["pitcher_id"].to_numpy(), X["season"].to_numpy())
            an = X["asof_pitcher_n"].to_numpy(dtype="float64")
            num = X["asof_pitcher_success_rate"].to_numpy(dtype="float64") * an \
                - np.nan_to_num(c["c_success"])
            den = np.maximum(an - np.nan_to_num(c["c_n"]), 0.0)
            prior = rp["asof_pitcher_success_rate"]
            if multik:
                for k in (50.0, 600.0):
                    X[f"std_success_k{int(k)}"] = (num + k * prior) / (den + k)
            if eb:
                pv = np.where(c["s_n"] > 0, c["s_success"] / np.maximum(c["s_n"], 1.0),
                              prior)
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

        if usage:
            X["log_season_n"] = np.log1p(X["season_n"])
            X["log_career_n"] = np.log1p(X["asof_pitcher_n"])
            X["season_progress"] = X["season_n"] / np.maximum(X["prevseason_n"], 1.0)
            X["bat_log_season_n"] = np.log1p(X["bat_season_n"])
        return X

    return build


base = make()
