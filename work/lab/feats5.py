"""v5 knobs on top of v4: F-regime handling, form baseline, drift signal."""
import numpy as np

from common import base_frame, add_v3
from feats4 import add_season_to_date

F_BREAK_SEASON = 2023   # game_type F changed definition here (.709 -> .473)


def build(d, ctx, f_flag=False, form_vs_std=False, drift=False,
          drop_season=False, **std_kw):
    X = add_v3(base_frame(d), ctx.priors)
    X = add_season_to_date(X, ctx, **std_kw)

    if f_flag:
        # one split instead of two: lets the tree quarantine the old F regime
        # without spending depth on season x game_type
        X["f_old_regime"] = ((X["game_type"] == 0) &
                             (X["season"] < F_BREAK_SEASON)).astype("float64")
    if form_vs_std:
        # recent-game form is more informative against this season's own level
        # than against a career average that spans the whole decline
        for w in (1, 3, 5):
            X[f"form{w}_vs_std"] = (X[f"asof_pitcher_prev{w}_game_success_rate"]
                                    - X["std_success"])
    if drift:
        # per-row measurement of "how much worse is this pitcher than last year",
        # whose row average is exactly the league drift we are trying to forecast
        X["drift_success"] = X["std_success"] - X["prevseason_success"]
        X["drift_middle"] = X["std_middle"] - X["prevseason_middle"]
    if drop_season:
        X = X.drop(columns=["season"])
    return X


def make(**kw):
    def f(d, ctx):
        return build(d, ctx, **kw)
    return f
