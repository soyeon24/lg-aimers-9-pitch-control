"""Tuning the season-to-date estimator itself.

The batter and pitcher season-to-date blocks carry almost all of v4's gain, so
the two knobs inside them are worth a sweep: how hard to shrink, and how wide a
window to reconstruct. A two-season window (this season plus last) is available
from the same carry table -- totals at the end of s-2 are c_n minus s_n -- and
sits between the noisy one-season and the stale career estimate.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import fe  # noqa: E402


def ctx_fn(tr):
    return fe.fit_context(tr)


def make(k_pit=None, k_bat=None, k_mix=None, window2=False):
    def build(d, ctx):
        old = (fe.K_PIT, fe.K_BAT, fe.K_MIX)
        if k_pit:
            fe.K_PIT = k_pit
        if k_bat:
            fe.K_BAT = k_bat
        if k_mix:
            fe.K_MIX = k_mix
        try:
            X = fe.build_features(d, ctx)
        finally:
            fe.K_PIT, fe.K_BAT, fe.K_MIX = old

        if window2:
            season = X["season"].to_numpy()
            c = fe._lookup(ctx, "pit", X["pitcher_id"].to_numpy(), season)
            an = X["asof_pitcher_n"].to_numpy(dtype="float64")
            n2 = np.nan_to_num(c["c_n"]) - np.nan_to_num(c["s_n"])   # end of s-2
            for r in ("success", "middle"):
                prior = ctx["rp"][f"asof_pitcher_{r}_rate"]
                base = np.nan_to_num(c["c_" + r]) - np.nan_to_num(c["s_" + r])
                cum = X[f"asof_pitcher_{r}_rate"].to_numpy(dtype="float64") * an
                X["win2_" + r] = fe._shrunk(cum - base, an - n2, prior, fe.K_PIT)

            cb = fe._lookup(ctx, "bat", X["batter_id"].to_numpy(), season)
            anb = X["asof_batter_n"].to_numpy(dtype="float64")
            n2b = np.nan_to_num(cb["c_n"]) - np.nan_to_num(cb["s_n"])
            prior = ctx["rp"]["asof_batter_success_rate"]
            base = np.nan_to_num(cb["c_success"]) - np.nan_to_num(cb["s_success"])
            cum = X["asof_batter_success_rate"].to_numpy(dtype="float64") * anb
            X["bat_win2_success"] = fe._shrunk(cum - base, anb - n2b, prior, fe.K_BAT)
        return X

    return build
