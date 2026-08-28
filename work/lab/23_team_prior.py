"""A team-level prior for pitchers with thin histories.

The weakest slices are pitchers with 1-500 career pitches (522.9) and rows early
in a season (669.6). For those the season-to-date reconstruction has almost
nothing to work with, and the model falls back to the league mean. A team's own
recent control rate is a better fallback than the league's -- and it is a
train-derived constant looked up per row, so it stays legal.

park_id failed badly (-25) when added as a categorical, but that asked the tree
to learn 14 identities from scratch. A rate is a different object: one ordered
number the tree can split on directly.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import fe
import harness
import lab_v6

TARGET = fe.TARGET_COL


def team_table(tr):
    """Per (team, season) success rate, shifted so season s sees only s-1."""
    out = {}
    for col, tag in (("pitcher_team_id", "pt"), ("batter_team_id", "bt")):
        g = tr.groupby([col, "season"])[TARGET].agg(["mean", "size"])
        g = g.reset_index()
        g["season"] = g["season"] + 1          # use in the following season
        out[tag] = {(int(t), int(s)): (m, n) for t, s, m, n in
                    zip(g[col], g["season"], g["mean"], g["size"])}
    out["league"] = float(tr[TARGET].mean())
    return out


def make(team_prior=True, rookie_only=False):
    def build(d, ctx):
        X = fe.build_features(d, ctx)
        if not team_prior:
            return X
        tt = ctx["_team"]
        for col, tag in (("pitcher_team_id", "pit_team"), ("batter_team_id", "bat_team")):
            key = list(zip(X[col].astype(int), X["season"].astype(int)))
            src = tt["pt" if tag == "pit_team" else "bt"]
            vals = np.array([src.get(k, (np.nan, 0))[0] for k in key], dtype="float64")
            X[tag + "_prev_rate"] = vals
        # the prior matters most where the pitcher's own record is thin
        X["team_minus_pitcher"] = X["pit_team_prev_rate"] - X["std_success"]
        if rookie_only:
            thin = X["career_prev_n"].to_numpy() < 500
            X.loc[~thin, "pit_team_prev_rate"] = np.nan
            X.loc[~thin, "team_minus_pitcher"] = np.nan
        return X

    return build


def ctx_fn(tr):
    ctx = fe.fit_context(tr)
    ctx["_team"] = team_table(tr)
    return ctx


if __name__ == "__main__":
    harness.bench(lab_v6.base, "base (v5 features)", seasons=(2024,), seeds=(0, 1),
                  ctx_fn=lab_v6.ctx_fn)
    harness.bench(make(), "+ team prev-season rate", seasons=(2024,), seeds=(0, 1),
                  ctx_fn=ctx_fn)
    harness.bench(make(rookie_only=True), "+ team prior, thin histories only",
                  seasons=(2024,), seeds=(0, 1), ctx_fn=ctx_fn)
