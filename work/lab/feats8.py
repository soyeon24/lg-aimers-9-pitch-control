"""Per-entity *conditional* career rates: platoon and count splits.

The distributed asof_* columns give one career rate per pitcher and one per
batter. Baseball's two largest and most durable individual effects are not in
there at all:

platoon   how a pitcher fares against left- versus right-handed batters, and
          symmetrically how a batter fares against left- versus right-handed
          pitchers. `same_hand` already tells the tree the league-average
          version of this; what it cannot say is that *this* pitcher has a big
          reverse split and that one has none.
count     how a pitcher holds the zone when he is behind in the count versus
          ahead. `count_state` gives the league-average shape; the per-pitcher
          deviation is the part that separates pitchers.

Both are computed the same way as the existing carry tables: totals per
(entity, season, bucket) accumulated through the end of the previous season,
learned from train.csv, looked up per row. No evaluation-set statistic.

Splits are noisy -- a bucket holds a fraction of an already small history -- so
each is shrunk toward that entity's own overall career rate rather than toward
the league. The feature handed to the tree is then a *deviation*: zero for a
pitcher with no evidence of a split, which is the right default.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import fe  # noqa: E402
import feats7  # noqa: E402

TARGET = "control_success"
SEASON_MUL = fe.SEASON_MUL
K_SPLIT = 300.0     # shrinkage of a bucket toward the entity's own career rate


def _bucket_carry(tr, key, code, nbucket, tag):
    """(entity, season) -> career n/successes per bucket, through season-1."""
    d = pd.DataFrame({"e": tr[key].to_numpy(dtype="int64"),
                      "s": tr["season"].to_numpy(dtype="int64"),
                      "b": code,
                      "y": tr[TARGET].to_numpy(dtype="float64")})
    g = d.groupby(["e", "s", "b"]).agg(n=("y", "size"), k=("y", "sum")).reset_index()
    ents = np.sort(d.e.unique())
    seasons = list(range(int(d.s.min()), int(d.s.max()) + 3))
    mi = pd.MultiIndex.from_product([ents, seasons], names=["e", "s"])

    out, names = [], []
    for bi in range(nbucket):
        sub = (g[g.b == bi].set_index(["e", "s"])[["n", "k"]]
               .reindex(mi).fillna(0.0))
        prev = sub.groupby(level=0).cumsum().groupby(level=0).shift(1)
        out += [prev["n"].to_numpy(), prev["k"].to_numpy()]
        names += [f"b{bi}_n", f"b{bi}_k"]

    keys = (mi.get_level_values(0).to_numpy(dtype="int64") * SEASON_MUL
            + mi.get_level_values(1).to_numpy(dtype="int64"))
    order = np.argsort(keys)
    return keys[order], np.column_stack(out)[order], names


# hand columns are already integer coded in the raw file (1/2)
def _hand_code(v):
    return (np.asarray(v) == 2).astype("int64")


def _count_code(tr):
    """0 = pitcher behind or even, 1 = pitcher ahead."""
    return (tr["strikes_before"].to_numpy() > tr["balls_before"].to_numpy()).astype("int64")


SPECS = [
    ("plat_pit", "pitcher_id", lambda tr: _hand_code(tr["batter_hand"]), 2, "batter_hand"),
    ("plat_bat", "batter_id", lambda tr: _hand_code(tr["pitcher_hand"]), 2, "pitcher_hand"),
    ("cnt_pit", "pitcher_id", _count_code, 2, "__count__"),
]


def fit(tr, specs=None):
    ctx = feats7.fit(tr)
    for tag, key, coder, nb, _ in (specs or SPECS):
        k, v, names = _bucket_carry(tr, key, coder(tr), nb, tag)
        ctx[tag + "_keys"], ctx[tag + "_vals"], ctx[tag + "_cols"] = k, v, names
    return ctx


def _apply(X, ctx, tag, idcol, sel, prefix, base_rate):
    c = fe._lookup(ctx, tag, X[idcol].to_numpy(), X["season"].to_numpy())
    v = {n: np.nan_to_num(c[n]) for n in ctx[tag + "_cols"]}
    n0, k0, n1, k1 = v["b0_n"], v["b0_k"], v["b1_n"], v["b1_k"]
    # each bucket shrunk toward the entity's own career rate -> a deviation that
    # is 0 when the entity has no evidence of a split
    d0 = (k0 + K_SPLIT * base_rate) / (n0 + K_SPLIT) - base_rate
    d1 = (k1 + K_SPLIT * base_rate) / (n1 + K_SPLIT) - base_rate
    X[prefix + "_dev"] = np.where(sel == 1, d1, d0)
    X[prefix + "_gap"] = d1 - d0
    X[prefix + "_n"] = np.log1p(np.where(sel == 1, n1, n0))
    return X


def make(platoon=True, count=True, regime=True):
    build7 = feats7.make(pit=True, bat=True)

    def build(d, ctx):
        X = build7(d, ctx) if regime else fe.build_features(d, ctx, extras=False)
        base_p = (X["career_clean"].to_numpy() if regime
                  else X["career_success"].to_numpy())
        base_p = np.where(np.isnan(base_p), ctx["rp"]["asof_pitcher_success_rate"],
                          base_p)
        base_b = (X["bat_career_clean"].to_numpy() if regime
                  else X["bat_career_success"].to_numpy())
        base_b = np.where(np.isnan(base_b), ctx["rp"]["asof_batter_success_rate"],
                          base_b)
        if platoon:
            X = _apply(X, ctx, "plat_pit", "pitcher_id",
                       _hand_code(X["batter_hand"].to_numpy()), "plat_pit", base_p)
            X = _apply(X, ctx, "plat_bat", "batter_id",
                       _hand_code(X["pitcher_hand"].to_numpy()), "plat_bat", base_b)
        if count:
            sel = (X["strikes_before"].to_numpy() > X["balls_before"].to_numpy()
                   ).astype("int64")
            X = _apply(X, ctx, "cnt_pit", "pitcher_id", sel, "cnt_pit", base_p)
        return X

    return build
