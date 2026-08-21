"""Lab harness: fast multi-season holdout evaluation.

Design points
-------------
* Splits are by season. For holdout season V we train on seasons < V and every
  derived constant (priors, per-entity carry tables, season-rate extrapolation)
  comes from seasons < V only -- exactly mirroring the real 2025 setting.
* `carry` tables let us recover *season-to-date* rates for any row from that
  row's own asof_* columns plus a per-entity constant learned from train.
  Inference stays row-independent, so the contest rule holds.
"""
import os
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "cache")
TARGET = "control_success"
EPS = 1e-6

CAT_MAPS = {
    "top_bottom": {"B": 0, "T": 1},
    "game_type": {"F": 0, "R": 1},
    "base_state": {"___": 0, "1__": 1, "_2_": 2, "__3": 3,
                   "12_": 4, "1_3": 5, "_23": 6, "123": 7},
}
CAT_COLS = list(CAT_MAPS)

PARAMS = dict(max_iter=400, learning_rate=0.05, max_leaf_nodes=15,
              l2_regularization=10.0, min_samples_leaf=10000, early_stopping=False)

_DF = None


def load():
    global _DF
    if _DF is None:
        t = time.time()
        _DF = pd.read_parquet(os.path.join(CACHE, "train.parquet"))
        print(f"[load] {_DF.shape} {time.time() - t:.1f}s", flush=True)
    return _DF


def logit(p):
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def comp_score(p, y):
    r = y.mean()
    return max(0.0, 100000 * (1 - np.mean((p - y) ** 2) / (r * (1 - r))))


def extrapolate(rates, ahead=1):
    y = np.asarray(rates, dtype=float)
    x = np.arange(len(y), dtype=float)
    slope, intercept = np.polyfit(x, y, 1)
    return float(intercept + slope * (len(y) - 1 + ahead))


# --------------------------------------------------------------- carry tables
PIT_RATES = ["success", "reverse", "middle", "ball", "strike"]
MIX_RATES = ["fastball", "breaking", "offspeed"]
BAT_RATES = ["success", "middle"]


def _season_end(df, key, ncol, rates, prefix, use_label):
    """Cumulative totals at the END of each (entity, season).

    The entity's last row of the season carries asof_* describing everything
    before it, so totals come back with at most one pitch of slack -- exact for
    `success`, where that row's own label is known.
    """
    idx = df.groupby([key, "season"])[ncol].idxmax()
    cols = [key, "season", ncol] + [prefix + r + "_rate" for r in rates]
    if use_label:
        cols.append(TARGET)
    last = df.loc[idx, cols]
    n = last[ncol].values.astype("float64")
    out = pd.DataFrame({key: last[key].values, "season": last["season"].values,
                        "n": n + (1.0 if use_label else 0.0)})
    for r in rates:
        v = last[prefix + r + "_rate"].values.astype("float64") * n
        if use_label and r == "success":
            v = v + last[TARGET].values.astype("float64")
        out[r] = v
    return out


def _grid(tbl, key, seasons):
    """Forward-filled cumulative-at-end-of-season grid over all entities."""
    ents = np.sort(tbl[key].unique())
    mi = pd.MultiIndex.from_product([ents, seasons], names=[key, "season"])
    g = tbl.set_index([key, "season"]).reindex(mi).groupby(level=0).ffill()
    return g


def _carry_frame(tbl, key, seasons, cols):
    """(entity, season) -> totals at end of s-1, plus season s-1 alone."""
    g = _grid(tbl, key, seasons)
    prev1 = g.groupby(level=0).shift(1)          # end of s-1
    prev2 = g.groupby(level=0).shift(2)          # end of s-2
    out = prev1.copy()
    out.columns = ["c_" + c for c in cols]
    d = prev1.values - prev2.values              # season s-1 only
    for i, c in enumerate(cols):
        out["s_" + c] = d[:, i]
    return out.reset_index()


class Ctx:
    """Constants fitted on the training portion only."""

    def __init__(self, tr):
        self.priors = {
            "success": float(tr["asof_pitcher_success_rate"].mean()),
            "middle": float(tr["asof_pitcher_middle_rate"].mean()),
        }
        # league means for every rate column -- always from train only
        self.rp = {c: float(tr[c].mean()) for c in tr.columns
                   if c.startswith("asof_") and c.endswith("_rate")}
        seasons = sorted(tr.season.unique())
        self.seasons = seasons
        self.rates = tr.groupby("season")[TARGET].mean().loc[seasons].values
        self.target_rate = extrapolate(self.rates)
        grid_seasons = list(range(min(seasons), max(seasons) + 3))

        pit = _season_end(tr, "pitcher_id", "asof_pitcher_n", PIT_RATES,
                          "asof_pitcher_", True)
        mix = _season_end(tr, "pitcher_id", "asof_pitcher_pitchmix_n", MIX_RATES,
                          "asof_pitcher_", False)
        bat = _season_end(tr, "batter_id", "asof_batter_n", BAT_RATES,
                          "asof_batter_", False)
        self.pit = _carry_frame(pit, "pitcher_id", grid_seasons, ["n"] + PIT_RATES)
        self.mix = _carry_frame(mix, "pitcher_id", grid_seasons, ["n"] + MIX_RATES)
        self.bat = _carry_frame(bat, "batter_id", grid_seasons, ["n"] + BAT_RATES)
        self.mix = self.mix.rename(columns={c: c + "_mix" for c in self.mix.columns
                                            if c.startswith(("c_", "s_"))})


# --------------------------------------------------------------- feature sets
def base_frame(d):
    X = d.drop(columns=[c for c in ("row_id", TARGET) if c in d.columns]).copy()
    for col, m in CAT_MAPS.items():
        X[col] = X[col].astype(str).map(m).astype("float64")
    return X


def add_v3(X, priors):
    b, s = X["balls_before"], X["strikes_before"]
    X["count_state"] = (b * 3 + s).astype("float64")
    X["count_diff"] = (s - b).astype("float64")
    X["two_strikes"] = (s == 2).astype("float64")
    X["three_balls"] = (b == 3).astype("float64")
    X["full_count"] = ((b == 3) & (s == 2)).astype("float64")
    X["first_pitch"] = ((b == 0) & (s == 0)).astype("float64")
    X["pitcher_ahead"] = (s > b).astype("float64")
    X["must_strike"] = ((b == 3) & (s < 2)).astype("float64")
    X["can_waste"] = ((s == 2) & (b < 2)).astype("float64")

    X["same_hand"] = (X["pitcher_hand"] == X["batter_hand"]).astype("float64")
    X["risp"] = ((X["runner_on_2b"] == 1) | (X["runner_on_3b"] == 1)).astype("float64")
    X["bases_loaded"] = (X["num_runners_on"] == 3).astype("float64")
    X["close_game"] = (X["score_diff_pitcher_team"].abs() <= 1).astype("float64")
    X["blowout"] = (X["score_diff_pitcher_team"].abs() >= 5).astype("float64")
    X["late_inning"] = (X["inning"] >= 7).astype("float64")
    X["log_li"] = np.log1p(X["li"])

    base_s = X["asof_pitcher_success_rate"].fillna(priors["success"])
    base_m = X["asof_pitcher_middle_rate"].fillna(priors["middle"])
    for w in (1, 3, 5):
        X[f"form{w}_success"] = X[f"asof_pitcher_prev{w}_game_success_rate"] - base_s
        X[f"form{w}_middle"] = X[f"asof_pitcher_prev{w}_game_middle_rate"] - base_m
    X["form_trend"] = (X["asof_pitcher_prev1_game_success_rate"]
                       - X["asof_pitcher_prev5_game_success_rate"])
    return X


def build_v3(d, ctx):
    return add_v3(base_frame(d), ctx.priors)


# --------------------------------------------------------------------- runner
def fit_predict(Xtr, ytr, Xva, Xref, seeds, params=None):
    p = dict(PARAMS)
    if params:
        p.update(params)
    ci = [Xtr.columns.get_loc(c) for c in CAT_COLS]
    pv = np.zeros(len(Xva))
    pr = np.zeros(len(Xref))
    for sd in seeds:
        m = HistGradientBoostingClassifier(categorical_features=ci, random_state=sd, **p)
        m.fit(Xtr, ytr)
        pv += m.predict_proba(Xva)[:, 1]
        pr += m.predict_proba(Xref)[:, 1]
    return pv / len(seeds), pr / len(seeds)


def run_split(build_fn, val_season, seeds=(0, 1), params=None, verbose=True):
    df = load()
    tr = df[df.season < val_season]
    va = df[df.season == val_season]
    ytr, yva = tr[TARGET].values, va[TARGET].values
    ctx = Ctx(tr)
    t = time.time()
    Xtr = build_fn(tr, ctx)
    Xva = build_fn(va, ctx)[list(Xtr.columns)]
    tfeat = time.time() - t

    last = (tr.season.values == val_season - 1)
    t = time.time()
    pv, pref = fit_predict(Xtr, ytr, Xva, Xtr[last], seeds, params)
    tfit = time.time() - t

    ref = float(pref.mean())
    adj = sigmoid(logit(pv) + logit(ctx.target_rate) - logit(ref))
    orc = sigmoid(logit(pv) + logit(yva.mean()) - logit(pv.mean()))
    res = dict(season=val_season, nfeat=Xtr.shape[1],
               raw=comp_score(pv, yva), adj=comp_score(adj, yva),
               oracle=comp_score(orc, yva),
               raw_mean=float(pv.mean()), ref_mean=ref,
               adj_mean=float(adj.mean()), true=float(yva.mean()),
               target=ctx.target_rate, tfeat=tfeat, tfit=tfit)
    if verbose:
        print(f"  V{val_season} feat={res['nfeat']:3d} adj={res['adj']:7.1f} "
              f"oracle={res['oracle']:7.1f} | true={res['true']:.4f} "
              f"target={res['target']:.4f} adjmean={res['adj_mean']:.4f} "
              f"refmean={ref:.4f} rawmean={res['raw_mean']:.4f} "
              f"({tfeat:.0f}s+{tfit:.0f}s)", flush=True)
    return res


def bench(build_fn, tag, val_seasons=(2022, 2023, 2024), seeds=(0, 1), params=None):
    print(f"--- {tag} ---", flush=True)
    rs = [run_split(build_fn, v, seeds, params) for v in val_seasons]
    print(f"  MEAN adj={np.mean([r['adj'] for r in rs]):7.1f} "
          f"oracle={np.mean([r['oracle'] for r in rs]):7.1f}", flush=True)
    return rs
