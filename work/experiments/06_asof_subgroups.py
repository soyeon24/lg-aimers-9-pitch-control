"""asof 그룹을 쪼개서 어느 조합이 최선인지 확인."""
import time
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

CAT_MAPS = {
    "top_bottom": {"B": 0, "T": 1},
    "game_type": {"F": 0, "R": 1},
    "base_state": {"___": 0, "1__": 1, "_2_": 2, "__3": 3,
                   "12_": 4, "1_3": 5, "_23": 6, "123": 7},
}
EPS = 1e-6
SEEDS = [0, 1]
PARAMS = dict(max_iter=400, learning_rate=0.05, max_leaf_nodes=15,
              l2_regularization=10.0, min_samples_leaf=10000, early_stopping=False)

df = pd.read_csv("C:/school/26summer/lgaimer/open/data/train.csv", encoding="utf-8-sig")
tr, va = df[df.season <= 2023], df[df.season == 2024]
ytr, yva = tr.control_success.values, va.control_success.values
PRIOR = {"success": float(tr.control_success.mean()),
         "middle": float(tr.asof_pitcher_middle_rate.mean()),
         "ball": float(tr.asof_pitcher_ball_rate.mean()),
         "reverse": float(tr.asof_pitcher_reverse_rate.mean())}


def base_features(d):
    X = d.drop(columns=[c for c in ("row_id", "control_success") if c in d.columns]).copy()
    for col, m in CAT_MAPS.items():
        X[col] = X[col].astype(str).map(m).astype("float64")
    return X


def add_count(X):
    b, s = X.balls_before, X.strikes_before
    X["count_state"] = (b * 3 + s).astype("float64")
    X["count_diff"] = (s - b).astype("float64")
    X["two_strikes"] = (s == 2).astype("float64")
    X["three_balls"] = (b == 3).astype("float64")
    X["full_count"] = ((b == 3) & (s == 2)).astype("float64")
    X["first_pitch"] = ((b == 0) & (s == 0)).astype("float64")
    X["pitcher_ahead"] = (s > b).astype("float64")
    X["must_strike"] = ((b == 3) & (s < 2)).astype("float64")
    X["can_waste"] = ((s == 2) & (b < 2)).astype("float64")
    return X


def add_situation(X):
    X["same_hand"] = (X.pitcher_hand == X.batter_hand).astype("float64")
    X["risp"] = ((X.runner_on_2b == 1) | (X.runner_on_3b == 1)).astype("float64")
    X["bases_loaded"] = (X.num_runners_on == 3).astype("float64")
    X["close_game"] = (X.score_diff_pitcher_team.abs() <= 1).astype("float64")
    X["blowout"] = (X.score_diff_pitcher_team.abs() >= 5).astype("float64")
    X["late_inning"] = (X.inning >= 7).astype("float64")
    X["log_li"] = np.log1p(X.li)
    return X


def _shrink(rate, n, prior, k=200):
    return (n.to_numpy() * rate.fillna(prior).to_numpy() + k * prior) / (n.to_numpy() + k)


def add_shrink(X):
    X["sh_pitcher_success"] = _shrink(X.asof_pitcher_success_rate, X.asof_pitcher_n, PRIOR["success"])
    X["sh_pitcher_middle"] = _shrink(X.asof_pitcher_middle_rate, X.asof_pitcher_n, PRIOR["middle"])
    X["sh_pitcher_reverse"] = _shrink(X.asof_pitcher_reverse_rate, X.asof_pitcher_n, PRIOR["reverse"])
    X["sh_batter_success"] = _shrink(X.asof_batter_success_rate, X.asof_batter_n, PRIOR["success"])
    X["matchup_success"] = X.sh_pitcher_success - X.sh_batter_success
    return X


def add_cold(X):
    X["log_pitcher_n"] = np.log1p(X.asof_pitcher_n)
    X["log_batter_n"] = np.log1p(X.asof_batter_n)
    X["pitcher_cold"] = (X.asof_pitcher_n < 50).astype("float64")
    X["batter_cold"] = (X.asof_batter_n < 50).astype("float64")
    return X


def add_form(X):
    base_s = X.asof_pitcher_success_rate.fillna(PRIOR["success"])
    base_m = X.asof_pitcher_middle_rate.fillna(PRIOR["middle"])
    for w in (1, 3, 5):
        X[f"form{w}_success"] = X[f"asof_pitcher_prev{w}_game_success_rate"] - base_s
        X[f"form{w}_middle"] = X[f"asof_pitcher_prev{w}_game_middle_rate"] - base_m
    X["form_trend"] = (X.asof_pitcher_prev1_game_success_rate
                       - X.asof_pitcher_prev5_game_success_rate)
    return X


def make(d, steps):
    X = base_features(d)
    for f in steps:
        X = f(X)
    return X


def comp_score(p, y):
    r = y.mean()
    return max(0.0, 100000 * (1 - np.mean((p - y) ** 2) / (r * (1 - r))))


def logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


rates = tr.groupby("season").control_success.mean().values
slope, intercept = np.polyfit(np.arange(len(rates), dtype=float), rates, 1)
target_rate = intercept + slope * len(rates)
last_mask = (tr.season == 2023).values


def run(tag, steps):
    t = time.time()
    Xtr, Xva = make(tr, steps), make(va, steps)
    ci = [Xtr.columns.get_loc(c) for c in CAT_MAPS]
    ms = [HistGradientBoostingClassifier(categorical_features=ci, random_state=s,
                                         **PARAMS).fit(Xtr, ytr) for s in SEEDS]
    pv = np.mean([m.predict_proba(Xva)[:, 1] for m in ms], axis=0)
    ref = np.mean([m.predict_proba(Xtr[last_mask])[:, 1] for m in ms], axis=0).mean()
    adj = 1 / (1 + np.exp(-(logit(pv) + logit(target_rate) - logit(ref))))
    print(f"{tag:34s} feat={Xtr.shape[1]:3d}  raw={comp_score(pv,yva):7.1f}  "
          f"+offset={comp_score(adj,yva):7.1f}  ({round(time.time()-t)}s)", flush=True)


run("카운트+상황 (asof 없음)", [add_count, add_situation])
run("카운트+상황+shrink", [add_count, add_situation, add_shrink])
run("카운트+상황+cold", [add_count, add_situation, add_cold])
run("카운트+상황+form", [add_count, add_situation, add_form])
run("전부 (재확인)", [add_count, add_situation, add_shrink, add_cold, add_form])
