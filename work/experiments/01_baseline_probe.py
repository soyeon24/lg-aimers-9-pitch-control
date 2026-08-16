import time, numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

T0 = time.time()
DATA = "C:/school/26summer/lgaimer/open/data/train.csv"
df = pd.read_csv(DATA, encoding="utf-8-sig")
print("loaded", df.shape, round(time.time()-T0, 1), "s")

CAT = ["top_bottom", "game_type", "base_state"]
DROP = ["row_id", "control_success"]

# drift check on asof features
print("\n--- season drift ---")
cols = ["control_success", "asof_pitcher_success_rate", "asof_batter_success_rate",
        "asof_pitcher_middle_rate", "li", "home_win_expectancy"]
print(df.groupby("season")[cols].mean().round(4).to_string())

def prep(d):
    X = d.drop(columns=[c for c in DROP if c in d.columns]).copy()
    for c in CAT:
        X[c] = X[c].astype("category").cat.codes.astype("float64")
    return X

def score(p, y):
    bs = np.mean((p - y) ** 2)
    r = y.mean()
    return max(0.0, 100000 * (1 - bs / (r * (1 - r)))), bs, r

# ---- holdout: train 2019-2023, validate 2024 (mimics the 1-year gap to 2025) ----
tr = df[df.season <= 2023]
va = df[df.season == 2024]
ytr, yva = tr.control_success.values, va.control_success.values
Xtr, Xva = prep(tr), prep(va)
cat_idx = [Xtr.columns.get_loc(c) for c in CAT]
print(f"\ntrain={Xtr.shape} valid={Xva.shape}  train_rate={ytr.mean():.4f} valid_rate={yva.mean():.4f}")

# 0) constant = train mean (the naive transfer)
for name, p in [("const=train_mean", np.full(len(yva), ytr.mean())),
                ("const=valid_mean(oracle)", np.full(len(yva), yva.mean())),
                ("const=2023_mean", np.full(len(yva), df[df.season == 2023].control_success.mean()))]:
    s, bs, r = score(p, yva)
    print(f"{name:28s} score={s:9.2f}  bs={bs:.6f}")

# 1) HistGBM with season feature
for use_season in [True, False]:
    Xa, Xb = (Xtr, Xva) if use_season else (Xtr.drop(columns=["season"]), Xva.drop(columns=["season"]))
    ci = [Xa.columns.get_loc(c) for c in CAT]
    t = time.time()
    m = HistGradientBoostingClassifier(
        max_iter=400, learning_rate=0.05, max_leaf_nodes=31,
        l2_regularization=1.0, min_samples_leaf=200,
        categorical_features=ci, early_stopping=False, random_state=0)
    m.fit(Xa, ytr)
    p = m.predict_proba(Xb)[:, 1]
    s, bs, r = score(p, yva)
    print(f"HGB season={str(use_season):5s}  score={s:9.2f}  bs={bs:.6f}  "
          f"pred_mean={p.mean():.4f} ({round(time.time()-t)}s)")
    # mean-shift corrected (what recalibrating the intercept would buy)
    p2 = np.clip(p + (yva.mean() - p.mean()), 1e-6, 1 - 1e-6)
    s2, bs2, _ = score(p2, yva)
    print(f"   +oracle mean-shift        score={s2:9.2f}  bs={bs2:.6f}")
    # shrink toward own mean
    for a in [0.9, 0.75, 0.5]:
        ps = p.mean() + a * (p - p.mean())
        s3, _, _ = score(ps, yva)
        print(f"   shrink a={a:<4}            score={s3:9.2f}")

print("\ntotal", round(time.time()-T0), "s")
