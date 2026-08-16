"""정규화 방향으로 한 번 더 스윕 + 시드 앙상블 효과 확인."""
import time, numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

CAT = ["top_bottom", "game_type", "base_state"]
df = pd.read_csv("C:/school/26summer/lgaimer/open/data/train.csv", encoding="utf-8-sig")

def prep(d):
    X = d.drop(columns=[c for c in ["row_id", "control_success"] if c in d.columns]).copy()
    for c in CAT:
        X[c] = X[c].astype("category").cat.codes.astype("float64")
    return X

def score(p, y):
    return max(0.0, 100000 * (1 - np.mean((p - y) ** 2) / (y.mean() * (1 - y.mean()))))

tr, va = df[df.season <= 2023], df[df.season == 2024]
ytr, yva = tr.control_success.values, va.control_success.values
Xtr, Xva = prep(tr), prep(va)
ci = [Xtr.columns.get_loc(c) for c in CAT]
rates = tr.groupby("season").control_success.mean().values
b, a = np.polyfit(np.arange(len(rates), dtype=float), rates, 1)
pred_rate = a + b * len(rates)

def fit(seed=0, **kw):
    p = dict(max_iter=400, learning_rate=0.05, max_leaf_nodes=15, l2_regularization=10.0,
             min_samples_leaf=2000, categorical_features=ci, early_stopping=False, random_state=seed)
    p.update(kw)
    return HistGradientBoostingClassifier(**p).fit(Xtr, ytr).predict_proba(Xva)[:, 1]

def rep(tag, pr):
    sh = score(np.clip(pr + (pred_rate - pr.mean()), 1e-6, 1 - 1e-6), yva)
    print(f"{tag:38s} raw={score(pr,yva):7.1f}  +trend={sh:7.1f}", flush=True)
    return sh

t = time.time()
rep("leaf15 l2=10 msl=2000 (현재 best)", fit())
rep("msl=5000", fit(min_samples_leaf=5000))
rep("msl=10000", fit(min_samples_leaf=10000))
rep("l2=50 msl=5000", fit(l2_regularization=50.0, min_samples_leaf=5000))
rep("leaf=20 msl=5000", fit(max_leaf_nodes=20, min_samples_leaf=5000))
rep("leaf=15 msl=5000 lr=.03 iter=700", fit(min_samples_leaf=5000, learning_rate=0.03, max_iter=700))
print(f"({round(time.time()-t)}s)")

# 시드 앙상블
ps = [fit(seed=s, min_samples_leaf=5000) for s in range(4)]
rep("seed 4개 평균 (msl=5000)", np.mean(ps, axis=0))
print(f"total {round(time.time()-t)}s")
