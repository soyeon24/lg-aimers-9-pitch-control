"""용량을 줄이는 방향으로 스윕 (probe2에서 키울수록 나빠짐을 확인)."""
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

def run(tag, **kw):
    t = time.time()
    p = dict(max_iter=400, learning_rate=0.05, max_leaf_nodes=31, l2_regularization=1.0,
             min_samples_leaf=200, categorical_features=ci, early_stopping=False, random_state=0)
    p.update(kw)
    m = HistGradientBoostingClassifier(**p).fit(Xtr, ytr)
    pr = m.predict_proba(Xva)[:, 1]
    sh = score(np.clip(pr + (pred_rate - pr.mean()), 1e-6, 1 - 1e-6), yva)
    print(f"{tag:38s} raw={score(pr,yva):7.1f}  +trend={sh:7.1f}  {round(time.time()-t)}s", flush=True)

run("iter=400 leaf=31 (기준)")
run("iter=200 leaf=31", max_iter=200)
run("iter=100 leaf=31", max_iter=100)
run("iter=400 leaf=15", max_leaf_nodes=15)
run("iter=200 leaf=15", max_iter=200, max_leaf_nodes=15)
run("iter=400 leaf=15 l2=10 msl=2000", max_leaf_nodes=15, l2_regularization=10.0, min_samples_leaf=2000)
run("iter=200 leaf=8", max_iter=200, max_leaf_nodes=8)
run("iter=600 leaf=8 lr=.03", max_iter=600, max_leaf_nodes=8, learning_rate=0.03)
