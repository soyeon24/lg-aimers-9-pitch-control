"""2024 홀드아웃으로 (a) 시즌 추세 기반 평균 보정, (b) 최근 시즌 가중치 검증."""
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

def trend_rate(rates, ahead=1):
    """시즌별 성공률에 선형추세를 적용해 ahead 시즌 뒤 값을 외삽."""
    x = np.arange(len(rates), dtype=float)
    b, a = np.polyfit(x, np.asarray(rates, dtype=float), 1)
    return a + b * (len(rates) - 1 + ahead)

tr, va = df[df.season <= 2023], df[df.season == 2024]
ytr, yva = tr.control_success.values, va.control_success.values
Xtr, Xva = prep(tr), prep(va)
ci = [Xtr.columns.get_loc(c) for c in CAT]

rates_tr = tr.groupby("season").control_success.mean().values
pred_rate = trend_rate(rates_tr)
print(f"train seasons rate = {np.round(rates_tr,4)}")
print(f"추세 외삽 2024 예측률 = {pred_rate:.4f} / 실제 = {yva.mean():.4f} "
      f"(오차 {pred_rate - yva.mean():+.4f})\n")

def run(tag, weights=None, **kw):
    t = time.time()
    p = dict(max_iter=400, learning_rate=0.05, max_leaf_nodes=31, l2_regularization=1.0,
             min_samples_leaf=200, categorical_features=ci, early_stopping=False, random_state=0)
    p.update(kw)
    m = HistGradientBoostingClassifier(**p).fit(Xtr, ytr, sample_weight=weights)
    pr = m.predict_proba(Xva)[:, 1]
    base = score(pr, yva)
    shift_trend = score(np.clip(pr + (pred_rate - pr.mean()), 1e-6, 1 - 1e-6), yva)
    shift_oracle = score(np.clip(pr + (yva.mean() - pr.mean()), 1e-6, 1 - 1e-6), yva)
    print(f"{tag:34s} raw={base:7.1f}  +추세보정={shift_trend:7.1f}  "
          f"(+오라클={shift_oracle:7.1f})  mean={pr.mean():.4f}  {round(time.time()-t)}s")

run("HGB 기본")
for half in [2, 3]:
    w = 0.5 ** ((2023 - tr.season.values) / half)
    run(f"HGB 최근가중(반감기 {half}시즌)", weights=w)
run("HGB max_iter=800 lr=.03", max_iter=800, learning_rate=0.03)
run("HGB leaf=63", max_leaf_nodes=63)
