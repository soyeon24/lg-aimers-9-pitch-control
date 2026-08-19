"""서로 다른 편향의 모델을 섞으면 Brier가 주는가.

절편·기울기 보정은 이미 최적점이라 남은 축은 예측 자체의 다양성뿐이다.
선형 모델(로지스틱)은 트리와 오차 구조가 크게 달라 블렌딩 이득이 기대된다.
용량이 다른 두 번째 HistGBM도 함께 잰다.
"""
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from evaluate import CAT_MAPS, PARAMS, comp_score, extrapolate, load_train, logit
from features import build_features

VAL = 2024
SEEDS = (0, 1)

df = load_train()
tr, va = df[df.season < VAL], df[df.season == VAL]
ytr, yva = tr.control_success.values, va.control_success.values
priors = {"success": float(tr["asof_pitcher_success_rate"].mean()),
          "middle": float(tr["asof_pitcher_middle_rate"].mean())}
target = extrapolate(tr.groupby("season").control_success.mean().values)
Xtr = build_features(tr, priors)
Xva = build_features(va, priors)[list(Xtr.columns)]
cat = list(CAT_MAPS)
ci = [Xtr.columns.get_loc(c) for c in cat]
last = (tr.season == VAL - 1).values
print(f"X={Xtr.shape}  target={target:.4f}  실제={yva.mean():.4f}\n", flush=True)


def score_with_offset(ptr_last, pva, tag):
    ref = ptr_last.mean()
    adj = 1 / (1 + np.exp(-(logit(pva) - logit(ref) + logit(target))))
    s = comp_score(adj, yva)
    print(f"{tag:34s} score={s:7.1f}  mean={adj.mean():.4f}", flush=True)
    return s


def fit_hgb(tag, **kw):
    t = time.time()
    p = dict(PARAMS); p.update(kw)
    ms = [HistGradientBoostingClassifier(categorical_features=ci, random_state=s,
                                         **p).fit(Xtr, ytr) for s in SEEDS]
    pv = np.mean([m.predict_proba(Xva)[:, 1] for m in ms], axis=0)
    pl = np.mean([m.predict_proba(Xtr[last])[:, 1] for m in ms], axis=0)
    print(f"  [{tag} 학습 {time.time()-t:.0f}s]", flush=True)
    return pv, pl


def fit_lr():
    t = time.time()
    num = [c for c in Xtr.columns if c not in cat]
    pipe = Pipeline([
        ("prep", ColumnTransformer([
            ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                              ("sc", StandardScaler())]), num),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat)])),
        ("clf", LogisticRegression(max_iter=300, C=1.0)),
    ])
    pipe.fit(Xtr, ytr)
    pv = pipe.predict_proba(Xva)[:, 1]
    pl = pipe.predict_proba(Xtr[last])[:, 1]
    print(f"  [로지스틱 학습 {time.time()-t:.0f}s]", flush=True)
    return pv, pl


hgb_v, hgb_l = fit_hgb("HGB leaf15")
score_with_offset(hgb_l, hgb_v, "HGB 단독 (현재 v2)")

lr_v, lr_l = fit_lr()
score_with_offset(lr_l, lr_v, "로지스틱 단독")

print()
for w in (0.1, 0.2, 0.3, 0.4):
    score_with_offset((1 - w) * hgb_l + w * lr_l, (1 - w) * hgb_v + w * lr_v,
                      f"HGB + 로지스틱 w={w}")

print()
hgb2_v, hgb2_l = fit_hgb("HGB leaf31", max_leaf_nodes=31, min_samples_leaf=2000,
                         l2_regularization=1.0)
score_with_offset(hgb2_l, hgb2_v, "HGB leaf31 단독")
for w in (0.3, 0.5):
    score_with_offset((1 - w) * hgb_l + w * hgb2_l, (1 - w) * hgb_v + w * hgb2_v,
                      f"HGB15 + HGB31 w={w}")
