"""확률의 '기울기'까지 보정하면 이득이 있는가.

지금은 logit 절편(평균)만 맞추고 있다. 모델이 과신(예측이 너무 퍼짐)이라면
기울기 a<1로 눌러야 Brier가 준다. Brier는 캘리브레이션에 극도로 민감하므로
이 축이 남아 있다면 값이 크다.

    z' = a * (logit(p) - logit(ref)) + logit(target)

a는 학습 데이터 홀드아웃(2024)에서 고르므로 평가 데이터를 보지 않는다.
2023 홀드아웃에서도 최적 a의 방향이 같은지 함께 확인한다.
"""
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from evaluate import CAT_MAPS, PARAMS, comp_score, extrapolate, load_train, logit
from features import build_features

SEEDS = (0, 1)
ALPHAS = (1.0, 0.95, 0.9, 0.85, 0.8, 0.7, 0.6)

df = load_train()


def evaluate(val_season):
    t = time.time()
    tr, va = df[df.season < val_season], df[df.season == val_season]
    ytr, yva = tr.control_success.values, va.control_success.values
    priors = {"success": float(tr["asof_pitcher_success_rate"].mean()),
              "middle": float(tr["asof_pitcher_middle_rate"].mean())}
    target = extrapolate(tr.groupby("season").control_success.mean().values)

    Xtr, Xva = build_features(tr, priors), build_features(va, priors)
    Xva = Xva[list(Xtr.columns)]
    ci = [Xtr.columns.get_loc(c) for c in CAT_MAPS]
    ms = [HistGradientBoostingClassifier(categorical_features=ci, random_state=s,
                                         **PARAMS).fit(Xtr, ytr) for s in SEEDS]
    pv = np.mean([m.predict_proba(Xva)[:, 1] for m in ms], axis=0)
    last = (tr.season == val_season - 1).values
    ref = np.mean([m.predict_proba(Xtr[last])[:, 1] for m in ms], axis=0).mean()

    print(f"\n=== val {val_season} (실제 {yva.mean():.4f}, 외삽 {target:.4f}) "
          f"[{time.time()-t:.0f}s] ===")
    zc, zt = logit(pv) - logit(ref), logit(target)
    for a in ALPHAS:
        p = 1 / (1 + np.exp(-(a * zc + zt)))
        print(f"  a={a:<5} score={comp_score(p, yva):7.1f}  mean={p.mean():.4f}  "
              f"sd={p.std():.4f}", flush=True)


evaluate(2024)
evaluate(2023)
