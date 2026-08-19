"""구 레짐 데이터가 도움이 되는가, 해가 되는가.

2022→2023 사이에 구조적 단절이 있다(2019~2022 학습 모델이 2023에서 상수보다 나쁨).
그렇다면 2025를 맞출 때 2019~2022 데이터가 오히려 방해일 수 있다.
학습 구간만 바꿔가며 2024 홀드아웃으로 확인한다.
"""
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from evaluate import CAT_MAPS, PARAMS, comp_score, extrapolate, load_train, logit
from features import build_features

VAL = 2024
SEEDS = (0, 1)

df = load_train()
va = df[df.season == VAL]
yva = va.control_success.values
full = df[df.season < VAL]

# 오프셋용 시즌 추세는 항상 전체 이력에서 뽑는다 (학습 구간과 별개 문제)
target = extrapolate(full.groupby("season").control_success.mean().values)
print(f"외삽 target={target:.4f}  실제 2024={yva.mean():.4f}\n")


def run(first_season):
    t = time.time()
    tr = full[full.season >= first_season]
    priors = {"success": float(tr["asof_pitcher_success_rate"].mean()),
              "middle": float(tr["asof_pitcher_middle_rate"].mean())}
    Xtr, Xva = build_features(tr, priors), build_features(va, priors)
    Xva = Xva[list(Xtr.columns)]
    ci = [Xtr.columns.get_loc(c) for c in CAT_MAPS]
    ms = [HistGradientBoostingClassifier(categorical_features=ci, random_state=s,
                                         **PARAMS).fit(Xtr, tr.control_success.values)
          for s in SEEDS]
    pv = np.mean([m.predict_proba(Xva)[:, 1] for m in ms], axis=0)
    last = (tr.season == VAL - 1).values
    ref = np.mean([m.predict_proba(Xtr[last])[:, 1] for m in ms], axis=0).mean()
    adj = 1 / (1 + np.exp(-(logit(pv) + logit(target) - logit(ref))))
    print(f"학습 {first_season}~{VAL-1}  n={len(tr):>9,}  "
          f"raw={comp_score(pv,yva):7.1f}  +offset={comp_score(adj,yva):7.1f}  "
          f"({time.time()-t:.0f}s)", flush=True)


for fs in (2019, 2021, 2022, 2023):
    run(fs)
