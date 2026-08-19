"""아이디어를 두 개의 시즌 홀드아웃에서 동시에 재는 하네스.

2024 한 해만 보고 판단했더니 홀드아웃 +55.8이 리더보드 +11.4로만 옮겨졌다.
측정 노이즈에 속지 않으려면 두 해에서 모두 개선되는 것만 채택해야 한다.

  from evaluate import dual_holdout
  dual_holdout(my_build_fn, "내 아이디어")
"""
import os
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

_HERE = os.path.dirname(os.path.abspath(__file__))
TRAIN_CSV = os.path.join(_HERE, "..", "open", "data", "train.csv")
TARGET = "control_success"
CAT_MAPS = {
    "top_bottom": {"B": 0, "T": 1},
    "game_type": {"F": 0, "R": 1},
    "base_state": {"___": 0, "1__": 1, "_2_": 2, "__3": 3,
                   "12_": 4, "1_3": 5, "_23": 6, "123": 7},
}
PARAMS = dict(max_iter=400, learning_rate=0.05, max_leaf_nodes=15,
              l2_regularization=10.0, min_samples_leaf=10000, early_stopping=False)
EPS = 1e-6
VAL_SEASONS = (2023, 2024)

_CACHE = {}


def load_train():
    if "df" not in _CACHE:
        t = time.time()
        _CACHE["df"] = pd.read_csv(TRAIN_CSV, encoding="utf-8-sig")
        print(f"train.csv 로드 {_CACHE['df'].shape} ({time.time() - t:.0f}s)", flush=True)
    return _CACHE["df"]


def base_frame(d):
    """공통 시작점 — id/target 제거 + 범주형 고정 매핑."""
    X = d.drop(columns=[c for c in ("row_id", TARGET) if c in d.columns]).copy()
    for col, m in CAT_MAPS.items():
        X[col] = X[col].astype(str).map(m).astype("float64")
    return X


def comp_score(p, y):
    r = y.mean()
    return max(0.0, 100000 * (1 - np.mean((p - y) ** 2) / (r * (1 - r))))


def logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def extrapolate(rates, ahead=1):
    y = np.asarray(rates, dtype=float)
    x = np.arange(len(y), dtype=float)
    slope, intercept = np.polyfit(x, y, 1)
    return float(intercept + slope * (len(y) - 1 + ahead))


def run_split(build_fn, val_season, seeds, params=None):
    """build_fn(df, priors) -> 피처 프레임. priors는 학습 데이터에서만 계산된다."""
    df = load_train()
    tr, va = df[df.season < val_season], df[df.season == val_season]
    ytr, yva = tr[TARGET].values, va[TARGET].values

    priors = {"success": float(tr["asof_pitcher_success_rate"].mean()),
              "middle": float(tr["asof_pitcher_middle_rate"].mean())}
    rates = tr.groupby("season")[TARGET].mean().values
    target_rate = extrapolate(rates)

    Xtr, Xva = build_fn(tr, priors), build_fn(va, priors)
    Xva = Xva[list(Xtr.columns)]
    ci = [Xtr.columns.get_loc(c) for c in CAT_MAPS]
    p = dict(PARAMS)
    if params:
        p.update(params)

    models = [HistGradientBoostingClassifier(categorical_features=ci, random_state=s,
                                             **p).fit(Xtr, ytr) for s in seeds]
    pv = np.mean([m.predict_proba(Xva)[:, 1] for m in models], axis=0)

    last = (tr.season == val_season - 1).values
    ref = np.mean([m.predict_proba(Xtr[last])[:, 1] for m in models], axis=0).mean()
    adj = 1 / (1 + np.exp(-(logit(pv) + logit(target_rate) - logit(ref))))
    return comp_score(adj, yva), Xtr.shape[1]


def dual_holdout(build_fn, tag, seeds=(0, 1), params=None):
    t = time.time()
    scores = []
    for vs in VAL_SEASONS:
        s, nfeat = run_split(build_fn, vs, seeds, params)
        scores.append(s)
    print(f"{tag:36s} feat={nfeat:3d}  "
          + "  ".join(f"{vs}={s:7.1f}" for vs, s in zip(VAL_SEASONS, scores))
          + f"  평균={np.mean(scores):7.1f}  ({time.time() - t:.0f}s)", flush=True)
    return scores
