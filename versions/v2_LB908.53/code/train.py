"""제구 성공 확률 모델 학습.

train.csv는 2019~2024, 평가 데이터는 2025다. 시즌마다 제구 성공률이
꾸준히 떨어지고 있어서, 모델을 그대로 쓰면 평균이 높게 나온다.
그래서 시즌별 성공률에 선형추세를 적용해 2025 성공률을 외삽하고,
그 차이를 logit 공간의 **고정 상수**로 모델에 저장한다.
(평가 데이터 통계를 보고 만드는 보정이 아니라 학습 데이터만으로 정해지는
 값이므로 "행 독립 추론" 규칙을 지킨다.)

  python train.py --validate   # 2024 홀드아웃으로 절차 전체를 검증
  python train.py              # 전체 학습 후 submit/model/ 에 저장
"""
import argparse
import os
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from features import CAT_COLS, TARGET_COL, build_features, compute_priors

HERE = os.path.dirname(os.path.abspath(__file__))
TRAIN_CSV = os.path.join(HERE, "..", "open", "data", "train.csv")
MODEL_DIR = os.path.join(HERE, "..", "submit", "model")

PARAMS = dict(max_iter=400, learning_rate=0.05, max_leaf_nodes=15,
              l2_regularization=10.0, min_samples_leaf=10000,
              early_stopping=False)
SEEDS = [0, 1, 2, 3]
EPS = 1e-6


def logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def comp_score(p, y):
    """대회 평가식. r은 평가 데이터의 실제 평균 성공률."""
    r = y.mean()
    return max(0.0, 100000 * (1 - np.mean((p - y) ** 2) / (r * (1 - r))))


def extrapolate_rate(season_rates, ahead=1):
    """시즌별 성공률에 선형추세를 적용해 ahead 시즌 뒤를 외삽."""
    y = np.asarray(season_rates, dtype=float)
    x = np.arange(len(y), dtype=float)
    slope, intercept = np.polyfit(x, y, 1)
    return float(intercept + slope * (len(y) - 1 + ahead))


def fit_ensemble(X, y, cat_idx):
    models = []
    for seed in SEEDS:
        t = time.time()
        m = HistGradientBoostingClassifier(
            categorical_features=cat_idx, random_state=seed, **PARAMS)
        m.fit(X, y)
        models.append(m)
        print(f"  seed {seed} 학습 완료 ({time.time() - t:.0f}s)", flush=True)
    return models


def predict_raw(models, X):
    return np.mean([m.predict_proba(X)[:, 1] for m in models], axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate", action="store_true",
                    help="2019~2023 학습 / 2024 검증으로 절차 전체를 시뮬레이션")
    args = ap.parse_args()

    t0 = time.time()
    df = pd.read_csv(TRAIN_CSV, encoding="utf-8-sig")
    print(f"train.csv 로드 {df.shape} ({time.time() - t0:.0f}s)")

    if args.validate:
        tr, va = df[df.season <= 2023], df[df.season == 2024]
    else:
        tr, va = df, None

    seasons = sorted(tr.season.unique())
    last_season = seasons[-1]
    rates = tr.groupby("season")[TARGET_COL].mean().loc[seasons].values
    target_rate = extrapolate_rate(rates)
    print(f"시즌별 성공률 {dict(zip(seasons, rates.round(4)))}")
    print(f"→ {last_season + 1} 외삽 성공률 = {target_rate:.4f}")

    priors = compute_priors(tr)
    print("priors =", {k: round(v, 4) for k, v in priors.items()})
    Xtr = build_features(tr, priors)
    ytr = tr[TARGET_COL].values
    columns = list(Xtr.columns)
    cat_idx = [columns.index(c) for c in CAT_COLS]

    print(f"학습 시작 X={Xtr.shape}")
    models = fit_ensemble(Xtr, ytr, cat_idx)

    # 보정 상수: 마지막 시즌 행에 대한 모델 평균 예측을 기준점으로 삼고,
    # 외삽한 다음 시즌 성공률까지의 차이를 logit 오프셋으로 굳힌다.
    last_mask = (tr.season == last_season).values
    ref_mean = float(predict_raw(models, Xtr[last_mask]).mean())
    logit_offset = float(logit(target_rate) - logit(ref_mean))
    print(f"기준({last_season}) 예측 평균 = {ref_mean:.4f} → logit_offset = {logit_offset:+.4f}")

    artifact = dict(models=models, columns=columns, cat_idx=cat_idx,
                    logit_offset=logit_offset, target_rate=target_rate, priors=priors,
                    ref_mean=ref_mean, trained_seasons=seasons, params=PARAMS,
                    seeds=SEEDS)

    if args.validate:
        Xva = build_features(va, priors, columns)
        yva = va[TARGET_COL].values
        raw = predict_raw(models, Xva)
        adj = sigmoid(logit(raw) + logit_offset)
        oracle = sigmoid(logit(raw) + (logit(yva.mean()) - logit(raw.mean())))
        print("\n=== 2024 홀드아웃 ===")
        print(f"실제 성공률       = {yva.mean():.4f} (외삽값 {target_rate:.4f}, "
              f"오차 {target_rate - yva.mean():+.4f})")
        print(f"보정 없음         score={comp_score(raw, yva):8.1f}  mean={raw.mean():.4f}")
        print(f"고정 오프셋 보정  score={comp_score(adj, yva):8.1f}  mean={adj.mean():.4f}")
        print(f"오라클 보정(상한) score={comp_score(oracle, yva):8.1f}")
        return

    os.makedirs(MODEL_DIR, exist_ok=True)
    path = os.path.join(MODEL_DIR, "model.joblib")
    joblib.dump(artifact, path, compress=3)
    print(f"\n저장 완료: {path} ({os.path.getsize(path) / 1e6:.1f} MB)")
    print(f"총 {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
