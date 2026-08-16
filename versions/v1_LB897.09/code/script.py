"""평가 서버 추론 스크립트.

  submit.zip/
    model/model.joblib   학습 결과 (HistGBM 시드 앙상블 + 고정 보정 상수)
    script.py            이 파일
    requirements.txt

평가 서버가 ./data 와 ./output 을 붙여준다. 각 행을 독립적으로 예측하며,
평가 데이터의 통계는 어떤 형태로도 사용하지 않는다. 시즌 간 성공률 하락에
대한 보정은 학습 데이터만으로 정해진 고정 상수(logit_offset)다.
"""
import os
import time

import joblib
import numpy as np
import pandas as pd

ID_COL = "row_id"
TARGET_COL = "control_success"
EPS = 1e-6

# 학습 때와 동일하게 고정된 범주형 매핑 (데이터에 따라 코드가 흔들리면 안 된다)
CAT_MAPS = {
    "top_bottom": {"B": 0, "T": 1},
    "game_type": {"F": 0, "R": 1},
    "base_state": {"___": 0, "1__": 1, "_2_": 2, "__3": 3,
                   "12_": 4, "1_3": 5, "_23": 6, "123": 7},
}

DATA_DIR = "./data"
MODEL_PATH = "./model/model.joblib"
OUT_PATH = "./output/submission.csv"


def build_features(df, columns):
    X = df.drop(columns=[c for c in (ID_COL, TARGET_COL) if c in df.columns]).copy()
    for col, mapping in CAT_MAPS.items():
        X[col] = X[col].astype(str).map(mapping).astype("float64")
    missing = [c for c in columns if c not in X.columns]
    if missing:
        raise ValueError(f"test 데이터에 없는 학습 피처: {missing}")
    return X[list(columns)]


def main():
    t0 = time.time()

    art = joblib.load(MODEL_PATH)
    models, columns = art["models"], art["columns"]
    offset = art["logit_offset"]
    print(f"모델 로드: {len(models)}개 시드, 피처 {len(columns)}개, "
          f"logit_offset={offset:+.4f} ({time.time() - t0:.1f}s)")

    test = pd.read_csv(os.path.join(DATA_DIR, "test.csv"), encoding="utf-8-sig")
    sub = pd.read_csv(os.path.join(DATA_DIR, "sample_submission.csv"), encoding="utf-8-sig")
    print(f"test={len(test)} submission={len(sub)} ({time.time() - t0:.1f}s)")

    X = build_features(test, columns)
    p = np.mean([m.predict_proba(X)[:, 1] for m in models], axis=0)

    # 시즌 드리프트 보정 — 행마다 동일하게 적용되는 상수 오프셋
    z = np.log(np.clip(p, EPS, 1 - EPS) / (1 - np.clip(p, EPS, 1 - EPS))) + offset
    p = 1.0 / (1.0 + np.exp(-z))
    p = np.clip(p, EPS, 1 - EPS)
    print(f"추론 완료 mean={p.mean():.4f} min={p.min():.4f} max={p.max():.4f} "
          f"({time.time() - t0:.1f}s)")

    pred = pd.Series(p, index=test[ID_COL].values)
    out = pd.DataFrame({ID_COL: sub[ID_COL]})
    out[TARGET_COL] = out[ID_COL].map(pred)
    n_missing = int(out[TARGET_COL].isna().sum())
    if n_missing:
        print(f"경고: 예측 없는 row_id {n_missing}건 → 학습 성공률로 대체")
        out[TARGET_COL] = out[TARGET_COL].fillna(art["target_rate"])

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    out.to_csv(OUT_PATH, index=False, encoding="utf-8")
    print(f"저장: {OUT_PATH} rows={len(out)} (총 {time.time() - t0:.1f}s)")


if __name__ == "__main__":
    main()
