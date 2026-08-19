"""제출 전 안전장치.

학습용 features.py와 제출용 script.py의 전처리는 손으로 두 군데를 맞춰야 해서
어긋나기 쉽다. 어긋나도 에러 없이 '조용히 틀린 예측'이 나오므로 반드시 확인한다.

    python verify_submit.py
"""
import importlib.util
import os
import sys

import joblib
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, HERE)

from features import build_features as train_build  # noqa: E402


def load_submit_module():
    path = os.path.join(ROOT, "submit", "script.py")
    spec = importlib.util.spec_from_file_location("submit_script", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    art = joblib.load(os.path.join(ROOT, "submit", "model", "model.joblib"))
    columns, priors = art["columns"], art["priors"]
    submit = load_submit_module()

    df = pd.read_csv(os.path.join(ROOT, "open", "data", "train.csv"),
                     encoding="utf-8-sig", nrows=20000)
    df = df.drop(columns=["control_success"])

    a = train_build(df, priors, columns)
    b = submit.build_features(df, priors, columns)

    assert list(a.columns) == list(b.columns) == list(columns), "컬럼 순서 불일치"
    av, bv = a.to_numpy(dtype="float64"), b.to_numpy(dtype="float64")
    same = np.isclose(av, bv, equal_nan=True)
    if not same.all():
        bad = [columns[i] for i in np.unique(np.where(~same)[1])]
        raise SystemExit(f"[FAIL] 전처리 불일치 컬럼: {bad}")
    print(f"[OK] 전처리 일치 - {len(columns)}개 피처 × {len(df)}행 완전 동일")

    w = art["blend_w"]
    pa = np.mean([m.predict_proba(a)[:, 1] for m in art["models_a"]], axis=0)
    pb = np.mean([m.predict_proba(a)[:, 1] for m in art["models_b"]], axis=0)
    p = (1 - w) * pa + w * pb
    z = np.log(p / (1 - p)) + art["logit_offset"]
    p = 1 / (1 + np.exp(-z))
    print(f"[OK] 예측 정상 - mean={p.mean():.4f} min={p.min():.4f} max={p.max():.4f}")
    print(f"   (참고: 2025 외삽 성공률 {art['target_rate']:.4f}, "
          f"위 표본은 2019 시즌이라 더 높게 나오는 게 정상)")


if __name__ == "__main__":
    main()
