"""학습과 추론이 반드시 공유해야 하는 전처리.

범주형은 데이터에 등장한 값으로 코드를 매기면 test.csv(5행 샘플)에서
train과 다른 코드가 붙는다. 그래서 매핑을 아래에 고정한다.
"""

ID_COL = "row_id"
TARGET_COL = "control_success"

CAT_MAPS = {
    "top_bottom": {"B": 0, "T": 1},
    "game_type": {"F": 0, "R": 1},
    "base_state": {"___": 0, "1__": 1, "_2_": 2, "__3": 3,
                   "12_": 4, "1_3": 5, "_23": 6, "123": 7},
}
CAT_COLS = list(CAT_MAPS)


def build_features(df, columns=None):
    """모델 입력 프레임 생성. columns를 주면 그 순서로 맞춘다."""
    X = df.drop(columns=[c for c in (ID_COL, TARGET_COL) if c in df.columns]).copy()
    for col, mapping in CAT_MAPS.items():
        # 미지의 범주는 결측으로 두고 HistGBM의 결측 분기에 맡긴다
        X[col] = X[col].astype(str).map(mapping).astype("float64")
    if columns is not None:
        X = X[list(columns)]
    return X
