"""학습과 추론이 반드시 공유해야 하는 전처리.

두 가지 원칙:
1) 범주형은 데이터에 등장한 값으로 코드를 매기면 test.csv(5행 샘플)에서 train과
   다른 코드가 붙는다. 그래서 매핑을 아래에 고정한다.
2) 파생 피처는 전부 '그 행 하나만 보고' 계산된다. 평가 데이터의 다른 행이나
   전체 분포를 쓰지 않으므로 대회의 행 독립 추론 규칙에 안전하다.
   리그 평균(priors)은 학습 데이터에서 계산해 모델과 함께 저장한다.
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

# 홀드아웃 ablation 결과: 카운트 + 상황 + 최근폼 조합이 최선(767.6).
# asof 스무딩(shrink)과 cold-start 플래그는 원본 asof_* 컬럼과 축이 겹쳐
# 오히려 점수를 깎아서 제외했다.


def compute_priors(df):
    """리그 평균 — 반드시 학습 데이터에서만 계산해 모델에 저장한다."""
    return {
        "success": float(df["asof_pitcher_success_rate"].mean()),
        "middle": float(df["asof_pitcher_middle_rate"].mean()),
    }


def build_features(df, priors, columns=None):
    import numpy as np

    X = df.drop(columns=[c for c in (ID_COL, TARGET_COL) if c in df.columns]).copy()
    for col, mapping in CAT_MAPS.items():
        # 미지의 범주는 결측으로 두고 HistGBM의 결측 분기에 맡긴다
        X[col] = X[col].astype(str).map(mapping).astype("float64")

    # --- 볼카운트 ---
    b, s = X["balls_before"], X["strikes_before"]
    X["count_state"] = (b * 3 + s).astype("float64")
    X["count_diff"] = (s - b).astype("float64")          # +면 투수 유리
    X["two_strikes"] = (s == 2).astype("float64")
    X["three_balls"] = (b == 3).astype("float64")
    X["full_count"] = ((b == 3) & (s == 2)).astype("float64")
    X["first_pitch"] = ((b == 0) & (s == 0)).astype("float64")
    X["pitcher_ahead"] = (s > b).astype("float64")
    X["must_strike"] = ((b == 3) & (s < 2)).astype("float64")   # 스트라이크를 넣어야 하는 상황
    X["can_waste"] = ((s == 2) & (b < 2)).astype("float64")     # 유인구 던질 여유

    # --- 경기 상황 ---
    X["same_hand"] = (X["pitcher_hand"] == X["batter_hand"]).astype("float64")
    X["risp"] = ((X["runner_on_2b"] == 1) | (X["runner_on_3b"] == 1)).astype("float64")
    X["bases_loaded"] = (X["num_runners_on"] == 3).astype("float64")
    X["close_game"] = (X["score_diff_pitcher_team"].abs() <= 1).astype("float64")
    X["blowout"] = (X["score_diff_pitcher_team"].abs() >= 5).astype("float64")
    X["late_inning"] = (X["inning"] >= 7).astype("float64")
    X["log_li"] = np.log1p(X["li"])

    # --- 최근 폼 (통산 대비 편차) ---
    base_s = X["asof_pitcher_success_rate"].fillna(priors["success"])
    base_m = X["asof_pitcher_middle_rate"].fillna(priors["middle"])
    for w in (1, 3, 5):
        X[f"form{w}_success"] = X[f"asof_pitcher_prev{w}_game_success_rate"] - base_s
        X[f"form{w}_middle"] = X[f"asof_pitcher_prev{w}_game_middle_rate"] - base_m
    X["form_trend"] = (X["asof_pitcher_prev1_game_success_rate"]
                       - X["asof_pitcher_prev5_game_success_rate"])

    if columns is not None:
        X = X[list(columns)]
    return X
