import os
import numpy as np, pandas as pd
pd.set_option("display.width", 200)
HERE = os.path.dirname(os.path.abspath(__file__))
df = pd.read_parquet(os.path.join(HERE, "cache", "train.parquet"))

print("=== row_id 샘플 ===")
print(df.row_id.head(5).tolist(), df.row_id.tail(3).tolist())
print("row_id 유일?", df.row_id.is_unique, "정렬됨?", df.row_id.is_monotonic_increasing)

print("\n=== 시즌별 ===")
g = df.groupby("season").agg(
    n=("control_success","size"), rate=("control_success","mean"),
    asof_succ=("asof_pitcher_success_rate","mean"),
    asof_mid=("asof_pitcher_middle_rate","mean"),
    asof_ball=("asof_pitcher_ball_rate","mean"),
    asof_strike=("asof_pitcher_strike_rate","mean"),
    asof_rev=("asof_pitcher_reverse_rate","mean"),
    asof_n=("asof_pitcher_n","mean"),
    asof_n_med=("asof_pitcher_n","median"),
    bat_succ=("asof_batter_success_rate","mean"),
    fb=("asof_pitcher_fastball_rate","mean"),
    li=("li","mean"),
    npitcher=("pitcher_id","nunique"),
)
print(g.round(4).to_string())

print("\n=== 시즌x월 성공률 ===")
print(df.pivot_table(index="game_month", columns="season", values="control_success",
                     aggfunc="mean").round(4).to_string())
print("\n=== 시즌x월 행수 ===")
print(df.pivot_table(index="game_month", columns="season", values="control_success",
                     aggfunc="size").to_string())

print("\n=== game_type ===")
print(df.groupby(["season","game_type"]).control_success.agg(["size","mean"]).round(4).to_string())

print("\n=== asof_pitcher_n 이 시즌마다 리셋되는가 ===")
# 각 투수의 시즌별 asof_pitcher_n 최소값
sub = df[df.pitcher_id.isin(df.pitcher_id.value_counts().head(3).index)]
print(sub.groupby(["pitcher_id","season"]).asof_pitcher_n.agg(["min","max","size"]).to_string())

print("\n=== 결측 ===")
na = df.isna().mean()
print(na[na>0].round(4).to_string())
