import os
import numpy as np, pandas as pd
pd.set_option("display.width", 250)
HERE = os.path.dirname(os.path.abspath(__file__))
df = pd.read_parquet(os.path.join(HERE, "cache", "train.parquet"))

print("=== rate by season x game_type ===")
piv = df.pivot_table(index="season", columns="game_type", values="control_success", aggfunc=["size","mean"])
print(piv.round(4).to_string())
sh = df.groupby("season").game_type.apply(lambda s:(s=="F").mean())
print("\nF share per season:", sh.round(4).to_dict())

print("\n=== F rate by season x month ===")
F = df[df.game_type=="F"]
print(F.pivot_table(index="game_month", columns="season", values="control_success", aggfunc="mean").round(3).to_string())
print("\n=== F count by season x month ===")
print(F.pivot_table(index="game_month", columns="season", values="control_success", aggfunc="size").to_string())

print("\n=== R rate by season x month ===")
R = df[df.game_type=="R"]
print(R.pivot_table(index="game_month", columns="season", values="control_success", aggfunc="mean").round(4).to_string())

print("\n=== what distinguishes F? ===")
for c in ["li","home_win_expectancy","asof_pitcher_n","asof_pitcher_success_rate","inning","pitcher_id","asof_pitcher_middle_rate","asof_pitcher_ball_rate","asof_pitcher_strike_rate","asof_pitcher_reverse_rate"]:
    print(f"  {c:32s} F={F[c].mean():10.4f}  R={R[c].mean():10.4f}")
print("  pitchers only in F:", len(set(F.pitcher_id)-set(R.pitcher_id)), " only in R:", len(set(R.pitcher_id)-set(F.pitcher_id)), " both:", len(set(R.pitcher_id)&set(F.pitcher_id)))
print("  teams F:", sorted(F.pitcher_team_id.unique())[:20])
print("  teams R:", sorted(R.pitcher_team_id.unique())[:20])
print("\n  F rows per team x season:")
print(F.pivot_table(index="pitcher_team_id", columns="season", values="control_success", aggfunc="size").to_string())
print("\n  F rate per team x season:")
print(F.pivot_table(index="pitcher_team_id", columns="season", values="control_success", aggfunc="mean").round(3).to_string())
