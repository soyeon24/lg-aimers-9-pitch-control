"""Can pitcher_id be matched to pitcher_trackman_id at all?

The README says the two id spaces are disjoint, which is true, but that only
rules out a direct join. If the two files cover the same games, a pitcher's
(season, team, hand, pitch count) signature might identify them. This probe
just measures how well the two sides line up before anyone invests in matching.
"""
import os

import numpy as np
import pandas as pd

import common

HERE = os.path.dirname(os.path.abspath(__file__))
TM = os.path.join(HERE, "..", "..", "open", "data", "trackman_history.csv")

cols = ["season", "game_date", "game_month", "game_dayofweek", "trackman_game_id",
        "pitcher_trackman_id", "pitcher_hand", "pitcher_team", "batter_team",
        "inning", "top_bottom", "balls_before", "strikes_before", "outs_before",
        "pitch_type_group", "rel_speed", "rel_height", "rel_side", "extension"]
tm = pd.read_csv(TM, usecols=cols, encoding="utf-8-sig")
print("trackman", tm.shape)
df = common.load()

print("\nrows per season")
print(pd.DataFrame({"main": df.groupby("season").size(),
                    "trackman": tm.groupby("season").size()}).to_string())

print("\nteam codes  main:", sorted(df.pitcher_team_id.unique()))
print("            trackman:", sorted(tm.pitcher_team.unique())[:30])
print("hand codes  main:", sorted(df.pitcher_hand.unique()),
      " trackman:", sorted(tm.pitcher_hand.unique())[:10])

print("\ndistinct pitchers per season   main / trackman")
print(pd.DataFrame({"main": df.groupby("season").pitcher_id.nunique(),
                    "trackman": tm.groupby("season").pitcher_trackman_id.nunique()}).to_string())

same = set(df.pitcher_team_id.astype(str)) & set(tm.pitcher_team.astype(str))
print("\nteam code overlap:", len(same), sorted(same)[:20])

# do per (season, team) pitch counts line up?
a = df.groupby(["season", "pitcher_team_id"]).size().rename("main")
b = tm.groupby(["season", "pitcher_team"]).size().rename("tm")
j = pd.concat([a, b], axis=1)
print("\nper (season, team) row counts, first 20:")
print(j.head(20).to_string())

# pitch-count signature per pitcher, one season, one team
S, T = 2024, 13
ma = df[(df.season == S) & (df.pitcher_team_id == T)].groupby(
    ["pitcher_id", "pitcher_hand"]).size().sort_values(ascending=False)
tb = tm[(tm.season == S) & (tm.pitcher_team.astype(str) == str(T))].groupby(
    ["pitcher_trackman_id", "pitcher_hand"]).size().sort_values(ascending=False)
print(f"\nseason {S} team {T}: main {len(ma)} pitchers / trackman {len(tb)} pitchers")
print("main top15   ", ma.head(15).values)
print("trackman top15", tb.head(15).values)
