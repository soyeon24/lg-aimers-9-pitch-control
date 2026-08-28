"""Is TrackMan matchable at all? A feasibility probe, not an implementation.

The two id spaces are disjoint and the team codes differ too (main data uses
12-25, TrackMan uses real names like KIA_TIG plus MIN_* for the second tier), so
nothing joins directly. But both files describe the same league, and a team's
schedule is close to a fingerprint: the multiset of (month, dayofweek) it played
in a season is unlikely to be shared by two teams.

If teams can be identified this way, pitchers could then be matched within
(season, team, hand) by pitch count, and their physical traits -- velocity,
spin, extension, and especially release-point consistency, which is the closest
thing to a direct command measurement -- could become features.

This only measures whether the fingerprints are distinctive enough to bother.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import common

HERE = os.path.dirname(os.path.abspath(__file__))
TM = os.path.join(HERE, "..", "..", "open", "data", "trackman_history.csv")

df = common.load()
tm = pd.read_csv(TM, usecols=["season", "game_date", "game_month", "game_dayofweek",
                              "trackman_game_id", "pitcher_trackman_id",
                              "pitcher_hand", "pitcher_team"], encoding="utf-8-sig")
print("trackman", tm.shape, flush=True)

tm_teams = sorted(tm.pitcher_team.unique())
first = [t for t in tm_teams if not t.startswith(("MIN_", "KBO_", "ACE_"))]
print(f"\nfirst-tier trackman teams ({len(first)}): {first}")
print(f"second-tier ({len(tm_teams) - len(first)}): "
      f"{[t for t in tm_teams if t not in first]}")

R = df[df.game_type == "R"]
main_teams = sorted(R.pitcher_team_id.unique())
print(f"\nmain R team ids ({len(main_teams)}): {main_teams}")


def fingerprint(g, month_col, dow_col):
    v = np.zeros(12 * 7)
    idx = (g[month_col].to_numpy() - 1) * 7 + g[dow_col].to_numpy()
    np.add.at(v, idx, 1)
    return v / max(v.sum(), 1)


def match_season(season):
    a = {t: fingerprint(R[(R.season == season) & (R.pitcher_team_id == t)],
                        "game_month", "game_dayofweek") for t in main_teams}
    sub = tm[(tm.season == season) & (tm.pitcher_team.isin(first))]
    b = {t: fingerprint(sub[sub.pitcher_team == t], "game_month", "game_dayofweek")
         for t in first}
    A = np.array([a[t] for t in main_teams])
    B = np.array([b[t] for t in first])
    # cosine similarity between schedule fingerprints
    A = A / np.linalg.norm(A, axis=1, keepdims=True)
    B = B / np.linalg.norm(B, axis=1, keepdims=True)
    return A @ B.T


for season in (2023, 2024):
    S = match_season(season)
    print(f"\n=== season {season}: schedule fingerprint similarity ===")
    order = np.argsort(-S, axis=1)
    for i, t in enumerate(main_teams):
        top = order[i][:2]
        gap = S[i, top[0]] - S[i, top[1]]
        print(f"  team {t}: best={first[top[0]]} ({S[i, top[0]]:.4f})  "
              f"2nd={first[top[1]]} ({S[i, top[1]]:.4f})  gap={gap:+.4f}")
    # a fingerprint is only useful if the best match is clearly ahead
    gaps = np.sort(S, axis=1)[:, -1] - np.sort(S, axis=1)[:, -2]
    print(f"  median gap {np.median(gaps):.4f}, min gap {gaps.min():.4f}")
    uniq = len(set(order[:, 0]))
    print(f"  distinct best matches: {uniq}/{len(main_teams)} "
          f"({'one-to-one' if uniq == len(main_teams) else 'COLLISIONS -- not identifying'})")
