"""One sweep over the four open questions.

A  offset scale + beta across four holdout seasons
B  how big is seed noise, so we know which gaps below are real
C  which season-to-date blocks to keep
D  capacity, now that the features carry much more signal
E  the old game_type F regime
"""
import time

import feats4
import feats5
import harness

t0 = time.time()

print("===== A. offset scale / beta, four seasons =====", flush=True)
harness.bench(feats4.build_v4, "v4 all", seasons=(2021, 2022, 2023, 2024))

print("\n===== B. seed noise on 2024 =====", flush=True)
for sd in (1, 2):
    harness.bench(feats4.build_v4, f"v4 all seed={sd}", seasons=(2024,), seeds=(sd,))

print("\n===== C. feature blocks (2024) =====", flush=True)
COMBOS = [
    ("pitcher only", dict(batter=False, mix=False, prev_season=False)),
    ("pitcher+batter", dict(batter=True, mix=False, prev_season=False)),
    ("pitcher+batter+mix", dict(batter=True, mix=True, prev_season=False)),
    ("pitcher+batter+prev", dict(batter=True, mix=False, prev_season=True)),
]
for tag, kw in COMBOS:
    harness.bench(feats5.make(**kw), tag, seasons=(2024,))
harness.bench(feats5.make(batter=True, mix=False, prev_season=False, form_vs_std=True),
              "pitcher+batter +form_vs_std", seasons=(2024,))
harness.bench(feats5.make(batter=True, mix=False, prev_season=True, drift=True),
              "pitcher+batter+prev +drift", seasons=(2024,))

print("\n===== D. capacity (2024, pitcher+batter) =====", flush=True)
PB = feats5.make(batter=True, mix=False, prev_season=False)
for tag, p in [
    ("leaf8", dict(max_leaf_nodes=8)),
    ("leaf15 iter200", dict(max_iter=200)),
    ("leaf15 msl30000", dict(min_samples_leaf=30000)),
    ("leaf15 lr.03 iter700", dict(learning_rate=0.03, max_iter=700)),
]:
    harness.bench(PB, tag, seasons=(2024,), params=p)

print("\n===== E. old F regime =====", flush=True)
harness.bench(feats5.make(batter=True, mix=False, prev_season=False, f_flag=True),
              "pitcher+batter +f_old flag", seasons=(2023, 2024))
harness.bench(PB, "pitcher+batter, drop 2019-2022 F from train", seasons=(2024,),
              train_filter=lambda d: ~((d.game_type == "F") & (d.season < 2023)))
harness.bench(feats5.make(batter=True, mix=False, prev_season=False, drop_season=True),
              "pitcher+batter, no `season` column", seasons=(2023, 2024))

print(f"\nTOTAL {time.time() - t0:.0f}s")
