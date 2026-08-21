"""Confirm the season-to-date jump on other holdout seasons, then ablate.

Order is deliberate: the confirmation runs first because a gain that shows up on
one season only is the exact trap that cost us on v2 (holdout +55.8 -> LB +11.4).
"""
import time

import common
import feats4

t0 = time.time()

common.bench(common.build_v3, "v3 baseline", val_seasons=(2022, 2023), seeds=(0,))
common.bench(feats4.build_v4, "v4 all", val_seasons=(2022, 2023), seeds=(0,))

print("\n===== ablation on 2024 =====", flush=True)
common.bench(feats4.make_builder(pitcher=True, batter=False, mix=False, prev_season=False),
             "v4 pitcher season-to-date only", val_seasons=(2024,), seeds=(0,))
common.bench(feats4.make_builder(pitcher=True, batter=False, mix=False, prev_season=True),
             "  + pitcher prev-season", val_seasons=(2024,), seeds=(0,))
common.bench(feats4.make_builder(pitcher=True, batter=True, mix=False, prev_season=True),
             "  + batter", val_seasons=(2024,), seeds=(0,))
common.bench(feats4.make_builder(pitcher=True, batter=True, mix=True, prev_season=True),
             "  + pitchmix (= v4 all)", val_seasons=(2024,), seeds=(0,))

print("\n===== capacity re-sweep with v4 features (2024) =====", flush=True)
for tag, p in [
    ("leaf31 msl2000 l2=1", dict(max_leaf_nodes=31, min_samples_leaf=2000, l2_regularization=1.0)),
    ("leaf31 msl10000 l2=10", dict(max_leaf_nodes=31, min_samples_leaf=10000, l2_regularization=10.0)),
    ("leaf15 msl10000 iter800", dict(max_iter=800)),
]:
    common.bench(feats4.build_v4, tag, val_seasons=(2024,), seeds=(0,), params=p)

print(f"\nTOTAL {time.time() - t0:.0f}s")
