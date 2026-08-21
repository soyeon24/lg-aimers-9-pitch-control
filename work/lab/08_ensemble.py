"""Ensemble size, capacity under averaging, and per-group calibration.

Going from one seed to two moved the 2024 holdout by about +12, which is much
larger than the seed-to-seed spread of a single fit. That says variance, not
bias, is what is left -- so the question is whether more capacity plus more
averaging beats less capacity, and how many seeds are worth paying for.
"""
import time

import harness
import lab_v6

t0 = time.time()

for n in (1, 2, 4):
    harness.bench(lab_v6.base, f"leaf15 x{n} seeds", seasons=(2024,),
                  seeds=tuple(range(n)), ctx_fn=lab_v6.ctx_fn)

for tag, p in [
    ("leaf31 l2=1 msl2000", dict(max_leaf_nodes=31, l2_regularization=1.0,
                                 min_samples_leaf=2000)),
    ("leaf31 l2=10 msl10000", dict(max_leaf_nodes=31)),
    ("leaf15 max_features=.6", dict(max_features=0.6)),
    ("leaf15 lr.03 iter700", dict(learning_rate=0.03, max_iter=700)),
]:
    harness.bench(lab_v6.base, tag + " x4 seeds", seasons=(2024,),
                  seeds=(0, 1, 2, 3), params=p, ctx_fn=lab_v6.ctx_fn)

print(f"\nTOTAL {time.time() - t0:.0f}s")
