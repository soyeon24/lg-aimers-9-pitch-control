"""Candidate features on top of the production builder, two seeds each.

Seed noise on this holdout is about +/-4 (three seeds of the same config landed
at 825.6 / 832.6 / 830.8), so single-seed gaps under ~8 mean nothing. Two seeds
halves that; only accept a candidate that clears roughly +8.
"""
import time

import harness
import lab_v6

t0 = time.time()
SEEDS = (0, 1)

harness.bench(lab_v6.base, "base (production fe)", seasons=(2024,), seeds=SEEDS,
              ctx_fn=lab_v6.ctx_fn)
for tag, kw in [
    ("+ park id", dict(park=True)),
    ("+ pitcher-batter contrast", dict(contrast=True)),
    ("+ multi-K shrinkage", dict(multik=True)),
    ("+ empirical-Bayes to prev season", dict(eb=True)),
    ("+ prev-game windows", dict(prevgames=True)),
    ("+ usage / workload", dict(usage=True)),
]:
    harness.bench(lab_v6.make(**kw), tag, seasons=(2024,), seeds=SEEDS,
                  ctx_fn=lab_v6.ctx_fn)

print(f"\nTOTAL {time.time() - t0:.0f}s")
