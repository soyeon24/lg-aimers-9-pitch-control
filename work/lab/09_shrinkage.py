"""Shrinkage strength and window width for the season-to-date block."""
import time

import harness
import lab_v7

t0 = time.time()
SEEDS = (0, 1)

harness.bench(lab_v7.make(), "base K_PIT=200 K_BAT=300", seasons=(2024,),
              seeds=SEEDS, ctx_fn=lab_v7.ctx_fn)
for tag, kw in [
    ("K_PIT=60", dict(k_pit=60)),
    ("K_PIT=600", dict(k_pit=600)),
    ("K_BAT=100", dict(k_bat=100)),
    ("K_BAT=1000", dict(k_bat=1000)),
    ("+ two-season window", dict(window2=True)),
]:
    harness.bench(lab_v7.make(**kw), tag, seasons=(2024,), seeds=SEEDS,
                  ctx_fn=lab_v7.ctx_fn)

print(f"\nTOTAL {time.time() - t0:.0f}s")
