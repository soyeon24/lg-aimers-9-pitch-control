"""How many nets is enough, and is seed diversity better than config diversity?

Individual nets are noisy -- solo scores across seeds ranged 704 to 777 in an
earlier run -- so averaging more of them should pay in a way that more tree
seeds no longer does (trees saturated at 4). The other question is whether
averaging *different* configurations (3, 4 and 6 epochs) beats averaging more
seeds of the single best one.

Each net costs about 25s with the trees cached, so this is the cheapest place
left to buy points.
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import lab_v6
import nnlib
from common import TARGET, comp_score, load, logit, sigmoid

VAL = 2024
EPS = 1e-6
CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache",
                     f"trees_{VAL}.npz")

df = load()
tr, va = df[df.season < VAL], df[df.season == VAL]
ytr, yva = tr[TARGET].to_numpy(), va[TARGET].to_numpy()
ctx = lab_v6.ctx_fn(tr)
Xtr = lab_v6.base(tr, ctx)
Xva = lab_v6.base(va, ctx)[list(Xtr.columns)]
z = np.load(CACHE)
ptree = 0.65 * z["p15"] + 0.35 * z["p31"]

prep = nnlib.fit_prep(Xtr)
Atr, Ava = nnlib.prep(Xtr, prep), nnlib.prep(Xva, prep)


def oracle(p):
    q = np.clip(p, EPS, 1 - EPS)
    return comp_score(sigmoid(logit(q) + logit(yva.mean()) - logit(q.mean())), yva)


def blend(pnn, w=0.35):
    return (1 - w) * ptree + w * pnn


print(f"trees only  {oracle(ptree):7.1f}\n", flush=True)

t = time.time()
e3 = [nnlib.train_nn(Atr, ytr, [Ava], sd, hidden=(256, 128), epochs=3)[1][0]
      for sd in range(12)]
print(f"[12 nets at e3, {time.time() - t:.0f}s]", flush=True)
print("  solo scores: " + " ".join(f"{oracle(p):.0f}" for p in e3), flush=True)
for n in (1, 2, 4, 8, 12):
    p = np.mean(e3[:n], axis=0)
    print(f"  {n:2d} nets  solo {oracle(p):7.1f}   blend "
          + "  ".join(f"w={w}:{oracle(blend(p, w)):7.1f}" for w in (0.3, 0.35, 0.4)),
          flush=True)

print("\n--- config diversity: 4 seeds each of e3 / e4 / e6 ---", flush=True)
t = time.time()
mixed = list(e3[:4])
for ep in (4, 6):
    mixed += [nnlib.train_nn(Atr, ytr, [Ava], sd, hidden=(256, 128), epochs=ep)[1][0]
              for sd in range(4)]
print(f"[8 more nets, {time.time() - t:.0f}s]", flush=True)
pm = np.mean(mixed, axis=0)
print(f"  12 mixed  solo {oracle(pm):7.1f}   blend "
      + "  ".join(f"w={w}:{oracle(blend(pm, w)):7.1f}" for w in (0.3, 0.35, 0.4)),
      flush=True)
print("\ndone", flush=True)
