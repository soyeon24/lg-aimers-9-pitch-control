"""Extend the net-weight sweep -- 29 stopped at the top of its own range.

Experiment 29 swept the net at 0.10 .. 0.35 and the best point was the last one,
which means the range was wrong, not that 0.35 is the answer. The reason the old
0.35 no longer bounds it: the net fitted on the extras view scores 867.1 alone
against the base view net's 835.1, so it is a much stronger member than the one
that weight was chosen for.

Everything here is arithmetic on experiment 29's cached member predictions, so
the sweep is free. It is deliberately coarse. Fine-tuning blend weights against
2024 is a known failure in this project -- v6 optimised them to 891.9 and the
same weights scored 886.0 on another season -- so what this is for is finding
the shape of the curve and a defensible flat spot, not its argmax.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import TARGET, comp_score, load, logit, sigmoid

VAL = 2024
EPS = 1e-6

df = load()
va = df[df.season == VAL]
yva = va[TARGET].to_numpy()
isF = va.game_type.to_numpy() == "F"
z = np.load(os.path.join("cache", f"restack_{VAL}.npz"))


def oracle(p, m=None):
    m = slice(None) if m is None else m
    y = yva[m]
    q = np.clip(p[m], EPS, 1 - EPS)
    return comp_score(sigmoid(logit(q) + logit(y.mean()) - logit(q.mean())), y)


T = 0.5 * (z["tree_A"] + z["tree_B"])
print(f"members: treeA {oracle(z['tree_A']):7.1f}  treeB {oracle(z['tree_B']):7.1f}  "
      f"tree mix {oracle(T):7.1f}  nnA {oracle(z['nn_A']):7.1f}  "
      f"nnB {oracle(z['nn_B']):7.1f}", flush=True)

print("\n--- net weight, extended ---", flush=True)
NETS = {"A": z["nn_A"], "B": z["nn_B"], "avg": 0.5 * (z["nn_A"] + z["nn_B"]),
        "both": None}
for tag in ("A", "B", "avg"):
    print(f"  net {tag:3s} " + "  ".join(
        f"{w:.2f}:{oracle((1 - w) * T + w * NETS[tag]):7.1f}"
        for w in (0.25, 0.35, 0.45, 0.55, 0.65, 0.75)), flush=True)

print("\n--- both nets as separate members (wA + wB of the net half) ---", flush=True)
for w in (0.35, 0.45, 0.55, 0.65):
    row = []
    for sb in (0.5, 0.7, 0.85, 1.0):
        p = (1 - w) * T + w * ((1 - sb) * z["nn_A"] + sb * z["nn_B"])
        row.append(f"B share {sb:.2f}:{oracle(p):7.1f}")
    print(f"  net w={w:.2f}  " + "  ".join(row), flush=True)

print("\n--- tree mix, re-checked at a high net weight ---", flush=True)
for w in (0.45, 0.55):
    row = []
    for wb in (0.4, 0.5, 0.6, 0.7):
        p = (1 - w) * ((1 - wb) * z["tree_A"] + wb * z["tree_B"]) + w * z["nn_B"]
        row.append(f"wB={wb:.1f}:{oracle(p):7.1f}")
    print(f"  net w={w:.2f}  " + "  ".join(row), flush=True)

print("\n--- F specialist on the retuned stack ---", flush=True)
F = {"A-leaf31": z["f_A_leaf31"], "A-leaf15": z["f_A_leaf15"],
     "A-ens": 0.5 * (z["f_A_leaf31"] + z["f_A_leaf15"]),
     "B-ens": 0.5 * (z["f_B_leaf31"] + z["f_B_leaf15"])}
for nw in (0.45, 0.55):
    S = (1 - nw) * T + nw * z["nn_B"]
    print(f"  [net w={nw:.2f}: {oracle(S):7.1f}]", flush=True)
    for tag, p in F.items():
        row = []
        for w in (0.10, 0.15, 0.20, 0.25, 0.30):
            q = S.copy()
            q[isF] = (1 - w) * S[isF] + w * p[isF]
            row.append(f"{w:.2f}:{oracle(q):7.1f}/{oracle(q, isF):6.1f}")
        print(f"    {tag:9s} " + "  ".join(row), flush=True)

print("\ndone", flush=True)
