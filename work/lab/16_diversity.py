"""More members for the blend.

v5's entire gain came from adding one weak-but-different model. The trees are
cached here, so each new candidate costs under a minute and the only question
that matters is what it adds *on top of* the existing tight+loose+MLP stack --
a candidate that scores well alone but agrees with what we have is worthless.

Everything tried here stays exportable to numpy (matmuls and table lookups), so
a winner can ship without adding a dependency to the evaluation image.
"""
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import harness
import lab_v6
import nnlib
from common import TARGET, comp_score, load, logit, sigmoid

VAL = 2024
TREE_SEEDS = (0, 1, 2, 3)
SEEDS = (0, 1)
EPS = 1e-6
CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache",
                     f"trees_{VAL}.npz")

torch.set_num_threads(max(1, (os.cpu_count() or 4) - 2))

df = load()
tr, va = df[df.season < VAL], df[df.season == VAL]
ytr, yva = tr[TARGET].to_numpy(), va[TARGET].to_numpy()
ctx = lab_v6.ctx_fn(tr)
Xtr = lab_v6.base(tr, ctx)
Xva = lab_v6.base(va, ctx)[list(Xtr.columns)]
gt = va.game_type.to_numpy()

if os.path.exists(CACHE):
    z = np.load(CACHE)
    p15, p31 = z["p15"], z["p31"]
    print("[trees from cache]", flush=True)
else:
    t = time.time()
    p15 = harness.pmean(harness.fit_models(Xtr, ytr, TREE_SEEDS), Xva)
    p31 = harness.pmean(harness.fit_models(
        Xtr, ytr, TREE_SEEDS,
        dict(max_leaf_nodes=31, l2_regularization=1.0, min_samples_leaf=2000)), Xva)
    np.savez_compressed(CACHE, p15=p15, p31=p31)
    print(f"[trees fitted {time.time() - t:.0f}s]", flush=True)


def oracle(p, y=None, mask=None):
    y = yva if y is None else y
    if mask is not None:
        p, y = p[mask], y[mask]
    p = np.clip(p, EPS, 1 - EPS)
    return comp_score(sigmoid(logit(p) + logit(y.mean()) - logit(p.mean())), y)


prep = nnlib.fit_prep(Xtr)
Atr, Ava = nnlib.prep(Xtr, prep), nnlib.prep(Xva, prep)
print(f"dense input {Atr.shape}", flush=True)


# ------------------------------------------------------------- entity indices
def entity_index(col):
    """Train ids get 1..n; anything unseen (a 2025 debutant) falls to slot 0."""
    uniq = np.sort(tr[col].unique())
    lut = {int(v): i + 1 for i, v in enumerate(uniq)}
    itr = np.array([lut[int(v)] for v in tr[col].to_numpy()], dtype="int64")
    iva = np.array([lut.get(int(v), 0) for v in va[col].to_numpy()], dtype="int64")
    return itr, iva, len(uniq) + 1


ptr, pva, n_pit = entity_index("pitcher_id")
btr, bva, n_bat = entity_index("batter_id")
print(f"entities: {n_pit} pitchers, {n_bat} batters "
      f"({(pva == 0).mean():.3%} / {(bva == 0).mean():.3%} unseen in {VAL})",
      flush=True)


class EmbMLP(nn.Module):
    def __init__(self, d, n_p, n_b, dp=16, db=8, hidden=(256, 128)):
        super().__init__()
        self.ep = nn.Embedding(n_p, dp)
        self.eb = nn.Embedding(n_b, db)
        nn.init.normal_(self.ep.weight, std=0.01)
        nn.init.normal_(self.eb.weight, std=0.01)
        layers, prev = [], d + dp + db
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.GELU(approximate="tanh")]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x, pi, bi):
        return self.net(torch.cat([x, self.ep(pi), self.eb(bi)], 1)).squeeze(-1)


def train_emb(seed, epochs=3, lr=2e-3, bs=8192, hidden=(256, 128)):
    torch.manual_seed(seed)
    model = EmbMLP(Atr.shape[1], n_pit, n_bat, hidden=hidden)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    xt = torch.from_numpy(Atr)
    pt, bt = torch.from_numpy(ptr), torch.from_numpy(btr)
    yt = torch.from_numpy(ytr.astype("float32"))
    n = len(xt)
    steps = epochs * (n // bs + 1)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps,
                                                pct_start=0.2)
    lossf = nn.BCEWithLogitsLoss()
    g = torch.Generator().manual_seed(seed)
    done = 0
    for _ in range(epochs):
        perm = torch.randperm(n, generator=g)
        for i in range(0, n, bs):
            j = perm[i:i + bs]
            opt.zero_grad()
            lossf(model(xt[j], pt[j], bt[j]), yt[j]).backward()
            opt.step()
            if done < steps - 1:
                sched.step()
                done += 1
    model.eval()
    with torch.no_grad():
        out = torch.sigmoid(model(torch.from_numpy(Ava),
                                  torch.from_numpy(pva),
                                  torch.from_numpy(bva))).numpy().astype("float64")
    return out


def train_mse(seed, epochs=3, lr=2e-3, bs=8192, hidden=(256, 128)):
    """Same net, squared error on the 0/1 label -- the competition's own loss."""
    torch.manual_seed(seed)
    model = nnlib.MLP(Atr.shape[1], hidden)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    xt = torch.from_numpy(Atr)
    yt = torch.from_numpy(ytr.astype("float32"))
    n = len(xt)
    steps = epochs * (n // bs + 1)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps,
                                                pct_start=0.2)
    g = torch.Generator().manual_seed(seed)
    done = 0
    for _ in range(epochs):
        perm = torch.randperm(n, generator=g)
        for i in range(0, n, bs):
            j = perm[i:i + bs]
            opt.zero_grad()
            ((torch.sigmoid(model(xt[j])) - yt[j]) ** 2).mean().backward()
            opt.step()
            if done < steps - 1:
                sched.step()
                done += 1
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(torch.from_numpy(Ava))).numpy().astype("float64")


# --------------------------------------------------------------- candidates
def avg(fn, **kw):
    t = time.time()
    out = np.mean([fn(sd, **kw) for sd in SEEDS], axis=0)
    return out, time.time() - t


members = {"tight": p15, "loose": p31}
pnn, el = avg(lambda sd: nnlib.train_nn(Atr, ytr, [Ava], sd, hidden=(256, 128),
                                        epochs=3)[1][0])
members["mlp"] = pnn
print(f"[mlp 256-128 e3 {el:.0f}s]", flush=True)

base = 0.65 * (0.65 * p15 + 0.35 * p31) + 0.35 * pnn
print(f"\nv5 stack                          {oracle(base):8.1f}   "
      f"R {oracle(base, mask=gt == 'R'):7.1f}  F {oracle(base, mask=gt == 'F'):7.1f}\n",
      flush=True)

CANDIDATES = [
    ("linear (no hidden layer)",
     lambda: avg(lambda sd: nnlib.train_nn(Atr, ytr, [Ava], sd, hidden=(),
                                           epochs=3)[1][0])),
    ("mlp 1024-512 e2",
     lambda: avg(lambda sd: nnlib.train_nn(Atr, ytr, [Ava], sd, hidden=(1024, 512),
                                           epochs=2)[1][0])),
    ("mlp 256-128-64 e3",
     lambda: avg(lambda sd: nnlib.train_nn(Atr, ytr, [Ava], sd,
                                           hidden=(256, 128, 64), epochs=3)[1][0])),
    ("mlp e3 squared-error loss", lambda: avg(train_mse)),
    ("mlp e3 + entity embeddings", lambda: avg(train_emb)),
    ("mlp e3 embeddings, 8 epochs", lambda: avg(train_emb, epochs=8)),
]

for tag, fn in CANDIDATES:
    p, el = fn()
    members[tag] = p
    line = []
    for w in (0.15, 0.25, 0.35):
        line.append(f"w={w}:{oracle((1 - w) * base + w * p):7.1f}")
    print(f"  {tag:32s} alone {oracle(p):7.1f}  ({el:.0f}s)", flush=True)
    print(f"    on top of v5 stack: " + "  ".join(line), flush=True)

np.savez_compressed(os.path.join(os.path.dirname(CACHE), f"members_{VAL}.npz"),
                    **{k: v for k, v in members.items()}, yva=yva,
                    gt=(gt == "R").astype("int8"))
print("\nmembers cached; done", flush=True)
