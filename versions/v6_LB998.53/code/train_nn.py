"""Fit the neural net and export it as plain numpy arrays.

    python train_nn.py --validate    # fitted on all but the last season
    python train_nn.py               # fitted on everything

Run this with the *system* python, which has torch. Nothing torch-shaped ends up
in the submission: this writes work/nn_weights.npz (weight matrices plus the
standardisation constants) and train_v4.py -- which runs under .venv-submit for
numpy 1.26 pickle compatibility -- folds those arrays into the artifact.

The net is a poor standalone model (777 against the tree blend's 870 on the 2024
holdout) and that is fine. What it contributes is disagreement: blending it in
at w=0.3 moved the holdout from 869.7 to 893.2.
"""
import argparse
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import fe
import nnpred

HERE = os.path.dirname(os.path.abspath(__file__))
TRAIN_CSV = os.path.join(HERE, "..", "open", "data", "train.csv")

# Tuned on the 2024 holdout with the tree predictions cached. The striking part
# is that training *longer* destroys it -- at 15 epochs the net scores a flat 0,
# because with a BSS around 0.01 there is almost nothing to fit and it memorises
# noise immediately. 3 epochs is the setting that held up on every usable
# holdout (2021 +27, 2022 +16, 2024 +23 over the tree blend); 6 epochs ties on
# 2024 but loses badly on 2021, so the less-overfit net wins on robustness.
HIDDEN = (256, 128)
EPOCHS = 3
BATCH = 8192
LR = 2e-3
WD = 1e-4
SEEDS = [0, 1, 2, 3]

torch.set_num_threads(max(1, (os.cpu_count() or 4) - 2))


class MLP(nn.Module):
    def __init__(self, d, hidden):
        super().__init__()
        layers = []
        prev = d
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.GELU(approximate="tanh")]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_one(A, y, seed, hidden, epochs, lr):
    torch.manual_seed(seed)
    model = MLP(A.shape[1], hidden)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=WD)
    xt, yt = torch.from_numpy(A), torch.from_numpy(y.astype("float32"))
    n = len(xt)
    steps = epochs * (n // BATCH + 1)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps,
                                                pct_start=0.2)
    lossf = nn.BCEWithLogitsLoss()
    g = torch.Generator().manual_seed(seed)
    done = 0
    for ep in range(epochs):
        perm = torch.randperm(n, generator=g)
        tot = 0.0
        for i in range(0, n, BATCH):
            j = perm[i:i + BATCH]
            opt.zero_grad()
            loss = lossf(model(xt[j]), yt[j])
            loss.backward()
            opt.step()
            if done < steps - 1:
                sched.step()
                done += 1
            tot += loss.item() * len(j)
        if ep % 5 == 0 or ep == epochs - 1:
            print(f"    ep{ep:3d} logloss {tot / n:.5f}", flush=True)
    model.eval()
    ws, bs = [], []
    for m in model.net:
        if isinstance(m, nn.Linear):
            ws.append(m.weight.detach().numpy().T.astype("float32").copy())
            bs.append(m.bias.detach().numpy().astype("float32").copy())
    return ws, bs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--seeds", type=int, default=len(SEEDS))
    args = ap.parse_args()
    out = args.out or os.path.join(
        HERE, "nn_weights_validate.npz" if args.validate else "nn_weights.npz")

    t0 = time.time()
    df = pd.read_csv(TRAIN_CSV, encoding="utf-8-sig")
    if args.validate:
        df = df[df.season < int(df.season.max())]
    print(f"train rows {df.shape} ({time.time() - t0:.0f}s)")

    ctx = fe.fit_context(df)
    X = fe.build_features(df, ctx)
    y = df[fe.TARGET_COL].to_numpy()
    prep = nnpred.fit_prep(X)
    A = nnpred.prep(X, prep)
    print(f"nn input {A.shape}", flush=True)

    payload = {"columns": np.array(list(X.columns)), "hidden": np.array(HIDDEN),
               "n_nets": np.array(args.seeds),
               "prep_med": prep["med"], "prep_ind": prep["ind"],
               "prep_mu": prep["mu"], "prep_sd": prep["sd"],
               "prep_bs": np.array(prep["bs"])}
    for k, sd in enumerate(SEEDS[:args.seeds]):
        t = time.time()
        print(f"  net seed {sd}", flush=True)
        ws, bs = train_one(A, y, sd, HIDDEN, EPOCHS, LR)
        for li, (w, b) in enumerate(zip(ws, bs)):
            payload[f"n{k}_W{li}"] = w
            payload[f"n{k}_b{li}"] = b
        payload[f"n{k}_layers"] = np.array(len(ws))
        print(f"  net seed {sd} done ({time.time() - t:.0f}s)", flush=True)

    np.savez_compressed(out, **payload)
    print(f"\nsaved {out} ({os.path.getsize(out) / 1e6:.1f} MB, "
          f"total {time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
