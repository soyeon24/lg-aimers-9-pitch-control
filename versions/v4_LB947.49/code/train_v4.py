"""Train the submission model.

    python train_v4.py --validate    # 2019-2023 -> 2024, the honest dry run
    python train_v4.py               # full fit -> submit/model/model.joblib

Everything the evaluation server needs ends up in one artifact: the fitted
trees, the feature context (league means and per-entity carry tables), the
column order, and a single logit offset for the season drift.

The offset
----------
Success rates fall every season, so a model fitted on 2019-2024 predicts too
high for 2025. We forecast the 2025 rate from train.csv alone (see forecast.py)
and freeze the gap as a constant in logit space. No evaluation-data statistic is
involved, and the constant is never tuned against leaderboard feedback -- the
rules treat that as the same violation.

It is applied at 85% strength. The features themselves already drift: the
season-to-date columns of a 2025 row carry 2025's level, so the model's mean
prediction moves down on its own by a fraction beta of the true drift (measured
at 0.18-0.25 on the 2024 holdout, and falling as training seasons accumulate).
A full-strength offset would add the drift a second time on top of that.
"""
import argparse
import os
import time

import joblib
import numpy as np
import pandas as pd

import fe
from forecast import forecast_next_rate

HERE = os.path.dirname(os.path.abspath(__file__))
TRAIN_CSV = os.path.join(HERE, "..", "open", "data", "train.csv")
MODEL_DIR = os.path.join(HERE, "..", "submit", "model")

# Capacity stays small on purpose. Sweeps at leaf 31 (-25) and 800 iterations
# (-25) both lose on the 2024 holdout; leaf 8 and 200 iterations tie. The signal
# is thin (BSS is around 0.01) and extra capacity only buys variance.
PARAMS = dict(max_iter=400, learning_rate=0.05, max_leaf_nodes=15,
              l2_regularization=10.0, min_samples_leaf=10000, early_stopping=False)
SEEDS = [0, 1, 2, 3, 4, 5, 6, 7]
OFFSET_SCALE = 0.85
EPS = 1e-6


def logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def comp_score(p, y):
    r = y.mean()
    return max(0.0, 100000 * (1 - np.mean((p - y) ** 2) / (r * (1 - r))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate", action="store_true",
                    help="hold out the last season and simulate the whole procedure")
    ap.add_argument("--seeds", type=int, default=len(SEEDS))
    args = ap.parse_args()

    from sklearn.ensemble import HistGradientBoostingClassifier

    t0 = time.time()
    df = pd.read_csv(TRAIN_CSV, encoding="utf-8-sig")
    print(f"train.csv {df.shape} ({time.time() - t0:.0f}s)")

    if args.validate:
        last = int(df.season.max())
        tr, va = df[df.season < last], df[df.season == last]
    else:
        tr, va = df, None

    ctx = fe.fit_context(tr)
    seasons = ctx["seasons"]
    prev_rate = float(tr[tr.season == seasons[-1]][fe.TARGET_COL].mean())
    forecast = forecast_next_rate(tr, verbose=True)
    target_rate = prev_rate + OFFSET_SCALE * (forecast - prev_rate)
    print(f"seasons {seasons}  last-season rate {prev_rate:.4f}")
    print(f"forecast {seasons[-1] + 1} = {forecast:.4f}  -> applied at "
          f"{OFFSET_SCALE:.2f} = {target_rate:.4f}")

    t = time.time()
    Xtr = fe.build_features(tr, ctx)
    columns = list(Xtr.columns)
    cat_idx = [columns.index(c) for c in fe.CAT_COLS]
    ytr = tr[fe.TARGET_COL].to_numpy()
    print(f"features {Xtr.shape} ({time.time() - t:.0f}s)")

    models = []
    for sd in SEEDS[:args.seeds]:
        t = time.time()
        m = HistGradientBoostingClassifier(categorical_features=cat_idx,
                                           random_state=sd, **PARAMS)
        m.fit(Xtr, ytr)
        models.append(m)
        print(f"  seed {sd} ({time.time() - t:.0f}s)", flush=True)

    prev_mask = (tr.season.to_numpy() == seasons[-1])
    ref_mean = float(np.mean([m.predict_proba(Xtr[prev_mask])[:, 1]
                              for m in models], axis=0).mean())
    logit_offset = float(logit(target_rate) - logit(ref_mean))
    print(f"reference mean on {seasons[-1]} = {ref_mean:.4f} "
          f"-> logit_offset = {logit_offset:+.4f}")

    artifact = dict(models=models, columns=columns, cat_idx=cat_idx, ctx=ctx,
                    logit_offset=logit_offset, target_rate=target_rate,
                    forecast=forecast, offset_scale=OFFSET_SCALE,
                    ref_mean=ref_mean, prev_rate=prev_rate,
                    trained_seasons=seasons, params=PARAMS, seeds=SEEDS[:args.seeds],
                    fe_version="v4")

    if args.validate:
        Xva = fe.build_features(va, ctx, columns)
        yva = va[fe.TARGET_COL].to_numpy()
        raw = np.mean([m.predict_proba(Xva)[:, 1] for m in models], axis=0)
        adj = sigmoid(logit(raw) + logit_offset)
        orc = sigmoid(logit(raw) + logit(yva.mean()) - logit(raw.mean()))
        gt = va.game_type.to_numpy()
        print(f"\n=== {int(va.season.iloc[0])} holdout ===")
        print(f"true rate {yva.mean():.4f} | forecast {forecast:.4f} "
              f"| applied {target_rate:.4f} | achieved {adj.mean():.4f}")
        print(f"model's own drift = {raw.mean() - ref_mean:+.4f} "
              f"(true drift {yva.mean() - prev_rate:+.4f})")
        print(f"score            {comp_score(adj, yva):8.1f}")
        print(f"  R rows only    {comp_score(adj[gt == 'R'], yva[gt == 'R']):8.1f}")
        print(f"  F rows only    {comp_score(adj[gt == 'F'], yva[gt == 'F']):8.1f}")
        print(f"oracle mean      {comp_score(orc, yva):8.1f}   (calibration-free ceiling)")
        return

    os.makedirs(MODEL_DIR, exist_ok=True)
    path = os.path.join(MODEL_DIR, "model.joblib")
    joblib.dump(artifact, path, compress=3)
    print(f"\nsaved {path} ({os.path.getsize(path) / 1e6:.1f} MB, "
          f"total {time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
