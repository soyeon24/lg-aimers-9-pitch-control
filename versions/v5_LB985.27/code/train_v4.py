"""Train the submission model.

    python train_v4.py --validate    # last season held out, the honest dry run
    python train_v4.py               # full fit -> submit/model/model.joblib

Everything the evaluation server needs ends up in one artifact: the fitted
models, the feature context (league means and per-entity carry tables), the
column order, and a single logit offset for the season drift.

The ensemble
------------
Three members, blended.

Two tree capacities: deeply regularised (leaf 15) is much better on regular
games, looser (leaf 31, l2 1, min_samples_leaf 2000) is much better on the
second-tier F games, which have far fewer rows per pitcher. On the 2024 holdout
they score R 858.8 / F 551.8 and R 839.6 / F 600.4 -- neither is good at both,
and blending beats switching by game_type (869.7 vs 866.8). The loose model's
share is held at 0.35 rather than the 2024-optimal 0.4 because on the 2021
holdout, where only two training seasons are available, it costs 47 points.

A small neural net at weight 0.35. It is a weak model on its own (835 against
the tree blend's 870) but it is wrong in different places, and that is worth
+27 / +16 / +23 on the 2021 / 2022 / 2024 holdouts -- the only three that are
usable, since 2023's F regime break makes even its oracle score 0.

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
import nnpred
from forecast import forecast_next_rate

HERE = os.path.dirname(os.path.abspath(__file__))
TRAIN_CSV = os.path.join(HERE, "..", "open", "data", "train.csv")
MODEL_DIR = os.path.join(HERE, "..", "submit", "model")

BASE = dict(max_iter=400, learning_rate=0.05, max_leaf_nodes=15,
            l2_regularization=10.0, min_samples_leaf=10000, early_stopping=False)
# (tag, parameter overrides, seeds, share of the tree half of the blend)
GROUPS = [
    ("tight", {}, [0, 1, 2, 3, 4, 5, 6, 7], 0.65),
    ("loose", dict(max_leaf_nodes=31, l2_regularization=1.0, min_samples_leaf=2000),
     [0, 1, 2, 3, 4, 5, 6, 7], 0.35),
]
NN_WEIGHT = 0.35     # the trees keep 1 - NN_WEIGHT between them
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


def group_predict(g, X):
    if g["kind"] == "nn":
        return nnpred.nn_predict(X, g["nn"])
    return np.mean([m.predict_proba(X)[:, 1] for m in g["models"]], axis=0)


def blend(groups, X):
    """Weighted mean over groups; each group is its own seed ensemble."""
    out = np.zeros(len(X))
    for g in groups:
        out += g["weight"] * group_predict(g, X)
    return np.clip(out, EPS, 1 - EPS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate", action="store_true",
                    help="hold out the last season and simulate the whole procedure")
    ap.add_argument("--seeds", type=int, default=None,
                    help="cap the seeds per group (for quick runs)")
    ap.add_argument("--nn", default=None,
                    help="nn_weights npz from train_nn.py; omit to skip the net")
    args = ap.parse_args()
    nn_path = args.nn or os.path.join(
        HERE, "nn_weights_validate.npz" if args.validate else "nn_weights.npz")

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

    use_nn = os.path.exists(nn_path)
    tree_share = (1.0 - NN_WEIGHT) if use_nn else 1.0

    groups = []
    for tag, over, seeds, weight in GROUPS:
        params = dict(BASE)
        params.update(over)
        used = seeds[:args.seeds] if args.seeds else seeds
        models = []
        for sd in used:
            t = time.time()
            m = HistGradientBoostingClassifier(categorical_features=cat_idx,
                                               random_state=sd, **params)
            m.fit(Xtr, ytr)
            models.append(m)
            print(f"  {tag} seed {sd} ({time.time() - t:.0f}s)", flush=True)
        groups.append(dict(kind="trees", tag=tag, models=models, params=params,
                           seeds=used, weight=weight * tree_share))

    if use_nn:
        nn_art = nnpred.load_weights(nn_path)
        if nn_art["columns"] != columns:
            raise SystemExit("nn weights were fitted on a different feature set "
                             "-- rerun train_nn.py")
        groups.append(dict(kind="nn", tag="nn", nn={"nets": nn_art["nets"],
                                                    "prep": nn_art["prep"]},
                           weight=NN_WEIGHT))
        print(f"  nn: {len(nn_art['nets'])} nets from {os.path.basename(nn_path)} "
              f"(weight {NN_WEIGHT})")
    else:
        print(f"  no {os.path.basename(nn_path)} -- trees only")

    prev_mask = (tr.season.to_numpy() == seasons[-1])
    ref_mean = float(blend(groups, Xtr[prev_mask]).mean())
    logit_offset = float(logit(target_rate) - logit(ref_mean))
    print(f"reference mean on {seasons[-1]} = {ref_mean:.4f} "
          f"-> logit_offset = {logit_offset:+.4f}")

    artifact = dict(groups=groups, columns=columns, cat_idx=cat_idx, ctx=ctx,
                    logit_offset=logit_offset, target_rate=target_rate,
                    forecast=forecast, offset_scale=OFFSET_SCALE,
                    ref_mean=ref_mean, prev_rate=prev_rate,
                    trained_seasons=seasons, nn_weight=NN_WEIGHT,
                    fe_version="v5")

    if args.validate:
        Xva = fe.build_features(va, ctx, columns)
        yva = va[fe.TARGET_COL].to_numpy()
        raw = blend(groups, Xva)
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
