"""Forecasting next season's league success rate.

This constant is the single most expensive number in the whole pipeline: the
score loses 401,000 * (error)^2 points, so a 0.005 miss costs ~10 and a 0.02
miss costs ~160. Backtests over 2021-2024 put every sensible rule's RMSE around
0.016-0.020, i.e. roughly 100-150 points of unavoidable tax. The goal here is
only to avoid making it worse than it has to be.

Two structural facts shape the rule:

* game_type F changed definition in 2023 (.709 -> .473, uniformly across every
  team). A plain linear fit through the pooled series reads that one-off step as
  trend and keeps extrapolating it. So R and F are forecast separately.
* No single extrapolator wins the backtest, and with four checkable seasons we
  cannot tell them apart. Averaging three of them is the variance-reducing move.

Everything here is computed from train.csv alone -- never from the evaluation
data, and never tuned against leaderboard feedback, which the rules treat the
same as reading the evaluation set.
"""
import numpy as np

TARGET_COL = "control_success"


def _linear(y, ahead=1):
    x = np.arange(len(y), dtype=float)
    slope, intercept = np.polyfit(x, y, 1)
    return float(intercept + slope * (len(y) - 1 + ahead))


def _mean_delta(y, ahead=1):
    return float(y[-1] + np.diff(y).mean() * ahead)


def _delta3(y, ahead=1):
    return float(y[-1] + np.diff(y)[-3:].mean() * ahead)


def _consensus(y, ahead=1):
    y = np.asarray(y, dtype=float)
    if len(y) < 3:
        return float(y[-1]) if len(y) == 1 else _mean_delta(y, ahead)
    return float(np.mean([_linear(y, ahead), _mean_delta(y, ahead), _delta3(y, ahead)]))


def forecast_next_rate(df, f_break_season=2023, verbose=False):
    """Success rate of the season right after the last one present in `df`."""
    seasons = sorted(int(s) for s in df.season.unique())
    piv = df.pivot_table(index="season", columns="game_type", values=TARGET_COL,
                         aggfunc="mean")
    share_f = float((df[df.season == seasons[-1]].game_type == "F").mean())

    r = piv["R"].loc[seasons].to_numpy(dtype=float)
    r_next = _consensus(r)
    d_r = r_next - r[-1]

    post = [s for s in seasons if s >= f_break_season]
    f_all = piv["F"].loc[seasons].to_numpy(dtype=float)
    if len(post) >= 2:
        f = piv["F"].loc[post].to_numpy(dtype=float)
        # only two post-break points most of the time, so its own slope is
        # fragile; average it with "F moves the way R moves"
        f_next = float(np.mean([_consensus(f), f[-1] + d_r]))
    elif len(post) == 1:
        f_next = float(piv["F"].loc[post[0]] + d_r)
    else:
        f_next = _consensus(f_all)

    rate = (1 - share_f) * r_next + share_f * f_next
    if verbose:
        print(f"  R {r.round(4).tolist()} -> {r_next:.4f} (d={d_r:+.4f})")
        print(f"  F {f_all.round(4).tolist()} -> {f_next:.4f}")
        print(f"  F share {share_f:.4f} -> blended {rate:.4f}")
    return float(rate)
