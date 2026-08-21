"""Which rule predicts next season's success rate best?

The offset constant is worth ~401,000 * (error)^2 points, so a 0.005 miss is
~10 points and a 0.02 miss is ~160. Backtest every rule on the seasons we can
actually check, and look at the structural story too: the 2022->2023 drop is
almost entirely a one-time step inside game_type F, which a plain linear fit
mistakes for trend.
"""
import numpy as np
import pandas as pd

import common

df = common.load()
TARGET = "control_success"

by_season = df.groupby("season")[TARGET].mean()
by_gt = df.pivot_table(index="season", columns="game_type", values=TARGET, aggfunc="mean")
share_f = df.groupby("season").game_type.apply(lambda s: (s == "F").mean())


def r_linear(y, ahead=1):
    return common.extrapolate(y, ahead)


def r_linear3(y, ahead=1):
    return common.extrapolate(y[-3:], ahead)


def r_last(y, ahead=1):
    return float(y[-1])


def r_mean_delta(y, ahead=1):
    d = np.diff(y)
    return float(y[-1] + d.mean() * ahead)


def r_median_delta(y, ahead=1):
    d = np.diff(y)
    return float(y[-1] + np.median(d) * ahead)


def r_delta3(y, ahead=1):
    d = np.diff(y)[-3:]
    return float(y[-1] + d.mean() * ahead)


def r_wlinear(y, ahead=1):
    x = np.arange(len(y), dtype=float)
    w = 0.7 ** (len(y) - 1 - x)
    b, a = np.polyfit(x, y, 1, w=w)
    return float(a + b * (len(y) - 1 + ahead))


RULES = [("linear", r_linear), ("linear3", r_linear3), ("last", r_last),
         ("mean_delta", r_mean_delta), ("median_delta", r_median_delta),
         ("delta3", r_delta3), ("wlinear.7", r_wlinear)]


def segmented(seasons, ahead=1):
    """R gets its own trend; F is extrapolated only from the post-2023 regime.

    The F step (0.709 -> 0.473 between 2022 and 2023, uniform across all teams)
    is a definition change, not a trend, so feeding it into a global line drags
    the forecast down for a reason that will not repeat.
    """
    yr = by_gt.loc[seasons, "R"].values
    r_next = common.extrapolate(yr, ahead)
    yf = by_gt.loc[seasons, "F"].values
    post = [s for s in seasons if s >= 2023]
    if len(post) >= 2:
        f_next = common.extrapolate(by_gt.loc[post, "F"].values, ahead)
    elif len(post) == 1:
        f_next = float(by_gt.loc[post[0], "F"] + (r_next - yr[-1]))
    else:
        f_next = common.extrapolate(yf, ahead)
    s = float(share_f.loc[seasons[-1]])
    return (1 - s) * r_next + s * f_next, r_next, f_next


print("season rates:", by_season.round(4).to_dict())
print("R:", by_gt["R"].round(4).to_dict())
print("F:", by_gt["F"].round(4).to_dict())
print("F share:", share_f.round(4).to_dict())

print("\n=== backtest: predict season V from seasons < V ===")
hdr = f"{'V':>6} {'true':>7} " + " ".join(f"{n:>13}" for n, _ in RULES) + f"{'segmented':>13}"
print(hdr)
errs = {n: [] for n, _ in RULES}
errs["segmented"] = []
for V in (2021, 2022, 2023, 2024):
    seasons = [s for s in by_season.index if s < V]
    y = by_season.loc[seasons].values
    true = by_season.loc[V]
    row = f"{V:>6} {true:7.4f} "
    for n, f in RULES:
        e = f(y) - true
        errs[n].append(e)
        row += f" {f(y):.4f}({e:+.3f})"
    sg, rn, fn = segmented(seasons)
    errs["segmented"].append(sg - true)
    row += f"  {sg:.4f}({sg - true:+.3f})"
    print(row)

print("\n=== error summary (points lost = 401000 * err^2) ===")
for n in list(errs):
    e = np.array(errs[n])
    print(f"  {n:14s} MAE={np.abs(e).mean():.4f}  RMSE={np.sqrt((e**2).mean()):.4f} "
          f"  avg points lost={401000 * (e ** 2).mean():7.1f}  bias={e.mean():+.4f}")

print("\n=== 2025 forecasts from all six seasons ===")
seasons = list(by_season.index)
y = by_season.values
for n, f in RULES:
    print(f"  {n:14s} {f(y):.4f}")
sg, rn, fn = segmented(seasons)
print(f"  {'segmented':14s} {sg:.4f}   (R {rn:.4f}, F {fn:.4f}, "
      f"F share {share_f.iloc[-1]:.3f})")
