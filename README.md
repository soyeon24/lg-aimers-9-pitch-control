<a id="english"></a>

# LG Aimers 9 Phase 2 — Pitch Control Success Probability

**English** · [한국어](#korean)

[Competition page](https://dacon.io/competitions/official/236743) · Leaderboard closed 2026-09-01 · Code + slides due 2026-09-07

Predict the probability that a single KBO pitch is a "control success." Train on 1.47M
pitches from 2019–2024, predict 250,000 pitches from 2025.

The metric is Brier Skill Score, so the **mean of the predictions** carries as much weight
as per-row discrimination. But training is 2019–2024 while evaluation is 2025, and the
success rate falls every year — half of this problem is drift correction, not resolution
(see [The shape of this problem](#the-shape-of-this-problem)).

**Final public leaderboard 998.53 — 437th of 1,087**, up from 897.09 at v1.
The final score is v6's. v7 was submitted before the deadline and did not beat it, which
makes it the one time in this project that a holdout gain failed to transfer at all — see
[How v7 ended](#how-v7-ended).

Throughout: **R** is `game_type` R, the top-tier KBO league; **F** is Futures, the
second-tier development league (12% of rows).

## Version history

Each version is one commit, and `versions/vN/` holds that moment's code, scores, and
reasoning in full.

| Version | Date | LB Public | 2024 holdout | Key change |
|---|---|---|---|---|
| [v1](versions/v1_LB897.09) | 08-17 | 897.09 | 710.1 | Baseline 47 features, HistGBM ×4 seeds + season-trend logit offset (−0.0503) |
| [v2](versions/v2_LB908.53) | 08-17 | 908.53 | 765.9 | 23 derived count / situation / recent-form features (70 total) |
| [v3](versions/v3) | 08-19 | 922.86 | 770.2 | Blend two HistGBM capacities (w=0.3). More seeds did nothing; the blend was the whole gain |
| [v4](versions/v4_LB947.49) | 08-22 | **947.49** | 855.9 | **Season-to-date reconstruction** + Futures regime isolation + segmented forecast |
| [v5](versions/v5_LB985.27) | 08-22 | **985.27** | 890.0 | Small MLP blend (w=0.35) + two tree capacities |
| [v6](versions/v6_LB998.53) | 08-22 | **998.53** | 895.6 | Futures-only model blended at 25%, on Futures rows only |
| [v7](versions/v7) | 08-29 | no gain | **917.6** | Blend two feature views (107 and 135 columns) |

v4 replaced the holdout harness, so v3 has two recorded scores — 770.2 on the old harness,
768.6 on the new one. The table shows the old value; the transfer table below uses the new one.

### Trust the holdout's sign, never its magnitude

This is the most expensive lesson in the project. How much of a 2024 holdout gain actually
reached the leaderboard varied wildly.

| Step | Holdout Δ | Leaderboard Δ | Transfer |
|---|---|---|---|
| v1 → v2 | +55.8 | +11.4 | 20% |
| v2 → v3 | +4.3 | +14.3 | 330% |
| v3 → v4 | +87.3 | +24.6 | 28% |
| v4 → v5 | +34.1 | +37.8 | 111% |
| v5 → v6 | +5.6 | +13.3 | 237% |
| v6 → v7 | +22.0 | none | **0%** |

From 20% to 330%, and then zero. **The ratio was never usable, and on the last step the
sign failed too.** For five rounds the holdout got the direction right and only the
magnitude wrong, which is why it was used for accept/reject decisions and never to predict
a score. v7 is the round where even that broke.

There is a reason transfer exceeded 100% at v5 and v6 — **the 2024 holdout structurally
underrates F.** The holdout model sees only one season (2023) of the new F regime, while
the real submission model sees two (2023 + 2024). That was the signal to invest more in F,
and v6 and v7 both went that way.

### What actually moved the score

Across seven versions only a handful of things were adopted; roughly sixty attempts were
rejected. The adopted ones fall on just three axes.

1. **Extracting more leak-free information** (v4, +24.6). The `asof_*` counters carry from
   train into test, which makes "how this player has done so far this season" recoverable
   row-independently. Career success rate correlates 0.19 with a player's actual results
   that year; the reconstructed season-to-date figure correlates 0.97.
2. **Blending models that are wrong in different ways** (v5, +37.8). An MLP worth 835 alone
   raises a tree blend worth 870. That is diversity, not accuracy.
3. **Giving a sacrificed minority its own model** (v6, +13.3). F is 12% of rows and was
   scoring under two-thirds of R.

v7 is 1 and 2 combined. The ensemble was fully saturated at v6 — bagging, ExtraTrees, and
residual stacking all landed below baseline — and what they had in common was that
**every one of them changed the model on top of the same feature table.** Splitting the
feature table in two showed that axis was not saturated at all.

## How v7 ended

v7 was submitted before the 2026-09-01 deadline and **did not improve on v6.** The exact
score was not recorded; v6's 998.53 remained the best and is the final public result. The
2024 holdout had said +22.0.

That is worth stating plainly, because this README spends its first screen arguing that the
holdout's sign can be trusted even when its magnitude cannot. Five rounds supported that.
The sixth did not. Anyone reading the v7 section below should read it as a well-measured
change that did not survive contact with the real evaluation set.

Two candidate explanations, neither verified:

- **The 2024 holdout has five training seasons; 2025 has six.** Every constant in v7 —
  the 0.5 view mix, the 0.35 net weight, the 0.25 F weight — was chosen on 2024 and
  confirmed on 2022. The section below already notes that 2021, with only two training
  seasons, disagreed by −29.8. The regime that made 2021 disagree may not have fully
  disappeared by 2025.
- **View B's gain concentrated in F, and F is 12% of rows.** The F transfer rate had been
  the highest of anything in the project (237% at v6), and that pattern may simply have run
  out.

What did verify: training, packaging, and the evaluation-server rehearsal all passed.

```
[OK] artifact loads (15.8 MB), tightA x4 + looseA x4 + tightB x4 + looseB x4
     + nn x4 w=.350 + f_spec x8 w=.25, 135 features
[OK] script.py matches fe.py on 20000 rows x 135 features
[OK] net inference matches (4 nets) -- numpy only, no torch
[OK] 250000 rows, wall clock 86.3s (limit 600s)
```

The first build used 8 seeds per tree group: 159.5s and 25.4 MB. Dropping to 4 seeds gave
86.3s and 15.8 MB — the reasoning is in ["Four seeds, not eight"](#four-seeds-not-eight).

The rehearsal's `predicted mean` is not a calibration preview. `verify_v4.py` takes the last
250k rows of train.csv and only rewrites `season` to 2025; those rows already carry
end-of-2024 asof counters, so every season-to-date feature collapses to zero. It checks
format, timing, and crashes — nothing else.

What the holdout and the rehearsal did establish:
- 2024 holdout end-to-end 917.6 (2 seeds), +21.9 over v6's 895.6
- On 2022, every constant (view mix 0.5, net on view B, F-specialist w=0.25) peaks in the
  same place
- The shipped `fe.py` matches the lab builder to the decimal (experiment 28's Aprod/Bprod)

## How to read this repository

- **Just the progression** — follow `git log` from v1, or the tags `v1`…`v7`, or each
  `versions/vN/NOTES.md`
- **Final methodology** — ["What changed in v7"](#what-changed-in-v7--the-axis-of-diversity-was-features-not-models)
  down through v4 (written newest-first)
- **The ~60 rejected attempts** — the rejection table in each version's section, and
  `work/lab/00`–`31`
- **Reproducing** — [Usage](#usage)

### Competition materials are not redistributed here

Everything under `open/` — the competition data (689 MB), the official data description,
and the provided baseline submission — is third-party material from Dacon / LG Aimers. It
is excluded from this repository and purged from its history, not merely untracked. Get it
from the [competition page](https://dacon.io/competitions/official/236743); the code here
expects it at `open/data/`.

The lab cache (`work/lab/cache/`) is also excluded — regenerate it with
`work/lab/00_cache.py`. Model binaries for the intermediate versions (v4–v6) are excluded
too; each is reproducible from that version's `code/` plus `train_v4.py`. The final v7
artifact is included, in `submit/` and `versions/v7/`.

### License

The code in this repository is MIT licensed ([LICENSE](LICENSE)). That covers this
repository's own code and write-ups only — it does not extend to the competition data,
documents, or baseline described above, which are not distributed here and remain under
their owners' terms.

## The shape of this problem

The metric is Brier Skill Score (`100000 × (1 − MSE / r(1−r))`), so only two things matter.

1. **The mean (calibration).** If the predicted mean is off by δ, roughly `401,000 × δ²`
   points disappear. Off by 0.01 costs 40 points; 0.02 costs 160. A constant prediction
   scores exactly 0.
2. **Resolution.** What is left is per-row discrimination, and the BSS here is on the order
   of 0.01 — the signal is very thin.

And `train.csv` is 2019–2024 while the evaluation data is 2025. The success rate falls
every year.

| Season | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 |
|---|---|---|---|---|---|---|
| All | .5647 | .5327 | .5328 | .5289 | .5000 | .4861 |
| R (top tier) | .5495 | .5269 | .5128 | .5037 | .5031 | .4897 |
| F (Futures) | .6892 | .5878 | .7038 | .7087 | **.4729** | .4593 |

## What changed in v7 — the axis of diversity was features, not models

v7 began by confirming v6's conclusion that the ensemble was saturated, and it confirmed it.
Swapping whole algorithm families raised nothing.

| Attempt | 2024 holdout | Verdict |
|---|---|---|
| v6 baseline | 892.4 (no specialist) / 897.9 | — |
| Row-bootstrap bagging, leaf15 ×4 | 828.6 alone → 889.8 blended | Rejected |
| ExtraTrees depth12 / depth20 | 613.7 / 544.6 alone → 890.0 / 888.7 | Rejected |
| Sequential residual stacking | 891.2 | Rejected |
| Deeper F specialist (4 train windows × 2 configs) | best 898.4 | +0.5, noise |
| F trained on the 2023+ regime only | 895.8 | Rejected — 25k rows is far too few |
| F-specific neural net | 890.0 | Rejected |
| Team's previous-season success rate | 831.2 | **Rejected (−29)** |
| Team prior applied only to thin-history pitchers | 842.2 | Rejected (−18) |
| 12 net seeds | 890.3 | Rejected |

The pattern is visible here. **Every one of them changed the model on top of the same
feature table.** MLP shape, tree capacity, bagging, different algorithms, residual
stacking — diversity was only ever sought in the model. The one axis never touched was the
feature table itself, and it was not saturated.

### Two feature views

`fe.py` now produces two sets of columns. `extras=False` gives the existing 107 (view A),
`extras=True` adds 28 more for 135 (view B). The same tree is trained on each and averaged.

| 2024 holdout (trees only, 4 seeds) | all | R | F |
|---|---|---|---|
| View A alone | 869.5 | 864.8 | 579.3 |
| View B alone | 878.0 | 870.4 | 613.2 |
| **A + 0.5·B** | **884.0** | 877.2 | 610.3 |

The proof that this is **diversity and not accuracy** sits in an intermediate step. A view
carrying only the regime family scores 868.2 alone — **lower** than A's 869.5 — yet mixing
them gives 874.8, above both. A weaker model raising the blend means, by definition, that
it is wrong in different places.

The mix ratio stays at 0.5. On 2024 the sweep 0.3/0.4/0.5/0.6 gives 880.6/882.7/884.0/884.4,
which is flat, and on 2022 it gives 2372.2/2373.5/**2373.7**/2372.8, peaking at 0.5.

### What went into view B

All of it is a per-row lookup of (player, season) constants learned from train.csv — the
same structure as the existing carry tables, so it stays within the "row-independent
prediction" rule.

**Regime split.** `asof_pitcher_success_rate` averages pitches from two eras measured under
different rules. But each pitcher's F share has a median of .19 with quartiles at .01 and
.89 — it is **spread across the entire range.** The degree of contamination differs
completely from pitcher to pitcher, so no single pooled ratio lets a tree undo it. The
career carry table is split into R / new F (>=2023) / old F (<2023), giving the model a
clean rate, a prior matched to that row's regime, and the contamination ratio separately.
About half of all R pitchers also appear in F each season, so this is not a rare corner.

**Platoon.** A pitcher's splits by batter handedness and a batter's by pitcher handedness.
`same_hand` only carried the league-average version, with individual deviation missing.

**Count.** The gap between a pitcher ahead in the count and behind in it.

Platoon and count both subdivide an already small history, so they are noisy. Each bucket is
therefore shrunk toward **that player's own career rate** rather than the league, and passed
as a difference — a player with no evidence of a split gets exactly 0.

The three families cannot be separated. Count alone adds +2.6 when blended and regime alone
+3.4, but all three together add +14.5.

### Moving the net to view B — the single biggest gain

The diversity gained in the trees showed up larger in the neural net.

| 2024, on top of the 884.0 tree blend | w=0.25 | w=0.35 | w=0.45 | w=0.55 |
|---|---|---|---|---|
| View A net (835.1 alone) | 901.6 | 903.0 | 901.3 | 896.5 |
| **View B net (867.1 alone)** | 910.2 | **915.0** | 916.6 | 914.9 |

The weight **stays at 0.35.** On 2024 the peak is 0.45 (916.6), but the 1.6 gap to 0.35 is
inside seed noise, and there is precedent below in this README for tuning a blend weight to
2024 and losing 5 points on another season. On 2022 the curve is still rising at 0.35 — so
0.35 sits **before** the peak on both holdouts, which is the safe side.

### The F specialist stays on view A

Since the regime split lifted F substantially, moving the F specialist to view B looks
natural — but measured, it is the opposite: 917.4 on view B versus **919.8** on view A. For
a model that only ever sees F rows, the regime split has little left to say. Two capacities
(leaf31 msl1000 + leaf15 msl2000) are averaged and blended at w=0.25 on F rows only.

### The final stack

| 2024 holdout (oracle) | |
|---|---|
| Trees, view A only | 869.5 |
| + view B average | 884.0 |
| + net on view A (w=0.35) | 903.0 |
| + net on **view B** (w=0.35) | **915.0** |
| + F specialist, view A, 2 capacities (w=0.25) | **919.8** |

Reproduced end-to-end through the real pipeline: oracle 918.9, **917.6** after the offset
(R 909.2 / F 654.8). That is **+21.9** over v6's 897.9.

### Four seeds, not eight

The first build used 8 seeds per tree group and took **159.5s** (limit 600s) with a 25.4 MB
zip. That was measured on a 16-core laptop, and HistGBM prediction is thread-parallel, so a
4-core evaluation server would land at 400–500s — right against the limit. Per the rules, an
error during `script.py` costs a submission.

Cutting to 4 seeds gives **86.3s / 15.8 MB** at essentially no cost. Seed averaging does not
raise the expectation, only lowers variance, and four is already flat — at the blend stage,
2 seeds gave 869.8 versus 869.5 for 4. In practice the 8-seed and 4-seed builds computed
baseline means of 0.4882 and 0.4881, with an identical offset of −0.0462.

The F specialist keeps its 8 (2 capacities × 4 seeds). F rows are only 12%, costing 3
seconds of inference, and F improvements had the highest leaderboard transfer of anything.

### The 2021 holdout is negative — and it is adopted anyway

Recorded honestly. Across three seasons, the effect of A + 0.5·B is:

| Holdout | Train seasons | F regime evaluated | all | R |
|---|---|---|---|---|
| 2021 | 2 | old | **−29.8** | −1.4 |
| 2022 | 3 | old | +12.0 | +18.5 |
| 2024 | 5 | new | +14.5 | +12.4 |
| 2025 (actual) | 6 | new | — | — |

Looking at R alone the numbers are −1.4 / +18.5 / +12.4 — never meaningfully negative.
2021's −29.8 comes entirely from F, and the fact that 2021's all is three times its R
(1536 vs 511) explains why: that year, `all` is effectively the task of predicting the
group gap between old-regime F (.70) and R (.51), and in 2025 neither that gap nor those
rows exist (`f_old_regime` is 0 on every row; F .46 / R .49). On 2024, `all` (869.5) and
R (864.8) are nearly identical, which shows which shape 2025 has.

And 2022 also evaluates old-regime F, yet comes out +12. What separates 2021 is not the F
regime but having **only two training seasons**, and 2025 has six. The same trap appeared
with the offset strength (see the v6 section below) — an early holdout pointing the wrong
way is a recurring pattern in this data.

**One suspected cause was ruled out.** `career_f_share` / `career_fold_share` looked like
they might blur the group gap `f_old_regime` already handles, so they were removed — and
2021 came out at 1506.7, **exactly identical**. Not the cause. (On 2024 removing them is
+1.7, but that trades +3.4 on R for −11.2 on F, so given the F transfer history they were
kept.)

### More net seeds make it worse

| Nets | Alone | Blended |
|---|---|---|
| 1 | 818.6 | 891.7 |
| 2 | 835.1 | **892.4** |
| 4 | 832.7 | 891.1 |
| 12 | 837.1 | 890.3 |

The standalone score rises while **the blend falls.** Averaging smooths the predictions until
they resemble the trees, and the very property that justified blending — being wrong in
different places — disappears. Varying the configs (mixing 3/4/6 epochs) gives 891.3, the
same story. There is nothing left to push in the ensemble.

### Where the model is weak (2024 holdout)

| Segment | Share | Score |
|---|---|---|
| Career 1–500 pitches | 6.8% | 522.9 |
| Under 50 pitches this season | 7.2% | 669.6 |
| 50–200 pitches this season | 17.0% | 752.7 |
| No career history (debut) | 19.9% | 1036.3 |

Thin-history pitchers are the weak point, but debut pitchers are actually the best predicted —
they can all simply be collapsed to the league average. The hardest group is the one with a
little history (career 1–500).

### TrackMan can be joined — it is just not worth it

The earlier conclusion ("cannot be matched") was **wrong.** Team codes are real names
(`KIA_TIG`) versus anonymized integers (12–25), but matching on the cosine similarity of a
(month × weekday) schedule fingerprint attaches **9 teams consistently across both 2023 and
2024**, and the remaining one (team 16 = KIA_TIG) falls out by elimination. Matching pitchers
by pitch count within (season, team, handedness) then attaches velocity, spin rate, and
release consistency.

It is not worth it. Control skill is already measured **directly** by the success rate, and
the "thin-history pitcher" group that physical measurements might help is 7–24% per the table
above — lifting that group by 100 points is about +5 overall. Against 2–4 hours of matching
work plus the risk of mismatches, the expected value is close to negative.

### Experiment 31 — take the level from the row instead of the forecast: rejected

Forecast RMSE 0.018 × 401,000 is an expected 130-point tax, larger than the entire v4→v7
resolution gain. And the row-mean of the season-to-date success rate tracks the league level
directly. Experiment 13 tried this as a post-hoc logit shift `z += γ·(a_i − ā)` — where the
same coefficient also catches cross-sectional noise — so experiment 31 measured the opposite
form: put the anchor in as an **offset with its coefficient fixed at 1** and let the tree fit
only the residual (fit `y − a` under squared error, predict `a + g`). That removes the
forecast from the pipeline entirely.

| Arm | 2024 | 2022 | 2021 |
|---|---|---|---|
| `clf_fc` (current) | **855.9** | 2281.0 | 1069.0 |
| `clf_none` (offset off) | 811.7 | **2355.8** | 1519.5 |
| `reg_off` (anchor offset, no forecast) | 811.2 | 2355.6 | **1589.1** |
| `reg_off_fc` | 854.6 | 2270.7 | 1163.9 |
| `anchor` alone | 316.2 | 1181.5 | 891.9 |

**The current setup wins on the representative season (2024), and the offset form does no
self-calibration whatsoever** (`reg_off` 811.2 ≈ `clf_none` 811.7). Even nailing the anchor
in at coefficient 1, the tree undoes just as much through the residual.

The reason is that the anchor's bias is not a constant. **The raw anchor sits about 0.005
below actual for 2019–2023 but 0.003 above it in 2024 — the sign flips.** So applying a bias
constant learned on the training seasons makes it worse (uncorrected +0.0029 → corrected
+0.0081). Used uncorrected it ties the current forecast (+0.0030), so there is no gain.
Shrinking it is worse still: early-season rows have empty accumulators and get shrunk toward
the previous season's league value, which is exactly the stale level being escaped, so the
bias grows in precisely the years the league dropped most (2023's −.029 drop → bias +0.0173).
The anchor's bias is proportional to the drift it is meant to measure — circular.

One side observation. **On 2021 only, `reg_off`'s oracle is 1613.7 against `clf`'s 1552.5,
+61** — resolution genuinely improved. On 2024 it is −1.3, so this looks like a
regularization effect that only appears with two training seasons. 2025 has six, so it does
not apply.

**`fe.py`'s season-to-date reconstruction was verified correct in the process.** All 253,507
rows of 2024 matched a within-season cumsum computed directly from the labels (0.00% n·k
mismatch), and only 0.15% of rows have den=0.

## Axes not yet opened

### train.csv contains five labels, not one

`asof_pitcher_n` is a counter that increases by exactly 1 per pitcher (verified across every
row of train.csv), so differencing consecutive rows of cumulative rate × n **recovers that
pitch's reverse / middle / ball / strike per row.** The rounding error is at most 0.015, and
checking the recovered success against `control_success` gives a **100% match across
1,474,300 rows** (the only unrecoverable rows are each pitcher's last pitch, 792 in total).

And the structure of the target becomes visible.

| reverse | middle | ball | strike | Rows | Success rate |
|---|---|---|---|---|---|
| 0 | 0 | 0 | 1 | 411,782 | .934 |
| 0 | 0 | 1 | 0 | 391,127 | .591 |
| 0 | 0 | 0 | 0 | 163,427 | .958 |
| reverse=1 or middle=1 | | | | 508,164 | **exactly .000** |

`control_success = 1 ⟹ reverse=0 ∧ middle=0` holds without exception. What was being solved
as a single binary target is in fact a multi-stage structure. What follows from that:

- **Multi-task.** The essence of this problem is that the signal is thin (BSS ~0.01), and yet
  the same rows carry four more labels with thicker signal (2024 R pitcher ICC: reverse 1085
  > success 816 > middle 656 > ball 335 > strike 285). Turning the net into 5 heads is the
  most direct move — the net is the largest blend contributor, and the fact that adding seeds
  makes it worse means what it needs is *material for being wrong differently*.
- **Product decomposition.** `P(success) = P(¬reverse) · P(¬middle|¬reverse) · P(success|…)`.
  Every ensemble so far averaged the same target; this multiplies different targets, so the
  way it is wrong is structurally different.
- **New batter-side tables.** The raw data gives batters only success and middle. The
  recovered labels allow batter ball/strike tendencies, and on 2024 R their batter ICCs are
  213 and 225 — larger than the existing batter success (104).

### Rescale the features instead of adding more

`asof_pitcher_success_rate` simply averages a 2019 pitch (success rate .565) with a 2024 one
(.486). v7's regime split only resolved this along the R/F axis and never touched the era
axis. Dividing each cumulative rate by the league level at the time it accumulated — turning
it into "skill relative to era" — is a renormalization of existing features rather than a
feature addition, which makes it a different kind of move from everything rejected so far.

## What changed in v6 — a Futures-only model, and mostly rejections

The v6 round was **almost entirely rejections**. That the ensemble was saturated is this
round's main conclusion, and only one thing survived.

### What survived: the F (Futures) specialist

F is 12% of rows but scores 583.6, under two-thirds of R's 890.2. A model trained on F rows
only (161k rows) is **283 alone** — far worse than the shared model's 583 — but in exchange
it never has to compromise with the other 88%. Blended at 25% on F rows only, F goes
583.6 → **638.3** and the total 892.4 → **897.9**. R rows are left untouched.

### What was rejected

| Attempt | 2024 holdout | Verdict |
|---|---|---|
| v5 stack (baseline) | 892.4 | — |
| Adding a linear model | 888.8 | Rejected |
| MLP 1024-512 / 256-128-64 / squared error | 891.7 / 892.3 / 892.1 | All null |
| **Player-embedding MLP** | 892.2 | Null (19.9% of 2024 rows are unseen pitchers) |
| Tree max_features 0.6 / 0.3 | 891.6 / 891.2 | Rejected |
| lr 0.02 × iter 1200 | 890.9 | Rejected |
| leaf63 msl500 | 892.5 | Null |
| Recent seasons only (>=2021 / >=2022) | 893.5 / 892.7 | Null |
| Optimizing blend weights | 891.9 (v5 weights 891.1) | Null — and 886.0 when applied to another season |
| Per-`game_type` blend weights | own season 896.2, other season 2375.9 (v5 2387.0) | Failed to generalize |

### Should the offset strength be lowered? No

Sweeping offset strength s across four seasons made s=0 (offset removed) look overwhelmingly
better (mean 1002.7 → 1169.0). Broken out by season, the story inverts.

| Holdout | Train seasons | Forecast error | Optimal s |
|---|---|---|---|
| 2021 | 2 | −0.0301 | 0 |
| 2022 | 3 | −0.0135 | 0 |
| 2023 | 4 | +0.0155 (F regime collapse, unforecastable) | — |
| 2024 | 5 | **+0.0011** | ~0.9 |

**The forecasting rule improves as training seasons accumulate.** Every signal saying "drop
the offset" came from early holdouts with only two or three seasons. 2025 uses six, so the
2024 holdout is the only representative observation, and there the current 0.85 is right
(solving with the v5 stack's own drift of −0.0020 gives optimal s=0.82). **Unchanged.**

## What changed in v5 — ensemble diversity

Resolution had already been squeezed out of features in v4. What remained was **blending
models that are wrong in different places.**

### The net: weak alone, a large gain when blended

A small MLP (256-128, GELU) scores 835 alone, far below the tree blend's 870. Yet blending it
at w=0.35 gives **2021 +27 / 2022 +16 / 2024 +23** — the same direction on all three usable
holdouts (2023 is unusable; even the oracle scores 0 there because of the F regime collapse).
Trees cut with axis-perpendicular steps while an MLP mixes every feature at every unit and
interpolates smoothly, so they are wrong in different places.

**More epochs collapse it.** 3 epochs 835 → 6 epochs 777 → **15 epochs 0**. With BSS around
0.01 there is almost no signal to fit, and everything past that is memorized noise. So it
stops at 3.

torch is used for training only. The artifact carries weight matrices and standardization
constants, and inference is three numpy matrix multiplies (`work/nnpred.py`).

### Two tree capacities: R and F want different ones

leaf15 scores R 858.8 / F 551.8; leaf31 (l2=1, msl=2000) scores R 839.6 / F 600.4. Each is
good at one side only. Blending beats switching on `game_type` (869.7 vs 866.8). The looser
model's weight is set to **0.35**, not 2024's optimum of 0.4 — on the 2021 holdout, which has
only two training seasons, 0.4 costs 47 points.

### Rejected in v5

| Attempt | 2024 holdout oracle | Verdict |
|---|---|---|
| Squared-error regression (matching the metric's loss) | 860.0 vs log-loss 860.9 | Null |
| Logit slope correction a≠1 | a=1.0 optimal (a=0.9 → 853.5) | Null; the model is not overconfident |
| Per-`game_type` offset | 853.6 vs 852.7 | Null |
| Season weighting (half-life 3/5) | 861.8 / 859.0 vs 860.9 | Null; volume wins |
| Shrinkage K tuning (60/600, 100/1000) | all at or below parity | Already at the optimum |
| Two-season window reconstruction | 858.3 | Null |

## What changed in v4

### 1. The `asof_*` counters carry from train into test → season-to-date is recoverable

`asof_pitcher_n` is a **career pitch counter that does not reset each season**. And that
counter carries straight into the evaluation data — confirmed with the published 5-row
`test.csv` sample (pitcher 21813 has 3085 pitches at the end of 2024 in train.csv, and 3465
in test.csv).

So subtracting **that player's career total at the end of the previous season** (a constant
learned from train.csv) from each row's asof value recovers how that player has done
**so far this season**.

```
season_n    = asof_n           − career pitches through the previous season
season_succ = asof_rate·asof_n − career successes through the previous season
```

Why this matters: `asof_pitcher_success_rate` is a career cumulative figure, so a
ten-year veteran's value barely moves. Across 249 pitchers with 200+ pitches in 2024, the
**correlation between career success rate and that year's actual rate is only 0.19**, while
the reconstructed season-to-date figure correlates **0.97**. The model had been judging
pitchers by a ten-year average. The prev1/3/5-game values are the opposite problem — too
noisy — and season-to-date fills the gap between them.

**Rule safety**: the subtracted value is a per-player constant learned from train.csv, and
everything else is that row's own columns. No other row of the evaluation data is consulted,
so the "row-independent prediction" clause is not violated.

Applied to pitchers (success / reverse / middle / ball / strike), batters (success / middle),
and the pitch mix. Contribution (2024 holdout oracle): pitchers +19, **batters +36**.

### 2. `game_type` F changed definition in 2023 — a step, not a trend

F is the Futures (second-tier) league, matching the `MIN_*` team markers in
`trackman_history.csv`. Its success rate collapses from .709 in 2022 to .473 in 2023 — and
**all 13 teams fall by the same amount.** The teams did not get worse; the measurement or
judgment standard changed.

Two things follow.

- **The 2022→2023 drop in the overall success rate (−.0289) is almost entirely this F step.**
  R moved only −.0006 over the same period. Fitting a straight line to the overall series
  mistakes a one-off step for a trend.
- **Old-regime F rows must be isolated.** The single flag
  `f_old_regime = (game_type==F) & (season<2023)` takes the 2024 holdout from 816 to 843
  (+27), and the F subset alone from 412 to 533. On 2025 rows this flag is always 0.

### 3. The holdout evaluation itself was broken

The old `work/evaluate.py` judged on the average of 2023 and 2024, but **the 2023 holdout
scores 0 even given oracle calibration.** A model trained on 2019–2022 predicts .71 for F
rows when the truth is .47 — worse than a constant. That zero dominated the average, and it
is why experiments 08–10 were all rejected.

Now `work/lab/harness.py` reports **all / R-only / F-only** separately, along with seed noise
(±4) and a drift diagnostic (beta).

### 4. Calibration: segmented forecast at 85% strength

`work/forecast.py` extrapolates R and F separately. R is the mean of linear, mean_delta, and
delta3; F uses only the post-2023 window and averages it with "F moves as much as R moves."
Backtest RMSE over 2021–2024 is 0.0182 (a plain line gives 0.0196), and the 2024 error is
+0.0011 (line: +0.0057).

The offset is applied at **85% strength** only. The season-to-date features already carry the
2025 level, so the model's own predicted mean drifts down by itself (0.18–0.25 of the true
drift on the 2024 holdout, decreasing as training seasons accumulate). Applying 100% would
subtract the drift twice.

On the 2024 holdout this procedure targets 0.4891 and achieves 0.4862, against a truth of
0.4861. Calibration loss is effectively zero, so **the adjusted score equalled the oracle.**

### 5. What was rejected

| Attempt | 2024 holdout oracle | Verdict |
|---|---|---|
| Baseline (v4) | 855.1 | — |
| Pitcher-vs-batter contrast (`std_success − bat_std_success`) | 860.7 | Adopted |
| Park ID (home team, derivable) | 830.3 | Rejected |
| Multiple shrinkage K | 852.0 | Rejected |
| Empirical Bayes from the previous season | 853.2 | Rejected |
| prev1/3/5 game window decomposition | 846.6 | Rejected |
| Pitch count / workload | 849.8 | Rejected |
| Dropping the `season` column | 686.3 | Rejected |
| leaf 31 / iter 800 | 791 / 790 | Rejected (more capacity loses) |

**TrackMan still could not be joined at this point.** `pitcher_trackman_id` and `pitcher_id`
have an empty intersection, and the team codes use different systems (real names like
`KIA_TIG` versus anonymized 12–25). Matching by season / team / pitch-count fingerprint is
theoretically possible, but what it yields is physical measurements like velocity and spin
rate, while control skill is already measured directly by the success rate — a poor return on
investment. (v7 revisited this and found the match *is* possible; see above. The conclusion
about its value did not change.)

## Environment trap

The evaluation server runs **numpy 1.26.4**. A pickle produced under numpy 2.x **fails to
load at all** on the server with a `numpy._core` ImportError. Always train and verify inside
`.venv-submit`.

```bash
python -m venv .venv-submit
.venv-submit/Scripts/python.exe -m pip install "numpy==1.26.4" "scikit-learn==1.8.0" "joblib==1.5.3" "pandas==2.2.3"
```

`requirements.txt` is left as comments only — using nothing beyond the server's default
packages makes the risk of an install error zero.

## Layout

```
open/                     competition distribution -- entirely excluded from git,
                          see "Competition materials are not redistributed here"
work/
  fe.py                   feature engineering — the single source shared by
                          training, validation, and submission.
                          extras=False -> view A (107), extras=True -> view B (135).
                          EXTRA_COLS is the only list separating the two views.
  nnpred.py               net preprocessing + inference (numpy only, no torch)
  forecast.py             next-season success rate forecast (R/F separated)
  train_nn.py             net training -> nn_weights.npz (system python, needs torch)
  train_v4.py             tree training + npz absorption -> submit/model/model.joblib
  build_submit.py         inlines fe.py + nnpred.py to generate submit/script.py + zip
  verify_v4.py            evaluation-server rehearsal (numpy 1.26 load / parity / timing)
  lab/                    experiment harness and records (00-31)
submit/                   contents of the submission zip
versions/                 per-version snapshots (code + weights + scores)
```

`submit/script.py` is **never edited by hand.** If its preprocessing diverges from training,
the result is silently wrong predictions with no error — the most expensive mistake available
in this competition. It is generated by inlining `fe.py` verbatim, and `verify_v4.py` checks
that both paths produce identical output.

## Usage

```bash
cd work
python train_nn.py --view B --validate                      # net (system python)
../.venv-submit/Scripts/python.exe train_v4.py --validate --seeds 2   # holdout

python train_nn.py --view B                                 # full net
../.venv-submit/Scripts/python.exe train_v4.py              # full trees + merge
python build_submit.py                                      # script.py + submit.zip
../.venv-submit/Scripts/python.exe verify_v4.py             # server rehearsal
python snapshot.py v7 --score <LB> --note "..."
```

`--view` must match `NN_VIEW` in `train_v4.py`. If they disagree, `train_v4.py` halts
immediately on a column-count mismatch during training — deliberately, so it cannot fail
silently.

## Competition rules worth noting

- 5 submissions per day. **Install errors are not counted, but an error during `script.py`
  is.**
- No external data or APIs; no 2025 TrackMan.
- No post-hoc correction using aggregate statistics of the evaluation data. **Tuning a
  constant against the leaderboard score falls under the same clause** — every correction
  must be derived from the training data alone.

---

<a id="korean"></a>

# LG Aimers 9기 Phase2 — 투구 제구 성공 확률 예측

[English](#english) · **한국어**

[대회 페이지](https://dacon.io/competitions/official/236743) · 리더보드 마감 2026-09-01 · 코드+PPT 마감 2026-09-07

KBO 투구 한 건이 "제구 성공"일 확률을 예측한다. 2019~2024 시즌 147만 투구로 학습해
2025 시즌 25만 투구를 맞히는 문제다.

평가식이 Brier Skill Score라서 행별 변별력만큼이나 **예측 평균**이 크게 걸린다.
그런데 학습은 2019~2024이고 평가는 2025인데 성공률이 매년 떨어진다 — 이 문제의 절반은
해상도가 아니라 드리프트 보정이다 (아래 [이 문제의 구조](#이-문제의-구조)).

**최종 리더보드 Public 998.53 — 1,087명 중 437위.** v1의 897.09에서 올라온 값이다.
최종 점수는 v6의 것이다. v7은 마감 전에 제출했지만 v6을 넘지 못했고, 이 프로젝트에서
홀드아웃 이득이 전혀 전이되지 않은 유일한 사례가 됐다 — [v7은 어떻게 끝났나](#v7은-어떻게-끝났나) 참고.

## 버전 발전사

각 버전은 커밋 하나이고, `versions/vN/`에 그 시점의 코드·점수·판단 근거가 통째로 남아 있다.

| 버전 | 날짜 | LB Public | 2024 홀드아웃 | 핵심 변경 |
|---|---|---|---|---|
| [v1](versions/v1_LB897.09) | 08-17 | 897.09 | 710.1 | 기본 47피처 HistGBM 4시드 + 시즌추세 logit 보정 |
| [v2](versions/v2_LB908.53) | 08-17 | 908.53 | 765.9 | 카운트·상황·최근폼 파생 23개 (총 70피처) |
| [v3](versions/v3) | 08-19 | 922.86 | 770.2 | 용량이 다른 HistGBM 블렌딩 (w=0.3). 시드 증가는 효과 0 |
| [v4](versions/v4_LB947.49) | 08-22 | **947.49** | 855.9 | **시즌 누적 성적 복원** + 2군 레짐 격리 + 세그먼트 예보 |
| [v5](versions/v5_LB985.27) | 08-22 | **985.27** | 890.0 | 소형 MLP 블렌딩(w=0.35) + 트리 용량 2종 |
| [v6](versions/v6_LB998.53) | 08-22 | **998.53** | 895.6 | F(2군) 전용 모델을 F 행에만 25% 블렌딩 |
| [v7](versions/v7) | 08-29 | 개선 없음 | **917.6** | 피처 뷰 2벌(107·135컬럼) 블렌딩 |

v4에서 홀드아웃 하네스를 갈아서 v3 점수가 두 번 찍힌다 — 구 하네스 770.2, 신 하네스 768.6.
표의 v3은 구 하네스 값이고, 아래 전이율 표는 신 하네스 기준이다.

### 홀드아웃은 크기를 못 믿고 부호만 믿는다

이 프로젝트에서 제일 값비싼 교훈이다. 2024 홀드아웃 개선분이 리더보드로 얼마나
전이되는지는 매번 달랐다.

| 구간 | 홀드아웃 Δ | 리더보드 Δ | 전이율 |
|---|---|---|---|
| v1 → v2 | +55.8 | +11.4 | 20% |
| v2 → v3 | +4.3 | +14.3 | 330% |
| v3 → v4 | +87.3 | +24.6 | 28% |
| v4 → v5 | +34.1 | +37.8 | 111% |
| v5 → v6 | +5.6 | +13.3 | 237% |
| v6 → v7 | +22.0 | 없음 | **0%** |

20%에서 330%까지 흩어지다가 마지막에 0이 됐다. **비율은 처음부터 못 쓸 물건이었고,
마지막 한 번은 부호까지 틀렸다.** 다섯 라운드 동안 홀드아웃은 방향만은 맞혔고 그래서
채택/기각 판정에만 쓰고 점수 예측에는 쓰지 않았다. v7은 그마저 깨진 라운드다.

전이율이 v5·v6에서 100%를 넘긴 데는 이유가 있다 — 2024 홀드아웃은 F(2군)를
구조적으로 과소평가한다. 홀드아웃 모델은 새 F 레짐을 2023년 한 해치만 보고 학습하는데
실제 제출 모델은 2023+2024 두 해치를 본다. 이게 "F 쪽에 더 투자하라"는 신호였고,
v6·v7이 그 방향이었다.

### 무엇이 실제로 점수를 올렸나

7개 버전에서 채택된 건 손에 꼽고, 나머지 60여 개 시도는 전부 기각됐다.
채택된 것들만 모으면 축이 세 개뿐이다.

1. **누수 없는 정보를 더 캐낸 것** (v4, +24.6). `asof_*` 카운터가 train에서 test로
   이어진다는 걸 발견해 "올 시즌 지금까지 성적"을 행 독립적으로 복원했다.
   통산 성공률과 그 해 실제 성적의 상관은 0.19인데, 복원한 시즌 누적은 0.97이다.
2. **다르게 틀리는 모델을 섞은 것** (v5, +37.8). 단독 835점짜리 MLP가 870점짜리
   트리 블렌드에 섞이면 올린다. 정확도가 아니라 다양성이다.
3. **다수에 맞추느라 희생되던 소수를 따로 본 것** (v6, +13.3). F는 전체의 12%인데
   점수가 R의 3분의 2도 안 됐다.

v7은 1번과 2번의 결합이다. 앙상블은 v6에서 완전히 포화였고 — 배깅·ExtraTrees·잔차
스태킹까지 전부 기준 아래였다 — 공통점은 **전부 같은 피처 테이블 위에서 모델만 바꿨다**는
것이었다. 피처 테이블을 두 벌로 나누자 그 축은 포화가 아니었다.

## v7은 어떻게 끝났나

v7은 2026-09-01 마감 전에 제출했고 **v6을 개선하지 못했다.** 정확한 점수는 기록해 두지
않았다. v6의 998.53이 그대로 최선으로 남았고 그게 최종 public 결과다. 2024 홀드아웃은
+22.0이라고 말했었다.

이건 분명히 적어야 한다. 이 README는 첫 화면 전체를 "홀드아웃은 크기는 못 믿어도 부호는
믿을 수 있다"는 주장에 쓰고 있다. 다섯 라운드가 그 주장을 뒷받침했고, 여섯 번째가
뒷받침하지 않았다. 아래 v7 절은 **잘 측정했지만 실제 평가셋과의 접촉에서 살아남지 못한
변경**으로 읽어야 한다.

검증되지 않은 가설 두 가지:

- **2024 홀드아웃은 학습 시즌이 5개고 2025는 6개다.** v7의 모든 상수 — 뷰 혼합 0.5,
  신경망 가중치 0.35, F 전용 0.25 — 는 2024에서 고르고 2022에서 확인한 것이다. 아래 절에
  이미 적혀 있듯 학습 시즌이 2개뿐인 2021은 −29.8로 반대를 가리켰다. 2021을 반대로 만든
  그 레짐이 2025에서 완전히 사라지지는 않았을 수 있다.
- **뷰 B의 이득은 F에 몰려 있었고 F는 전체의 12%다.** F의 전이율은 이 프로젝트에서
  가장 높았지만(v6에서 237%), 그 패턴이 그냥 소진됐을 수 있다.

검증된 것: 학습·패키징·평가 서버 리허설은 전부 통과했다.

```
[OK] artifact loads (15.8 MB), tightA x4 + looseA x4 + tightB x4 + looseB x4
     + nn x4 w=.350 + f_spec x8 w=.25, 135 features
[OK] script.py matches fe.py on 20000 rows x 135 features
[OK] net inference matches (4 nets) -- numpy only, no torch
[OK] 250000 rows, wall clock 86.3s (limit 600s)
```

처음엔 트리 그룹당 8시드로 빌드했고 159.5초 / 25.4MB였다. 4시드로 줄여
86.3초 / 15.8MB가 됐다 — 이유는 아래 "시드는 8개가 아니라 4개다" 절에 있다.

리허설의 `predicted mean`은 캘리브레이션 예고가 아니다. `verify_v4.py`는
train.csv 마지막 25만 행의 `season`만 2025로 바꿔 쓰는데, 그 행들의 asof 카운터가
이미 2024년 말 값이라 시즌 누적이 전원 0으로 붕괴한다. 형식·시간·크래시 점검용이다.

홀드아웃과 리허설이 실제로 확인해 준 것:
- 2024 홀드아웃 端-to-端 917.6 (2시드), v6 895.6 대비 +21.9
- 2022에서 모든 상수(뷰 혼합 0.5, 신경망 뷰 B, F 전용 w=0.25)가 같은 위치에서 최적
- 배포되는 `fe.py`가 랩 빌더와 소수점까지 일치 (실험 28의 Aprod/Bprod)

## 저장소를 읽는 순서

- **발전 과정만 보고 싶다면** — `git log` 를 v1부터 따라가거나, 태그 `v1`~`v7`, 또는 각 `versions/vN/NOTES.md`
- **최종 방법론** — 아래 "v7에서 바뀐 것"부터 "v4에서 바뀐 것"까지 (역순 서술)
- **기각된 시도 60여 개** — 각 버전 절의 기각 표, 그리고 `work/lab/00~31`
- **재현** — 아래 "사용법"

### 대회 자료는 재배포하지 않는다

`open/` 아래 전부 — 대회 데이터(689MB), 공식 데이터 설명서, 제공된 베이스라인 제출본 —
는 Dacon / LG Aimers의 제3자 자료다. 이 저장소에서 제외했고, 단순히 추적 해제한 게
아니라 히스토리에서도 제거했다. [대회 페이지](https://dacon.io/competitions/official/236743)에서
받으면 되고, 이 저장소의 코드는 `open/data/` 위치를 기대한다.

랩 캐시(`work/lab/cache/`)도 제외했다 — `work/lab/00_cache.py`로 재생성한다.
중간 버전(v4~v6)의 모델 바이너리도 제외했고, 각 버전의 `code/`와 `train_v4.py`로
재현된다. 최종 v7 아티팩트는 `submit/`과 `versions/v7/`에 들어 있다.

### 라이선스

이 저장소의 코드는 MIT다 ([LICENSE](LICENSE)). 저장소 자체의 코드와 문서만 해당하며,
위에 적은 대회 데이터·문서·베이스라인에는 미치지 않는다. 그것들은 여기서 배포하지 않고
각 소유자의 조건을 따른다.

## 이 문제의 구조

평가식이 Brier Skill Score(`100000 × (1 − MSE / r(1−r))`)라서 두 가지가 전부다.

1. **평균(캘리브레이션).** 예측 평균이 실제보다 δ 어긋나면 약 `401,000 × δ²`점이 날아간다.
   0.01 어긋나면 40점, 0.02면 160점. 상수 예측은 정확히 0점.
2. **해상도.** 남는 건 행별 변별력인데 이 문제의 BSS는 0.01 수준이라 신호가 아주 얇다.

그리고 `train.csv`는 2019~2024, 평가 데이터는 2025다. 성공률이 매년 떨어진다.

| 시즌 | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 |
|---|---|---|---|---|---|---|
| 전체 | .5647 | .5327 | .5328 | .5289 | .5000 | .4861 |
| R(1군) | .5495 | .5269 | .5128 | .5037 | .5031 | .4897 |
| F(2군) | .6892 | .5878 | .7038 | .7087 | **.4729** | .4593 |

## v7에서 바뀐 것 — 다양성의 축이 모델이 아니라 피처였다

v7은 "앙상블이 포화됐다"는 v6의 결론을 확인하는 것으로 시작했고, 실제로 확인됐다.
알고리즘 계열을 통째로 갈아도 아무것도 안 올랐다.

| 시도 | 2024 홀드아웃 | 판정 |
|---|---|---|
| v6 기준 | 892.4 (전용모델 없음) / 897.9 | — |
| 행 부트스트랩 배깅 leaf15 ×4 | 단독 828.6 → 블렌드 889.8 | 기각 |
| ExtraTrees depth12 / depth20 | 단독 613.7 / 544.6 → 890.0 / 888.7 | 기각 |
| 잔차 순차 스태킹 | 891.2 | 기각 |
| F 전용 모델 심화 (학습구간 4종 × 설정 2종) | 최선 898.4 | +0.5, 노이즈 |
| F를 2023+ 레짐만으로 학습 | 895.8 | 기각 — 25k행은 너무 적다 |
| F 전용 신경망 | 890.0 | 기각 |
| 팀 직전시즌 성공률 | 831.2 | **기각 (−29)** |
| 팀 사전정보를 이력 얇은 투수에만 | 842.2 | 기각 (−18) |
| 신경망 시드 12개 | 890.3 | 기각 |

여기서 공통점이 보인다. **전부 같은 피처 테이블 위에서 모델만 바꾼 것이다.**
MLP 모양, 트리 용량, 배깅, 다른 알고리즘, 잔차 스태킹 — 다양성을 모델에서만 찾았다.
한 번도 안 건드린 축이 피처 테이블 자체였고, 거기는 포화가 아니었다.

### 두 개의 피처 뷰

`fe.py`가 이제 두 벌의 컬럼을 만든다. `extras=False`가 기존 107개(뷰 A),
`extras=True`가 여기에 28개를 더한 135개(뷰 B)다. 같은 트리를 양쪽에 각각 학습시켜
평균낸다.

| 2024 홀드아웃 (트리만, 4시드) | all | R | F |
|---|---|---|---|
| 뷰 A 단독 | 869.5 | 864.8 | 579.3 |
| 뷰 B 단독 | 878.0 | 870.4 | 613.2 |
| **A + 0.5·B** | **884.0** | 877.2 | 610.3 |

이게 정확도가 아니라 **다양성**이라는 증거는 중간 단계에 있다. 레짐 계열만 넣은 뷰는
단독 868.2로 A(869.5)보다 **낮은데**, 섞으면 874.8로 둘 다보다 높다. 낮은 모델을
섞어서 올라간다는 건 정의상 다르게 틀린다는 뜻이다.

혼합비는 0.5로 둔다. 2024는 0.3/0.4/0.5/0.6에서 880.6/882.7/884.0/884.4로 평평하고,
2022도 2372.2/2373.5/**2373.7**/2372.8로 0.5가 꼭짓점이다.

### 뷰 B에 들어간 것

전부 train.csv에서 학습한 (선수, 시즌) 상수를 행별로 조회하는 형태다 — 기존 캐리
테이블과 같은 구조라 "행 독립 예측" 조항을 그대로 지킨다.

**레짐 분리.** `asof_pitcher_success_rate`는 측정 규칙이 다른 두 시대의 투구를 섞어
평균낸 값이다. 그런데 투수별 F 비중이 중앙값 .19에 사분위가 .01 / .89로 **전 구간에
퍼져 있다.** 오염 정도가 투수마다 완전히 달라서 통합 비율 하나로는 트리가 되돌릴 수
없다. 커리어 캐리 테이블을 R / 신규F(≥2023) / 구F(<2023)로 쪼개서 깨끗한 비율,
그 행의 레짐에 맞는 사전값, 오염 비율을 따로 준다. 매 시즌 R 투수의 절반가량이
F에도 등판하므로 이건 드문 구석이 아니다.

**플래툰.** 투수의 좌우 상대별 성적, 타자의 좌우 투수별 성적. `same_hand`는 리그
평균판만 담고 있고 개인 편차가 빠져 있었다.

**카운트.** 투수가 유리한 카운트일 때와 불리할 때의 차이.

플래툰·카운트는 이미 작은 이력을 또 쪼개는 것이라 노이즈가 크다. 그래서 각 버킷을
리그가 아니라 **그 선수 자신의 커리어 비율** 쪽으로 수축시키고, 차이값으로 넘긴다 —
분할 증거가 없는 선수는 정확히 0이 된다.

세 계열은 따로 떼면 안 된다. 카운트만 넣은 뷰는 섞어도 +2.6, 레짐만은 +3.4인데,
셋을 다 넣으면 +14.5다.

### 신경망도 뷰 B로 옮긴다 — 여기가 제일 컸다

트리에서 얻은 다양성이 신경망에서 더 크게 나왔다.

| 2024, 트리 블렌드 884.0 위에 | w=0.25 | w=0.35 | w=0.45 | w=0.55 |
|---|---|---|---|---|
| 뷰 A 신경망 (단독 835.1) | 901.6 | 903.0 | 901.3 | 896.5 |
| **뷰 B 신경망 (단독 867.1)** | 910.2 | **915.0** | 916.6 | 914.9 |

가중치는 **0.35 그대로 둔다.** 2024는 0.45가 꼭짓점(916.6)이지만 0.35와의 차이
1.6은 시드 노이즈 안이고, README 아래에 적힌 대로 블렌드 가중치를 2024에 맞췄다가
다른 시즌에서 5점을 잃은 전례가 있다. 게다가 2022에서는 0.35까지 계속 오른다 —
즉 0.35는 양쪽 홀드아웃 모두에서 꼭짓점 **이전**이라 안전한 쪽이다.

### F 전용 모델은 뷰 A에 남긴다

레짐 분리가 F를 크게 올렸으니 F 전용 모델도 뷰 B로 옮기는 게 자연스러워 보이는데,
재보면 반대다: 뷰 B 기반 917.4 vs 뷰 A 기반 **919.8**. F 행만 보는 모델에게는
레짐 분리가 더 해 줄 말이 별로 없다. 용량 두 개(leaf31 msl1000 + leaf15 msl2000)를
평균내고 w=0.25로 F 행에만 섞는다.

### 최종 스택

| 2024 홀드아웃 (오라클) | |
|---|---|
| 트리 뷰 A만 | 869.5 |
| + 뷰 B 평균 | 884.0 |
| + 신경망 뷰 A (w=0.35) | 903.0 |
| + 신경망 **뷰 B** (w=0.35) | **915.0** |
| + F 전용 뷰 A 2용량 (w=0.25) | **919.8** |

실제 파이프라인으로 端-to-端 재현: 오라클 918.9, 오프셋 적용 후 **917.6**
(R 909.2 / F 654.8). v6의 897.9 대비 **+21.9**.

### 시드는 8개가 아니라 4개다

트리 그룹당 8시드로 먼저 빌드했더니 추론이 **159.5초**(제한 600초), zip이 25.4MB였다.
16코어 노트북 기준이고 HistGBM 예측은 스레드 병렬이라, 4코어 평가 서버라면 400~500초로
한도에 붙는다. 규칙상 `script.py` 실행 중 오류는 제출 횟수가 차감된다.

4시드로 줄이면 **86.3초 / 15.8MB**가 되고 비용은 사실상 없다. 시드 평균은 기댓값을
올리는 게 아니라 분산만 줄이고 4개면 이미 평평하다 — 블렌드 단계에서 2시드 869.8 vs
4시드 869.5였다. 실제로 8시드와 4시드가 계산한 기준 평균은 0.4882 vs 0.4881,
오프셋은 −0.0462로 완전히 같았다.

F 전용 모델은 8개(용량 2종 × 4시드)를 그대로 둔다. F 행이 12%뿐이라 추론에서 3초밖에
안 쓰는데, F 개선은 리더보드 전이율이 가장 높았던 구간이다.

### 2021 홀드아웃은 음수다 — 그런데 채택한다

정직하게 적는다. 세 시즌에서 A + 0.5·B의 효과는 이렇다.

| 홀드아웃 | 학습 시즌 | 평가 F 레짐 | all | R |
|---|---|---|---|---|
| 2021 | 2 | 구 | **−29.8** | −1.4 |
| 2022 | 3 | 구 | +12.0 | +18.5 |
| 2024 | 5 | 신 | +14.5 | +12.4 |
| 2025 (실제) | 6 | 신 | — | — |

R만 보면 −1.4 / +18.5 / +12.4로 의미 있게 음수인 적이 없다. 2021의 −29.8은 전부
F에서 나오고, 2021의 all이 R의 세 배(1536 vs 511)인 것이 그 이유를 말해 준다 —
그 해 all은 사실상 "구F(.70) 대 R(.51)" 집단 격차를 맞히는 문제이고, 2025에는
그 격차도 그 행도 없다(`f_old_regime`이 전 행 0, F .46 / R .49). 2024에서는
all(869.5)과 R(864.8)이 거의 같다는 것이 2025가 어느 쪽 모양인지를 보여 준다.

그리고 2022도 구 레짐 F를 평가하는데 +12다. 2021을 가르는 건 F 레짐이 아니라
**학습 시즌이 2개뿐**이라는 것이고, 2025는 6개다. 오프셋 강도 때도 같은 함정이
있었다(아래 v6 항목) — 초기 홀드아웃이 반대 방향을 가리키는 건 이 데이터에서
반복되는 패턴이다.

**의심한 가설 하나는 기각됐다.** `career_f_share` / `career_fold_share`가 이미
`f_old_regime`이 담당하는 집단 격차와 겹쳐서 흐리는 것 아니냐고 보고 빼봤는데,
2021이 1506.7로 **완전히 동일**했다. 원인이 아니다. (2024에서는 빼는 쪽이 +1.7인데
R에서 +3.4를 얻고 F에서 −11.2를 내주는 거래라, F 전이율 이력을 감안해 유지했다.)

### 신경망은 시드를 늘리면 나빠진다

| 신경망 수 | 단독 | 블렌드 |
|---|---|---|
| 1 | 818.6 | 891.7 |
| 2 | 835.1 | **892.4** |
| 4 | 832.7 | 891.1 |
| 12 | 837.1 | 890.3 |

단독 점수는 올라가는데 **블렌드는 내려간다.** 평균낼수록 예측이 매끄러워져 트리와
비슷해지고, 블렌딩의 근거였던 "다르게 틀리는 성질" 자체가 사라지기 때문이다.
설정 다양화(3/4/6에폭 혼합)도 891.3으로 마찬가지. 앙상블은 더 밀 데가 없다.

### 모델이 약한 구간 (2024 홀드아웃)

| 구간 | 비중 | 점수 |
|---|---|---|
| 통산 1~500구 | 6.8% | 522.9 |
| 시즌 50구 미만 | 7.2% | 669.6 |
| 시즌 50~200구 | 17.0% | 752.7 |
| 통산 이력 없음(데뷔) | 19.9% | 1036.3 |

이력이 얇은 투수가 약점이지만, 데뷔 투수는 오히려 가장 잘 맞는다 — 모두 리그 평균으로
수렴시키면 되기 때문이다. 어중간하게 이력이 있는 쪽(통산 1~500구)이 가장 어렵다.

### TrackMan은 붙일 수 있다 — 다만 값어치가 없다

이전 결론("매칭 불가")은 **틀렸다.** 팀 코드는 실명(`KIA_TIG`) vs 익명(12~25)이지만,
(월 × 요일) 일정 지문의 코사인 유사도로 맞추면 2023·2024 두 시즌에서 **9개 팀이
일관되게 붙고** 남은 하나(팀 16 = KIA_TIG)는 소거법으로 정해진다. 그 다음 (시즌, 팀,
좌우) 안에서 투구 수로 투수를 맞추면 구속·회전수·릴리스 일관성을 붙일 수 있다.

그런데 값어치가 없다. 제구 실력은 이미 성공률로 **직접** 측정되고 있고, 물리량이 도울
만한 "이력 얇은 투수"는 위 표대로 7~24%인데 그 구간이 100점 올라도 전체 +5 수준이다.
2~4시간짜리 매칭 작업에 매칭 오차 위험까지 감안하면 기대값이 음수에 가깝다.

### 실험 31 — 레벨을 예보 대신 행에서 받기: 기각

예보 RMSE 0.018 × 401,000 = 기댓값 130점의 세금이라, 이건 v4→v7 해상도 이득 전체보다
크다. 그런데 시즌 누적 성공률의 행 평균은 리그 레벨을 직접 따라간다. 실험 13은 이걸
사후 로짓 이동 `z += γ·(a_i − ā)`로 시도했고 — 같은 계수가 횡단면 노이즈에도 걸린다 —
실험 31은 반대 형태를 쟀다: 앵커를 **계수 1로 고정한 오프셋**으로 넣고 트리는 잔차만
학습(`y − a`를 제곱오차로 적합, 예측 = `a + g`). 그러면 예보가 파이프라인에서 사라진다.

| 팔 | 2024 | 2022 | 2021 |
|---|---|---|---|
| `clf_fc` (현행) | **855.9** | 2281.0 | 1069.0 |
| `clf_none` (오프셋 끔) | 811.7 | **2355.8** | 1519.5 |
| `reg_off` (앵커 오프셋, 예보 없음) | 811.2 | 2355.6 | **1589.1** |
| `reg_off_fc` | 854.6 | 2270.7 | 1163.9 |
| `anchor` 단독 | 316.2 | 1181.5 | 891.9 |

**대표성 있는 2024에서 현행이 최고고, 오프셋 형태는 자가 캘리브레이션을 전혀 못 한다**
(`reg_off` 811.2 ≈ `clf_none` 811.7). 앵커를 계수 1로 박아 넣어도 트리가 잔차에서
그만큼 되돌려놓는다.

이유는 앵커 편차가 상수가 아니라는 것이다. **원시 앵커는 2019~2023에서 실제보다 약
0.005 낮은데 2024에서는 0.003 높다 — 부호가 뒤집힌다.** 그래서 학습 시즌에서 배운
편차 상수를 적용하면 오히려 나빠진다(무보정 +0.0029 → 보정 후 +0.0081). 무보정으로
써도 현행 예보(+0.0030)와 동률이라 이득이 없다. 수축을 걸면 더 나쁘다: 시즌 초 행은
누적이 비어 직전 시즌 리그값으로 수축되는데 그게 곧 벗어나려던 낡은 레벨이고, 그래서
리그가 크게 떨어진 해일수록 편차가 커진다(2023 하락 −.029 → 편차 +0.0173). 즉 앵커의
편차가 드리프트 크기에 비례해서 순환이다.

부수 관측 하나. **2021에서만 `reg_off`의 오라클이 1613.7로 `clf`의 1552.5보다 +61
높다** — 해상도가 실제로 올랐다. 2024에서는 −1.3이라 학습 데이터가 2시즌뿐일 때만
나오는 정규화 효과로 보인다. 2025는 6시즌이므로 해당 없다.

**`fe.py`의 시즌 누적 재구성은 이 과정에서 정확성이 확인됐다.** 2024년 253,507행 전부
라벨에서 직접 계산한 within-season cumsum과 일치했고(n·k 불일치 0.00%), den=0인 행은
0.15%뿐이다.

## 아직 안 열어본 축

### train.csv에는 라벨이 5개 들어 있다

`asof_pitcher_n`은 투수별로 정확히 1씩 증가하는 연속 카운터라(train.csv 전 행 확인),
누적 rate × n의 연속 행 차분으로 **그 투구의 reverse / middle / ball / strike를 행마다
복원**할 수 있다. 반올림 오차 최대 0.015로 안전하고, 복원한 success를 `control_success`와
대조하면 1,474,300행에서 **일치율 100%**다(복원 불가는 각 투수의 마지막 투구 792행뿐).

그리고 타깃의 구조가 드러난다.

| reverse | middle | ball | strike | 행 수 | 성공률 |
|---|---|---|---|---|---|
| 0 | 0 | 0 | 1 | 411,782 | .934 |
| 0 | 0 | 1 | 0 | 391,127 | .591 |
| 0 | 0 | 0 | 0 | 163,427 | .958 |
| reverse=1 또는 middle=1 | | | | 508,164 | **정확히 .000** |

`control_success = 1 ⟹ reverse=0 ∧ middle=0`이 예외 없이 성립한다. 단일 이진 타깃으로
풀던 문제가 사실 다단계 구조였다. 여기서 나오는 것:

- **멀티태스크.** BSS가 0.01 수준이라 신호가 얇은 게 이 문제의 본질인데, 같은 행에
  신호가 두꺼운 라벨이 4개 더 있다(2024 R 기준 투수 ICC: reverse 1085 > success 816 >
  middle 656 > ball 335 > strike 285). 신경망을 5-헤드로 바꾸는 게 제일 직접적이다 —
  신경망은 블렌드 최대 기여자인데 시드를 늘리면 오히려 나빠진다는 건 "더 다르게 틀릴
  재료"가 필요하다는 뜻이다.
- **곱 분해.** `P(성공) = P(¬reverse) · P(¬middle|¬reverse) · P(성공|…)`. 지금까지의
  앙상블은 전부 같은 타깃의 평균이었고, 이건 다른 타깃을 곱하는 것이라 틀리는 방식이
  구조적으로 다르다.
- **타자쪽 신규 테이블.** 원본은 타자에 success·middle만 준다. 복원 라벨로 타자
  ball/strike 성향을 만들 수 있고, 2024 R 기준 타자 ICC가 213 / 225로 이미 있는 타자
  success(104)보다 크다.

### 피처를 더하지 말고 척도를 바꾼다

`asof_pitcher_success_rate`는 성공률 .565였던 2019 투구와 .486인 2024 투구를 그냥
평균낸 값이다. v7의 레짐 분리는 R/F 축으로만 이걸 풀었고 시대 축은 안 건드렸다. 각
누적 rate를 그게 쌓인 시점의 리그 레벨로 나눠 "시대 대비 실력"으로 바꾸는 건 피처
추가가 아니라 기존 피처의 재정규화라, 지금까지 기각된 것들과 종류가 다르다.

## v6에서 바뀐 것 — 2군 전용 모델, 그리고 대부분의 기각

v6 라운드는 **거의 전부 기각**이었다. 앙상블이 포화됐다는 게 이 라운드의 주된 결론이고,
살아남은 건 하나뿐이다.

### 살아남은 것: F(2군) 전용 모델

F는 전체의 12%인데 점수는 583.6으로 R(890.2)의 3분의 2도 안 된다. F 행만으로 학습한
모델은 **단독으로는 283점**(공용 모델의 583보다 한참 나쁨)이지만, 대신 나머지 88%와
타협할 필요가 없다. F 행에만 25% 섞으니 F 583.6 → **638.3**, 전체 892.4 → **897.9**.
R 행은 손대지 않는다.

### 기각된 것

| 시도 | 2024 홀드아웃 | 판정 |
|---|---|---|
| v5 스택 (기준) | 892.4 | — |
| 선형 모델 추가 | 888.8 | 기각 |
| MLP 1024-512 / 256-128-64 / 제곱오차 | 891.7 / 892.3 / 892.1 | 전부 무효 |
| **선수 임베딩 MLP** | 892.2 | 무효 (2024 행의 19.9%가 미학습 투수) |
| 트리 max_features 0.6 / 0.3 | 891.6 / 891.2 | 기각 |
| lr 0.02 × iter 1200 | 890.9 | 기각 |
| leaf63 msl500 | 892.5 | 무효 |
| 최근 시즌만 학습 (≥2021 / ≥2022) | 893.5 / 892.7 | 무효 |
| 블렌드 가중치 최적화 | 891.9 (v5 가중치 891.1) | 무효 — 게다가 다른 시즌에 적용하면 886.0으로 하락 |
| game_type별 블렌드 가중치 | 자기 시즌 896.2, 다른 시즌 2375.9 (v5 2387.0) | 일반화 실패 |

### 오프셋 강도를 낮춰야 하는가 — 아니다

4개 시즌에서 오프셋 강도 s를 쓸어봤더니 겉보기엔 s=0(오프셋 제거)이 압도적이었다
(평균 1002.7 → 1169.0). 하지만 시즌별로 뜯으면 이야기가 뒤집힌다.

| 홀드아웃 | 학습 시즌 수 | 예보 오차 | 최적 s |
|---|---|---|---|
| 2021 | 2 | −0.0301 | 0 |
| 2022 | 3 | −0.0135 | 0 |
| 2023 | 4 | +0.0155 (F 레짐 붕괴, 예보 불가) | — |
| 2024 | 5 | **+0.0011** | ~0.9 |

**예보 규칙은 학습 시즌이 쌓일수록 좋아진다.** "오프셋을 빼라"는 신호는 전부 시즌이
2~3개뿐이던 초기 홀드아웃에서 나온 것이다. 2025는 6시즌을 쓰므로 2024 홀드아웃이
유일하게 대표성 있는 관측이고, 거기서는 현행 0.85가 맞다 (v5 스택의 자체 드리프트
−0.0020을 넣고 풀면 최적 s=0.82). **변경하지 않는다.**

## v5에서 바뀐 것 — 앙상블 다양성

해상도는 v4에서 이미 피처로 뽑을 만큼 뽑았다. 남은 건 **서로 다르게 틀리는 모델을 섞는 것**.

### 신경망: 혼자서는 약한데 섞으면 크게 오른다

작은 MLP(256-128, GELU)는 단독 835점으로 트리 블렌드(870)보다 한참 나쁘다. 그런데
w=0.35로 섞으면 **2021 +27 / 2022 +16 / 2024 +23**. 쓸 수 있는 세 홀드아웃 전부에서
같은 방향이다 (2023은 F 레짐 붕괴로 오라클조차 0점이라 판정에 쓸 수 없다).
트리는 축에 수직인 계단으로 자르고 MLP는 모든 피처를 매 유닛에서 섞어 매끄럽게 보간하니,
틀리는 자리가 다르다.

**에폭을 늘리면 무너진다.** 3에폭 835 → 6에폭 777 → **15에폭 0점**.
BSS가 0.01 수준이라 맞출 신호 자체가 거의 없고, 그 이상은 전부 노이즈 암기다.
그래서 3에폭에서 멈춘다.

torch는 학습에만 쓴다. 아티팩트에 들어가는 건 가중치 행렬과 표준화 상수뿐이고
추론은 numpy 행렬곱 3번이다 (`work/nnpred.py`).

### 트리 두 종류: R과 F가 원하는 용량이 다르다

leaf15는 R에서 858.8 / F에서 551.8, leaf31(l2=1, msl=2000)은 R 839.6 / F 600.4.
둘 다 한쪽만 잘한다. 섞는 쪽이 game_type으로 갈아끼우는 것보다 낫다 (869.7 vs 866.8).
느슨한 쪽 비중은 2024 최적값 0.4가 아니라 **0.35**로 둔다 — 학습 시즌이 2개뿐인
2021 홀드아웃에서는 0.4가 47점을 깎기 때문이다.

### v5에서 기각된 것

| 시도 | 2024 홀드아웃 오라클 | 판정 |
|---|---|---|
| 제곱오차 회귀(평가식과 동일한 손실) | 860.0 vs log-loss 860.9 | 무효 |
| 로짓 기울기 보정 a≠1 | a=1.0이 최적 (a=0.9 → 853.5) | 무효, 모델은 이미 과신하지 않음 |
| game_type별 오프셋 분리 | 853.6 vs 852.7 | 무효 |
| 시즌 가중치(반감기 3/5) | 861.8 / 859.0 vs 860.9 | 무효, 데이터 양이 이긴다 |
| shrinkage K 조정 (60/600, 100/1000) | 전부 동률 이하 | 이미 최적점 |
| 2시즌 창 복원 | 858.3 | 무효 |

## v4에서 바뀐 것

### 1. `asof_*` 카운터는 train에서 test로 이어진다 → 시즌 누적 성적을 복원할 수 있다

`asof_pitcher_n`은 시즌마다 리셋되지 않는 **통산 투구 수 카운터**다. 그리고 이 카운터는
평가 데이터까지 그대로 이어진다 — 배포된 `test.csv` 5행 샘플로 확인했다
(투수 21813은 train.csv의 2024년 말에 3085구, test.csv에서 3465구).

그래서 각 행의 asof 값에서 **그 선수의 직전 시즌 말 누적치**(train.csv에서 뽑은 상수)를
빼면 그 선수가 **올 시즌 지금까지** 어땠는지가 복원된다.

```
season_n    = asof_n           − 직전 시즌 말 통산 투구 수
season_succ = asof_rate·asof_n − 직전 시즌 말 통산 성공 수
```

이게 중요한 이유: `asof_pitcher_success_rate`는 통산 누적이라 10년차 투수는 값이 거의
안 움직인다. 2024년 249명(200구 이상)에 대해 **통산 성공률과 그 해 실제 성공률의 상관은
0.19**뿐인데, 복원한 시즌 누적은 **0.97**이다. 그동안 모델은 10년 평균으로 투수를
판단하고 있었다. prev1/3/5경기 값은 반대로 너무 noisy하고, 시즌 누적이 그 사이를 메운다.

**규칙 안전성**: 빼는 값은 train.csv에서 학습한 선수별 상수이고, 나머지는 그 행 자신의
컬럼이다. 평가 데이터의 다른 행을 보지 않으므로 "행 독립 예측" 조항에 걸리지 않는다.

투수(성공/반대성/가운데/볼/스트라이크) · 타자(성공/가운데) · 구종믹스 전부에 적용.
기여도(2024 홀드아웃 오라클): 투수 +19, **타자 +36**.

### 2. `game_type` F는 2023년에 정의가 바뀌었다 (추세가 아니라 계단)

F는 2군(Futures) 경기다 — `trackman_history.csv`의 `MIN_*` 팀 구분과 일치한다.
F 성공률이 2022 .709 → 2023 .473으로 무너지는데, **13개 팀 전부 같은 폭으로** 떨어진다.
팀이 못해진 게 아니라 측정/판정 기준이 바뀐 것이다.

여기서 두 가지가 따라 나온다.

- **2022→2023 전체 성공률 하락(−.0289)은 거의 전부 이 F 계단 때문이다.** R은 같은 기간
  −.0006밖에 안 움직였다. 전체 시계열에 직선을 맞추면 일회성 계단을 추세로 오해한다.
- **옛 레짐 F 행은 격리해야 한다.** `f_old_regime = (game_type==F) & (season<2023)`
  플래그 하나로 2024 홀드아웃 816 → 843 (+27), F 서브셋만 보면 412 → 533.
  2025 행에서는 이 플래그가 항상 0이다.

### 3. 홀드아웃 평가 자체가 망가져 있었다

기존 `work/evaluate.py`는 2023·2024 두 시즌 평균으로 판정했는데, **2023 홀드아웃은
오라클 캘리브레이션을 줘도 0점**이다. 2019~2022로 학습한 모델이 F 행을 .71로 예측하는데
실제는 .47이라 상수보다 나쁘다. 이 0점이 평균을 지배해서 실험 08~10이 전부 기각됐던 것이다.

지금은 `work/lab/harness.py`가 **전체 / R만 / F만**을 따로 찍고, 시드 노이즈(±4)와
드리프트 진단(beta)까지 같이 낸다.

### 4. 캘리브레이션: 세그먼트 예보 + 85% 강도

`work/forecast.py`가 R과 F를 따로 외삽한다. R은 linear·mean_delta·delta3의 평균,
F는 2023 이후 구간만 쓰고 "R이 움직이는 만큼 F도 움직인다"와 평균낸다.
2021~2024 백테스트 RMSE 0.0182 (단순 직선 0.0196), 2024년 오차 +0.0011 (직선 +0.0057).

오프셋은 **85% 강도**로만 넣는다. 시즌 누적 피처 자체가 2025년 수준을 담고 있어서
모델 예측 평균이 스스로 내려가기 때문이다(2024 홀드아웃에서 실제 드리프트의 0.18~0.25,
학습 시즌이 쌓일수록 감소). 100%로 넣으면 드리프트를 두 번 빼게 된다.

2024 홀드아웃에서 이 절차는 목표 0.4891 → 실제 달성 평균 0.4862, 진짜 값 0.4861.
캘리브레이션 손실이 사실상 0이라 **adj 점수가 오라클과 같았다**.

### 5. 기각된 것들

| 시도 | 2024 홀드아웃 오라클 | 판정 |
|---|---|---|
| 기준 (v4) | 855.1 | — |
| 투타 대비 (`std_success − bat_std_success`) | 860.7 | 채택 |
| 구장 ID (홈팀 = 파생 가능) | 830.3 | 기각 |
| 다중 shrinkage K | 852.0 | 기각 |
| 직전 시즌으로 경험적 베이즈 | 853.2 | 기각 |
| prev1/3/5 경기 창 분해 | 846.6 | 기각 |
| 투구 수/워크로드 | 849.8 | 기각 |
| `season` 컬럼 제거 | 686.3 | 기각 |
| leaf 31 / iter 800 | 791 / 790 | 기각 (용량 키우면 손해) |

**TrackMan은 여전히 붙일 수 없다.** `pitcher_trackman_id`와 `pitcher_id`는 교집합 0개고
팀 코드도 실명(`KIA_TIG`) vs 익명(12~25)으로 체계가 다르다. 시즌·팀·투구수 지문으로
매칭하는 건 이론상 가능하지만, 얻는 건 구속·회전수 같은 물리량뿐이고 제구 실력은 이미
성공률로 직접 측정되고 있어 투자 대비가 나쁘다.

## 환경 함정

평가 서버는 **numpy 1.26.4**다. numpy 2.x로 만든 pickle은 서버에서 `numpy._core`
ImportError로 **로드 자체가 실패**한다. 반드시 `.venv-submit`에서 학습·검증할 것.

```bash
python -m venv .venv-submit
.venv-submit/Scripts/python.exe -m pip install "numpy==1.26.4" "scikit-learn==1.8.0" "joblib==1.5.3" "pandas==2.2.3"
```

`requirements.txt`는 주석만 둔다 — 서버 기본 패키지만 쓰면 설치 오류 위험이 0이다.

## 구조

```
open/                     대회 배포본 — 전부 git 제외,
                          "대회 자료는 재배포하지 않는다" 절 참고
work/
  fe.py                   피처 엔지니어링 — 학습/검증/제출이 공유하는 유일한 원본
                          extras=False → 뷰 A(107), extras=True → 뷰 B(135).
                          EXTRA_COLS가 두 뷰를 가르는 유일한 목록이다.
  nnpred.py               신경망 전처리 + 추론 (numpy만, torch 없음)
  forecast.py             다음 시즌 성공률 예보 (R/F 분리)
  train_nn.py             신경망 학습 → nn_weights.npz (시스템 python, torch 필요)
  train_v4.py             트리 학습 + npz 흡수 → submit/model/model.joblib
  build_submit.py         fe.py+nnpred.py를 인라인해서 submit/script.py 생성 + zip
  verify_v4.py            평가 서버 리허설 (numpy 1.26 로드 / 전처리 일치 / 시간)
  lab/                    실험 하네스와 기록 (00~31)
submit/                   제출 zip의 내용물
versions/                 버전별 스냅샷 (코드 + 가중치 + 점수)
```

`submit/script.py`는 **손으로 고치지 않는다.** 전처리가 학습과 어긋나면 에러 없이 조용히
틀린 예측이 나오는데, 이게 이 대회에서 가장 비싼 실수다. `fe.py`를 그대로 인라인해서
생성하고, `verify_v4.py`가 두 경로의 출력이 완전히 같은지 확인한다.

## 사용법

```bash
cd work
python train_nn.py --view B --validate                      # 신경망 (시스템 python)
../.venv-submit/Scripts/python.exe train_v4.py --validate --seeds 2   # 홀드아웃

python train_nn.py --view B                                 # 전체 신경망
../.venv-submit/Scripts/python.exe train_v4.py              # 전체 트리 + 합치기
python build_submit.py                                      # script.py + submit.zip
../.venv-submit/Scripts/python.exe verify_v4.py             # 서버 리허설
python snapshot.py v7 --score <LB> --note "..."
```

`--view`는 `train_v4.py`의 `NN_VIEW`와 반드시 같아야 한다. 어긋나면 학습 때
`train_v4.py`가 컬럼 수 불일치로 즉시 멈춘다 — 조용히 틀리지 않게 하려고 넣은 것이다.

## 규칙 메모

- 1일 제출 5회. **설치 오류는 차감 없음, `script.py` 실행 중 오류는 차감됨.**
- 외부 데이터·API 금지, 2025년 Trackman 금지.
- 평가 데이터 전체 통계를 이용한 사후 보정 금지. **리더보드 점수를 보고 상수를 튜닝하는
  것도 같은 조항에 걸린다** — 보정값은 반드시 학습 데이터에서만 유도할 것.
