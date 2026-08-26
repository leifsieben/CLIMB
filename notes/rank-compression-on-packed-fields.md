# "Descriptors are actively harmful on three datasets" -- they are not

**2026-08-26, Leif:** *"I find it VERY surprising that there should be three random sets that for
no reason descriptors are actively harmful on. XGBoost should be able to ignore irrelevant
dimensions. So this most likely either a problem in our evaluation or a problem in how we run
XGBoost."*

Correct instinct. It is the evaluation, and specifically the METRIC we summarise with. Two of the
three cases are not losses at all.

## What the three cases actually are

fig_A's classification mean for ECFP4+desc was 3.93 while its other three categories were 1.0-1.5.
Three of fourteen datasets carried it: BBBP (rank 12), cyp2c9-substrate (10), pkis2-ret-cls (10).

**BBBP -- descriptors HELP, and the rank is an artefact.**

    ECFP4       0.8792 ROC-AUC      between-fold SD 0.0354      seed SD 0.0030
    ECFP4+desc  0.9056 ROC-AUC      between-fold SD 0.0354      seed SD 0.0022

ECFP4+desc beats bare ECFP4 by +0.026 and still takes rank 12 of 13, because the whole field is
packed into 0.0737 ROC-AUC on a dataset whose fold-to-fold SD is 0.0354. The ordering inside that
field is decided by less noise than the folds themselves carry. Note also that TEST-SET variation
is 16x the seed variation -- any error bar built from seeds understates this dataset by that
factor.

**cyp2c9-substrate -- a tie charged as a rank.**

    ECFP4       0.3576 pr_auc +- 0.0088 (9 head seeds)
    ECFP4+desc  0.3514 pr_auc +- 0.0124
    delta -0.0062 = 0.57 pooled SD

Rank 10 of 13 for six thousandths of a pr_auc, well inside seed noise alone.

**pkis2-ret-cls -- real, and it has an equal and opposite sibling.**

    pkis2-ret-wt-cls-v2   n_test=106   ECFP4 0.7207 +-0.0214   +desc 0.5671 +-0.0298   -0.1536  (5.9 SD)
    pkis2-kit-wt-cls-v2   n_test=116   ECFP4 0.5133 +-0.0136   +desc 0.6398 +-0.0166   +0.1265  (8.4 SD)

Two sibling kinase panels, same size, same metric, same head, same feature blocks -- and the
descriptor block swings pr_auc by ~0.14 in OPPOSITE directions. A systematic failure of XGBoost to
ignore irrelevant columns would be one-directional. This is small-n variance on a fixed split, and
the seed SD cannot see it: Polaris splits are FIXED, so nine head seeds resample the head and never
the 106 test molecules. The reproducibility across seeds says the two models differ on THOSE
molecules; it says nothing about whether the difference generalises.

Across the whole category the descriptor block HELPS on 11 of 14 classification datasets
(mean +0.256 of the field's own spread).

## So is it XGBoost?

`heads_v2.HEAD_HPARAMS["xgb"]` is one fixed configuration for every feature block:
n_estimators=600, max_depth=6, lr=0.08, subsample=0.8, colsample_bytree=0.8, min_child_weight=2,
early_stopping_rounds=40. It is never retuned per block, and 2,265 mixed sparse-binary + dense-
continuous columns on a 534-row training set is a regime where depth and early stopping matter.
That is a genuine limitation and worth stating -- but it cannot produce opposite-signed effects of
equal size on two sibling tasks, so it is not the explanation for what was flagged.

## The actual defect: mean rank on a packed field

Rank averaging assumes an ordering is information. On a dataset where the whole field sits inside
the test-set noise, it is not -- and mean rank then charges a coin flip as a full rank. It is also
ASYMMETRIC for a leading arm: ECFP4+desc is rank 1 on 6 of 14 classification datasets, so it cannot
gain past 1 but can lose to 13. Classification is the most volatile category (median within-arm
rank SD 3.33, against 1.83 activity cliffs and 2.71 regression) because its median test set is 135
molecules and 4 of 14 use pr_auc.

`tasksuites.wide_ranks(summary=...)` now takes "mean" or "median"; `fig_A.SUMMARY` selects it.

    ECFP4+desc classification:   mean 3.93   ->   median 2.00
    published ordering:          IDENTICAL, Kendall tau +1.000, no arm moves a place

The conclusion does not depend on the choice, which is the reason it is safe to report the more
robust one.

## Two mistakes made while investigating this, both the same mistake

Twice I filtered per-dataset score rows by task and NOT by metric, pooling pr_auc with roc_auc, and
once nef1 with roc_auc. The first produced "seed SD 0.17" (it is 0.02) and the second produced
"BBBP ECFP4+desc 0.9461" (it is 0.9056). Both looked plausible and neither failed. A long-format
table keyed (task, seed, metric) read with two of the three keys returns a confident wrong number
-- the same shape as every other silent failure in this repo. Filter on the manifest's
`primary_metric`, never on the task alone.

---

# Five summary schemes compared, and why ties are the wrong fix (2026-08-26)

Leif asked whether rounding to the third or fourth digit and giving indistinguishable models a
shared rank would stop noise moving the ranking, and said he prefers the mean to the median.

## The measurements

| scheme | ECFP4+desc classification | Kendall tau vs mean |
|---|---|---|
| mean rank | 3.93 | -- |
| median rank | 2.00 | +0.994 |
| 10% trimmed mean rank | 3.50 | +0.974 |
| 20% trimmed mean rank | 3.10 | +0.974 |
| mean z-score (effect size, not rank) | +0.77 | +0.949 |
| mean rank, midranks tied within 1 replicate SD | 4.18 | +0.949 |
| mean rank, midranks tied within 2 replicate SD | 4.46 | +0.949 |

**No arm moves more than one place under any scheme.** That is the important line: the ordering is
not an artefact of the summary, so the mean can be reported with the median as a robustness note.

## Why ties BACKFIRE

Ties look neutral and are not. ECFP4+desc is **outright best on 42 of 65 datasets**. A tie merges
a win into a midrank, so the arm that wins most has the most to lose and an arm that never wins
(CLIMB unsupervised: best on 0 of 65) can only gain. Ties transfer rank mass from winners to the
middle. Measured, they made the number they were meant to fix worse: 3.93 -> 4.18 -> 4.46.

Rounding to a fixed decimal is the same rule with an arbitrary threshold, and it has a further
problem: 0.001 is not the same quantity in ROC-AUC, RMSE, NEF1% and pr_auc, so one constant cannot
serve 65 heterogeneous datasets.

## Why the tie THRESHOLD is also unavailable

The only per-dataset noise scale we have is the replicate SD, and for a fixed-split benchmark that
is head-init variance with the test set held fixed. It is wrong in both directions at once: at
1 SD it ties **593 of 845 cells** (70% of the field) yet still does NOT tie BBBP, where the gap to
the leader is 21 replicate SD but only 1.3 *fold* SD. On BBBP fold-to-fold variation is 16x the
replicate SD. A tie rule needs the test-set sampling scale, which fixed splits do not expose.

## Why z-scores do not fix it either

Replacing ranks with within-dataset z-scores keeps effect size, which sounds like the principled
answer. It is not, for this problem: z normalises by the FIELD'S OWN SPREAD, and on a packed field
that spread IS the noise. BBBP contributes rank 12 under ranks and z = -1.20 under z-scores --
equally punishing. Same flaw, different clothes.

## What actually works

A ROBUST SUMMARY, because it does not need to know the noise scale at all. The median is unmoved
by how the tail is ordered and has no free parameter; a 20% trimmed mean is the middle ground for
anyone who wants the word "mean". Both are one line in `fig_A.SUMMARY` /
`tasksuites.wide_ranks(summary=)`.

Reported: the MEAN, per Leif's preference, with the robustness table above as the caption's
warrant.

---

# Is bare ECFP4's range across categories real? (Leif, 2026-08-26)

Yes, and it is the most chemically interpretable thing on the plate. In METRIC units, not ranks --
z against the 13-arm field on each dataset, oriented so + is better, averaged within category:

| category | n | bare ECFP4 | ECFP4+desc | descriptors add (fraction of field span) |
|---|---|---|---|---|
| Activity cliffs | 30 | **+1.28** | +1.50 | +0.058 |
| Virtual screening | 2 | **+0.94** | +1.97 | +0.275 |
| Regression | 19 | **+0.20** | +1.60 | +0.416 |
| Classification | 14 | **-0.10** | +0.77 | +0.256 |

Bare ECFP4 swings from +1.28 (well above the field) on activity cliffs to **-0.10 (below the field
average)** on classification. That is not rank compression -- it is the same picture in raw metric
units, and it is exactly what the representation predicts: ECFP4 is a pure SUBSTRUCTURE code, so it
is excellent where the answer is structural (matched pairs across a cliff, rare-active retrieval)
and mediocre where the answer is physicochemical (ADMET regression and classification, which turn
on logP, TPSA, molecular weight -- quantities the descriptor block supplies and a fingerprint does
not encode).

The mirror image is that the descriptor block's benefit is SMALLEST on activity cliffs (+0.058 of
the field span) and LARGEST on regression (+0.416). Two ways of measuring the same fact.

So the wide range is a result to report, not a defect to smooth. ECFP4+desc is the arm that is
uniformly strong (+0.77 to +1.97 across all four), which is why it leads overall.
