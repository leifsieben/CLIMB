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
