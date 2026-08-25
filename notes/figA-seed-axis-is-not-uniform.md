# "3 seeds" means two different things in fig_A, and the caption has to say so

2026-08-23, Leif: "there don't exist 3 seeds for the foundation model so head seeds is the best we
can do." Correct, and it is what the spec asks for. This records the consequence.

## The asymmetry

Every ranked arm carries three replicate dirs, but the axis differs by arm:

    CLIMB arms (unsup, sup_*, u2s_*, s2u_dense)   three PRETRAININGS; head seeds pinned to the
                                                  base triple. Spread includes pretraining
                                                  variability.
    literature CLMs (chemberta_mtr, molformer,    one released checkpoint, so three HEAD INITS on
      molformer_c3, selfies_ted)                  disjoint seed triples. Spread EXCLUDES
                                                  pretraining variability.
    ECFP / ECFP+desc anchors                      same as the CLMs -- no pretraining stage exists.
    random_encoder                                three random inits, i.e. its own pretraining axis.

Nothing is wrong with any of these individually; each is the strongest replicate available for that
arm. But "3 seeds" is one label over two estimands, and a reviewer counting directories cannot see
the difference.

## Why it does NOT contaminate the drawn interval

fig_A1's error bar is NOT a seed spread. From allsuites.wide_ranks: `se_rank` is the standard error
across the FOUR CATEGORY means, inflated by the design effect. Seeds enter only through each arm's
point estimate. Measured head-seed spread on the literature CLMs is <=0.003 macro RMSE on
MoleculeACE, so the point estimates are stable and the asymmetry does not distort the ranking.

## What the caption must therefore avoid

Do NOT write "three pretraining seeds" as a property of the panel -- it is false for four of the
fourteen arms. Say that every arm carries three replicates on the strongest axis available to it,
name which arms vary the pretraining and which vary the head, and state that the interval is the
spread across the four task categories rather than across seeds.

Related: [[replicate-axis-depends-on-arm]] in memory, which is the same distinction for the
concatenation figures.

---

# The probe head is also not uniform, and that is deliberate

2026-08-23, Leif: "the heads are fixed that needs no ablation right now."

fig_A1 ranks 13 arms across two head types:

    ecfp, ecfp_desc                          XGBoost
    the 8 CLIMB arms + 3 literature CLMs     frozen encoder + MLP

The rule is REPRESENTATION AT THE HEAD THAT SUITS IT, not one head for all. SI fig f is the
evidence: the preference is representation-dependent, so forcing a single head handicaps whichever
representation it does not suit. Measured on MoleculeACE macro RMSE, same vectors:

    CLIMB unsup        MLP 0.7781   XGB 0.8057    MLP by 0.028
    CLIMB sup, desc    MLP 0.7738   XGB 0.7949    MLP by 0.021
    CheMeleon frozen   MLP 0.8251   XGB 0.6867    XGB by 0.138

## The exposure, stated rather than hidden

The three literature CLMs (chemberta_mtr, molformer_c3, selfies_ted) run at frozen+MLP and have
NOT been measured at XGBoost. So they are reported at a head assumed to suit them, while ECFP4 is
reported at a head measured to suit it.

The risk is not theoretical: CheMeleon is the closest analogue -- a frozen pretrained embedding
from outside this lab -- and its head preference is 0.138, larger than the whole span from
ECFP4+desc (0.673) to CLIMB unsup (0.778). If one of the three behaves that way it would be
reported near last while belonging mid-table.

Leif weighed this and ruled no ablation now. Recorded so the answer to "why is ChemBERTa only at
one head" is a decision with a known magnitude rather than an oversight, and so the ablation can be
run later without re-deriving why it would matter. Cost if wanted: 3 arms through the XGBoost
probe, the same run already done for CheMeleon.

## Two further non-uniformities the caption should carry

EQUAL WEIGHT PER CATEGORY, not per dataset. Activity cliffs holds 30 datasets and virtual screening
3, so each VS dataset carries ~8.3% of the headline against ~0.83% for each MoleculeACE target -- a
10x ratio. Deliberate (the axis is task type, not benchmark size) but not self-evident.

RANKS DISCARD EFFECT SIZE. An arm 0.001 better on 40 datasets outranks one 0.05 better on 26. That
is the price of combining ROC-AUC, NEF1%, RMSE and Spearman in one number; the per-category columns
expose it partly, the mean rank cannot.
