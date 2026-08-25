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
