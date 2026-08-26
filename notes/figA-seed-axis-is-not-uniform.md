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

---

# The three literature CLMs, measured (2026-08-26)

They landed and are promoted into `arms.py`. Three things the caption has to get right.

## Parameter count is INVERSE to the name, and inverse to rank

Measured from the checkpoints by the compute session, not read off the model card title:

| arm | name implies | actual params | fig_A rank |
|---|---|---|---|
| ChemBERTa-2 (`ChemBERTa-77M-MTR`) | 77M | **3.4M** | 4.47 (2nd of 13) |
| MoLFormer (`MoLFormer-c3-1.1B`) | 1.1B | **44.4M** | 6.07 (6th) |
| SELFIES-TED | -- | **358.1M** | 7.22 (8th) |

The number in each name is PRETRAINING DATA, not parameters. Two consequences:

1. **Never print a name's number as a size.** "The 1.1B model loses to the 77M model" is a
   sentence someone would write from the names, and it is backwards: 44.4M loses to 3.4M.
2. **Rank order is exactly inverse to parameter count** across all three, 3.4M > 44.4M > 358.1M.
   n = 3, so this is an observation and not a scaling claim -- but it is the opposite of what a
   reader assumes, so it is worth one sentence rather than being left for them to get wrong.

`params_m` is stored per arm in `arms.py` so a caption can use the real number.

## ChemBERTa-2 places second, tied with ECFP4

4.47 against ECFP4's 4.47, behind only ECFP4+desc (1.91), and ahead of EVERY CLIMB arm --
supervised-desc is 5.63. The three literature CLMs bracket the CLIMB arms rather than sitting
below them. This is the headline the figure now carries and it does not flatter us.

## Wong and FartDB are still out

4 of 13 and 2 of 13 arms as of 2026-08-26. Virtual screening therefore rests on CBS + HIV, two of
its three datasets, and classification on 14 of 15. Both are stated in the legend as "+1 pending".

## CBS discriminates -- the saturation warning does not hold

The compute session flagged that plain ECFP4 hits NEF1 = 1.0000 in 3 of 5 CBS folds (mean 0.8900,
ROC-AUC 0.9948, 8-10 positives per fold) and asked whether CBS is degenerate like BBBP. Checked
through `allsuites.wide_table` -- the figure's own resolver, not a reimplementation of it -- over
the arms fig_A actually ranks:

    CBS NEF1 spread 0.2304 across the field, SD 0.0716   (ECFP4+desc 0.9300 .. sup_sparse 0.6996)
    MolNet:HIV      spread 0.1278,           SD 0.0373
    Spearman(CBS, HIV) over the shared arms  +0.721

CBS is the MORE discriminating of the two available virtual-screening datasets, by roughly 2x,
and it agrees with HIV about the ordering. Per-fold saturation is real and the fold-to-fold SD is
large next to the between-arm spread, so CBS is NOISY -- but that is a different thing from
degenerate. BBBP was nef1 EXACTLY 1.0 for every feature block: spread zero, no information at all.
CBS's spread is 0.23. Keep it in the suite.

---

# unsup_100M has ONE pretraining, and how much that costs (2026-08-26)

**Leif:** *"the difference is that unsup_100M has only one seed right? That's ok for me right now
but it means we have slight apples-to-oranges comparison."*

Correct. Its three replicates are HEAD seeds inside one directory; the other CLIMB arms carry three
separate PRETRAININGS. So its point estimate contains one draw from the pretraining distribution
and the others average three. Measured rather than asserted:

**Pretraining-seed SD, from `unsup`'s three pretrainings** (same objective and architecture as
unsup_100M), per dataset, as a fraction of the 14-arm field spread:

    median 0.078   IQR [0.046, 0.143]   over all 65 datasets

Examples in raw units: BACE 0.8694 / 0.8625 / 0.8424 (sd 0.0140), Tox21 sd 0.0024, QM7 sd 0.84.

**What that does to its position.** Perturbing only unsup_100M by one resampled pretraining draw
on every dataset, 400 simulations:

    observed place                2nd of 14
    simulated place               median 2nd, 95% interval [2, 4]
    stays 2nd                     87% of draws
    stays top-3                   89%
    stays ahead of ChemBERTa-2    ~89%   (ChemBERTa is 4th observed)

So the exposure is real and bounded. "unsup_100M places second" is robust; "unsup_100M is ahead of
ChemBERTa-2" holds in roughly nine draws out of ten and should be written with that hedge, not as a
bare fact. The honest caption form is that it is a single pretraining run and its interval does not
include pretraining variability, with the 87%/89% figures available if a referee asks.

**Six of fourteen rows are now head-seed rather than pretraining-seed replicates** -- ecfp,
ecfp_desc, the three literature CLMs, and unsup_100M. The caption's "3 seeds means two estimands"
clause needs that count, not the old five of thirteen.

Two more pretrainings of unsup_100M would close it. At the measured MLM throughput (~833 s per 1M
forward passes) that is ~23 GPU-hours each, so ~46 hours for the pair -- more than the supervised
rungs in flight, and buying a hedge rather than a result. Not recommended unless a referee asks.
