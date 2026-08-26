# sup_mixed was PRETRAINED on WONG, and Wong is now an evaluation dataset

2026-08-26. Found while answering Leif's question "what is actually the difference between
desc+sparse and mixed". The answer is two label families, and one of them is a benchmark we now
score on.

## The difference he asked about

From the runs' own metadata.json, not from the names:

    skip_dense_plus_sparse_8M   objectives {mtr 0.5, supervised 0.5}
                                families   PCBA, L1000_MCF7, L1000_VCAP
    skip_mixed_8M               objectives {mtr 0.5, supervised 0.5}
                                families   PCBA, L1000_MCF7, L1000_VCAP, PCQM, WONG

Identical objective weighting. The ONLY difference is that `mixed` adds PCQM and WONG.

## Why that matters now and did not in July

WONG is an SFT label family: "antibiotic screen, 4 label columns, 34,652 molecules" (README §
supervised families). Wong S. aureus joined the EVALUATION suite on 2026-08-26 as one of the three
virtual-screening datasets. The same screen is on both sides.

The project HAS a leakage blocklist -- `configs/eval_blocklist.json`, 34,301 canonical SMILES --
and `skip_dense_8M`'s config points at it, so this is not a missing mechanism. It is a blocklist
built from the SIX MolNet tasks of July (ESOL, BBBP, BACE, Tox21, QM7, HIV) and never rebuilt when
Wong, CBS, Ames, MoleculeACE, Polaris and FartDB entered the suite. A dedup that was complete when
written stopped being complete when the eval set grew, and nothing in the pipeline notices, because
a blocklist cannot report what it was never asked about.

## The signature, measured on the figure's own resolver

Wong NEF1%, the 14 ranked arms:

    sup_mixed        0.4583   rank  1 of 14    <- trained on WONG
    unsup_100M       0.4034         2
    ecfp             0.3727         3
    ecfp_desc        0.3719         4
    ...
    sup_sparse       0.2099        13          <- trained on PCBA + L1000, NOT WONG
    u2s_sparse       0.1381        14          <- trained on PCBA + L1000, NOT WONG

THE INTERNAL CONTROL IS WHAT MAKES THIS MORE THAN A COINCIDENCE. Three ranked arms train on assay
labels. The two that train on assay labels WITHOUT WONG place 13th and 14th of 14 on Wong. The one
that trains on WONG places 1st. So the pattern is not "assay pretraining helps antibiotic
screening" -- it is specific to the family that is the eval set.

sup_mixed is otherwise mid-field: activity cliffs 9.33, classification 5.08, regression 6.05. Its
virtual-screening mean of 3.33 is carried by Wong; it is 5th on CBS and 4th on HIV.

## Blast radius

24 runs in climb_v2_phase2 list WONG in supervised_families -- every skip_mixed_*, skip_minimol_*,
u2s_mixed_* and u2s_minimol_* rung. Of those, only `skip_mixed_8M{,_s1,_s2}` is drawn in the paper,
as fig_A's `sup_mixed`. The scaling ladders draw skip_mixed_* rungs in the SI cuts.

Effect on the fig_A headline is small because Wong is 1 of 64 datasets -- sup_mixed 7.28 -> 7.38
with Wong dropped, no arm reorders -- but its virtual-screening category mean moves 3.33 -> 4.50,
and any sentence of the form "the mixed objective is strong on virtual screening" rests on a cell
the arm was trained for.

## What is NOT yet established

Molecule-level overlap between the WONG SFT family and Wong's eval folds has not been measured
here: the family parquet and the Wong eval CSV are both S3-only, and `fold_values.csv` carries no
SMILES. The structural case is strong (same screen, 34,652 vs 39,121 molecules, blocklist built
before Wong existed) but the NUMBER is the thing to get. Requested from the compute session.

Two questions for that measurement, in order:
  1. What fraction of Wong's eval TEST molecules appear in the WONG SFT family?
  2. Does the same audit turn up overlap for CBS, Ames, MoleculeACE, Polaris or FartDB against
     PCBA/L1000/PCQM? The blocklist covers none of them either, so Wong is the one we noticed and
     not necessarily the only one.

Related: [[absence-claims-go-stale-silently]] -- the blocklist is an absence claim ("no eval
molecule is in training") that was true when written, cannot fail loudly, and stopped being true
when the suite grew.

---

# The number: 88.5%, and the audit that followed it (2026-08-26)

Measured by the compute session on canonical keys:

    Wong        39,039 unique eval molecules
                34,568 (88.5%) present in the WONG SFT family
                 4.38% of them on eval_blocklist.json
                => at least 84% were in skip_mixed_8M's SUPERVISED training data, WITH LABELS

The blocklist mechanism WORKS -- pretrain_v2 loads it, passes it to data_v2, and prints how many
molecules it drops from SFT. It covers 4.38% of Wong because it was built before Wong was an eval
set. A working mechanism pointed at a stale target.

So sup_mixed's 1st of 14 on Wong is training on the test set. It is out of fig_A as of e8a8263,
replaced by sup_dense_sparse per Leif's decision, and it is now drawn in no paper figure.

## It is not only Wong -- but the exposure that remains looks harmless, and that is measured

Eval molecules overlapping SFT families, worst first (lower bounds; families are sampled at 400k):

    Polaris:bioavailability-ma    62.2% L1000_MCF7   23.9% WONG   (78.4% blocked)
    Polaris:dili                  51.6% L1000_MCF7   41.7% WONG   (59.0% blocked)
    Polaris:half-life-obach       49.9% L1000_MCF7   22.4% WONG   (63.6% blocked)
    Polaris:cyp2c9/2d6/3a4-sub    ~40%  L1000_MCF7   ~12%  WONG   (~55% blocked)
    Polaris:bbb-martins           23.0% L1000_MCF7   11.1% WONG   (49.5% blocked)
    Polaris:pgp-broccatelli       16.4% L1000_MCF7    7.6% WONG   (21.0% blocked)
    MolNet/Polaris Ames            4.1% L1000_MCF7   18.2% PCBA   (28.2% blocked)
    FartDB                         6.6% PCBA                      ( 7.7% blocked)
    MoleculeACE                   <1%                             ( 1.0% blocked)
    CBS                           <1%                             ( 0.5% blocked)

CBS and MoleculeACE are CLEAN. Virtual screening does not collapse; it loses one of three datasets.

L1000_MCF7 recurs because it is a drug-like compound panel and TDC's ADMET sets are drug
molecules -- a shared-chemistry mechanism, not a name collision. The opposite of how Wong was found.

## The control that bounds it, computed here

Three ranked arms train on PCBA + L1000_MCF7 + L1000_VCAP: sup_sparse, u2s_sparse, and
sup_dense_sparse (the arm that just replaced sup_mixed -- so the swap is NOT automatically clean).
If residual leakage were buying them anything, they should do relatively better on the high-overlap
datasets than elsewhere. Mean rank on the ten high-overlap sets vs the other 54:

    arm             high-overlap    rest    delta
    sup_sparse           10.50     11.39    -0.89
    u2s_sparse           10.80     11.96    -1.16
    sup_dense             4.30      6.15    -1.85     <- MTR only, CANNOT be leaked
    unsup                 7.90      8.06    -0.16     <- MLM, cannot be leaked
    ecfp_desc             3.50      1.56    +1.94

THE ASSAY-TRAINED ARMS GAIN LESS THAN sup_dense DOES, and sup_dense has no assay exposure at all.
So the apparent gain is a property of those datasets -- small, noisy Polaris tasks where the
fingerprint anchor is weakest -- and not evidence of a leakage benefit. Consistent with the 50-78%
block rates on exactly those tasks.

This BOUNDS the concern; it does not close it. Absence of a detectable benefit is not proof of no
leakage, and the decisive number is overlap MINUS blocklist per family, which is being recomputed.
Wong needed no such subtlety: 88.5% overlap against 4.4% blocked is not a residual question.

## The bigger question, which is Leif's and not ours

24 phase-2 runs list WONG in supervised_families, and the blocklist is a build-time artefact of an
eval suite that has grown five times since. Rebuilding it properly means rebuilding against the
CURRENT suite and re-training every affected run. That is a wave, not a fix.

---

# The replacement arm is CLEANER, not clean, and the exposure is systematic (2026-08-26)

From the three runs' own metadata, all seeds verified:

    skip_mixed_8M               PCBA, L1000_MCF7, L1000_VCAP, PCQM, WONG
    skip_dense_plus_sparse_8M   PCBA, L1000_MCF7, L1000_VCAP

No WONG, no PCQM. But Wong's eval molecules overlap the families it DOES train on:

    Wong x L1000_MCF7   2,418 molecules   6.2%     EXACT (11,718-molecule family, under the cap)
    Wong x PCBA           267 molecules   0.7%     lower bound (PCBA is sampled at 400k)

So sup_dense_sparse saw roughly 7% of Wong's eval molecules with labels, against sup_mixed's 88.5%.
Twelve-fold smaller, and not zero. The honest sentence is "trains on ~7% of Wong's molecules
through L1000_MCF7", NOT "the contaminated arm is out".

## It is not about this arm, and that is the point

EVERY assay-label SFT arm trains on those same families, so every one carries the same ~7% on Wong
and every non-SFT arm -- unsup, unsup_100M, s2u, the anchors, the three literature CLMs, and the
descriptor-only MTR arms -- carries none. That is a small SYSTEMATIC advantage to one group of rows
on one dataset, and it does not cancel when values become ranks.

fig_A now prints the affected arms in its CAPTION FACTS block, DERIVED from each run's
supervised_families rather than listed, so an arm whose recipe changes cannot slip out of the
sentence.

## The empirical bound on it, which is strong

The two other arms carrying the identical ~7% exposure are the WORST TWO ARMS IN THE FIELD on Wong:

    sup_sparse    0.2099   rank 13 of 13
    u2s_sparse    0.1381   rank 14 of 14 (pre-swap field)

If 7% of a dataset's molecules seen with labels during pretraining bought a meaningful advantage,
those two would not be last on exactly that dataset. Combined with the high-overlap control above
-- where sup_dense, which cannot be leaked, gains MORE than the assay arms do -- the reading is
that this exposure is real, declared, and not doing measurable work.

## A measurement upgrade worth propagating

L1000_MCF7 holds 11,718 molecules and L1000_VCAP 7,800, both far below the 400k sampling cap the
audit used. So every L1000 figure in the table above is EXACT rather than a lower bound. Only PCBA
and WONG are sampled and therefore floors. This matters most for the Polaris rows, where
L1000_MCF7 is the dominant source -- those numbers are not going to grow.
