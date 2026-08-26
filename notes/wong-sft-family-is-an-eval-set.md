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
