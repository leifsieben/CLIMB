# The suite xgb bases do not reproduce across environments

2026-08-21. Measured, not inferred.

## What was tested

`scripts/xgb_seed_replicates_run.sh` stage 0 rebuilds `moleculeace/unsup_8M__xgb` into a scratch
dir in a freshly built venv, holding EVERYTHING else fixed -- same encoder checkpoint, same
tokenizer, same data, same head seeds 42/117/709 -- and diffs the shared cells of `results.csv`
against the published base.

    630 shared cells, max |delta| 0.233949

That is orders of magnitude above float noise and far above the ~0.95% arm64/x86 float
reduction-order difference already documented elsewhere in this repo.

## What it is NOT

Not the `fp_desc` failure (scores computed from stale predictions). The published base was
rescored from the `test_predictions.csv` beside it, against `chemeleon_suite/data` ground truth,
on the `y [pEC50/pKi]` column:

    90 (task, seed) pairs, max |recomputed - published| = 4.44e-16

The published scores match their own predictions exactly. They are correct as computed. The fresh
environment simply computes DIFFERENT PREDICTIONS from the same inputs.

## Why the fix is a full-trio rebuild

`chemeleon_suite_run.py` has no `--features_npz` path, so suite cells cannot use the pinned
reference env that the MolNet and CBS cells use. The venv that produced the published bases exists
on no box. Therefore replicates built today would differ from their base by an amount that is
partly environment, and the seed spread would stop being a pretraining measurement. Rebuilding all
three cells in one env restores the estimand at the cost of moving the point estimate.

## The consequence nobody had queued

If the environment moves an xgb cell by 0.23, then ANY arm whose replicates were built in a
different env from its base has an interval that is partly an env measurement. The mtimes:

    moleculeace/chemeleon_frozen__xgb      2026-08-19 21:13
    moleculeace/chemeleon_frozen__xgb_s1   2026-08-20 16:47
    moleculeace/chemeleon_frozen__xgb_s2   2026-08-20 17:57

Base one wave apart from its replicates, and `chemeleon_frozen_xgb` is ranked and finishes #2
overall. Two hypotheses predict opposite answers and the same one-cell test settles it: drift in
xgboost contaminates chemeleon_frozen too; drift in the encoder forward pass (torch/transformers)
leaves it clean because it never runs the encoder. Test queued with the compute session
2026-08-21: rebuild that one cell at its own published seeds and diff.

## Preserved

The published bases are not in git. Before the rebuild overwrote them:

    s3://climb-s3-bucket/experiments/_preserved/published_xgb_bases_20260821/
      {moleculeace,polaris}/{unsup_8M__xgb,skip_dense_8M__xgb}/

Do not clean that prefix -- it is what quantifies how far the point estimates moved.
