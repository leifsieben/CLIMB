# Why two runs of the concat pipeline disagree, and what we tested

2026-08-20. Recorded because the conclusion is a limitation on the paper's XGBoost numbers, not a
bug that got fixed.

## The observation

`analysis/rigor/concat_redundancy_stereo.csv` (the table fig_F draws) and
`concat_redundancy_climb_v2.csv` (produced later on an EC2 c5.4xlarge) disagree in every cell.

    ESOL cell      stereo     this laptop        v2
    fp+desc        0.7281        0.7281       0.7350
    CLM            1.3235        1.3235       1.3327
    desc+CLM       0.7512        0.7512       0.7671
    fp+desc+CLM    0.7564        0.7564       0.7766

This laptop reproduces stereo on 4 of 4 shared cells to four decimals and v2 on 0 of 7.
Evidence: `analysis/rigor/_verify_esol_local.csv`.

Of the 40 rows the two tables share, 34 differ outright and the 6 that "match" are nef1 sitting on
quantised rungs (1.0000, 0.9500, 0.9000) — coarse metric, not agreeing computation. Nothing
reproduces between them.

## What it is NOT

* **Not the encoder.** All 10 embedding-free rows (`fp`, `desc`, `fp+desc` — RDKit plus XGBoost,
  no forward pass) moved. Numerical noise in the encoder cannot reach those cells.
* **Not the code.** The peer session md5'd `descriptors_v2.py`, `featurize_v2.py`, `heads_v2.py`,
  `eval_v2.py` and `scripts/concat_redundancy.py` on the box against the repo. Identical, all five.
* **Not the libraries.** python 3.9, numpy 2.0.2, rdkit 2025.09.2, xgboost 2.1.4, sklearn 1.6.1 on
  both machines.
* **Not stale environment.** No `OMP_*`, `MKL_*`, `CONCAT_*` or `FP_VARIANT` set on the box.
* **NOT THREAD COUNT.** `heads_v2.py:214` passes `n_jobs=0` ("all available"), which made the
  16-vs-12 core difference the leading hypothesis: XGBoost's `hist` builder sums histograms in a
  parallel reduction, so thread count changes floating-point order, split gains, and eventually a
  tree. Tested and rejected on this machine:

      OMP_NUM_THREADS=2 / default(12) / 16      all 7 cells bit-identical
      n_jobs forced to 2 and to 12 at fit()     identical again

  Five configurations, seven blocks, one number each. Evidence:
  `_verify_esol_local.csv`, `_verify_esol_t2.csv`, `_verify_esol_t16.csv`,
  `_probe_nj2.csv`, `_probe_nj12.csv` in `analysis/rigor/`.

  The env-var arm alone would have been worthless — `n_jobs=0` may resolve from hardware and ignore
  `OMP_NUM_THREADS`, so a null there could just mean the lever is disconnected. Forcing `n_jobs` at
  `fit()` (`scratchpad/njobs_probe.py`, patching `fit` rather than `__init__`, which sklearn's
  `get_params` rejects) is a lever that cannot be ignored, and it also shows no effect.

## What is left

CPU architecture. The box is Intel with AVX-512; this laptop is Apple silicon. SIMD width changes
the reduction order in the same way threads would have, and it fits every part of the signature:
every row moves including embedding-free ones, deterministic within a machine, immune to matching
versions and matching code, independent of the encoder. Unlike thread count it cannot be
configured away.

Untested, because confirming it needs an x86 box running the same code — the stereo box is gone.

## What follows for the paper

1. **fig_F keeps the stereo table.** It is the one the figure was drawn from AND the one a second
   machine reproduces. v2 is not adopted.
2. **One environment per table.** A merged panel would carry the machine difference inside a
   within-panel delta — the confound audit check 13 exists to catch.
3. **The reproducibility claim is machine-class-scoped.** "Reproducible" here means reproducible on
   the same CPU architecture. The paper should not claim more.
4. **Size it before dismissing it.** ESOL fp+desc moves 0.95% relative between machines. On
   MoleculeACE the ECFP4-vs-ECFP4+desc gap is 1.7%. The effect is the same order as differences
   the paper reports between arms, so cross-machine comparisons of XGBoost cells are not safe even
   though within-figure ones are (fig_F is one table; the mainline anchors are all EC2).
5. **Pinning `n_jobs` is NOT the fix** and should not be presented as one. It touches every XGBoost
   number in the paper and the variable it controls has been tested and excluded.
