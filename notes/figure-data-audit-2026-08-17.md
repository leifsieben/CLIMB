# Audit: what the figure layer actually reads (2026-08-17)

Triggered by the CheMeleon/QM7 anomaly. Scope: for every arm in `figures/arms.py`, is the source
(a) current, (b) complete — 3 pretraining seeds x 3 head seeds x 5 folds, (c) internally consistent.
Reproduce with `python3 scripts/audit_six_panel_sources.py`.

## FIXED — real bugs found

### 1. CheMeleon arm was mislabelled (caused the QM7 anomaly)
`arms.py` had ONE arm, `"chemeleon"`, with `label="end2end"` and `probe="frozen"` but
`src=chemeleon_frozen` for every panel. So the **frozen** arm's numbers were plotted on the
**end2end** comparison in Fig A2. That is the whole "CheMeleon QM7 = 268.8" report.
- **The e2e arm was never broken**: `chemeleon_e2e{,_s1,_s2}` QM7 = 198.7 / 199.9 / 199.9 (fold
  sd ~4.2-4.6), verified independently from the 6838-row per-molecule OOF.
- **Fix**: split into `chemeleon_e2e` (probe=e2e, 3 seeds) and `chemeleon_frozen` (probe=frozen,
  1 seed), each internally consistent. `figures/fig_A2.py` MODELS repointed to `chemeleon_e2e`.

### 2. The frozen CheMeleon QM7 failure is REAL (not a harness bug) — report, don't "fix"
`chemeleon_frozen` QM7 = 281 pooled; per-fold 212 / 227 / **434** / 292 / 242. Worse than
predicting the training mean (sigma = 228.7).
- NOT the 2026-08-07 target-centering bug: predictions are correctly centred and scaled
  (pred mean -1540 vs true -1531; pred sd 229 vs true sd 223). A centering bug shows a wrong mean
  or a collapsed sd.
- fold2 is bad across all three head seeds (370 / 529 / 404) => a property of the fold+features,
  not head init. The frozen CheMeleon embedding fails to encode atomization energy on some
  scaffold folds. Report with the fold spread visible.

### 3. CBS summary was 2 days stale and missing 5 arms
`experiment_cbs/cbs_nef1_summary.csv` was built 2026-08-14 while results landed through 08-16, and
`build_cbs_summary.py`'s ARMS list never included the four gap recipes. Also the four CBS-gap
results existed **on S3 only** — never pulled locally, so the figure session could not see them.
- **Fix**: synced `cbs_benchmark` from S3; added `sup_only:{mixed,minimol_full}`,
  `unsup2sup:{mixed,minimol_full}` and `sup2unsup:dense` to ARMS; rebuilt. Now 19 arms.
- Caveat: the 4 gap arms are **1 pretraining seed** (only the base encoder was run on CBS), vs 3
  for the others. Their `n_seeds=1` / `sd=0` is real, not a bug.

### 4. CBS seed sd was the population sd
`build_cbs_summary._std` used ddof=0 over pretraining seeds. At n=3 that understates the spread by
sqrt(3/2) = 22%, and disagreed with the sample sd the figure layer uses. Changed to ddof=1.
**All CBS error bars are now ~22% larger** — means are unchanged.

## OPEN — needs action

### 5. `s2u_dense` (forgetting arm) MolNet is hold-out, not 5-fold CV
Its MolNet eval landed in `s2u_dense_from8M_s*/moleculenet/` (single scaffold hold-out, written by
`launch_v2_wave`), whereas every other arm is read from `moleculenet_cv/` (5-fold). These are NOT
comparable and must not be pooled. Needs a `--cv_folds 5` re-eval into `moleculenet_cv/` before the
arm appears in any MolNet panel. Its CBS and MoleculeACE results are on the correct protocol.

### 6. `chemeleon_e2e` has no `moleculenet_summary.csv`
Its runner wrote `suite_summary.json` (per-dataset `<DS>_MEAN`/`_STD` over folds) + per-molecule
OOF, but not the per-fold `moleculenet_summary.csv` the aggregator globs. Per-fold cells are not
recoverable (no fold column in its OOF). The loader needs a `suite_summary.json` fallback for
e2e-style arms; MEAN/STD there are exactly the point estimate and fold spread.

## NOT bugs — known, documented asymmetries
- **Replication differs by arm type.** CLIMB frozen arms pool 45 cells (3 pretraining x 3 head x 5
  folds). The e2e and anchor arms store 5 per-fold values (seed-averaged) under `<metric>` rather
  than `<metric>_cell`; the aggregator's `_cell` -> plain fallback handles this, so nothing is
  dropped, but their error bars measure fold variance while CLIMB's also include pretraining-seed
  variance. Already noted in the figure script.
- **`random_baseline_00` is inconsistent within itself**: BACE has only aggregate rows while
  Tox21/QM7 have `_cell` rows. Handled by the same fallback.
- **MoleculeACE has no native CheMeleon e2e** (frozen only). Burns' published values in
  `chemeleon_suite/reference/reference_long.csv` are the e2e reference there, and are point
  estimates with no OOF, so they cannot be bootstrapped.
