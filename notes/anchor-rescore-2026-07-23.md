# Note for the AWS/compute session — anchor re-score (2026-07-23)

**Context.** Between 2026-07-22 and 07-23 the phase2 `moleculenet_cv` eval was re-scored (13 runs:
both classical anchors + all eight 8M CLIMB arms), and the retrained H1 wave `climb_v2_h1` landed.
The notebook session adopted the new snapshot (figures/tables regenerated, `figure_data_manifest.json`
rebuilt, everything committed). The new numbers look directionally *more* correct — but two things
about the **anchor** pass need a fix / confirmation at the source. Neither corrupts a reported metric;
both are about provenance and being able to state "this was a fix, not a regression."

## What changed (evidence)

- **Eval set refreshed (all arms):** HIV +7 molecules (41,120→41,127), Tox21 +96 (93,876→93,972),
  and **Lipophilicity dropped** entirely. Small, applied to every arm.
- **Anchors re-scored ~6h later than the CLIMB arms:** CLIMB arms at 07-22 ~18:0x, `ecfp4_anchor` /
  `fp_desc_anchor` at 07-23 00:53 / 01:20. So the anchors went through a separate, later pass.
- **ecfp4 ESOL RMSE 0.702 → 0.765** on the *same* 1,128 molecules (no set change, no dup rows).
  New OOF R²≈+0.40. CLIMB ESOL barely moved (0.487→0.490), so the shift is anchor-specific — a
  change in the anchor's CV folds or featurization, not a global re-split.

## Action item 1 — fix the anchor `test_predictions` double-append

`ecfp4_anchor` and `fp_desc_anchor` `moleculenet_cv/test_predictions.csv` have **HIV and Tox21 rows
duplicated exactly** (e.g. HIV 82,254 = 41,127 × 2), with **identical** `y_pred` — i.e. the dump was
appended twice for those two tasks. `eval_v2.evaluate()` unlinks `test_predictions.csv` at the start
of each call, so this most likely came from a second, separate eval call into the same `output_dir`
(the "HIV/Tox21 folded in on a later pass" pattern) that appended instead of replacing.

- **Impact: none on reported metrics.** AUC is identical with/without dedup (HIV 0.8000, Tox21 0.7384);
  the `suite_summary` means come from eval_v2's internal per-fold scoring, not the CSV; and the A1
  table already `.drop_duplicates(["mol_index","output_index"])` before its paired tests.
- **But** the raw CSVs are 2× size and will mislead anyone who reads them directly. Please re-dump
  those two files (or de-dup them) so the prediction files match the other arms.

## Action item 2 — confirm the anchor ESOL fold/featurization change was intentional

The ESOL anchor moved a lot (0.702→0.765) with an identical molecule set, and only for the anchors.
Please confirm what changed in the **later anchor pass** — CV fold assignment, ECFP4 featurization,
or XGBoost config — so we can state in the paper that the new number is a **fix** (bare ECFP4 is a
weak solubility featurizer; R²≈0.4 and lagging Morgan+desc is the expected behaviour) rather than an
unexplained regression. If the fold seed changed between the CLIMB pass and the anchor pass, re-score
the anchors on the **same** folds the CLIMB arms used so the two are strictly comparable.

## Not touched here

The notebook session owns `climb_figures.ipynb`; it did **not** modify `figure_data` (S3-owned) or
re-run any eval. These two items are for the compute session to resolve at the source.
