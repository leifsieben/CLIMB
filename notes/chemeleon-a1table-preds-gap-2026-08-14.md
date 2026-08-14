# ⚠️ CheMeleon e2e: A1-table bootstrap blocked — per-molecule preds not synced (preserve before teardown)

**From:** notebook session · **Date:** 2026-08-14

You reported CheMeleon e2e MoleculeNet **COMPLETE (all 7 tasks, n=3)** and asked me to wire the A1 table +
bootstrap re-run, then said the box self-stopped and you're tearing it down.

**The A1.b FIGURE is done and correct** — all 6 core panels show CheMeleon (e2e), HIV included (NEF1% 0.706),
+16.0% mean lift over the frozen floor (≈ fp_desc's +16.2%). Committed.

**The A1 TABLE is BLOCKED**, and the blocker is destroyed by teardown:

- Shared disk / S3 for `chemeleon_e2e` has ONLY the aggregate:
  `figure_data/climb_v2_phase2/chemeleon_e2e/moleculenet_cv/{suite_summary.json, verified.json}`
  (`_seeds:[0,1,2]` — the summary aggregates the 3 seeds).
- There is **no `test_predictions.csv`**. `scripts/best_model_bootstrap.py` (`arm_preds` → per-molecule OOF
  `test_predictions.csv`) needs those to run the scaffold cluster-bootstrap. No preds → CheMeleon cannot get
  co-best / beats-no_pretrain columns in Table A1.b.

## Ask (before teardown — else gone permanently; violates the keep-everything-for-repro rule)

1. Sync CheMeleon's per-molecule OOF predictions to shared local + S3:
   `figure_data/climb_v2_phase2/chemeleon_e2e/moleculenet_cv/test_predictions.csv`
   - seed-0 dump is the minimum (bootstrap reads `runs[0]`);
   - if per-seed dumps exist, sync all three as `chemeleon_e2e_s1/_s2` dirs, each with
     `moleculenet_cv/{suite_summary.json, test_predictions.csv}`. **Per-seed dirs also fix the figure error
     bars**: the single aggregate dir makes the notebook fall back to fold+seed `_STD` (`n_seeds=1`) instead of
     the 3-seed spread every other arm shows in A1.b.
2. Columns to match every other arm: `dataset, mol_index, output_index, y_true, y_pred` (+ `fold` if available).
   HIV ranks on NEF1%, so raw per-molecule scores are needed, not the reduced metric.

## Not a gap (do not "fix")

- No `moleculenet/` (single-split) eval exists for CheMeleon → A1.a stays all-pending for it, and it can't get a
  hold-out CI. Expected. (Only run the hold-out if you want CheMeleon in A1.a too.)
- `chemeleon_frozen` stays excluded per the e2e-only call.

**When `test_predictions.csv` lands:** ping me → I re-run `scripts/best_model_bootstrap.py`, add a
`chemeleon_e2e` entry to cell-10 `ARMS` + `ARM_RUNS`, rebuild/execute the notebook, `verify_notebook_sync`,
and commit the table update.
