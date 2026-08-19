# Polaris scores clobbered by the hERG top-up — event + recovery (2026-08-17)

## What happened
The 2026-08-16 18:50 hERG top-up run (commit `2ca8832`, two-venv polaris scoring) **rewrote**
`figure_data/chemeleon_suite/polaris/<dir>/polaris_scores.csv` instead of appending to it. The
five headline arms lost their other 27 Polaris tasks (28 → 1, hERG only):

- `skip_dense_8M` · `skip_dense_plus_sparse_8M` · `skip_sparse_all_8M` · `unsup_8M` · `u2s_dense_from8M`

S3 holds the same truncated files (synced 18:50:08) — no recovery there. The six-panel suite is
UNAFFECTED (it only reads hERG from Polaris). Fig A1's Polaris suite was silently degraded for
those five arms (their Polaris "mean rank" was hERG alone).

## Recovery (no re-run needed for figure values)
`chemeleon_suite/summaries/polaris_summary.csv` (built 2026-08-13 from the SAME per-seed runs)
holds every (model, task, metric) mean. Verified consistent: its skip_dense_8M hERG mean matches
the rewritten file's 3-seed mean to 7 decimals (0.7705940). `figures/allsuites.py::_polaris` now
fills tasks MISSING from a truncated `polaris_scores.csv` from that summary (per-seed granularity
is lost for the recovered tasks; A1 averages per task anyway). A1 is back to 65-66/66 coverage
for all 16 mainline arms.

## If per-seed values are ever needed again for those 5 arms × 27 tasks
Re-score Polaris from the saved encoders (the harness that wrote the 28-task files originally) —
the summary's std/n_seeds=3 confirms 3 eval seeds existed but their per-seed values are gone.

## Prevention
Any top-up that writes `polaris_scores.csv` must append/merge, never rewrite. (Same class of
guard as the `_cell` vs ensemble-row rule: the file format supports multiple tasks; treat it as
append-only.)
