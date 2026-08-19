# B1p1 label-efficiency — replot DONE (frozen arms), e2e wired for auto-clear

**From:** notebook session · **Date:** 2026-08-06 · **Commit:** 6f364db (on top of your c252bcd)

## What I did
Replotted Fig B1p1 from your per-task fraction data
(`analysis/rigor/label_efficiency_fractions_summary.csv`). Capping is gone — every core
task now shows 5 distinct `n_train` points; x-axis is `n_train` (per-task); regression is
native units; HIV keeps ROC-AUC (train-vs-test-gap panel needs a shared train/test metric).

Touched: `notebook_cells/14.py` (plot), `notebook_cells/28.py` (inventory/blocker).
Notebook rebuilt + executed + manifest checked + `verify_notebook_sync.py` = OK.

## What you need to do when the e2e arm lands
Nothing on my side is hard-coded to 4 arms. Both cells key off the CSV:
- **Plot** (`le()` in cell 14) reads `arm=='e2e'` rows automatically; the legend flips from
  "NOT RUN" to a live line as soon as the rows exist.
- **Inventory blocker** (cell 28) clears when `'e2e' in set(_led.arm)`.

So: **append the e2e rows to `label_efficiency_fractions_summary.csv`** (same schema:
`arm,task,task_type,metric,split,fraction,pct,n_train,mean,std,n_cells`, with `arm='e2e'`),
at the **same 5 per-task fractions** so it shares the grid. Then ping me (or just re-run
`build_figure_notebook.py` + execute) and B1p1 finalizes. No cell edits needed.

## Note
Neither c252bcd nor 6f364db is pushed yet (branch is ahead of origin by 2). I'm holding the
push for the user; if you push as part of your batch, my commit rides along fine.
