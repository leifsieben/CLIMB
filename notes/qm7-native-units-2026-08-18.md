# QM7 was reported in two different units — and why the fix is a re-run, not a conversion

**Date:** 2026-08-18 · **Status:** re-eval in flight (38 frozen runs + 3 e2e runs)

## If you are here because you saw a QM7 RMSE of ~0.85

That number is **z-scored**, not kcal/mol. It is a **stale artifact**, not current behaviour.
Do not "fix" it by multiplying by a sigma. Re-run the eval.

## The defect

The QM7 column of `figure_data/six_panel/mainline_8M.csv` mixed two conventions:

| convention | arms | example |
|---|---|---|
| z-scored (stale) | 15 | `sup_dense` 0.8507, `ecfp` 0.9360, `e2e_no_pretrain` 0.8546 |
| native kcal/mol | 3 | `s2u_dense` 197.8, `chemeleon_e2e` 199.5, `chemeleon_frozen` 268.8 |

Max/min across the panel was **327x**. Plotted together the arms are not comparable: `ecfp` got a
±0.02 whisker on a 215.9-scale bar, invisible next to `chemeleon_e2e`'s ±6.6.

The error bars in `a2_errorbars.csv` were **not** the bug — they matched the bar to 0.00% on all
eight arms, faithfully mirroring whichever unit each arm was stored in.

## Root cause — the sentence worth keeping

**`eval_v2.py` has been correct for a month.** `_fit_target_scaler` fits on the current fold's
TRAINING labels only, and `_unscale_preds` inverse-transforms predictions back to native units
*before any metric is computed* — deliberately, to avoid both cross-fold leakage and
"RMSE in standardized units mislabelled as physical". `finetune_e2e_v2.py` calls the very same
helpers (line ~275).

So the normalized values are **stored results that predate that fix** — their
`moleculenet_summary.csv` files date to **2026-07-22**. We were not choosing between two
conventions. We were bringing stored results up to code that was already right.

That is why the fix is a re-run and not a conversion.

## Why NOT to convert by a sigma

1. Each fold is z-scored by **its own training sigma**, so no single constant is correct.
   Measured two ways on `skip_dense_8M`: **228.656** (affine fit of native vs normalized `y_true`)
   vs **228.9** (ratio of native to normalized RMSE). Close, not equal — a conversion bakes a
   systematic error into every QM7 number.
2. Converting leaves the per-fold predictions z-scored, so the next consumer inherits the problem.
3. The S3 and local copies of `test_predictions.csv` are **different runs**, not two encodings of
   one run: their `y_true` is a pure affine map of each other, but `y_pred` differ by up to
   **173 kcal/mol**. A conversion path would first have to pick which run is canonical.

## What was run

- `scripts/qm7_native_reeval.py` — 14 arms via `eval_v2` (11 CLIMB × 3 pretraining seeds,
  `random_encoder` × 3, `ecfp`, `ecfp_desc`) = 38 runs.
  `random_encoder` is a **frozen** probe, so it belongs here, not on the e2e path.
- `scripts/qm7_native_e2e.py` — `e2e_no_pretrain` (3 replicates) via `finetune_e2e_v2`.
  It fine-tunes the **same saved weights** `random_baseline_XX` was frozen at, mirroring
  `run_e2e_random.py`; a freshly seeded encoder would make the frozen and e2e bars differ in two
  things instead of the one they exist to isolate.

## Two traps that shaped the implementation

- **Output goes to `moleculenet_cv_qm7native/`, never into `moleculenet_cv/`.** `eval_v2` clears
  the run's rows in its `output_dir` before writing, so re-using the existing dir with
  `--datasets QM7` would have silently destroyed that dir's BACE/BBBP/ESOL/HIV/Tox21 rows.
- **The completion gate checks the UNIT**, not just presence: a run counts as done only if its
  minimum QM7 fold RMSE exceeds 10. If the stale convention ever returns it fails the gate rather
  than quietly passing.

## How it hid for so long

`aws s3 sync` stamps the local file's mtime with the S3 object's `LastModified`. A synced
`mainline_8M.csv` therefore showed an old date while its contents had been replaced underneath —
so a stale file looked untouched. Check contents, not mtime.
