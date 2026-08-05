# Handoff → compute session: two data-correctness fixes + how they unify with your stats-rigor work

**From:** notebook session · **Date:** 2026-08-05
**Why you:** you own the eval pipeline / AWS re-runs and the stats-rigor protocol. These two fixes
change the point estimates your `analysis/rigor/*` were computed on, so they need a recompute; and the
vocab wave (yours) is affected. Nothing here overturns the thesis.

## The two fixes (both in `eval_v2._load_moleculenet`; `finetune_e2e_v2` inherits them via import)

1. **Tox21 missing labels were fake negatives.** DeepChem encodes missing multitask labels as `y=0`
   with `w=0` (NOT NaN). The loader returned only `.y` and dropped `.w`, so **16,012 missing Tox21
   cells** (of 93,876; the manuscript's own 77,864 valid / 5,858 pos / 72,006 neg confirm this) were
   used as true inactives in head training, validation early-stopping, ROC-AUC, NEF and the paired
   tests. The whole pipeline masks by NaN, so the masks were silent no-ops. **Fix:** `y[w==0]=NaN`.
   Effect: Tox21 AUC rises **~+0.015…0.020 per arm** (verified end-to-end; heads were being trained on
   fake negatives). Regression test in `tests/test_moleculenet_labels.py` locks the counts.

2. **Regression targets: native units + per-fold scaler (no leakage).** The loader used DeepChem's
   default `NormalizationTransformer` (fit on DeepChem's train split), then we concatenate+re-split into
   our CV folds — leaking the normalization stats across folds AND leaving RMSE in standardized units
   mislabelled as log mol/L / physical. **Fix:** load with `transformers=[]` (native targets); fit a
   target scaler PER SPLIT on train labels only; inverse-transform predictions before scoring. ESOL now
   **~1.03 log mol/L**, QM7 **~197 native** (were 0.49 / 0.87 normalized). Rankings unchanged;
   classification untouched (scaler is identity; ROC-AUC/NEF are rank-invariant).

**Unaffected and NOT re-run:** HIV, BBBP, BACE (single-task; both fixes are no-ops there). Their rows
are preserved byte-identical by the merge.

## Re-run status (this session, local, from checkpoints — no retraining)

- Driver: `scripts/rerun_corrected_tasks.py` re-evals only Tox21/ESOL/QM7(/Lipo) per run and **merges**
  into each run's existing `moleculenet{,_cv}/` outputs, keeping HIV/BBBP/BACE untouched. Idempotent
  (`.corrected_v2.json` markers), interruption-safe.
- **In progress:** `climb_v2_phase2` (81), `climb_v2_ablation_dedup` (10), `climb_v2_h1` (30) frozen arms.
- **Pending:** e2e arms (`_eval_ceiling*`, `e2e_random_0*`) via `finetune_e2e_v2` — only ESOL is affected;
  and the **vocab wave (yours)** — see below.

## What you need to know for the stats-rigor unification

- `analysis/rigor/comparison_fdr_bootstrap.csv` and `seed_variance.csv` were computed on **pre-fix**
  `figure_data` (I checked: ESOL is still normalized, Tox21 still pre-mask in them). They are **stale**.
- After the batch, the notebook session will re-run **`rigor_report.py` / `compare_many(n_boot=1000)`
  on the corrected `figure_data`**, regenerating `analysis/rigor/*`, then redraw A1 + T1/T2 with the new
  CIs/FDR. We reuse your `compare_models.py`/`rigor_report.py` verbatim — same method, corrected data.
- Your **HIV-NEF1%** and **BACE** claim-drops are single-task, unaffected by our fixes, and stand.

## Vocab wave (yours) — minor, your call

`climb_v2_vocab` runs also score Tox21/ESOL/QM7, so `figSV_vocab` is technically affected (ESOL/QM7
panels rescale to native units; Tox21 shifts slightly). The near-null conclusion is unchanged. We
**excluded** it from our batch because each vocab run has its own tokenizer and it's your wave. It's
SI/minor — no urgency. If you want it fully consistent: re-run the 8 vocab runs on Tox21/ESOL/QM7 with
the fixed `eval_v2` (their per-run tokenizers) and regenerate `vocab_cv_summary.csv`. Say the word and
the notebook session will do it instead.

## Git

We're on `v2-redux`; your `compare_models.py`/`rigor_report.py`/README §8.1/§10 are already present
here. We'll reconcile `v2-redux` ↔ `main` at the end so the paper sees corrected data + rigorous stats
unified, and will ping when the batch + rigor recompute are done.
