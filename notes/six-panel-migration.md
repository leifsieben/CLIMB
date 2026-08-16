# Six-panel benchmark migration (2026-08-16)

**Decision (user, 2026-08-16):** replace the MoleculeNet-centric benchmark suite with a fixed
**6-panel suite** used *consistently across every figure* (mainline, ablation, AND scaling). A
benchmark suite must be (a) diverse across MPP task types and (b) challenging (not saturated).
MoleculeNet alone fails (b) — QM7 RMSE is flat across all data fractions (199→204), Tox21/BACE
plateau by ~25% data — so scaling/label-efficiency claims measured on it are ambiguous.

**CheMeleon is explicitly OUT of these ablation/scaling plots** (kept only as a curiosity
comparator elsewhere). These figures are CLIMB arms + standard anchors only.

## The canonical 6 panels

| # | Panel | Task type | Metric | Split | Source suite |
|---|---|---|---|---|---|
| 1 | **MoleculeACE** (30 ChEMBL targets) | potency regression + activity cliffs | **macro-mean RMSE** over 30 targets (overall + cliff subset) | provided MoleculeACE splits, 3 seeds | `chemeleon_suite/moleculeace` |
| 2 | **CBS** (Truong 2026) | rare-active virtual screen (0.41% actives) | **NEF1%** | UMAP-cluster 5-fold | `cbs_benchmark` |
| 3 | **BACE** | binding classification | ROC-AUC | scaffold 5-fold | MoleculeNet |
| 4 | **BBBP** | CNS/ADMET classification | ROC-AUC | scaffold 5-fold | MoleculeNet |
| 5 | **Tox21** | multi-task toxicity (12 assays) | mean ROC-AUC over 12 subtasks | scaffold 5-fold | MoleculeNet |
| 6 | **QM7** | quantum property regression | RMSE (atomization energy) | scaffold 5-fold | MoleculeNet |

Coverage: 2 regression (potency, quantum) + 3 classification (binding, ADMET, toxicity) + 1
rare-active enrichment. **HIV and ESOL/Lipophilicity are dropped** from headline — HIV's
rare-active story is carried better by CBS; ESOL/Lipo are physchem regressions already
represented by QM7's regression + MoleculeACE's regression.

### MoleculeACE aggregation — why it is defensible
All 30 targets are pKi/pEC50 (log-molar potency), so RMSE is a common physical unit. Per-target
label SD is tightly clustered (min 0.93, median 1.11, max 1.96), so a plain macro-mean is not an
artifact of one high-variance assay. Recipe:
- **Macro-mean** of per-target RMSE (each target = 1 unit; NOT pooled-molecule RMSE, which would
  weight big targets 6× and confound size with difficulty). Matches van Tilborg (MoleculeACE) &
  Burns (CheMeleon, 0.666) convention → directly comparable to published numbers.
- **Uncertainty = target-level cluster bootstrap** (resample the 30 targets, recompute macro-mean
  → 95% CI). Same cluster-bootstrap machinery as the CBS/headline rigor protocol. Average the 3
  seeds within each target first.
- Show the **30-target distribution** (strip/box) behind each method's mean.
- Split **cliff vs non-cliff** RMSE — the cliff subset is the un-saturated "challenge" number.
- **SI robustness:** re-rank on normalized RMSE (RMSE ÷ per-target label SD); ranking should be
  invariant (closes the "scale-dependent average" critique). Zero new compute.

## Best-two CLIMB arms (recomputed on the new suite)
On MoleculeACE macro-mean RMSE the best two *frozen* CLIMB arms are **`sup_only:dense` (0.774)**
and **`unsup_only` (0.777)** — statistically tied, ahead of the next cluster (mixed 0.791,
dense_plus_sparse 0.812). They come from **different methodologies** (supervised multi-task
regression vs unsupervised MLM), which supports the intended narrative: *"the best two CLIMB
models arise from very different pre-training methodologies, so we focus on these."*
- Caveat: `unsup2sup:dense` ties them (0.777) but is a hybrid; picking the two *pure* exemplars
  is the clean choice.
- This **flips the old MoleculeNet verdict** (which favored `dense_plus_sparse` as #2) — a further
  reason the MoleculeNet ranking should not drive the narrative.
- Anchors still win outright (fp_desc 0.676, ecfp4/Morgan-XGB 0.688), consistent with the paper's
  central thesis.

## Central results location
All re-aggregated / re-evaluated 6-panel results land under **`figure_data/six_panel/`**:
```
figure_data/six_panel/
  mainline_8M.csv          # Wave 1: all arms × 6 panels @ 8M (re-aggregated from existing)
  mainline_8M_bootstrap.csv# target-cluster-bootstrap CIs for the MoleculeACE panel
  scaling_a2.csv           # Wave 2: compute ladder × 6 panels
  scaling_h1.csv           # Wave 2: canonical/enumerated × 6 panels
  scaling_vocab.csv        # Wave 2: vocab × 6 panels
  labeleff_fractions.csv   # Wave 3: fraction grid × 6 panels (incl e2e arm — needs retrain)
  README.md                # schema + provenance per file
```

## Per-figure audit & migration status

Legend: **frozen** = re-eval needs only the saved encoder (cheap); **e2e** = per-task fine-tune
(needs retraining on the new panels). "Have ckpt?" = encoder available on S3/local.

| Figure (cell) | What it shows | Have ckpt? | Probe | New panels needed | Wave | Status |
|---|---|---|---|---|---|---|
| A1 mainline battery | arms vs anchors @ 8M | yes (results on disk) | frozen+e2e | MoleculeACE ✓, CBS ✓(frozen), 4×MolNet ✓ | 1 | re-aggregate, ~0 compute |
| A1 ablation matrix | readout/objective ablation | yes | frozen | same | 1 | re-aggregate |
| A2 compute ladder (12) | metric vs compute {2M…96M} | yes (S3 `climb_v2_phase2`) | frozen | MoleculeACE + CBS (4×MolNet already done) | 2 | frozen re-eval |
| H1 canon/enum (26) | metric vs data-fraction | yes (S3 `climb_v2_h1`, 30 enc) | frozen | MoleculeACE + CBS | 2 | frozen re-eval |
| SV vocab (32) | metric vs vocab | yes (S3 `climb_v2_vocab`, 8 enc) | frozen | MoleculeACE + CBS | 2 | frozen re-eval |
| Label-eff / crossover | metric vs train-N (fractions) | frozen arms: yes; **e2e arm: NO** | frozen+e2e | MoleculeACE + CBS at fractions | 3 | **RETRAIN e2e arm** |

### Gaps that require NEW compute
1. **Wave 2 (frozen, cheap ~10 GPU-h):** MoleculeACE (30 tasks) + CBS frozen-probe on every
   scaling encoder (A2 ~12, H1 30, vocab 8 ≈ 50 encoders). `s3 sync` (~8 GB) + loop
   `eval_v2.evaluate(featurizer="encoder", head="mlp", cv_folds=5)`. No retraining.
2. **Wave 3 (RETRAIN, GPU):** the e2e-after-pretraining arm has never been fine-tuned on
   MoleculeACE or CBS at the fraction grid. Also `sup_only:mixed` / `minimol_full` lack CBS
   frozen results (small top-up, folds into Wave 2). The e2e fraction grid is the real retrain:
   {best-two encoders} × {MoleculeACE 30 + CBS} × {5 fractions} × {3 seeds}.

## Execution waves
- **Wave 1 — mainline re-aggregation (local, ~0 compute):** `scripts/six_panel_aggregate.py`
  → `figure_data/six_panel/mainline_8M.csv` + bootstrap. **DONE.**
- **Wave 2 — scaling frozen re-eval:** `scripts/six_panel_frozen_reeval.py` +
  `scripts/six_panel_w2_run.sh` (gated self-shutdown). **LAUNCHED 2026-08-16** on
  **i-073b5c44d553791d9** (g5.xlarge A10G on-demand, us-east-1f, 98.92.104.143). 61 scaling
  encoders (A2 23 + H1 30 + vocab 8) × MoleculeACE frozen. MoleculeACE results land in
  `figure_data/chemeleon_suite/moleculeace/<prefix>/` (+ S3), one home for all arms+scales.
  Box self-stops only when `figure_data/SIX_PANEL_W2_DONE` is written (all encoders' MoleculeACE
  verified). ~5–10 GPU-h. Aggregation into `six_panel/scaling_{a2,h1,vocab}.csv` is a follow-up
  local step once results are back.
  - **CBS scaling panel BLOCKED:** the raw `data/cbs.csv` (with the UMAP provided-fold column)
    lived only on the terminated CBS box — not on S3, not local. Molecule labels are recoverable
    from `cbs_benchmark/*/moleculenet_cv/test_predictions.csv`, but the FOLD assignment is lost, so
    CBS-per-scaling-point can't be reproduced comparably to the 8M CBS numbers. Driver auto-skips
    CBS while `data/cbs.csv` is absent and will pick it up on re-run once staged. DECISION NEEDED:
    locate the original cbs.csv, or regenerate folds (breaks cross-comparability with the 8M panel).
- **Wave 3 — e2e retraining, best-two arms only (`unsup_only`, `sup_only:dense`):** full-data +
  fraction-grid e2e on the new panels. e2e fraction knob = `finetune_e2e_v2.evaluate_finetuned(
  train_subsample, subsample_seed)`, hold-out only (NOT cv). Template: `scripts/chemeleon_suite_e2e.py`.
  STATUS: **set up next** (own box or chained after Wave 2); CBS sub-panel shares the same blocker.

## Cross-session
The ipynb (notebook/figures) session was informed of: the 6-panel definition, the MoleculeACE
aggregation recipe, the central `figure_data/six_panel/` location, and the best-two finding.
