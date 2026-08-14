# Fig CheMeleon (mainline) — cross-task model comparison (Burns 2025 suite): spec

**Status:** DRAFT SPEC. Core finding solid (CheMeleon≈XGBoost tie; CLMs below both), but the **CLM
positioning is NOT final** — a recipe-robustness check (stronger fine-tune LR/epochs on the largest
tasks) + a `chemeleon_frozen` re-run (adds a model row) are running. **Hold the build until the peer
re-pings** with final numbers + HF/zip package. Placement: **mainline** (user, 2026-08-14).

## Source
Burns et al. 2025 CheMeleon suite, replicated on all CLIMB 8M models + Morgan/XGBoost baselines + 14
published baselines. 28 Polaris/TDC tasks + 30 MoleculeACE activity-cliff tasks. Each track's OWN
fixed split, task-defined primary metric, 3 seeds. Methodology: `chemeleon_suite/METHODOLOGY.md`.
Ready-made peer version: `chemeleon_suite/summaries/model_comparison_rank.{csv,png}`.

## Why mean-rank (not scaled metric)
Polaris mixes Pearson/Spearman/R²/ROC-AUC/PR-AUC — can't average raw metrics. Use **mean rank across
tasks** (1=best, direction-aware) with a **bootstrap 95% CI over tasks**; overlapping CIs = tied
across the suite. (Burns' per-task panels use min-max/baseline-scaled metric; that's per-task, not
aggregable — the mean-rank forest is the honest cross-task summary.)

## Figure
- **2×2 forest plot**, one panel per track: `mace_overall`, `mace_cliff`, `mace_noncliff`, `polaris`.
- Each panel: models on the **y-axis ordered by mean rank** (best at top), **mean rank ± 95% CI** as
  horizontal whiskers (x-axis = mean rank, 1..11; lower = better). Dashed **reference band = best
  model's CI** (Burns style) so ties are visible.
- Input: `chemeleon_suite/summaries/model_comparison_rank.csv`
  (cols: `panel, model, mean_rank, ci_lo, ci_hi, n_tasks`; n=30 MolACE / 28 Polaris).
- **Colours:** A1.a `rc_color` for our arms; distinct for the 3 published refs. Frozen vs e2e = same
  base colour, **filled marker = e2e, open marker = frozen**. Mapping:
  - `CheMeleon (e2e)` → published (distinct, e.g. purple); `XGBoost (fp+desc)` → `rc_color("fp_desc")`;
    `XGBoost (fp)` → `rc_color("ecfp4")`.
  - `unsup_8M (*)` → `rc_color("unsup_only")`; `sup_dense_8M (*)` → `rc_color("sup_only:dense")`;
    `sup_dense+sparse_8M` → `rc_color("sup_only:dense_plus_sparse")`; `sup_sparse_8M` →
    `rc_color("sup_only:sparse_all")`; `no_pretrain_random` → `rc_color("no_pretrain")`;
    `no_pretrain_e2e` → `rc_color("no_pretrain_e2e")`.
- Highlight the **top tier** (CheMeleon + XGBoost fp+desc, overlapping on all 4 panels).

## Headline numbers (current CSV — CLM ranks may shift with the robustness check)
Top tier tied on every panel: MolACE overall CheMeleon 1.80 [1.47,2.20] vs XGBoost(fp+desc) 2.03
[1.77,2.30]; cliffs 2.33 vs 2.37; non-cliff 1.70 vs 2.17; Polaris 2.68 vs 2.82. XGBoost(fp) third
(~2.5–2.7, Polaris 5.6). Best CLIMB CLM ~4.6+ (non-overlapping with the top pair). `no_pretrain_e2e`
last (~9–10.6).

## Caption caveats (peer)
1. `CheMeleon (e2e)` is their fine-tuned/published pipeline; our entries include BOTH frozen probes AND
   fine-tuned (e2e) CLMs — even fine-tuned, our CLMs don't reach the top pair.
2. `no_pretrain_e2e` (fine-tune a 41M transformer from scratch on tiny tasks) is WORST — it barely
   trains (predicts ≈ mean); pretraining matters, yet the pretrained-then-FT CLM still loses to XGBoost.
3. Reproduces **van Tilborg et al.**'s own MoleculeACE finding (classical descriptors ≥ deep models on
   activity cliffs) — supports it being a real regime effect, not our deployment.

## Complementary (optional second panel/table)
Dominance table (`chemeleon_suite/summaries/`, coming): pairwise sign-test across tasks, BH-FDR<0.05,
#models each significantly beats / #beaten-by.

## Gates before finalizing (peer to confirm)
1. **Recipe-robustness check** (stronger FT LR/epochs on largest tasks) — confirms the e2e deployment
   is fair before we assert "CLMs lose even fine-tuned".
2. **`chemeleon_frozen` re-run** (frozen probe of CheMeleon's fingerprint) — ADDS a model row; rebuild.
3. **Reproduction**: `chemeleon_suite/summaries/*` committed; raw suite outputs on HF/S3 (peer package).
4. Exact published-baseline set / labels finalized (14 baselines mentioned; CSV currently has 11 models
   — confirm which published refs make the plot).

## Cells / commit
New mainline cell pair (appended, e.g. 43/44), `save_fig("figCheMeleon_suite_rank")`. Commit the
`chemeleon_suite/summaries/*.csv` inputs. Do NOT touch existing figures/datasets.
