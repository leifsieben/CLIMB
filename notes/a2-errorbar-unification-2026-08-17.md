# A2 error-bar unification — implementation spec (2026-08-17)

**STATUS: IMPLEMENTED 2026-08-17** in `scripts/six_panel_aggregate.py` (`panel_stats`,
`mol_dir_summaries`, `mace_seed_macros`, `_cv_dir`) + `figures/fig_A2.py` (`_err` reads
`sd_total`, bootstrap override removed, n/a cells rendered). Verified: sup_dense Tox21 0.0023 →
0.0478, BACE 0.0044 → 0.0253, QM7 0.30 → 6.15; anchors unchanged; chemeleon_e2e healthy
(QM7 199.5 ± 4.9); values did not move, only whiskers. One self-introduced bug found + fixed
during implementation (case-sensitive `_MEAN`→`_STD` replace after lowercasing read the MEAN as
the SD). Also landed alongside: `figures/allsuites.py` Polaris clobber recovery +
`figures/fig_A1.py` ≥60/66 coverage floor — see notes/polaris-clobber-recovery-2026-08-17.md.

**For the LLM currently editing `figures/` + `scripts/`: this is the agreed spec. Please implement
exactly this; the audit behind it is summarized at the bottom so you don't have to redo it.**

## User decisions (2026-08-17, final)

1. **Panel #4 = hERG.** Keep the 2026-08-16 swap; BBBP stays dropped. The 6 panels are fixed:
   MoleculeACE · CBS · BACE · hERG · Tox21 · QM7.
2. **A2 error bars: ONE estimand for every bar — the total single-run SD** ("how much does one
   replicate evaluation of this whole panel vary"). No per-family definitions.
3. **Fig A1 (66-dataset ranking) stays as-is.** Audited, no methodological mistake found — do not
   rebuild it on the 6 panels.

## What to change

### 1. `scripts/six_panel_aggregate.py`

- **`seed_stats()`** — add the pooled SD across **all ensemble fold cells** as a new returned /
  emitted quantity, `sd_total`:
  - CLIMB arms: SD over 15 cells (3 pretraining-seed dirs × 5 folds).
  - anchors / CheMeleon (1 dir): SD over their 5 folds — numerically identical to today's value.
  - Keep writing `sd_seeds`, `n_seeds`, `n_cells` alongside (transparency), but `sd_total` is what
    the figure reads.
- **MoleculeACE** — compute per-(dir, eval-seed) macro-means over the 30 targets
  (`subset=="overall"`, `metric=="rmse"`); `sd_total` = SD across those macro-means (3 values today;
  becomes 9 when the `_s1/_s2` pretraining top-up lands — `mace_seed_dirs()` already auto-picks up
  new dirs, so no code change needed then). Emit in the MoleculeACE row's `extra` as
  `sd_total=...;n_seeds=...` next to the existing `cliff=...;noncliff=...`.
  Keep `mainline_8M_bootstrap.csv` unchanged (95% target-cluster CI is for the paper text, not the
  figure).
- **hERG** — unchanged: `sd_evalseeds` over the 3 eval seeds IS the total single-run SD for a
  single-encoder panel.

### 2. `figures/fig_A2.py`

- `_err()`: read `sd_total` first; fall back to `sd_evalseeds` (hERG). **Stop** overriding
  MoleculeACE with the bootstrap half-width (delete the `load_bootstrap()` block in `table()`).
- Docstring updates:
  - Error-bar paragraph → the single estimand: "±1 SD of one replicate evaluation of the panel:
    CLIMB arms 3 pretraining seeds × 5 folds (15 cells); ECFP/ECFP+desc/CheMeleon 5 folds (no
    pretraining stage to replicate); hERG 3 eval seeds (one provided split); MoleculeACE 3
    eval-seed macro-means (pretraining-seed top-up pending)."
  - Fix the TEST_N overclaim: they are hand-typed constants (values verified against raw files
    2026-08-17: BACE 1513, Tox21 7823, QM7 6838, CBS 10445, hERG 132, MoleculeACE 9802) — either
    say so or actually read them.

### 3. Re-run + sanity check

```
python3 scripts/six_panel_aggregate.py && python3 -m figures.fig_A2
```

Expected: CLIMB whiskers grow (sup_dense Tox21 0.0023 → ~0.048; BACE 0.0044 → ~0.025; QM7 0.30 →
~6.1), anchor whiskers unchanged, MoleculeACE whiskers shrink to the small eval-seed SD. Bars/values
must not move — only the whiskers change.

## Open items (do NOT silently "fix" these in aggregation)

- **CheMeleon arm still sources `chemeleon_frozen`** while its label says "end2end". When the e2e
  rerun lands (`figure_data/climb_v2_phase2/chemeleon_e2e{,_s1,_s2}/moleculenet_cv/`, CBS
  `cbs_benchmark/chemeleon_e2e_s{0,1,2}`, MoleculeACE dir), repoint `figures/arms.py` chemeleon
  `src` to the 3 e2e dirs (→ proper 3-seed `sd_total`). Until then the QM7 cell (268.8 ± 94, frozen
  probe's fold-2 divergence to 427.7) stays flagged in the docstring.
- **`u2s_dense_sparse` MoleculeACE = 1.080 is one MOLECULE's fault** — RESOLVED 2026-08-17, it is
  a genuine arm property, not a harness bug. The arm's noncliff RMSE 7.94 on CHEMBL1862_Ki is
  driven by a single test molecule, `CC1=CN2CC(=O)NN=C2C=C1` (fused bicyclic N–N-ring hydrazide,
  assay-floor pKi 4.00, not in train), predicted at **−72.3 / −69.8 / −57.1** across all 3 eval
  seeds. The other 93 noncliff predictions are sane (median 7.1); without the molecule the arm's
  macro is ~0.92. The explosion lives in the FROZEN features (consistent across head seeds;
  predictions correctly scaled elsewhere) — the encoder emits an extreme activation on this
  out-of-distribution chemotype and the unbounded linear head extrapolates. Same molecule is
  predicted at 4.6–8.1 by every other arm (anchors nail it: 4.58–4.96). The arm shows 8 more
  milder extremes on 3 other targets (cyclic peptide +18, lipid +14.9); the healthy sibling
  u2s_dense_from8M has 0/29,406. Handling: aggregation stays neutral (value remains 1.08); report
  in caption/text. Optional SI robustness: median per-target RMSE alongside the macro-mean.
- **Single-pretraining-seed panels**: MoleculeACE (1 seed × 3 eval seeds) and hERG (1 run × 3 eval
  seeds) until the top-up lands. Missing CBS for `sup_mixed`, `sup_minimol`, `u2s_mixed`,
  `u2s_minimol`; `s2u_dense` fully pending (run in flight).

## Audit summary (why this is safe — verified 2026-08-17)

- Aggregation independently reproduced by hand: sup_dense BACE 0.8490 (3 dirs × 5 ensemble folds),
  ecfp BACE 0.8711 ± 0.0351 (5 folds). Re-running the aggregator reproduces `mainline_8M.csv`
  byte-identically.
- Values use ensemble fold rows, never `_cell` (the 2026-08-16 bug stays fixed).
- The old mixed definitions, quantified: sup_dense Tox21 drew ±0.0023 while its own fold-SD is
  0.0477 (20×); ECFP drew ±0.0472. Whisker lengths were definition artifacts, not stability.
- TEST_N constants verified against raw prediction files (see above).
- A1's rank rescaling + design-effect SE correction reviewed — sound; user confirmed A1 unchanged.
