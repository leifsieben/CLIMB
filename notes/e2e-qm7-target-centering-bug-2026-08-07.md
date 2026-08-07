# ⚠️ e2e label-efficiency arm: QM7 is broken (target-centering) — do NOT ship B1p1 with it yet

> ## ✅ RESOLVED 2026-08-07 (commit f5b0923)
> Root cause confirmed exactly as diagnosed below: `finetune_e2e_v2` / `finetune_v2` did not
> standardize regression targets after `_load_moleculenet` switched to native units, so QM7's
> ~−1531 offset was unreachable. **Fix:** standardize regression targets (fit on train, unscale
> predictions) via `eval_v2._fit_target_scaler`, matching the frozen arm. The 3 e2e **regression**
> tasks (ESOL/Lipo/QM7) were re-run on GPU; classification e2e + frozen arms unchanged.
> **QM7 e2e is now ~200–205 RMSE** (< 228.7 chance) and all 5 arms on QM7 sit within 200–226 →
> panel readable. The corrected data is in `analysis/rigor/label_efficiency_fractions_all_summary.csv`.
> Blast radius was B1p1 only (main-wave `e2e_random` QM7 was already sane ~200). **Safe to ship B1p1.**


**From:** notebook session · **Date:** 2026-08-07 · **Re:** your 54a8704 combined 5-arm file

## The finding
The e2e (no_pretrain end-to-end) arm's **QM7** RMSE is ~1490–1527 at every fraction. QM7 native
targets are μ≈−1531.14, σ≈228.66, so **predict-zero** gives RMSE = √(μ²+σ²) ≈ **1548**. The e2e arm
is landing right at predict-zero — it never learns QM7's large target offset.

Decisive tell: the e2e QM7 **train** RMSE is ALSO ~1500 (1530→1496 over the fractions). It can't fit
even its own training set, so this is not overfitting or a hard-task story — the regression head is
not being trained against a centered/standardized target.

```
                predict-mean (chance) = 228.7      predict-zero = 1548.1
e2e QM7 train RMSE:  1530.6 1534.7 1527.7 1518.7 1496.4
e2e QM7 test  RMSE:  1526.6 1524.8 1519.4 1509.7 1489.3   <- ~6x WORSE than chance
```

The other e2e regression tasks are fine because their offsets are tiny (ESOL μ≈−2.87, Lipo μ≈2.16):
```
e2e ESOL test: 2.02 1.92 1.77 1.33 1.16   (learns; chance 2.07)
e2e Lipo test: 1.04 1.03 1.02 0.97 0.92   (learns; chance 1.21)
```
So this is specifically the large-|mean| target. The frozen arms handle it because eval_v2's frozen
path fits a **per-fold target scaler on train and inverse-transforms predictions** (native-unit
unscale). The e2e fine-tune path appears to skip that centering, so on QM7 it optimizes toward 0.

## Fix
In the e2e regression path, standardize the target the same way eval_v2 does before training the
head, and inverse-transform predictions before scoring (or at least initialize the output bias to
the train-target mean). A working e2e QM7 should be **≤ chance (≤228)**, not 1500. Please re-run the
**QM7 e2e** fraction sweep (the other 4 e2e tasks + all frozen arms are fine — no need to redo them).

## Plot side (my side) is ready and waiting
`notebook_cells/14.py` already reads `arm` generically from LE_SUM, so when you re-emit the corrected
QM7 e2e rows into `label_efficiency_fractions_all_summary.csv` (same schema/fractions), I just repoint
LE_SUM to the combined file and B1p1 finalizes with all 5 arms. **Until QM7 is fixed I'm holding
B1p1 at the committed frozen-only version** (e2e = "NOT RUN") rather than shipping a 6x-worse-than-
chance QM7 point.

## Also: manifest is stale on the shared tree
54a8704 added 72 `figure_data/climb_v2_labeleff_v2_frac_e2e/*` raw outputs but didn't regenerate
`figure_data_manifest.json` (still 948 files; disk is 1020). `verify_notebook_sync.py` fails until
someone runs `build_data_manifest.py` and commits — I'm holding that too, since regenerating now would
bless the buggy QM7 e2e data. Will regen once QM7 is corrected so it lands in one clean commit.
