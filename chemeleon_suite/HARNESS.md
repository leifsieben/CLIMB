# CheMeleon suite — harness (implementation & build log)

Engineering companion to [`METHODOLOGY.md`](METHODOLOGY.md) (the scientific protocol). Documents what is
built, how to run it, what is validated. Everything is isolated under `chemeleon_suite/` +
`figure_data/{chemeleon_suite,cbs_benchmark,climb_v2_phase2/chemeleon_*}` + new S3/HF prefixes — **no
prior results are overwritten**.

## Components (all under `scripts/`)

| Script | Purpose | Status |
|---|---|---|
| `chemeleon_suite_reference.py` | Parse 14 baseline `.md` tables → `reference/reference_long.csv` | ✅ run (11,600 rows) |
| `chemeleon_suite_fetch_polaris.py` | Download 28 Polaris/TDC tasks → `data/polaris/*.csv` + manifest | ✅ run — **NO login** (polaris-lib 0.13, `.venv_polaris`) |
| `chemeleon_suite_run.py` | FROZEN-probe runner: fixed split; MoleculeACE scored locally (+cliff/non-cliff), Polaris emits per-seed `test_predictions.csv`. Fast head (batch 512), OOD prediction clip | ✅ run — 30/30 frozen dirs |
| `chemeleon_suite_e2e.py` | ModernBERT e2e fine-tune per fixed split (CLIMB arms). `--lr/--epochs/--patience/--only_tasks/--suffix` for the recipe test | ✅ run — 6/6 e2e dirs + `*_e2e_tuned` |
| `chemeleon_suite_frozen_all.py` / `_e2e_all.py` | Battery drivers with `--arms`/`--tracks` partition for multi-box parallelism | ✅ run |
| `chemeleon_suite_score_polaris.py` | Score Polaris via official `benchmark.evaluate()` (hidden labels) → `polaris_scores.csv`; `.venv_polaris` | ✅ run (28/28) |
| `chemeleon_suite_leakage.py` | canonical-key leakage gate | ✅ run (22/16705, 5 TDC tasks) |
| `chemeleon_suite_summary.py` | Aggregate our results + reference → win-rate + cliff tables | ✅ built + run |
| `chemeleon_suite_plots.py` | Cross-task mean-rank forest plots (4 panels, bootstrap 95% CI over tasks) | ✅ built + run |
| `cbs_chemprop_e2e.py` | **CBS** native chemprop e2e (vanilla + `--from-foundation CHEMELEON`), provided folds, NEF1% | ✅ run (6 run-dirs) |
| `molnet_chemprop_e2e.py` | **MoleculeNet** CheMeleon-foundation e2e, scaffold-5fold, reuses eval_v2 loader/folds/metrics | 🔄 running (ESOL done, HIV last) |
| `chemeleon_bench.py` | CheMeleon **frozen** fingerprint probe on the 7 MoleculeNet tasks (+ CBS, skipped when handled elsewhere) | ✅ run (MoleculeNet 7/7) |
| `build_cbs_summary.py` | Aggregate the CBS battery → `experiment_cbs/cbs_nef1_summary.csv` (incl chemeleon/chemprop arms) | ✅ built + run |
| `molnet_box_bootstrap.sh` | Reproducible py3.12 chemprop-venv fix (numpy<2 stack + deepchem ragged patch) | ✅ built |
| `chemeleon_suite_toxcast_knn.py` | Track C: embed → kNN classify | ⛔ not built (follow-on) |

## Data layout (self-contained, versioned)

```
chemeleon_suite/  METHODOLOGY.md HARNESS.md  tasks/{polaris(28),moleculeace(30)}.txt
  data/moleculeace/CHEMBL*.csv (30)   data/polaris/*.csv (28) + polaris_manifest.json
  reference/{polaris,moleculeace}/*.md (14 baselines) + reference_long.csv
  leakage/pretrain_vs_testsets.json   summaries/*.csv + recipe_test_verdict.md
figure_data/chemeleon_suite/<track>/<model>[_e2e|_e2e_tuned]/{results,test_predictions}.csv + verified.json
figure_data/cbs_benchmark/<arm>/moleculenet_cv/suite_summary.json    # CBS (data/cbs.csv is on the box only)
figure_data/climb_v2_phase2/chemeleon_{frozen,e2e}/moleculenet_cv/    # MoleculeNet A1 CheMeleon arms
```

## Validation

- Frozen runner smoke (`ecfp4 × MoleculeACE`): mean overall RMSE 0.758 vs ref RF_Morgan 0.694 (expected
  small gap), **cliff RMSE > non-cliff in 26/30** → activity-cliff signal captured.
- `chemeleon_frozen` OOD fix: unbounded MLP over CheMeleon embeddings diverged (RMSE 1.843, preds to 344);
  the ±25% train-range clip fixed it (RMSE 0.826, sane preds) and is a no-op for well-behaved arms.
- Native chemprop path smoke-tested end-to-end on CPU (train→best.pt→predict→NEF1%) before the batteries.
- MoleculeNet load fix smoke-verified: BBBP loads (2050→2039, 11 unparseable dropped by deepchem's
  valid_inds), CheMeleon features all-finite, scaffold folds generate.

## Leakage decision

`leakage/pretrain_vs_testsets.json`: **22 / 16,705 (0.132%)** test compounds in the 12M corpus — all in 5
TDCommons tasks (ames, bbb-martins, ld50-zhu, pgp-broccatelli, vdss-lombardo); MoleculeACE + other Polaris
clean. Summary step **excludes those 22** (raw predictions keep all rows; `leakage/leaked_pairs.csv` is the
exclusion list).

## Known env gotchas (chemprop venv, py3.12)

1. **deepchem needs numpy<2 AND a ragged-array patch.** `tensorflow-cpu` (deepchem's TF import) pulls a
   numpy-2 stack that breaks deepchem, and deepchem's `RawFeaturizer` does `np.asarray(features)` on a list
   where failed molecules are empty arrays → strict-numpy `ValueError`. Fix = `molnet_box_bootstrap.sh`
   (pin `tensorflow-cpu==2.16.2`/numpy 1.26.4/scipy 1.13.1 + patch `deepchem/feat/base_classes.py` line
   ~289 to fall back to `dtype=object`). **CBS is immune** (custom-CSV path, no deepchem) — which is why CBS
   frozen worked but MoleculeNet frozen first crashed.
2. **transformers absent in the chemprop venv** — `heads_v2._TorchHead.fit` now falls back to a local
   `set_seed`, so the frozen CheMeleon probe needs no transformers install.
3. `pkill -f <script>` self-matches the ssh command running it; kill by exact `ps` match or PGID.

## Build log

- 2026-08-13: scaffold, task lists (28/30), reference (14 models), MoleculeACE vendored; frozen runner +
  reference parser + leakage + Polaris fetch/score built & validated. Data acquisition complete.
- 2026-08-13: **frozen battery (30/30) + ModernBERT e2e battery (6/6)** run on AWS g4dn boxes (T4), 3 seeds,
  synced to S3. Fast frozen head; per-arm parallel drivers.
- 2026-08-14: **recipe test** (`skip_dense_8M_e2e_tuned`, 4 largest tasks/track) → verdict in
  `summaries/recipe_test_verdict.md` (still loses to XGBoost(fp+desc)). **CBS** CheMeleon arms
  (`chemprop_e2e` 0.462, `chemeleon_e2e` 0.784±0.009, `chemeleon_frozen` 0.788) — XGBoost(fp+desc) 0.930
  leads; CheMeleon mid-pack. **MoleculeNet** CheMeleon frozen done (mixed: worse on regression, on-par on
  classification); CheMeleon-foundation e2e running on dedicated box `i-01ec1c…` (ESOL 0.706, HIV last).
  Runners now record `_recipe` + support `SAVE_MODELS=1`.
- OPEN: ToxCast track C (kNN) not built/run.
