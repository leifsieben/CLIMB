# CheMeleon suite — harness (implementation & build log)

Engineering companion to `METHODOLOGY.md` (the scientific protocol). Documents what is built, how to run
it, what is validated, and what remains. Everything is isolated under `chemeleon_suite/` +
`figure_data/chemeleon_suite/` + a new S3/HF prefix — **no prior results are overwritten**.

## Components (all under `scripts/`)

| Script | Purpose | Status |
|---|---|---|
| `chemeleon_suite_reference.py` | Parse the 14 baseline `.md` tables → `chemeleon_suite/reference/reference_long.csv` (track,model,task,seed,metric,value) | ✅ built + run (11,600 rows) |
| `chemeleon_suite_fetch_polaris.py` | Download the 28 Polaris/TDC tasks → `chemeleon_suite/data/polaris/*.csv` + manifest (target col, primary metric, type) | ✅ built + **run — NO LOGIN NEEDED** (polaris-lib 0.13, py3.12 `.venv_polaris`). 28/28 fetched. Test labels hidden by design → test CSV holds inputs only |
| `chemeleon_suite_run.py` | FROZEN-probe runner: fixed split, our featurize→standardize→head; MoleculeACE scored locally (+cliff/non-cliff RMSE), Polaris emits per-seed `test_predictions.csv` | ✅ built + **validated** both tracks (ecfp4×MoleculeACE and ×Polaris) |
| `chemeleon_suite_score_polaris.py` | Score Polaris predictions via official `benchmark.evaluate()` (hidden labels) → `polaris_scores.csv`. Runs in `.venv_polaris` | ✅ built + **validated** (28/28 tasks; ordering ecfp4 ≤ RF_Morgan ≤ CheMeleon confirmed) |
| `chemeleon_suite_leakage.py` | canonical-key leakage gate: pretraining corpus vs suite test sets | ✅ built + **run**: MoleculeACE 0/9130 clean; incl Polaris **22/16705 = 0.132%** (5 TDC tasks) |
| `chemeleon_suite_e2e.py` (chemprop) | End-to-end fine-tune runner (ModernBERT for CLIMB arms; chemprop `--from-foundation CheMeleon`) per fixed split | ⛔ TODO (GPU; build on box, verify torch/CUDA first) |
| `chemeleon_suite_toxcast_knn.py` | Track C: embed → kNN classify (no head), random + cluster 5-fold | ⛔ TODO |
| `build_chemeleon_suite_summary.py` | Aggregate our results + reference → per-(track,task,model,metric) mean±std, cliff/non-cliff, win-rate, Tukey HSD | ⛔ TODO |

## Data layout (self-contained, versioned)

```
chemeleon_suite/
  METHODOLOGY.md            # protocol (splits, seeds=3 {42,117,709}, metrics, models, leakage)
  HARNESS.md                # this file
  tasks/{polaris_tasks.txt(28), moleculeace_tasks.txt(30)}
  data/moleculeace/CHEMBL*.csv        # vendored (30), cols: smiles, y [pEC50/pKi], cliff_mol, split
  data/polaris/*.csv                  # fetched post-auth (28) + polaris_manifest.json
  reference/{polaris,moleculeace}/*.md   # 14 baselines' PUBLISHED numbers (verbatim from their repo)
  reference/reference_long.csv           # parsed tidy form
  leakage/pretrain_vs_testsets.json      # leakage gate result
figure_data/chemeleon_suite/<track>/<model>/results.csv + verified.json   # OUR raw results (NEW path)
```

## How to run (order)

```bash
# 0. reference numbers (done)
python scripts/chemeleon_suite_reference.py

# 1. data: MoleculeACE vendored already. Polaris (user, after browser login):
polaris login                                   # or: python3 -c "import polaris as po; po.login()"
python3 scripts/chemeleon_suite_fetch_polaris.py

# 2. leakage gate (must pass/report before headline):
python scripts/chemeleon_suite_leakage.py

# 3. FROZEN probes — one call per (model, track). Baselines + CLIMB frozen + CheMeleon frozen:
python scripts/chemeleon_suite_run.py --track moleculeace --featurizer ecfp4    --model ecfp4     --head mlp
python scripts/chemeleon_suite_run.py --track moleculeace --featurizer fp_desc  --model fp_desc   --head xgb
python scripts/chemeleon_suite_run.py --track moleculeace --featurizer chemeleon --model chemeleon_frozen --head mlp
python scripts/chemeleon_suite_run.py --track moleculeace --featurizer encoder  --model unsup_8M \
    --encoder figure_data/climb_v2_phase2/unsup_8M/encoder --tokenizer figure_data/_tokenizer --head mlp
#   ... repeat --track polaris and for every CLIMB 8M encoder (see METHODOLOGY §3)

# 4. e2e (TODO), toxcast kNN (TODO), summary (TODO)
```

Seeds default to `{42,117,709}` (3, per user). Encoder/chemeleon featurizers are z-scored; Morgan ones are not
(head = xgb for those, mlp for learned embeddings) — identical to the CLIMB frozen-probe protocol.

## Validation (2026-08-13, GPU-free)

- Runner smoke: `ecfp4 × MoleculeACE × linear head × 3 seeds` → all 30 tasks, exit 0.
  - Mean overall RMSE **0.758** vs reference **RF_Morgan 0.694** (ours linear vs their RF → expected small gap;
    real runs use mlp/xgb). Numbers in-range → loader, split, target col (`y [pEC50/pKi]`), metrics correct.
  - **cliff RMSE > non-cliff in 26/30 tasks** → activity-cliff signal captured; cliff subset logic correct.
- Reference parser: 14 models × (28 polaris + 30 moleculeace) × 5 seeds → 11,600 rows.

## Leakage decision

Gate result (`leakage/pretrain_vs_testsets.json`): **22 / 16,705 unique test compounds (0.132%) appear in
our 12M PubChem pretraining corpus** — all in 5 TDCommons tasks (ames, bbb-martins, ld50-zhu,
pgp-broccatelli, vdss-lombardo); MoleculeACE + all other Polaris tasks are 0. Decision: the **summary step
excludes those 22 leaked test molecules** from the affected tasks' scoring (raw predictions keep all rows so
scores stay reproducible; `leakage/leaked_pairs.csv` is the exclusion list). Report both flagged + filtered.

## Open items / risks

1. **e2e on box.** pip pulled `torch 2.13 / cu13` into `~/venvs/chemeleon`; the T4 driver may not support cu13.
   Verify `torch.cuda.is_available()` and pin torch to the driver before e2e (or use the py3.9 climb venv for
   ModernBERT e2e and the chemeleon venv only for CheMeleon's own chemprop e2e).
2. **Polaris scoring env split:** predictions are produced in the eval env (torch/our code); scoring runs in
   `.venv_polaris`. Two-step by design (polaris-lib needs py≥3.10; our encoder stack is py3.9). Keep aligned.
3. **ToxCast track C** endpoints not yet enumerated (~20 of 617); different eval mode (kNN).
4. **Summary/HSD tooling** not yet built — needed before any headline comparison to CheMeleon Table 1/2.

## Build log

- 2026-08-13: scaffold, task lists (28/30), reference (14 models), MoleculeACE vendored, frozen runner +
  reference parser + leakage script built & validated (frozen/ref); Polaris fetch script pending user auth.
  GPU runs (all encoders' frozen featurization, e2e, kNN) DEFERRED — user capacity-constrained. Box (g4dn/T4)
  stopped, env preserved.
