# CheMeleon evaluation suite — CLIMB replication

**Status (2026-08-14):** Tracks A (Polaris) + B (MoleculeACE) frozen **and** e2e **COMPLETE**; recipe
test complete; the CheMeleon comparator was also carried onto the **CBS** rare-actives VS benchmark
(complete) and onto the **main MoleculeNet A1 figure** (frozen + CheMeleon-foundation e2e both
**COMPLETE**, all 7 tasks × 3 seeds). All AWS boxes torn down; results durable in S3 + local. Track C
(ToxCast kNN) **not run**.
> **MoleculeNet CheMeleon e2e (A1.b, final):** ESOL 0.706 · Lipophilicity 0.598 · QM7 195.6 (RMSE↓) ·
> BBBP 0.906 · BACE 0.882 · Tox21 0.832 · HIV 0.808 (ROC↑). e2e closes the frozen arm's regression gap
> (ESOL 1.242→0.706, QM7 268.8→195.6 ≈ our skip_dense_8M); competitive-but-not-dominant on classification.
**Why this exists:** replicate the evaluation suite of Burns et al. 2025 (CheMeleon, arXiv 2506.15792;
repo `JacksonBurns/chemeleon`) and report **all CLIMB 8M models + our baselines** on it. Everything is
versioned, isolated, and reproducible; nothing here overwrites prior results — all outputs land under
NEW paths (see §6).

---

## 1. Task tracks

| Track | Source | #tasks | Eval mode | Split | Primary metric(s) | Status |
|---|---|---|---|---|---|---|
| **A. Polaris/TDC** | Polaris Hub + TDCommons | **28** (`tasks/polaris_tasks.txt`) | frozen probe + e2e | **official Polaris/TDC test split** (fixed) | task-defined (ROC-AUC/PR-AUC cls; MAE/R²/Spearman reg) | ✅ done |
| **B. MoleculeACE** | van Tilborg et al. (ChEMBL activity cliffs) | **30** (`tasks/moleculeace_tasks.txt`) | frozen probe + e2e | **van Tilborg split as-is** (series held out) | **overall / cliff / non-cliff test RMSE** | ✅ done |
| **C. ToxCast kNN** | MoleculeNet ToxCast | ~20 | kNN probe | cluster 5-fold CV | balanced acc / sens / spec | ⛔ not run (scoped as follow-on) |

Tracks A+B = **58 core tasks**. Two adjacent benchmarks reuse the same CheMeleon comparator code and
are documented here for provenance (they are NOT part of the 58):
- **CBS** — external CBS-inhibitor virtual-screening set (Truong et al. 2026), provided 5 folds, **NEF1%**.
- **MoleculeNet A1** — the paper's main 7-task figure (scaffold 5-fold CV); CheMeleon added as a comparator.

## 2. Protocol — MATCH Burns et al. (comparability is the point)

Do **NOT** use CLIMB's usual scaffold CV for tracks A/B — it would not be comparable to CheMeleon Table 1/2.
- **Splits:** each track uses its OWN fixed split (Polaris official; MoleculeACE van Tilborg).
- **Seeds:** Burns used 5 = {42,117,709,1701,9001}. **We use 3 = {42,117,709}** for BOTH frozen and e2e
  (user decision — 5 seeds on e2e is far too slow). Ours are a strict subset of Burns'; the MEANS are
  directly comparable, only the CI width differs. State this in captions.
- **Metric per task:** the task's own primary metric. MoleculeACE additionally reports **cliff vs
  non-cliff RMSE separately** + a one-sided consistency test and win-rate across the 30 tasks.
- **Regression is native-unit:** target scaler fit on train only, predictions inverse-transformed before
  scoring, so RMSE is in physical units with no cross-fold leakage.
- **OOD prediction clip:** regression predictions are bounded to the train target range ±25% (uniform
  across arms; a no-op for well-behaved features — added because an unbounded MLP over CheMeleon's
  heavier-tailed embeddings blew up on a few OOD test molecules).

## 3. Models

**Frozen-probe pipeline** = identical to CLIMB's frozen-featurizer protocol (embed → z-score → MLP head),
so our encoders and CheMeleon are apples-to-apples. Fast head: batch 512, 60 epochs, patience 8
(launch-overhead-bound otherwise; applied uniformly).

**CLIMB 8M (frozen probe):** `unsup_8M`, `skip_dense_8M`, `skip_sparse_all_8M`,
`skip_dense_plus_sparse_8M`, `skip_minimol_full_8M`, `skip_mixed_8M`, the `u2s_*_from8M` set,
`random_baseline_00` (no_pretrain). 30 frozen run-dirs (15 models × 2 tracks) — all verified.

**End-to-end fine-tuned (ModernBERT, 3 arms × 3 seeds):** `unsup_8M`, `skip_dense_8M` (best supervised —
wins CLIMB regression + CBS, ~tie on AUC), `no_pretrain_e2e` (from `random_baseline_00`). 6 run-dirs.
Recipe: `finetune_e2e_v2.finetune_predict`, lr 2e-5, 20 epochs, patience 5, batch 32, linear head.
- **Recipe test** (`summaries/recipe_test_verdict.md`): re-ran `skip_dense_8M` e2e with a stronger recipe
  (lr 5e-5, 40 epochs, patience 8) on the 4 largest tasks of each track (`*_e2e_tuned`). Verdict: tuning
  helps ~0.025–0.036 on regression, neutral on ROC, and **still loses to XGBoost(fp+desc) on all 8** — the
  "small-data" explanation for the e2e arm is refuted.

**External / baselines:**
- **CheMeleon frozen** — our run, `chemeleon_suite_run.py --featurizer chemeleon` (frozen fingerprint +
  our MLP head). Present on both tracks.
- **CheMeleon e2e on A/B** — the **published** Burns numbers from `reference/…/CheMeleon.md` (we did NOT
  re-run native chemprop for Polaris/MoleculeACE — the published means are the fair comparison there).
- **Native chemprop e2e** (`--from-foundation CHEMELEON`) — run by us **only** on CBS + MoleculeNet, where
  no published numbers exist (see §3a).
- **Classical:** `ecfp4` (Morgan+XGB), `fp_desc` (Morgan+217 RDKit descriptors+XGB).
- **Reference (published, NOT re-run):** `reference/{polaris,moleculeace}/*.md` → `reference_long.csv`.

### 3a. Native chemprop e2e comparators (CBS + MoleculeNet only)

Run natively with `chemprop 2.3.1` in the py3.12 `~/venvs/chemeleon` venv. Two arms:
- `chemprop_e2e` — vanilla D-MPNN from scratch (no foundation).
- `chemeleon_e2e` — D-MPNN initialised from the **CheMeleon foundation** (`--from-foundation CHEMELEON`),
  fine-tuned end-to-end. This IS the CheMeleon model, e2e.

Both reuse `eval_v2`'s dataset loader, scaffold-fold generator (seed 0), and `heads_v2.compute_metric` /
`compute_nef`, so numbers are 1:1 with our arms. Per fold we train 3 chemprop seeds {0,1,2} and average
the test predictions (mirrors eval_v2's head-seed averaging); the error bar is the spread across folds.
- **CBS:** provided 5 folds (fold col 1..5), NEF1% headline; `--epochs 40 --split-sizes 0.9 0.1 0.0
  --class-balance`. Scripts: `scripts/cbs_chemprop_e2e.py`.
- **MoleculeNet:** scaffold 5-fold CV (seed 0), ROC-AUC/RMSE primary + NEF1%; `--epochs 50 --patience 15`.
  Script: `scripts/molnet_chemprop_e2e.py` (CheMeleon-foundation arm only, per user, 3 seeds).

## 4. Data-leakage gate (run before reporting)

Canonical-key intersection of the 12M PubChem pretraining corpus vs the UNION of all track-A+B **test**
compounds → `leakage/pretrain_vs_testsets.json` (+ `leaked_pairs.csv`).
**RESULT:** 22 / 16,705 unique test compounds (0.132%) are in the corpus — all in 5 TDCommons tasks
(ames, bbb-martins, ld50-zhu, pgp-broccatelli, vdss-lombardo); MoleculeACE + all other Polaris tasks are
clean (0). The summary step excludes those 22 (raw predictions keep all rows so scores stay reproducible).

## 5. Data acquisition

- **Polaris:** `pip install polaris-lib` (py≥3.10). **NO LOGIN NEEDED** — public benchmarks load without
  auth. `scripts/chemeleon_suite_fetch_polaris.py` pulls all 28 (`.venv_polaris`, py3.12, polaris-lib 0.13).
  Test labels are hidden by design → predict, then score via `benchmark.evaluate()`
  (`scripts/chemeleon_suite_score_polaris.py`, run in `.venv_polaris`).
- **MoleculeACE:** 30 targets vendored to `data/moleculeace/CHEMBL*.csv` (van Tilborg split as-is).
- **CBS:** `data/cbs.csv` (external, provided fold column). **MoleculeNet:** deepchem loaders (see §7 env).

## 6. Outputs (isolated; NOTHING overwritten)

```
chemeleon_suite/           METHODOLOGY.md, HARNESS.md, tasks/, reference/, leakage/, summaries/
figure_data/chemeleon_suite/<track>/<model>[_e2e|_e2e_tuned]/results.csv|test_predictions.csv|verified.json
figure_data/cbs_benchmark/<arm>/moleculenet_cv/suite_summary.json     # CBS arms incl chemeleon/chemprop
figure_data/climb_v2_phase2/chemeleon_{frozen,e2e}/moleculenet_cv/    # MoleculeNet CheMeleon arms
s3://climb-s3-bucket/experiments/{chemeleon_suite,cbs_benchmark,climb_v2_phase2}/...   # durable backup
chemeleon_suite/summaries/*.csv + recipe_test_verdict.md              # tidy tables + the e2e verdict
experiment_cbs/cbs_nef1_summary.csv                                    # CBS figure input (build_cbs_summary.py)
```
Completion is provable per run: `verified.json` is written only when all tasks×seeds produced metrics.
Skip logic reads that / achieved work, never file existence.

## 7. Reproduce

```bash
# --- Tracks A/B: frozen battery (12 CLIMB 8M encoders + ecfp4/fp_desc/chemeleon_frozen, both tracks) ---
python scripts/chemeleon_suite_frozen_all.py            # driver; --arms/--tracks partition for parallel boxes
# per-arm form:
python scripts/chemeleon_suite_run.py --track moleculeace --featurizer chemeleon --model chemeleon_frozen --head mlp
python scripts/chemeleon_suite_run.py --track polaris --featurizer encoder --model unsup_8M \
    --encoder figure_data/climb_v2_phase2/unsup_8M/encoder --tokenizer figure_data/_tokenizer --head mlp
# --- Tracks A/B: e2e (ModernBERT), 3 arms × 3 seeds ---
python scripts/chemeleon_suite_e2e_all.py               # unsup_8M / skip_dense_8M / no_pretrain_e2e
# --- Polaris scoring (hidden test labels) ---
.venv_polaris/bin/python scripts/chemeleon_suite_score_polaris.py figure_data/chemeleon_suite/polaris/<model>
# --- CBS + MoleculeNet CheMeleon comparators (chemeleon venv; see env note) ---
~/venvs/chemeleon/bin/python scripts/cbs_chemprop_e2e.py        # CBS: chemprop_e2e + chemeleon_e2e, 3 seeds
~/venvs/chemeleon/bin/python scripts/molnet_chemprop_e2e.py     # MoleculeNet: chemeleon_e2e, 3 seeds
~/venvs/chemeleon/bin/python scripts/chemeleon_bench.py         # CheMeleon FROZEN on the 7 MoleculeNet tasks
# --- summaries ---
python scripts/chemeleon_suite_summary.py               # tracks A/B (win-rate + cliff tables)
python scripts/build_cbs_summary.py                     # CBS NEF1% per arm
python scripts/chemeleon_suite_plots.py                 # cross-task mean-rank forest plots (A/B)
```

### Environment (chemprop / CheMeleon path)
`~/venvs/chemeleon` (py3.12, chemprop 2.3.1, deepchem 2.5.0). The MoleculeNet load path needs deepchem,
whose TF import drags in a numpy-2 stack that (a) breaks deepchem and (b) makes its RawFeaturizer reject
RDKit-unparseable molecules (hypervalent atoms in a few rows). **`scripts/molnet_box_bootstrap.sh`** pins
a coherent numpy<2 stack (`tensorflow-cpu==2.16.2` / numpy 1.26.4 / scipy 1.13.1) and patches deepchem's
one ragged-array line to fall back to an object array (pre-numpy-1.24 behavior; its `valid_inds` then
drops the failed molecules, e.g. BBBP 2050→2039). CBS does NOT need this (custom-CSV path, no deepchem).
`heads_v2` falls back to a local `set_seed` when transformers is absent (chemprop venv has none).

## 8. Reproducibility of the trained chemprop models

We do not persist a `.pt` for every fold by default (105 MoleculeNet + 30 CBS checkpoints), but the models
are **deterministically regenerable** from the pinned recipe, which each run records in
`suite_summary.json["_recipe"]`:
- **Foundation:** CheMeleon `chemeleon_mp.pt`, MD5 `6a80b54fdb7de37ef0374d302f01e8ce`, from Zenodo record
  15460715 (auto-downloaded to `~/.chemprop/`; cite arXiv 2506.15792). This is the only external weight.
- **Recipe:** chemprop 2.3.1, `--pytorch-seed`/`--data-seed` ∈ {0,1,2}, split-sizes 0.9/0.1/0.0,
  `--class-balance` (cls), CBS `--epochs 40`, MoleculeNet `--epochs 50 --patience 15`.
- **Folds:** CBS = provided fold column; MoleculeNet = `eval_v2._scaffold_kfold_indices(seed=0)` (exported
  deterministically). Per-fold test predictions are saved (CBS `per_fold.csv`; A/B `test_predictions.csv`).
- **To keep the actual weights**, run either chemprop runner with `SAVE_MODELS=1` → each fold's `best.pt`
  lands in `<run>/models/`. (Reproduction is seed-deterministic modulo GPU/cuDNN nondeterminism.)

## 9. Provenance / decisions log

- 2026-08-13: task lists from `JacksonBurns/chemeleon@main` (Polaris=28, MoleculeACE=30); reference tables
  copied to `reference/`. Seeds set to **3** {42,117,709} for frozen + e2e (5 too slow on e2e).
- 2026-08-13: Polaris needs **NO login** (public); built `.venv_polaris` (polaris-lib 0.13). Leakage gate
  run. Frozen (30/30) + e2e (6/6) batteries run on AWS g4dn boxes (T4).
- 2026-08-14: added native chemprop e2e (`--from-foundation CHEMELEON`) for **CBS** and **MoleculeNet**;
  recipe test complete. Fixed `chemeleon_frozen` OOD divergence via the ±25% prediction clip. Resolved the
  deepchem/numpy env issue on py3.12 (bootstrap script + one-line deepchem patch). **MoleculeNet e2e
  completed (all 7 × 3 seeds); all boxes terminated, results in S3 + local.**
- RESOLVED (were open): Polaris login (not needed); box torch/CUDA (both venvs verified). **STILL OPEN:**
  ToxCast track C not enumerated/run.
