# CheMeleon evaluation suite — CLIMB replication

**Status:** scaffolding + task lists locked (2026-08-13). GPU runs DEFERRED (account capacity constrained).
**Why this exists:** replicate the evaluation suite of Burns et al. 2025 (CheMeleon, arXiv 2506.15792;
repo `JacksonBurns/chemeleon`) and report **all CLIMB 8M models + our baselines** on it. These datasets
are considered more trustworthy than MoleculeNet; **this suite may become the paper's headline benchmark**,
so everything here is versioned, isolated, and reproducible. Nothing in this directory overwrites prior
results — all outputs land under NEW paths (see "Outputs" below).

---

## 1. Task tracks

| Track | Source | #tasks | Eval mode | Split | Primary metric(s) |
|---|---|---|---|---|---|
| **A. Polaris/TDC** | Polaris Hub + TDCommons | **28** (`tasks/polaris_tasks.txt`) | frozen probe / e2e | **official Polaris/TDC test split** (fixed) | task-defined (ROC-AUC/PR-AUC for cls; MAE/R²/Spearman for reg) |
| **B. MoleculeACE** | van Tilborg et al. (ChEMBL activity cliffs) | **30** (`tasks/moleculeace_tasks.txt`) | frozen probe / e2e | **van Tilborg split as-is** (series held out) | **overall / cliff / non-cliff test RMSE** |
| **C. ToxCast kNN** | MoleculeNet ToxCast (`dc.molnet.load_toxcast`) | ~20 (TODO enumerate) | **kNN probe** (embed → kNN, no head) | random + agglomerative-cluster 5-fold CV | balanced acc, sensitivity, specificity |

Tracks A+B = **58 tasks** (the core; matches the user's brief). Track C is a distinct non-parametric
probe and is scoped as a follow-on.

## 2. Protocol — MATCH Burns et al. (comparability is the point)

Do **NOT** use CLIMB's usual 5-fold scaffold CV here — it would not be comparable to CheMeleon Table 1/2.
- **Splits:** each track uses its OWN fixed split (Polaris official; MoleculeACE van Tilborg; ToxCast CV).
- **Seeds:** Burns used 5 seeds = {42, 117, 709, 1701, 9001}. **We use 3 seeds = {42, 117, 709}** for
  BOTH frozen and e2e (user decision 2026-08-13 — 5 seeds on the e2e arms takes far too long). Ours are a
  strict subset of Burns' seeds. Comparability caveat for captions: our error bars come from 3 seeds; the
  reference published means used 5 — the MEANS remain directly comparable, only the CI width differs.
- **Metric per task:** the task's own primary metric (NOT free choice). MoleculeACE additionally reports
  **cliff vs non-cliff RMSE separately** (the number the CliffPFN thesis needs) + one-sided t-test
  (RMSE_cliff − RMSE_noncliff > 0) "consistency rate", and win-count/win-rate across the 30 tasks.
- **Significance:** Tukey HSD (α=0.05) across models per task (as Burns did) for win/loss designation.

## 3. Models

**Head/probe pipeline** = identical to CLIMB frozen-featurizer protocol (embed → z-score → MLP head),
so our encoders and CheMeleon are apples-to-apples.

**CLIMB 8M (frozen probe, all):** `unsup_8M`, `skip_dense_8M`, `skip_sparse_all_8M`,
`skip_dense_plus_sparse_8M`, `skip_minimol_full_8M`, `skip_mixed_8M`, the 6 `u2s_*_from8M`,
`random_baseline_0{0,1,2}` (no_pretrain).

**End-to-end fine-tuned (3 models × 3 seeds):** `unsup_8M` (unsup-only), **`skip_dense_8M` (best
supervised — see note), `e2e_random` (no_pretrain_e2e).
> *Best-supervised pick:* from CLIMB A1b 5-fold CV, `skip_dense_8M` wins both regression tasks + CBS and
> is within 0.004 AUC of the top; `skip_dense_plus_sparse_8M` has the marginal AUC edge. Chose
> `skip_dense_8M` as the canonical dense-MTR representative.

**External / baselines:**
- CheMeleon (frozen fingerprint + native chemprop e2e) — our re-run, `--featurizer chemeleon`.
- Classical: `ecfp4` (Morgan+XGB), `fp_desc` (Morgan+desc+XGB).
- **Reference (published, NOT re-run):** parsed from `reference/{polaris,moleculeace}/*.md` — CheMeleon,
  Chemprop(_Mordred), MoLFormer, MolCLR, minimol, fastprop, MLP_PLR_Pretrained, PCA_MLP(_Prefitted),
  RF, RF_Mordred, RF_Morgan, RF_Morgan_Physchem. Pulling their numbers avoids RF-implementation variance.

## 4. Data-leakage gate (MUST run before reporting)

MoleculeACE/Polaris test compounds are ChEMBL/assay molecules; our Phase-1 SSL corpus is a PubChem draw.
Compute canonical-key intersection between the pretraining corpus and the UNION of all track-A+B **test**
compounds. Output: `chemeleon_suite/leakage/pretrain_vs_testsets.json` (+ `leaked_pairs.csv`).

**RESULT (2026-08-13):** 22 / 16,705 unique test compounds (0.132%) are in the 12M corpus — all in 5
TDCommons tasks (ames, bbb-martins, ld50-zhu, pgp-broccatelli, vdss-lombardo); MoleculeACE + all other
Polaris tasks are clean (0). **Decision:** the summary step excludes those 22 leaked test molecules from the
affected tasks (raw predictions retain all rows; report flagged + filtered).

## 5. Data acquisition

- **Polaris:** `pip install polaris-lib` (py≥3.10). **NO LOGIN NEEDED** (verified 2026-08-13: public
  benchmarks load without auth). `scripts/chemeleon_suite_fetch_polaris.py` pulls all 28. Test labels are
  hidden by design → score via `benchmark.evaluate()` (`scripts/chemeleon_suite_score_polaris.py`).
- **MoleculeACE:** `pip install MoleculeACE` + `git clone molML/MoleculeACE`; use pre-assigned split.
- **ToxCast:** `dc.molnet.load_toxcast()`.

## 6. Outputs (isolated; NOTHING overwritten)

```
chemeleon_suite/                      # code, task lists, methodology, reference numbers (in git)
  METHODOLOGY.md, tasks/, reference/, leakage/
figure_data/chemeleon_suite/<track>/<model>/<seed>/...     # raw per-run predictions + metrics (NEW path)
s3://climb-s3-bucket/experiments/chemeleon_suite/...        # durable backup (NEW prefix)
HF lsieben/climb-results/chemeleon_suite/...                # published (NEW folder)
chemeleon_suite/summaries/*.csv       # tidy per-(track,task,model,metric) mean±std + win/HSD tables
```

Completion is provable per (model, track): a `verified.json` written only when all tasks×seeds produced
metrics. Skip logic reads that, never file existence.

## 7. Reproduce

```bash
# 1. task lists are fixed in tasks/*.txt (do not regenerate silently)
# 2. acquire data (§5); run leakage gate (§4)
# 3. frozen probes (all CLIMB 8M + CheMeleon + classical), 5 seeds:
python scripts/chemeleon_suite_run.py --track polaris    --mode frozen   # TODO
python scripts/chemeleon_suite_run.py --track moleculeace --mode frozen
# 4. e2e (3 models × 3 seeds):
python scripts/chemeleon_suite_run.py --track polaris    --mode e2e
# 5. summarize + Tukey HSD + cliff/noncliff + win tables:
python scripts/build_chemeleon_suite_summary.py
```

## 8. Provenance / decisions log

- 2026-08-13: task lists extracted from `JacksonBurns/chemeleon@main` `analysis/{polaris,moleculeace}_results/CheMeleon.md`.
  Polaris=28, MoleculeACE=30. Seeds {42,117,709,1701,9001}. Reference baseline tables copied to `reference/`.
- Compute: g5/g6 capacity-blocked in us-east-1f; box is g4dn.2xlarge (T4). chemprop venv = py3.12
  `~/venvs/chemeleon` (chemprop 2.3.1). GPU runs deferred pending user capacity + Polaris auth.
- 2026-08-13: user set seeds = **3** {42,117,709} for both frozen and e2e (5 too slow on e2e).
- OPEN: (a) Polaris interactive login; (b) ToxCast 20-endpoint enumeration; (c) verify torch/CUDA on box
  for chemprop e2e (pip pulled torch 2.13/cu13 — may mismatch driver).
