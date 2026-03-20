# CLIMB Pretraining & Evaluation Guide

This repo supports end‑to‑end tokenizer training, hyperparameter sweeps, pretraining (unsupervised, supervised, mixed), streaming large datasets, and spot‑friendly execution on AWS. Below is a practical how‑to.

## Robust Experiment Workflow (Current)

The current end-to-end path for the paper-scale experiment matrix is now:

1. Define one experiment spec in [configs/experiment_spec_example.yaml](/Users/lsieben/VSCode/CLIMB/configs/experiment_spec_example.yaml).
2. Expand it into a fully resolved manifest:
   ```bash
   python3 scripts/generate_experiment_manifest.py \
     --spec configs/experiment_spec_example.yaml \
     --output experiments/robust_matrix/manifest.json
   ```
3. Validate the selected runs before launch:
   ```bash
   python3 scripts/preflight_experiment.py \
     --manifest experiments/robust_matrix/manifest.json
   ```
4. Launch smoke runs or a main wave:
   ```bash
   python3 scripts/launch_experiment_wave.py \
     --manifest experiments/robust_matrix/manifest.json \
     --stage smoke \
     --preflight \
     --resume
   ```
   After smoke finishes, backfill ETA estimates for the rest of the matrix:
   ```bash
   python3 scripts/calibrate_manifest.py \
     --manifest experiments/robust_matrix/manifest.json
   ```
5. Monitor active nodes with the manifest-aware monitor:
   ```bash
   bash watch_pretrain.sh \
     experiments/robust_matrix/manifest.json \
     configs/cluster_config_example.json
   ```
6. Aggregate raw results after runs complete:
   ```bash
   python3 scripts/aggregate_experiment_results.py \
     --manifest experiments/robust_matrix/manifest.json \
     --output experiments/robust_matrix/aggregate/raw_results.csv
   python3 scripts/candidate_moleculenet_score.py \
     --input experiments/robust_matrix/aggregate/raw_results.csv \
     --output experiments/robust_matrix/aggregate/candidate_scores.csv
   ```

What is new in this workflow:
- `s3://` inputs are accepted directly for tokenizer paths, tokenized unsupervised shards, and supervised parquet sources.
- Remote shard access uses a small local read-through cache in `~/.cache/climb_s3` unless `CLIMB_S3_CACHE_DIR` overrides it.
- Every resolved run gets a saved config, run context, metrics log, metadata file, backup target, and MoleculeNet suite output.
- `pretrain_pipeline.py --resume` now resumes from the latest checkpoint or `spot_checkpoint` when present.
- Background S3 backup is handled by [scripts/sync_run_to_s3.py](/Users/lsieben/VSCode/CLIMB/scripts/sync_run_to_s3.py) during launches and writes `backup_status.json` for monitoring.

Manifest shape and default matrix:
- Smoke runs: `3`
- Unsupervised baseline runs: `15`
- Supervised family/order runs: `25`
- Unsupervised fixed-budget coverage runs: `5`
- Mixed fixed-budget runs: `15`

Representative files for this workflow:
- [experiment_manifest.py](/Users/lsieben/VSCode/CLIMB/experiment_manifest.py)
- [scripts/generate_experiment_manifest.py](/Users/lsieben/VSCode/CLIMB/scripts/generate_experiment_manifest.py)
- [scripts/preflight_experiment.py](/Users/lsieben/VSCode/CLIMB/scripts/preflight_experiment.py)
- [scripts/launch_experiment_wave.py](/Users/lsieben/VSCode/CLIMB/scripts/launch_experiment_wave.py)
- [scripts/run_moleculenet_suite.py](/Users/lsieben/VSCode/CLIMB/scripts/run_moleculenet_suite.py)
- [scripts/monitor_cluster.py](/Users/lsieben/VSCode/CLIMB/scripts/monitor_cluster.py)

---

## Current Data Sources, Processing, and Storage (AWS)

### A) Dataset sources and how they are used
- Unsupervised filtered PubChem source shards: `s3://climb-s3-bucket/tokenized_sources/pubchem_filtered/`.
- Unsupervised training-ready tokenized shards: `s3://climb-s3-bucket/tokenized_sources/pubchem_filtered_tokenized_pkl/`.
- Tokenizer artifact used for current runs: `s3://climb-s3-bucket/tokenizer_10M/` (contains `tokenizer.json`).
- Canonical supervised source parquet for pretraining: `s3://climb-s3-bucket/preparing_datasets/supervised_wide.parquet`.
- Canonical supervised tokenized parquet for pretraining: `s3://climb-s3-bucket/tokenized/supervised_wide_parquet/`.
- Downstream evaluation datasets: MoleculeNet tasks loaded via DeepChem in `evaluate_model.py` (e.g., BBBP, Tox21, ESOL, FreeSolv, Lipophilicity).
- Legacy multi-task examples in `configs/config_multitask_example.yaml` still reference local CSVs (`data/bbbp.csv`, `data/bace.csv`, etc.), but the current robust experiment pipeline uses the fused supervised wide parquet above.

Processing path used in this repo:
1. Build or load tokenizer (`train_tokenizer.py` or prebuilt tokenizer from S3).
2. Tokenize raw SMILES (single-file flow with `tokenize_dataset.py`, or sharded flow with `tokenizing_datasets.py`).
3. Run MLM hyperparameter search (`hyperparameter_search.py`) on tokenized shards.
4. Run pretraining (`train_model.py` / `pretrain_pipeline.py`), then evaluate (`evaluate_model.py` / `evaluate_all_moleculenet.sh`).

### B) Current S3 structure and access
Known active paths:
- `s3://climb-s3-bucket/tokenizer_10M/`
- `s3://climb-s3-bucket/tokenized_sources/pubchem_filtered/`
- `s3://climb-s3-bucket/tokenized_sources/pubchem_filtered_tokenized_pkl/`
- `s3://climb-s3-bucket/preparing_datasets/supervised_wide.parquet`
- `s3://climb-s3-bucket/tokenized/supervised_wide_parquet/`

Canonical storage inventory:
- `s3://climb-s3-bucket/tokenizer_10M/`
  - Format: tokenizer directory containing `tokenizer.json` and tokenizer metadata files.
  - Used by: unsupervised pretraining, supervised pretraining, MoleculeNet evaluation.
- `s3://climb-s3-bucket/tokenized_sources/pubchem_filtered/`
  - Format: filtered unsupervised parquet source shards.
  - Important: this is not the current training-ready unsupervised input for the smoke/main matrix.
- `s3://climb-s3-bucket/tokenized_sources/pubchem_filtered_tokenized_pkl/`
  - Format: training-ready unsupervised tokenized pickle shards.
  - Loader support: direct S3 streaming for legacy pickle shards.
  - Required content: `{"data": [{"input_ids": [...], "attention_mask": [...]}, ...]}`.
- `s3://climb-s3-bucket/preparing_datasets/supervised_wide.parquet`
  - Format: fused supervised source parquet with canonical SMILES column plus task label columns.
  - Used by: source-of-truth supervised dataset; can be re-tokenized reproducibly.
- `s3://climb-s3-bucket/tokenized/supervised_wide_parquet/`
  - Format: tokenized parquet shards with all original supervised label columns preserved and added `input_ids` + `attention_mask`.
  - Current artifact: 55 parquet shards generated on 2026-03-17 from `supervised_wide.parquet` with `tokenize_supervised_parquet.py`.
  - Used by: robust smoke runs and main supervised/mixed experiment matrix.
- `s3://climb-s3-bucket/tokenized/supervised_wide/`
  - Format: older tokenized pickle shards.
  - Important: do not use for supervised training in the robust pipeline. These shards do not contain labels.

Inspect available prefixes:
```bash
aws s3 ls s3://climb-s3-bucket/
aws s3 ls s3://climb-s3-bucket/ --recursive | grep tokenizer.json
```

Download artifacts to EC2:
```bash
mkdir -p ~/artifacts/tokenizer_10M ~/data/pubchem_filtered ~/data/pubchem_filtered_tokenized_pkl ~/data/supervised_wide_tokenized
aws s3 sync s3://climb-s3-bucket/tokenizer_10M ~/artifacts/tokenizer_10M
aws s3 sync s3://climb-s3-bucket/tokenized_sources/pubchem_filtered ~/data/pubchem_filtered
aws s3 sync s3://climb-s3-bucket/tokenized_sources/pubchem_filtered_tokenized_pkl ~/data/pubchem_filtered_tokenized_pkl
aws s3 cp s3://climb-s3-bucket/preparing_datasets/supervised_wide.parquet ~/data/supervised_wide.parquet
aws s3 sync s3://climb-s3-bucket/tokenized/supervised_wide_parquet ~/data/supervised_wide_tokenized
```

### C) README status / outdated items now corrected
- Hyperparameter sweep entrypoint is `hyperparameter_search.py` (not `run_hyperopt.py`).
- Sweep now supports both `.pkl` and `.parquet` tokenized inputs.
- Current tokenizer location is documented as `s3://climb-s3-bucket/tokenizer_10M/`.
- AWS examples now align with the currently used S3 prefixes.
- The robust experiment spec now points to `s3://climb-s3-bucket/tokenized_sources/pubchem_filtered_tokenized_pkl/` for unsupervised training-ready shards.
- The robust experiment spec now points to the labeled tokenized supervised parquet at `s3://climb-s3-bucket/tokenized/supervised_wide_parquet/`.
- The legacy unlabeled supervised pickle shards are explicitly marked as non-canonical.
- Clean AWS bootstrap now requires the pinned `requirements.txt` in this repo, including the Python-3.9-safe `urllib3==1.26.20` and no `datasets` dependency.

### D) Current CLIMB smoke worker setup
- Current CLIMB smoke pool: `3 x g5.2xlarge` in `us-east-1d`.
- Worker template: Deep Learning Base AMI with Single CUDA (Amazon Linux 2023) `2026-02-20`.
- Per-worker storage: `1000 GB gp3` root volume.
- Per-worker runtime layout:
  - repo: `/home/ec2-user/CLIMB`
  - venv: `/home/ec2-user/venvs/climb`
  - run artifacts: `/home/ec2-user/artifacts/robust_matrix`
- Cluster monitor config for the current smoke pool:
  - `configs/cluster_config_smoke_20260317.json`
- Recommended G5 choice for this project:
- `g5.xlarge` is generally too CPU/RAM constrained for S3 streaming plus evaluation.
- `g5.2xlarge` is the default sweet spot for one-A10G runs.
- `g5.4xlarge` is useful when the extra CPU/RAM is actually needed, but it does not buy an additional GPU.
- Mixed token-budget runs are configured with `total_epochs: 1`, but the pipeline now explicitly enables both phases when `0 < supervised_fraction < 1`; the token budget, not the epoch count, is the real limiter there.
- RoBERTa-style encoder configs should use `max_position_embeddings: 514` when training on 512-token sequences.

---

## 1) Train a tokenizer (BPE) and store/reuse it
1. Prepare raw SMILES text (one per line) or a CSV with a `SMILES` column.
2. Train the tokenizer (example command; adjust paths):
   ```bash
   python train_tokenizer.py \
     --input data/unsup_smiles.smi \
     --output tokenizer \
     --vocab_size 1000 \
     --min_frequency 2
   ```
   (If you use the notebook `playing_around_with_the_tokenizer.ipynb`, point it to your SMILES file and export to a directory with `tokenizer.json`.)
3. Reuse it by pointing training/eval scripts to the tokenizer directory (the folder that contains `tokenizer.json`):
   - `train_model.py --tokenizer tokenizer ...`
   - `pretrain_pipeline.py --config ...` (config references `tokenizer_path`)
   - `evaluate_model.py --tokenizer tokenizer ...`

---

## 2) Hyperparameter sweep (MLM)
Goal: minimize MLM eval loss on a held‑out split, then fix the best hyperparameters for future runs.
1. Prepare tokenized unsupervised data (`.pkl` or `.parquet`) with `input_ids` + `attention_mask`:
   ```bash
   python tokenize_dataset.py \
     --tokenizer tokenizer \
     --input data/unsup.smi \
     --output data/unsup.pkl
   ```
2. Run the sweep:
   ```bash
   python hyperparameter_search.py \
     --config experiments/config.yaml \
     --tokenizer ~/artifacts/tokenizer_10M \
     --train_data ~/data/pubchem_filtered \
     --output hp_search_results \
     --n_trials 20 \
     --study_name mlm_hpo_wide \
     --load_if_exists \
     --pruner hyperband \
     --dataloader_num_workers 4 \
     --log_file hp_search_results/logs/worker0.log
   ```
   - The sweep searches lr, batch_size, warmup_steps, hidden size/layers/heads, weight_decay.
   - Optimizes `eval_loss` on MLM.
   - Uses persistent Optuna storage at `hp_search_results/optuna_study.db` by default.
   - Best params are snapshotted during search to `hp_search_results/best_hyperparameters.json`.
   - Study state snapshots are written to `hp_search_results/study_snapshot.json`.
3. Multi-GPU parallel workers (single host with shared storage):
   ```bash
   scripts/run_hpo_workers.sh \
     0,1,2,3 \
     configs/config_hyperopt_unsup.yaml \
     ~/artifacts/tokenizer_10M \
     ~/data/pubchem_filtered \
     hp_search_results \
     40 \
     --study_name mlm_hpo_wide \
     --load_if_exists \
     --pruner hyperband \
     --bf16
   ```
4. Optional periodic backup to S3:
   ```bash
   scripts/backup_hpo_to_s3.sh hp_search_results s3://climb-s3-bucket/hpo_backups/mlm_hpo_wide
   ```
3. Use the best params in future configs/runs (copy them into your model/training config).

---

## 3) Streaming & spot instances on AWS
### Streaming
Use directories of tokenized shards (`*.pkl`) instead of one huge pickle:
- `train_model.py` will stream if `--unsup_data` or `--sup_data` points to a directory:
  ```bash
  python train_model.py \
    --config configs/config_prototyping_50_50.yaml \
    --tokenizer tokenizer \
    --unsup_data /mnt/data/unsup_shards \
    --sup_data /mnt/data/sup_shards \
    --streaming_max_samples 2000000 \
    --unsup_weight 0.5 \
    --sup_weight 0.5 \
    --task mlm \
    --output experiments/streaming_run
  ```
- When streaming, set `training.max_steps` (in the config) to control total compute rather than dataset length.

### Spot instances
- Add `--spot` to `train_model.py` / `pretrain_pipeline.py` / multi‑task training to register a SIGTERM handler that checkpoints to `<output>/spot_checkpoint/` before exit.
- Resume from the saved checkpoint by passing it as the pretrained model path.

### Performance notes for supervised streaming
The supervised family streaming pipeline is CPU‑bound (parquet scan + tokenization + label masking). To keep the GPU fed:
- Increase `supervised_training.batch_size` (try 64 or 128 on A10G).
- Increase `supervised_training.dataloader_num_workers` (8–16 on g5.4xlarge).
- Increase `supervised_training.streaming_batch_rows` (e.g., 16384 or 32768).

If throughput is still low, the next step is pre‑tokenizing supervised SMILES (store `input_ids` + `attention_mask` alongside labels) to remove per‑step tokenization.

### Token budget guidance
For ramp experiments, 50B tokens is not required. Start with **10B total tokens** and run ramps at:
- 10/25/50/75/100% of 10B → 1B, 2.5B, 5B, 7.5B, 10B
Scale up only if the loss‑vs‑tokens curve is still improving.

**Comparability note:** the ramp experiments are **token‑budgeted**, not dataset‑fraction‑limited.
Each run stops after a target number of tokens are seen, even though the full dataset is streamed.

**Planned Stage 2:** run a second series where we use random **data‑subset fractions** (10/25/50/75/100%) **while keeping the same compute budget**. This isolates data‑coverage effects from compute‑budget effects.

#### Decision: 1B token budget for coverage and mixed experiments (worker2)

**Decided 2026-03-19.** All 20 runs on worker2 (5 unsupervised coverage ablations + 15 mixed MLM/supervised ratio runs) use a **1B token budget** instead of the originally planned 10B. Worker0's unsupervised ramp-up (1M → 10B, 15 runs) is **kept at 10B** as a scaling control.

**Rationale:**

1. **Wall-clock time:** 20 × 10B runs on a single g5.2xlarge ≈ 62 days. 20 × 1B ≈ 6–7 days, fitting within the target 7–10 day window.

2. **Literature comparison:** 1B tokens is well within the range used by published chemical language models at similar or larger scales:
   - ChemBERTa (2020): ~300M–2B tokens, 125M params
   - MolBERT: ~80M tokens, 85M params
   - smi-ted (2024): ~3B tokens, 110M params
   - ChemBERTa-2 (2022): up to 15B tokens, 85M params
   At 13M params, 1B tokens gives ~77 tokens/param — well into the overtraining regime for BERT-style models. This is desirable for representation quality.

3. **Scale defence for 13M params:** The 13M parameter count (hidden_size=256, num_layers=10, num_heads=8, intermediate=1024) is intentional:
   - Enables a 60+ experiment ablation matrix on a 3-node spot cluster within the project timeline.
   - Correctly sized relative to supervised dataset scale (~1.5M unique molecules, ~3,288 tasks).
   - SMILES sequences are short (30–50 tokens median), so the encoder does not need large capacity to capture token-level dependencies.
   - Overtrained at 1B tokens — but overtraining a small model produces better representations for fine-tuning than undertraining a large model with the same compute budget.
   - Generalisation of the findings to larger models is a stated limitation; the paper argues the relative ordering of pretraining strategies is architecture-agnostic.

4. **Scaling control preserved:** worker0 runs 1M → 10B token ramp-up experiments. These provide direct evidence of whether increasing token budget beyond 1B produces continued downstream improvement, supporting (or qualifying) the 1B choice.

**Manifest impact:** `manifest.json` on worker2 was patched in-place; `total_tokens` changed from `10000000000` → `1000000000` for all 20 `unsupervised_fixed_budget` and `mixed_fixed_budget` runs. The per-run config YAMLs were also updated before restart.

---

## 4) Unsupervised pre-train run (MLM)
Use `pretrain_pipeline.py` or `train_model.py`.
### Using `pretrain_pipeline.py` (recommended for budget control)
1. Create a config (see `configs/pretraining_pipeline_example.yaml`). Key fields:
   - `tokenizer_path`: directory with `tokenizer.json`.
   - `compute_budget.total_epochs`: total epochs across phases.
   - `compute_budget.supervised_fraction`: set to `0.0` for pure MLM.
   - `unsupervised_data`: list of pickles or directories of pickled shards (must be pre‑tokenized).
   - `mlm_training`: batch_size, lr, mlm_probability, etc.
2. Run:
   ```bash
   python pretrain_pipeline.py \
     --config configs/pretraining_pipeline_example.yaml \
     --log_file logs/pretrain.log \
     --spot   # optional, for spot instances
   ```
   - Data must be tokenized already.
   - Compute budget is set in `compute_budget`; `mlm_epochs` is derived from it.

### Using `train_model.py` directly
```bash
python train_model.py \
  --config experiments/config.yaml \
  --tokenizer tokenizer \
  --unsup_data data/unsup.pkl \
  --unsup_weight 1.0 \
  --sup_weight 0.0 \
  --task mlm \
  --output experiments/model_unsup_only \
  --spot
```

---

## 5) Supervised pre-train run (multi-task)
Use `train_multitask.py` (pooled multi-task, masked losses).
1. Prepare a multi-task config (see `configs/config_multitask_example.yaml`):
   - `pretrained_model_path`: encoder to start from (MLM-pretrained) or leave empty to train from scratch.
   - `tokenizer_path`: tokenizer directory.
   - `tasks`: list of tasks (name, type, metric).
   - `data_sources`: CSV/pickle paths + label mappings; labels are pooled by unique SMILES, missing labels masked.
   - `training`: batch_size, lr, epochs, etc.
2. Run:
   ```bash
   python train_multitask.py \
     --config configs/config_multitask_example.yaml \
     --log_file logs/multitask.log \
     --spot
   ```
   - Encoder runs once; only heads with labels for a molecule contribute to loss.

### Pre-tokenize supervised parquet (CPU pre-processing)
For large supervised runs, pre-tokenize SMILES into a tokenized parquet to remove per-step tokenization cost:
```bash
python tokenize_supervised_parquet.py \
  --tokenizer /path/to/tokenizer_10M \
  --input /path/to/all_datasets_fused_standardized.parquet \
  --output-dir /path/to/supervised_tokenized_parquet \
  --max-length 512 \
  --batch-rows 4096 \
  --shard-rows 200000 \
  --drop-smiles
```
Current canonical AWS artifact:
```text
s3://climb-s3-bucket/tokenized/supervised_wide_parquet/
```
Notes:
- This artifact preserves the original label columns from `s3://climb-s3-bucket/preparing_datasets/supervised_wide.parquet`.
- It adds `input_ids` and `attention_mask` so the supervised pipeline can stream pre-tokenized examples from S3.
- The current export consists of `55` shards and is the path used by `configs/experiment_spec_example.yaml`.
Then point `pretrain_pipeline.py` at the tokenized output:
```yaml
supervised_parquet_path: s3://climb-s3-bucket/preparing_datasets/supervised_wide.parquet
supervised_tokenized_parquet_path: s3://climb-s3-bucket/tokenized/supervised_wide_parquet
```
When `supervised_tokenized_parquet_path` is set, the pipeline uses `input_ids` + `attention_mask` columns directly and skips SMILES tokenization.
If `supervised_tokenized_parquet_path` is unset, the pipeline falls back to `supervised_parquet_path` and tokenizes SMILES online, which is substantially slower and should be avoided for full AWS runs.

---

## 6) Combined run (MLM + supervised)
Two approaches:
1. **Sequential (pipelines):** MLM then supervised fine‑tune.
   - Use `pretrain_pipeline.py` with `compute_budget.supervised_fraction` > 0 to run MLM then supervised within one script.
   - Or run `train_model.py` (MLM) → `train_multitask.py` (supervised) manually.
   - Control proportions via `compute_budget.supervised_fraction` and per‑phase configs (batch_size/epochs).
2. **Simultaneous (joint MLM + supervised):**
   - `MultiTaskModel` supports an MLM head alongside task heads (`include_mlm_head=True`), but `train_multitask.py` currently trains supervised-only. To train jointly in one loop, add a mixed collator/batch with both `mlm_labels` and `labels` and call `model.forward(...)` with both weights. If you want this wired up, say so and we’ll add it.

Phase order:
- In `pretrain_pipeline.py` the order is MLM first, then supervised. To reverse it, run supervised first (via `train_multitask.py`), save the encoder, then run MLM (`train_model.py`) loading that encoder.

Proportions:
- `compute_budget` in `PretrainingConfig` controls the epoch split between MLM and supervised.
- Within a single loop (if joint training is added), use loss weights for MLM vs supervised heads.

---

## Evaluation (MoleculeNet)
Use `evaluate_all_moleculenet.sh` or `evaluate_model.py` directly.
```bash
./evaluate_all_moleculenet.sh \
  experiments/prototyping_sup_unsup_50_50 \
  experiments/moleculenet_evals \
  tokenizer \
  10   # optional: early_stopping_patience (default 10)
```
This runs per-dataset fine-tuning with frozen encoder and saves per‑dataset results plus an aggregate `all_evaluations.txt`.

**Early stopping:** fine-tuning now stops automatically when `eval_loss` does not improve for `early_stopping_patience` epochs (default: 10). With `--num_epochs 50` (the default), this prevents overfitting the classification head on small datasets such as FreeSolv (642 molecules) or BBBP (2039 molecules). The best checkpoint by `eval_loss` is loaded before test evaluation.

Direct single-dataset example:
```bash
python evaluate_model.py \
  --pretrained_model experiments/prototyping_sup_unsup_50_50 \
  --dataset moleculenet \
  --dataset_name Tox21 \
  --output experiments/moleculenet_evals/Tox21 \
  --tokenizer tokenizer \
  --freeze_encoder \
  --early_stopping_patience 10 \
  --predictions_csv experiments/moleculenet_evals/Tox21/test_predictions.csv  # optional
```
- If you want a human-readable predictions file (SMILES, prediction, label), pass `--predictions_csv <path>`. If omitted, predictions are saved only as `.npy`.

---

## Key scripts
- `train_tokenizer.py` / `tokenize_dataset.py`: tokenizer training, data tokenization.
- `tokenizing_datasets.py`: shard large parquet/CSV/text SMILES inputs into tokenized pickle shards.
- `train_model.py`: flexible MLM/supervised mix (streaming-aware, spot-aware).
- `pretrain_pipeline.py`: orchestrates MLM + supervised with compute budgets.
- `train_multitask.py`: pooled multi-task supervised pretraining (masked losses).
- `hyperparameter_search.py`: Optuna sweep for MLM.
- `evaluate_model.py`, `evaluate_all_moleculenet.sh`: downstream evaluation.
- `scripts/generate_experiment_manifest.py`: resolve the full smoke + main matrix from one YAML spec.
- `scripts/preflight_experiment.py`: validate tokenizer/data/backup/eval dependencies before launch.
- `scripts/launch_experiment_wave.py`: launch a selected wave, run backups, and trigger full MoleculeNet evaluation.
- `scripts/monitor_cluster.py`, `watch_pretrain.sh`: monitor manifest runs across multiple workers.
- `scripts/aggregate_experiment_results.py`, `scripts/candidate_moleculenet_score.py`: aggregate raw outputs and compute provisional summary scores.

---

## Methodology Draft (Data, Tokenization, and Optimization)

This section is a first-draft, paper-oriented description of the current CLIMB methodology as implemented in this repository and current AWS storage.

### 1) Data provenance

Unsupervised pretraining source:
- PubChem-124M molecular strings dataset (SMILES/SELFIES/InChI/IUPAC):
  - [PubChem-124M-SMILES-SELFIES-InChI-IUPAC](https://huggingface.co/datasets/hheiden/PubChem-124M-SMILES-SELFIES-InChI-IUPAC)
- In our current pipeline, filtered/tokenized unsupervised shards are stored in:
  - `s3://climb-s3-bucket/tokenized_sources/pubchem_filtered/` (filtered parquet source shards)
  - `s3://climb-s3-bucket/tokenized_sources/pubchem_filtered_tokenized_pkl/` (training-ready tokenized pickle shards; generated for this workflow)

Supervised/downstream sources:
- Graphium benchmark dataset collection (MoleculeNet-oriented):
  - [Graphium datasets documentation](https://graphium-docs.datamol.io/stable/datasets.html)
- In code, downstream evaluation and task definitions are aligned with MoleculeNet tasks (e.g., BBBP, Tox21, ESOL, FreeSolv, Lipophilicity) via `evaluate_model.py` and multi-task configs.
- Current supervised pretraining source-of-truth:
  - `s3://climb-s3-bucket/preparing_datasets/supervised_wide.parquet`
- Current supervised pretraining artifact used for robust experiments:
  - `s3://climb-s3-bucket/tokenized/supervised_wide_parquet/`
  - format: parquet shards containing original task labels plus `input_ids` and `attention_mask`
- Legacy supervised tokenized pickles:
  - `s3://climb-s3-bucket/tokenized/supervised_wide/`
  - not suitable for current supervised pretraining because labels are missing

Tokenizer artifact used across current runs:
- `s3://climb-s3-bucket/tokenizer_10M/` (contains `tokenizer.json`).

### 2) Data preparation and normalization workflow

Current operational workflow:
1. Retrieve source artifacts from S3 (tokenizer + unsupervised shards).
2. Convert unsupervised parquet shards into training-ready tokenized shards compatible with CLIMB training scripts.
3. Use these tokenized shards consistently for hyperparameter optimization and unsupervised pretraining.
4. Use supervised datasets for downstream evaluation/multi-task phases.

Practical storage pattern:
- Raw/filtered unsupervised source shards: parquet.
- Unsupervised training ingest format: pickled dictionaries with key `data`, where each element has:
  - `input_ids`
  - `attention_mask`
- Supervised source-of-truth format: one fused parquet with canonical SMILES and many label columns.
- Supervised training ingest format for robust experiments: tokenized parquet shards with:
  - original label columns preserved,
  - `input_ids`,
  - `attention_mask`

This format is consumed by:
- `hyperparameter_search.py`
- `train_model.py`
- streaming dataset loaders in `data.py`
- supervised parquet streaming in `supervised_streaming.py`

### 3) Tokenizer training and reuse policy

Tokenizer training script:
- `train_tokenizer.py` (BPE tokenizer training from SMILES text/CSV inputs).

Tokenizer reuse in current experiments:
- A pre-trained tokenizer artifact is loaded from `tokenizer_10M`.
- The same tokenizer is used consistently across:
  - hyperparameter search,
  - unsupervised pretraining,
  - supervised/multi-task training,
  - downstream evaluation.

Rationale:
- Holding tokenizer fixed prevents vocabulary/segmentation changes from confounding hyperparameter comparisons or downstream model comparisons.

### 4) Tokenization procedure for unsupervised corpus

Sharded tokenization pipeline:
- `tokenizing_datasets.py` streams parquet in batches, selects a SMILES column, tokenizes with `PreTrainedTokenizerFast`, and writes shard files.
- Default sequence handling:
  - truncation enabled,
  - max length typically 512 tokens,
  - no fixed padding during shard creation (padding performed by collators at training time).

Resulting shard content:
- `{"data": [{"input_ids": [...], "attention_mask": [...]}, ...]}`

Why sharding:
- Keeps memory usage bounded.
- Improves I/O reliability on cloud instances.
- Enables streaming or capped-sample experiments.

### 5) Hyperparameter optimization protocol (Optuna)

Optimization script:
- `hyperparameter_search.py`

Objective:
- Minimize MLM validation loss (`eval_loss`) on unsupervised tokenized data.

Search space (current implementation):
- `learning_rate` (log-uniform)
- `batch_size` (categorical)
- `warmup_steps`
- `hidden_size`
- `num_hidden_layers`
- `num_attention_heads` (conditional on hidden size)
- `weight_decay`

Training/evaluation behavior in each trial:
- Uses masked language modeling objective.
- Trains for fixed epochs per trial.
- Evaluates on held-out split (or optional explicit eval set).
- Stores best trial parameters in JSON and full study object for reproducibility.

Resource-aware safeguards:
- Hyperparameter search supports bounded loading via:
  - `--max_samples`
  - `--max_train_shards`
  - `--max_eval_samples`
  - `--max_eval_shards`
- This prevents loading the full corpus into RAM on smaller GPU instances.

### 6) Transfer of optimized settings to all experiments

Protocol used for comparability:
1. Run Optuna on unsupervised corpus (MLM).
2. Select best trial by minimum validation loss.
3. Freeze these hyperparameters as the default training recipe.
4. Reuse the same hyperparameter set across all subsequent experiments (unsupervised, supervised, and mixed) unless an experiment explicitly studies ablations.

This enforces a consistent optimization baseline and avoids per-task overfitting of training settings.

### 7) Open issues and action items

#### ACTION: PCBA pretraining/evaluation overlap — characterize and report

**Observation:** PCBA (`PCBA__` prefix) is one of the supervised pretraining families in `supervised_streaming.py` and also appears as a MoleculeNet evaluation dataset in `evaluate_model.py`. This is a potential train/test data relationship that must be understood before publishing.

**Why it may not be a simple leak:** The supervised pretraining heads are discarded after training — only the encoder is kept. PCBA labels are not directly accessible at evaluation time. It is an open empirical question whether PCBA pretraining gives a visible boost on the PCBA MoleculeNet benchmark (which tests the same assay family) vs. the encoder having merely learned useful chemical representations from the signal.

**Action items:**
1. Compute molecule-level overlap between `supervised_wide.parquet` PCBA columns and the MoleculeNet PCBA train/val/test splits (by canonical SMILES or InChI key).
2. Run a controlled ablation: compare encoder performance on PCBA evaluation for (a) runs pretrained with the PCBA family vs. (b) runs pretrained without it, holding everything else constant.
3. In the paper: explicitly state the overlap, report the ablation, and discuss whether observed PCBA performance is driven by representation quality or implicit label memorization.

#### ACTION: Add random-encoder evaluation baseline

Five replicates of the full MoleculeNet evaluation suite run on a randomly initialized encoder (same architecture, no pretraining) must be added to the experiment matrix. Without this baseline, no claim about pretraining benefit can be supported by the data. The five replicates quantify variance due to head initialization and fine-tuning randomness alone. These runs are cheap (no pretraining, just the evaluation pipeline).

#### ACTION: Fix `StreamingSupervisedFamilyDataset` multi-worker data duplication

`supervised_streaming.py:StreamingSupervisedFamilyDataset.__iter__` does not implement `torch.utils.data.get_worker_info()` sharding. When `dataloader_num_workers > 0` (the recommended setting for G5 instances), every worker loads the full parquet independently, causing each molecule to be presented `num_workers` times per pass. This inflates token counts and effective epochs. Fix by sharding the parquet batches across workers using `worker_info.id` and `worker_info.num_workers`, mirroring the pattern already in `StreamingTokenizedDataset.__iter__` (`data.py:232`).

### 8) Reproducibility and artifact management

Artifact locations:
- Tokenizer: `s3://climb-s3-bucket/tokenizer_10M/`
- Unsupervised filtered source shards: `s3://climb-s3-bucket/tokenized_sources/pubchem_filtered/`
- Unsupervised tokenized shards used by current experiment spec: `s3://climb-s3-bucket/tokenized_sources/pubchem_filtered_tokenized_pkl/`
- Supervised source parquet: `s3://climb-s3-bucket/preparing_datasets/supervised_wide.parquet`
- Supervised tokenized parquet: `s3://climb-s3-bucket/tokenized/supervised_wide_parquet/`
- Hyperparameter outputs (typical local path): `hp_search_results/` with:
  - `best_hyperparameters.json`
  - `optuna_study.db`
  - `study_snapshot.json`

**Training seed:** both `MLMTrainingConfig` and `MultiTaskTrainingConfig` now have a `seed` field (default `42`). The seed is forwarded to `TrainingArguments` in both the unsupervised and supervised phases, which calls `set_seed()` before training and seeds dataloader workers. Set it explicitly in your YAML config to ensure reproducibility across replicates:
```yaml
mlm_training:
  seed: 42
supervised_training:
  seed: 42
```

Suggested reporting fields for paper appendix:
- exact S3 prefixes used,
- code commit hash,
- config file path(s),
- tokenizer identifier/path,
- sample/shard caps used during HP search,
- number of trials and selected trial ID,
- final selected hyperparameters,
- `seed` value used for all training phases.

---

## Paper Analysis Plan

Central hypothesis: **Does unsupervised pretraining benefit the performance of chemical language models?**

---

### Aggregate performance scores

Two scores are computed for every model and reported side-by-side throughout the paper:

| Score | Tasks included | Purpose |
|-------|---------------|---------|
| **Score_full** | All MoleculeNet tasks (BBBP, Tox21, SIDER, ClinTox, MUV, HIV, PCBA, ESOL, FreeSolv, Lipophilicity) | Completeness; PCBA caveat noted |
| **Score_unbiased** | All tasks **excluding PCBA** | Primary metric; free of supervised-pretraining/evaluation overlap risk |

Construction: for each classification task compute ROC-AUC; for regression tasks compute RMSE normalised by the random-encoder-baseline RMSE on that task. Z-score across tasks before averaging so that high-variance tasks (e.g. PCBA with 128 sub-tasks) do not dominate. Report both scores in every table and every plot where a single performance number appears. The gap between them is itself informative about PCBA leakage magnitude.

---

### Planned plots

#### Figure 1 — Unsupervised scaling curve
- **X:** unsupervised token budget (log scale: 1M → 10B), worker0 data.
- **Y:** Score_unbiased (mean ± std over 3 reps).
- **Reference line:** random-encoder baseline (mean ± std over 5 reps).
- **Purpose:** primary evidence for/against "more unsupervised pretraining helps".

#### Figure 2 — Supervised scaling curve
- **X:** supervised molecules / tokens seen (log scale), worker1 data.
- **Y:** Score_unbiased (mean ± std over 5 reps).
- **Reference line:** same random-encoder baseline.
- **Faint lines:** individual family orderings (5 orderings as thin coloured lines), bold line = mean. The spread between orderings is the family-ordering variance band.
- **Purpose:** primary evidence for/against "more supervised pretraining helps", and whether ordering matters.

#### Figure 3 — 3D surface: unsupervised × supervised → performance
- **X:** unsupervised tokens (log scale, 0 → 10B).
- **Y:** supervised tokens (0 → 1B).
- **Z:** Score_unbiased.
- Rendered as both a 3D surface and a 2D heatmap. The heatmap is the primary figure; 3D is supplementary.
- **Purpose:** identifies the optimal mix of unsupervised and supervised pretraining. If the maximum lies away from both axes, balanced pretraining is genuinely better than either alone.
- Data sources: worker0 (unsup axis), worker1 (sup axis), worker2 mixed ratios (interior points).

#### Figure 4 — PCBA leakage diagnostic
- Side-by-side: Score_full vs. Score_unbiased for each run type (unsup-only, sup-only, mixed) with PCBA pretraining highlighted.
- Additionally, a per-run scatter: (Score_full − Score_unbiased) on the Y axis vs. run type / token budget on X. If PCBA pretraining inflates Score_full relative to Score_unbiased, the gap is large and systematic.
- **Purpose:** demonstrates whether PCBA's inclusion in supervised pretraining biases evaluation. Motivates Score_unbiased as primary metric.
- Follow-up (if gap is large): run models without PCBA in the supervised family set and compare PCBA evaluation directly — confirms leakage vs. genuine representation learning.

#### Figure 5 — Molecule overlap confusion matrix
Split all molecules in the MoleculeNet evaluation sets into four groups based on their presence in the pretraining corpora:

| Group | In unsupervised corpus? | In supervised corpus? |
|-------|------------------------|----------------------|
| A | ✓ | ✓ |
| B | ✓ | ✗ |
| C | ✗ | ✓ |
| D | ✗ | ✗ |

For each group, compute per-task and aggregate evaluation performance. Present as a 2×2 confusion matrix heatmap with performance as the cell colour. Key questions:
- Do molecules seen in both corpora score best (memorisation vs. representation)?
- Do novel molecules (group D) still benefit from pretraining, or does performance collapse to baseline?
- Is supervised exposure more important than unsupervised exposure for seen molecules?
This directly characterises whether the model is generalising or memorising. Matching is done by canonical SMILES (RDKit canonicalisation + salt stripping).

#### Figure 6 — Permutation analysis (supervised family ordering)
- Box plots: one box per family ordering (5 orderings × 5 reps), Score_unbiased on Y axis.
- Supplementary: per-task breakdowns to check if specific tasks are ordering-sensitive.
- **Purpose:** determines whether the sequence of supervised families during pretraining is a meaningful design choice. If all boxes overlap, the finding is robust to ordering.

#### Figure 7 — Coverage ablation
- **X:** % of unsupervised corpus used (10, 25, 50, 75, 100% of PubChem-filtered shards).
- **Y:** Score_unbiased. All runs at 1B token budget (worker2).
- Separates "how much chemical-space breadth" from "how much compute". A flat curve means coverage beyond a threshold adds nothing; a rising curve means broader coverage matters.

---

### Planned follow-up experiments

The following experiments are not in the current matrix and will be launched after the main wave completes.

#### F1 — Random-encoder baseline (5 runs) [CRITICAL]
No pretraining; same architecture initialised randomly. Full MoleculeNet evaluation suite with 5 different random seeds. Establishes the zero-pretraining performance floor and its variance. Every performance claim in the paper requires this baseline.

#### F2 — Reversed training order: supervised → unsupervised (3 runs)
Standard order is unsupervised first, then supervised. Run 3 replicates with the order reversed at matched total compute (1B supervised + 1B unsupervised). Prediction: reversed order degrades performance because supervised fine-tuning overwrites generic representations before they are consolidated by MLM. Confirming this supports the conventional pipeline; a null result would be a notable finding.

#### F3 — PCBA leave-one-out (conditional)
Pursue only if Figure 4 shows a large, systematic (Score_full − Score_unbiased) gap. Train models with PCBA excluded from supervised families; compare PCBA evaluation directly. Distinguishes label memorisation from representation learning as the driver of PCBA evaluation performance.

#### F4 — Full fine-tuning vs. frozen encoder (optional)
All main evaluations freeze the encoder and train only the MoleculeNet head. Run a parallel sweep with full end-to-end fine-tuning on 3 representative models (random baseline, best unsup-only, best mixed). If pretraining helps with frozen encoder but not with full fine-tuning, the benefit is fragile. If it helps either way, it is robust.

---

### Hypothesis resolution framework

| Sub-question | Data source | Positive evidence | Negative evidence |
|---|---|---|---|
| Does any unsupervised pretraining help? | Figure 1 vs. random baseline | Curve significantly above baseline at any token budget | Curve overlaps baseline CI throughout |
| Does more unsupervised data monotonically help? | Figure 1 shape | Log-linear increase with flattening | Flat or non-monotone curve |
| Does supervised pretraining help independently? | Figure 2 vs. random baseline | Same as above for sup curve | Same negative |
| Is the combination better than either alone? | Figure 3 interior vs. axes | Interior maximum above both axes | Optimum lies on an axis (pure strategy wins) |
| Is the benefit robust to training order? | F2 | Reversed order within CI of standard order | Large degradation under reversal |
| Is the benefit from representation or memorisation? | Figure 5 | Group D (novel molecules) scores near group A | Group D collapses to random baseline |
| Is PCBA evaluation biased? | Figure 4 | Large systematic Score_full − Score_unbiased gap for supervised runs | Gap is small and uniform across run types |

---

### Stage 2: Model scaling experiments

To be planned once the main wave results are in hand. The exact design will depend on whether unsupervised pretraining shows a benefit — the working assumption is that it will not, which means the data ceiling is set by the supervised corpus (~1.5M labelled molecules) rather than the unlabelled PubChem pool.

#### Motivation
The current 13M-parameter model was chosen for ablation-matrix throughput, not for peak performance. Stage 2 asks: given the best pretraining recipe identified in Stage 1, how far does performance continue to scale with model size? This is the Chinchilla framing applied to chemical language models: hold the dataset size D fixed (at the supervised corpus ceiling) and vary N (parameters).

#### F5 — Best-recipe large-model run (1 run, details TBD)
Train a single large model (size TBD, likely 100–200M params) using the best pretraining configuration identified from the main matrix (optimal unsup/sup mix, optimal token budget, best family ordering). Compare its Score_unbiased against the 13M model trained identically. This establishes whether the qualitative findings from Stage 1 hold at a scale relevant to practitioners, and whether the 13M model is an adequate proxy for larger models.

#### F6 — Chinchilla-style model-size scaling curve
Fix dataset size D at the supervised corpus (1.5M molecules, ~1B tokens), fix the training recipe to the Stage 1 optimum, and train models at four parameter counts:

| Model | Hidden | Layers | Heads | Intermediate | ~Params |
|-------|--------|--------|-------|-------------|---------|
| CLIMB-13M | 256 | 10 | 8 | 1024 | 13M |
| CLIMB-50M | 512 | 12 | 8 | 2048 | ~50M |
| CLIMB-100M | 768 | 12 | 12 | 3072 | ~100M |
| CLIMB-200M | 1024 | 16 | 16 | 4096 | ~200M |

Run 3 replicates per size (budget permitting; at minimum 1 rep per size + 3 reps at 13M from the main wave). Plot Score_unbiased vs. log(N). Key questions:
- Does performance continue to improve log-linearly with N, or is there an inflection point?
- At what model size does the supervised-data ceiling become the binding constraint (i.e., the curve flattens)?
- Does the optimal unsup/sup mix shift as N grows (larger models may extract more signal from unsupervised data)?

**Architecture note:** keep `max_position_embeddings: 514` and tokenizer fixed across all sizes. Only scale hidden_size, num_layers, num_heads, intermediate_size. This ensures token representations are comparable across sizes and only capacity changes.

**Compute note:** CLIMB-200M at 1B tokens on a g5.2xlarge is approximately 3–4× slower than CLIMB-13M (rough estimate: ~30–35 hours per run). Budget ~1 week of EC2 time for the full F6 series at 1 rep per size; ~3 weeks for 3 reps. Replicates can be parallelised across a 3-node cluster identical to the current setup.
