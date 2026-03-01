# CLIMB Pretraining & Evaluation Guide

This repo supports end‑to‑end tokenizer training, hyperparameter sweeps, pretraining (unsupervised, supervised, mixed), streaming large datasets, and spot‑friendly execution on AWS. Below is a practical how‑to.

---

## Current Data Sources, Processing, and Storage (AWS)

### A) Dataset sources and how they are used
- Unsupervised pretraining corpus: filtered PubChem tokenized shards stored in S3 at `s3://climb-s3-bucket/tokenized_sources/pubchem_filtered/`.
- Tokenizer artifact used for current runs: `s3://climb-s3-bucket/tokenizer_10M/` (contains `tokenizer.json`).
- Downstream evaluation datasets: MoleculeNet tasks loaded via DeepChem in `evaluate_model.py` (e.g., BBBP, Tox21, ESOL, FreeSolv, Lipophilicity).
- Multi-task supervised pretraining inputs: user-provided CSV files referenced by `configs/config_multitask_example.yaml` (`data/bbbp.csv`, `data/bace.csv`, etc.).

Processing path used in this repo:
1. Build or load tokenizer (`train_tokenizer.py` or prebuilt tokenizer from S3).
2. Tokenize raw SMILES (single-file flow with `tokenize_dataset.py`, or sharded flow with `tokenizing_datasets.py`).
3. Run MLM hyperparameter search (`hyperparameter_search.py`) on tokenized shards.
4. Run pretraining (`train_model.py` / `pretrain_pipeline.py`), then evaluate (`evaluate_model.py` / `evaluate_all_moleculenet.sh`).

### B) Current S3 structure and access
Known active paths:
- `s3://climb-s3-bucket/tokenizer_10M/`
- `s3://climb-s3-bucket/tokenized_sources/pubchem_filtered/`

Inspect available prefixes:
```bash
aws s3 ls s3://climb-s3-bucket/
aws s3 ls s3://climb-s3-bucket/ --recursive | grep tokenizer.json
```

Download artifacts to EC2:
```bash
mkdir -p ~/artifacts/tokenizer_10M ~/data/pubchem_filtered
aws s3 sync s3://climb-s3-bucket/tokenizer_10M ~/artifacts/tokenizer_10M
aws s3 sync s3://climb-s3-bucket/tokenized_sources/pubchem_filtered ~/data/pubchem_filtered
```

### C) README status / outdated items now corrected
- Hyperparameter sweep entrypoint is `hyperparameter_search.py` (not `run_hyperopt.py`).
- Sweep now supports both `.pkl` and `.parquet` tokenized inputs.
- Current tokenizer location is documented as `s3://climb-s3-bucket/tokenizer_10M/`.
- AWS examples now align with the currently used S3 prefixes.

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
     --n_trials 20
   ```
   - The sweep searches lr, batch_size, warmup_steps, hidden size/layers/heads, weight_decay.
   - Optimizes `eval_loss` on MLM.
   - Best params saved to `hp_search_results/best_hyperparameters.json`.
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
  tokenizer
```
This runs per-dataset fine-tuning with frozen encoder and saves per‑dataset results plus an aggregate `all_evaluations.txt`.
Direct single-dataset example:
```bash
python evaluate_model.py \
  --pretrained_model experiments/prototyping_sup_unsup_50_50 \
  --dataset moleculenet \
  --dataset_name Tox21 \
  --output experiments/moleculenet_evals/Tox21 \
  --tokenizer tokenizer \
  --freeze_encoder \
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

---

## Methodology Draft (Data, Tokenization, and Optimization)

This section is a first-draft, paper-oriented description of the current CLIMB methodology as implemented in this repository and current AWS storage.

### 1) Data provenance

Unsupervised pretraining source:
- PubChem-124M molecular strings dataset (SMILES/SELFIES/InChI/IUPAC):
  - [PubChem-124M-SMILES-SELFIES-InChI-IUPAC](https://huggingface.co/datasets/hheiden/PubChem-124M-SMILES-SELFIES-InChI-IUPAC)
- In our current pipeline, filtered/tokenized unsupervised shards are stored in:
  - `s3://climb-s3-bucket/tokenized_sources/pubchem_filtered/` (parquet source shards)
  - `s3://climb-s3-bucket/tokenized_sources/pubchem_filtered_tokenized_pkl/` (training-ready tokenized pickle shards; generated for this workflow)

Supervised/downstream sources:
- Graphium benchmark dataset collection (MoleculeNet-oriented):
  - [Graphium datasets documentation](https://graphium-docs.datamol.io/stable/datasets.html)
- In code, downstream evaluation and task definitions are aligned with MoleculeNet tasks (e.g., BBBP, Tox21, ESOL, FreeSolv, Lipophilicity) via `evaluate_model.py` and multi-task configs.

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
- Model-training ingest format: pickled dictionaries with key `data`, where each element has:
  - `input_ids`
  - `attention_mask`

This format is consumed by:
- `hyperparameter_search.py`
- `train_model.py`
- streaming dataset loaders in `data.py`

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

### 7) Reproducibility and artifact management

Artifact locations:
- Tokenizer: `s3://climb-s3-bucket/tokenizer_10M/`
- Unsupervised tokenized shards: `s3://climb-s3-bucket/tokenized_sources/pubchem_filtered_tokenized_pkl/`
- Hyperparameter outputs (typical local path): `hp_search_results/` with:
  - `best_hyperparameters.json`
  - `optuna_study.pkl`

Suggested reporting fields for paper appendix:
- exact S3 prefixes used,
- code commit hash,
- config file path(s),
- tokenizer identifier/path,
- sample/shard caps used during HP search,
- number of trials and selected trial ID,
- final selected hyperparameters.
