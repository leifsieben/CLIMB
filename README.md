# CLIMB Pretraining & Evaluation Guide

This repo supports end‑to‑end tokenizer training, hyperparameter sweeps, pretraining (unsupervised, supervised, mixed), streaming large datasets, and spot‑friendly execution on AWS. Below is a practical how‑to.

---

## 1) Train a tokenizer (BPE) and store/reuse it
1. Prepare raw SMILES text (one per line) or a CSV with a `SMILES` column.
2. Train the tokenizer (example command; adjust paths):
   ```bash
   python train_tokenizer.py \
     --input data/unsup_smiles.smi \
     --output tokenizer \
     --vocab_size 32000 \
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
1. Prepare tokenized unsupervised data (pickle list/dict with `input_ids`, `attention_mask`):
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
     --tokenizer tokenizer \
     --train_data data/unsup.pkl \
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
  --freeze_encoder
```

---

## Key scripts
- `train_tokenizer.py` / `tokenize_dataset.py`: tokenizer training, data tokenization.
- `train_model.py`: flexible MLM/supervised mix (streaming-aware, spot-aware).
- `pretrain_pipeline.py`: orchestrates MLM + supervised with compute budgets.
- `train_multitask.py`: pooled multi-task supervised pretraining (masked losses).
- `hyperparameter_search.py`: Optuna sweep for MLM.
- `evaluate_model.py`, `evaluate_all_moleculenet.sh`: downstream evaluation.
