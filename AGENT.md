# What

I want to pre-train a chemical language model using both supervised and unsupervised datasets. This should be as modular as possible. As of right now, I believe the tokenizer training and tokenization part to work quite well. Now the dataset preparation and training need to be refactored. For a given pre-training run I want to fix the compute budget (you can assume that sup and unsup training are similarly expensive) which will then define the n_epochs, how much of the compute budget goes to sup and to unsup (e.g. 50/50, 100/0 etc) and then which datasets. This last part is easy for unsupervised training where I'll just have a large pool of tokenized SMILES for MLM. The supervised datasets are harder. I'll have many different datasets each with many tasks that I'll have to pool first, then deduplicate so we can use them during training like this: For a given tokenized SMILES pass it thorugh the encoder part of the model. Then use that representation for n heads where n is the number of tasks I have (across al sup datasets) for that given molecule, then backprop the aggregate loss (equal weight for all tasks). So we train end-to-end here, in the end we just discard the task heads and only keep the encoder stack. I will do one hyperparameter Opt run with MLM and then keep the same hyperparameters for all iterations (for comparability). I will also use the same BPE tokenizer for all runs. But I'll vary how much unsup and sup pre-training I do to then see how this impacts the downstream performance. I think it makes sense structuring the model in this way as well, one encoder part with many heads where we even treat MLM as one more head (just to keep it DRY)

# Results

In the end, once we have a succesful set of hyperparameters. We will use the same hyperparameters for all runs. We start with ramping up 0 to 100% of only supervised or unsupervised data just to see the scaling laws in terms of pretraining. Then we will also use permutations (training the various supervised datasets in different sequences) just to see if this has an impact. We will also train combinations of supervised and unsupervised data. Typically, we will train unsupervised then supervised but surely we'll do a couple experiments to see how this impacts performance if we do it the other way around. 

Performance is always measured by MoleculeNet performance. Finetune on that data and then measure on this consistent, multi-task benchmark. We then need to define some aggregate score and we can essentially make a 3D plot where the z-axis is the aggregate score, and x- and y-axes are the supervised and unsupervised amount of data. Naturally, we will always use the same parameter size and in general keep everything consistent/the same to ensure we isolate just the impact of pre-training on downstream performance. 

# How

* Keep the codebase as simple as possible. Use as many standard, well-maintained, common packages (transformers, pytorch, scipy, etc) as possible.
* Always keep the version on the local machine, the git, and the EC2 instance up to date and consistent code wise.
* Make sure you alwyas ask the user for permission to spin up EC2 instances. You have a limit of 16 vCPU on AWS.
* Ultimately the model should be light-weight, fast, and easy to interface.
* Assume that all molecules will be inputted as SMILES. For now, we don't need to support other formats. Do check whether a SMILES is valid and we will have to do graph construction as well.

## Shell scripting rules

* **Never use `declare -A` (associative arrays) in bash scripts.** macOS ships with bash 3.2, which does not support associative arrays. String keys silently evaluate to index 0, so all lookups return the same value — a bug that is very hard to notice. Use `case` statements for key→value lookups instead.

# Pretraining Data

**Source file:** `datasets/all_datasets_fused_standardized.parquet`

This is the master pretraining dataset containing:
- ~1.5M molecules
- ~3,288 tasks (assay endpoints)
- SMILES column: `SMILES_std` (standardized)
- Wide format with NaN for missing labels

Use this dataset for all pretraining runs. The multi-task data loader handles missing labels via masked loss.

# Experiment Decisions

## Token budget: 1B tokens for coverage and mixed experiments

**Decided 2026-03-19.**

All coverage ablation and mixed MLM/supervised ratio runs (worker2, 20 total) use a **1B token budget**. Worker0's unsupervised ramp-up experiments are **kept at 10B** as a scaling control.

Key reasons:
- 10B × 20 runs on a single g5.2xlarge ≈ 62 days; 1B × 20 ≈ 6–7 days.
- 1B tokens is within the range of published chemical language models (ChemBERTa: 300M–2B, smi-ted: 3B, MolBERT: 80M).
- At 13M parameters, 1B tokens ≈ 77 tokens/param — well into the overtraining regime for BERT-style encoders, which is desirable for downstream fine-tuning.
- The 13M param model is intentionally small to fit the 60+ run ablation matrix on 3 spot nodes within the project timeline.
- Worker0's 10B ramp-up provides the scaling control needed to assess whether more tokens improve downstream performance.

**When running new experiments, use `total_tokens: 1000000000` (1B) for all unsupervised_fixed_budget and mixed_fixed_budget runs unless explicitly studying token-budget scaling.**

## Model architecture: 13M parameters

**Decided during initial HPO.** RoBERTa-style encoder with:
- `hidden_size: 256`
- `num_hidden_layers: 10`
- `num_attention_heads: 8`
- `intermediate_size: 1024`
- `max_position_embeddings: 514` (512-token sequences + 2 special tokens)

This size is appropriate for: the SMILES domain (short sequences, ~30–50 tokens median), the supervised dataset scale (~1.5M molecules), and the ablation matrix scope. See README "Decision: 1B token budget" section for full defence.

## Training seed

Both `MLMTrainingConfig` and `MultiTaskTrainingConfig` have a `seed` field (default `42`). Set explicitly in YAML configs for reproducibility across replicates. The seed is forwarded to HuggingFace `TrainingArguments`, which calls `set_seed()` and seeds dataloader workers.