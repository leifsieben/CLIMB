# Streaming Data and Spot Instance Workflow

## Streaming large tokenized datasets

1. Tokenize your SMILES shards as many pickle files as you need (e.g., `shard_000.pkl`, `shard_001.pkl`, …) and place them either directly in a directory (all `.pkl` files) or provide a mix of directories/files.
2. When calling `train_model.py`, point `--unsup_data` and/or `--sup_data` at the directory instead of a single pickle.
3. The loader now detects directories and streams samples via `StreamingMixedDataset`, so no single file needs to fit in memory. Use `--streaming_max_samples` to cap how many samples are consumed per epoch if you want a deterministic compute budget.

Example:
```
python train_model.py \
  --config configs/config_prototyping_50_50.yaml \
  --tokenizer tokenizer \
  --unsup_data /mnt/data/tokenized/unsup_shards \
  --sup_data /mnt/data/tokenized/sup_shards \
  --streaming_max_samples 2000000 \
  --unsup_weight 0.5 \
  --sup_weight 0.5 \
  --task mlm \
  --output experiments/streaming_run
```

When streaming, the trainer relies on `max_steps` (set in the config) instead of dataset length.

## Spot instance safety

1. Enable the `--spot` flag on any CLI entry point (`train_model.py`, `pretrain_pipeline.py`, `train_multitask.py` via its config) to register a SIGTERM handler that saves a checkpoint immediately when AWS notifies you the instance is interrupting.
2. The handler saves a checkpoint under `.../spot_checkpoint/` inside your configured output directory and exits cleanly.
3. Resume from that checkpoint by pointing `--pretrained_model` (or equivalent) to the `spot_checkpoint` directory.

Because the handler is wired through `utils.register_spot_handler`, `pretrain_pipeline.py`, `train_model.py`, and the multi-task trainer all respond consistently to spot interruptions.
