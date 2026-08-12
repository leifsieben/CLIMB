"""Experiment B — launch manifest for the `wiki_real` arm (English Wikipedia pretraining through the
frozen SMILES tokenizer). 3 seeds, 8M forward passes, cloned bit-for-bit from `unsup_8M` with only
the corpus swapped. Wave `climb_v2_expB` (separate from the paper waves). Comparators are reused:
`real` = climb_v2_phase2/unsup_8M, `no_pretrain` = random_baseline — both already native-eval'd under
climb_v2_expA/_baselines, so this experiment adds exactly one arm.

Usage:  python scripts/build_expB_manifest.py --out experiments/climb_v2_expB/manifest.json
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

WAVE = "climb_v2_expB"
WIKI_PKL = "s3://climb-s3-bucket/tokenized_sources/pubchem_filtered_wiki_pkl/"
BACKUP_ROOT = f"s3://climb-s3-bucket/experiments/{WAVE}"

BASE_PRETRAIN = {
    "tokenizer_path": "s3://climb-s3-bucket/tokenizer_10M",
    "unsupervised_data_paths": [WIKI_PKL],
    "unsupervised_raw_smiles_paths": ["s3://climb-s3-bucket/tokenized_sources/pubchem_filtered/"],
    "supervised_tokenized_parquet_path": "s3://climb-s3-bucket/tokenized/supervised_wide_parquet/",
    "descriptor_stats_path": "configs/descriptor_stats.json",
    "descriptor_precompute_dir": "s3://climb-s3-bucket/tokenized_sources/pubchem_descriptors/",
    "eval_blocklist_path": "s3://climb-s3-bucket/configs/eval_blocklist.json",
    "model": {"hidden_size": 512, "num_hidden_layers": 12, "num_attention_heads": 8,
              "intermediate_size": 1536, "max_position_embeddings": 256, "vocab_size": 1000},
    "training": {"learning_rate": 0.0002, "batch_size": 256, "warmup_ratio": 0.05, "weight_decay": 0.01,
                 "mlm_probability": 0.3, "train_max_length": 128, "supervised_regression_loss": "mae",
                 "mtr_loss": "mse", "uncertainty_weighting": True, "seed": 42, "bf16": True,
                 "log_every_steps": 50, "save_every_steps": 15000, "dataloader_num_workers": 6},
    "evaluation": {"pool": "mean", "standardize": "zscore", "head": "mlp", "max_length": 256,
                   "head_seeds": [0, 1, 2]},
    "unsupervised_subset_fraction": None,
    "selection": {"objectives": {"mlm": 1.0}, "supervised_families": None, "init_encoder_path": None,
                  "pretraining_seed": 0, "total_forward_passes": 8000000, "augmentation": "canonical",
                  "unsupervised_subset_fraction": None},
}


def _run(run_id: str, seed: int):
    pc = copy.deepcopy(BASE_PRETRAIN)
    pc["run_id"] = run_id
    pc["selection"]["pretraining_seed"] = seed
    return {
        "run_id": run_id, "run_type": "wiki_transfer", "stage": "expB", "requires_pretrain": True,
        "output_dir": f"experiments/{WAVE}/{run_id}",
        "backup_s3_uri": f"{BACKUP_ROOT}/{run_id}",
        "evaluation_output_dir": f"experiments/{WAVE}/{run_id}/moleculenet",
        "pretrain_config": pc, "selection": copy.deepcopy(pc["selection"]),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=f"experiments/{WAVE}/manifest.json")
    a = ap.parse_args()
    runs = [_run("wiki_real_8M" if s == 0 else f"wiki_real_8M_s{s}", s) for s in (0, 1, 2)]
    manifest = {"name": WAVE, "results_root": f"experiments/{WAVE}", "s3_backup_root": BACKUP_ROOT,
                "tokenizer_path": "s3://climb-s3-bucket/tokenizer_10M", "runs": runs}
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2))
    print(f"wrote {out} with {len(runs)} runs:")
    for r in runs:
        print(f"  {r['run_id']:18} seed={r['pretrain_config']['selection']['pretraining_seed']} "
              f"data={r['pretrain_config']['unsupervised_data_paths'][0].split('/')[-2]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
