"""Experiment A — build the launch manifest for the synthetic-statistics ladder.

Five NEW pretraining runs (all 8M forward passes, frozen-probe eval), cloned bit-for-bit from the
canonical `unsup_8M` phase-2 config so they differ in EXACTLY one thing each:

  unigram_8M, unigram_8M_s1, unigram_8M_s2   — real pkl corpus swapped for the materialized
      unigram-resample corpus (token marginal only). pretraining_seed 0/1/2.
  corrupt_mlm_8M_s1, corrupt_mlm_8M_s2       — real corpus, selection.corruption=shuffle_tokens,
      pretraining_seed 1/2 (fills in the two missing seeds; s0 already exists in climb_v2_phase2).

Everything else — model, optimizer, schedule, budget, mlm_probability, augmentation, eval protocol —
is identical to unsup_8M, so the ladder isolates the statistic under test.

The runs land in a SEPARATE wave (`climb_v2_expA`) so the paper's climb_v2_phase2 wave stays
pristine until the result is vetted. The ladder's other rungs (unsup_only, shuffle s0, no_pretrain)
are reused in place from climb_v2_phase2; scripts/build_expA_ladder_summary.py stitches everything
into one tidy CSV for the notebook session.

Usage (on the box):
    python scripts/build_expA_manifest.py --out experiments/climb_v2_expA/manifest.json
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

WAVE = "climb_v2_expA"
REAL_PKL = "s3://climb-s3-bucket/tokenized_sources/pubchem_filtered_tokenized_pkl/"
UNIGRAM_PKL = "s3://climb-s3-bucket/tokenized_sources/pubchem_filtered_unigram_pkl/"
BIGRAM_PKL = "s3://climb-s3-bucket/tokenized_sources/pubchem_filtered_bigram_pkl/"
BACKUP_ROOT = f"s3://climb-s3-bucket/experiments/{WAVE}"

# Canonical unsup_8M pretrain_config (verbatim from climb_v2_phase2/unsup_8M/config.yaml).
BASE_PRETRAIN = {
    "tokenizer_path": "s3://climb-s3-bucket/tokenizer_10M",
    "unsupervised_data_paths": [REAL_PKL],
    "unsupervised_raw_smiles_paths": ["s3://climb-s3-bucket/tokenized_sources/pubchem_filtered/"],
    "supervised_tokenized_parquet_path": "s3://climb-s3-bucket/tokenized/supervised_wide_parquet/",
    "descriptor_stats_path": "configs/descriptor_stats.json",
    "descriptor_precompute_dir": "s3://climb-s3-bucket/tokenized_sources/pubchem_descriptors/",
    "eval_blocklist_path": "s3://climb-s3-bucket/configs/eval_blocklist.json",
    "model": {
        "hidden_size": 512, "num_hidden_layers": 12, "num_attention_heads": 8,
        "intermediate_size": 1536, "max_position_embeddings": 256, "vocab_size": 1000,
    },
    "training": {
        "learning_rate": 0.0002, "batch_size": 256, "warmup_ratio": 0.05, "weight_decay": 0.01,
        "mlm_probability": 0.3, "train_max_length": 128, "supervised_regression_loss": "mae",
        "mtr_loss": "mse", "uncertainty_weighting": True, "seed": 42, "bf16": True,
        "log_every_steps": 50, "save_every_steps": 15000, "dataloader_num_workers": 6,
    },
    "evaluation": {
        "pool": "mean", "standardize": "zscore", "head": "mlp", "max_length": 256,
        "head_seeds": [0, 1, 2],
    },
    "unsupervised_subset_fraction": None,
    "selection": {
        "objectives": {"mlm": 1.0}, "supervised_families": None, "init_encoder_path": None,
        "pretraining_seed": 0, "total_forward_passes": 8000000, "augmentation": "canonical",
        "unsupervised_subset_fraction": None,
    },
}


def _run(run_id: str, *, data_paths, seed: int, corruption: str | None, run_type: str):
    pc = copy.deepcopy(BASE_PRETRAIN)
    pc["run_id"] = run_id
    pc["unsupervised_data_paths"] = list(data_paths)
    pc["selection"]["pretraining_seed"] = seed
    if corruption is not None:
        pc["selection"]["corruption"] = corruption
    sel = copy.deepcopy(pc["selection"])
    return {
        "run_id": run_id,
        "run_type": run_type,
        "stage": "ladder",
        "requires_pretrain": True,
        "output_dir": f"experiments/{WAVE}/{run_id}",
        "backup_s3_uri": f"{BACKUP_ROOT}/{run_id}",
        "evaluation_output_dir": f"experiments/{WAVE}/{run_id}/moleculenet",
        "pretrain_config": pc,
        "selection": sel,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=f"experiments/{WAVE}/manifest.json")
    a = ap.parse_args()

    runs = []
    # unigram-resample arm, 3 seeds
    for s in (0, 1, 2):
        rid = "unigram_8M" if s == 0 else f"unigram_8M_s{s}"
        runs.append(_run(rid, data_paths=[UNIGRAM_PKL], seed=s, corruption=None,
                         run_type="synthetic_unigram"))
    # shuffle_tokens missing seeds (s0 = corrupt_mlm_8M already exists in phase2)
    for s in (1, 2):
        runs.append(_run(f"corrupt_mlm_8M_s{s}", data_paths=[REAL_PKL], seed=s,
                         corruption="shuffle_tokens", run_type="synthetic_shuffle"))
    # bigram-resample arm, 3 seeds (added 2026-08-11: closes the ladder gap between unigram and
    # shuffle — preserves LOCAL adjacency, destroys the per-molecule multiset)
    for s in (0, 1, 2):
        rid = "bigram_8M" if s == 0 else f"bigram_8M_s{s}"
        runs.append(_run(rid, data_paths=[BIGRAM_PKL], seed=s, corruption=None,
                         run_type="synthetic_bigram"))

    manifest = {
        "name": WAVE,
        "results_root": f"experiments/{WAVE}",
        "s3_backup_root": BACKUP_ROOT,
        "tokenizer_path": "s3://climb-s3-bucket/tokenizer_10M",
        "runs": runs,
    }
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2))
    print(f"wrote {out} with {len(runs)} runs:")
    for r in runs:
        sel = r["pretrain_config"]["selection"]
        print(f"  {r['run_id']:22} seed={sel['pretraining_seed']} "
              f"corruption={sel.get('corruption','-')} data={r['pretrain_config']['unsupervised_data_paths'][0].split('/')[-2]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
