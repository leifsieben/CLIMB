"""Emit a small SFT learning-rate sweep manifest (warm-start confound check).

The ablation found unsup->sup <= the MLM base under a frozen probe. That may be an
artifact of fine-tuning the warm-started encoder at the SAME LR (2e-4) used for
from-scratch training, which can destroy pretrained features. This sweeps the SFT-phase
LR over a fixed base encoder for dense (MTR) and sparse_all, so we can see whether a
lower LR recovers (or beats) the base. Base defaults to the round-1 unsup_only encoder.

    python scripts/make_lrsweep_manifest.py --base_spec configs/v2_phase2.yaml \
        --base_encoder experiments/climb_v2/unsup_only_seed0/encoder \
        --results_root experiments/climb_v2_lrsweep \
        --s3_backup s3://climb-s3-bucket/experiments/climb_v2_lrsweep \
        --output experiments/climb_v2_lrsweep/manifest.json
"""
from __future__ import annotations
import argparse, copy, json, sys
from pathlib import Path
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))
from config_v2 import SUPERVISED_GROUPS

LRS = [2.0e-4, 1.0e-4, 5.0e-5, 2.0e-5]
TYPES = [("dense", {"mtr": 1.0}, None),
         ("sparse_all", {"supervised": 1.0}, SUPERVISED_GROUPS["sparse_all"])]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base_spec", required=True)
    p.add_argument("--base_encoder", required=True)
    p.add_argument("--results_root", required=True)
    p.add_argument("--s3_backup", required=True)
    p.add_argument("--sup_budget", type=int, default=2_000_000)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    spec = yaml.safe_load(open(args.base_spec))
    runs = []
    for lr in LRS:
        for name, objs, fams in TYPES:
            lr_tag = f"{lr:.0e}".replace("-0", "e-").replace("e-0", "e-")
            rid = f"lr{lr:.0e}_{name}".replace("-0", "-")
            training = copy.deepcopy(spec.get("training", {}))
            training["learning_rate"] = lr
            training["save_every_steps"] = 0  # short runs, save at end only
            pc = {
                "run_id": rid,
                "tokenizer_path": spec["tokenizer_path"],
                "unsupervised_raw_smiles_paths": spec.get("unsupervised_raw_smiles_paths"),
                "supervised_tokenized_parquet_path": spec.get("supervised_tokenized_parquet_path"),
                "descriptor_stats_path": spec.get("descriptor_stats_path"),
                "model": spec.get("model", {}),
                "training": training,
                "evaluation": spec.get("evaluation", {}),
                "selection": {
                    "objectives": objs,
                    "supervised_families": fams,
                    "init_encoder_path": args.base_encoder,
                    "pretraining_seed": 0,
                    "total_forward_passes": args.sup_budget,
                    "augmentation": "canonical",
                },
            }
            runs.append({
                "run_id": rid, "run_type": "lrsweep", "stage": "lrsweep",
                "requires_pretrain": True,
                "output_dir": f"{args.results_root}/{rid}",
                "backup_s3_uri": f"{args.s3_backup}/{rid}",
                "evaluation_output_dir": f"{args.results_root}/{rid}/moleculenet",
                "pretrain_config": pc, "selection": pc["selection"],
            })
    man = {"name": "climb_v2_lrsweep", "results_root": args.results_root,
           "s3_backup_root": args.s3_backup, "tokenizer_path": spec["tokenizer_path"], "runs": runs}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(man, indent=2))
    print(f"wrote {len(runs)} runs ({[r['run_id'] for r in runs]}) to {args.output}")


if __name__ == "__main__":
    main()
