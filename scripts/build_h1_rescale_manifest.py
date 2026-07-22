"""Rebuild the H1 canonical-vs-enumerated sweep — the one wave whose encoders were lost.

The round-1 `climb_v2/scaling_*` runs backed up config, metrics and evals but NEVER an
`encoder/` prefix, and no copy survives on any box. So H1 could not be re-scored on HIV, and a
reviewer could not reproduce or inspect those models at all. They are cheap (2M forward passes
each), so they are simply retrained.

Two things change versus round 1, both deliberate:

  * THREE pretraining seeds instead of one. This is the bigger fix. The round-1 lines carry no
    error bar at all, while the enumerated-minus-canonical differences they show (0.018 on BBBP,
    0.058 on BACE, 0.075 on ESOL, sign-flipping between adjacent fractions) are the same size as
    head-seed noise. As it stood H1 could support neither "enumeration helps" nor "it doesn't".
  * The current eval suite, so HIV/NEF1% and the train metrics exist and H1 stops being the one
    figure missing a panel.

The 2M-FP budget is kept exactly as round 1, so this is a faithful reproduction of the same
experiment rather than a new one.

Usage:
    python scripts/build_h1_rescale_manifest.py --out experiments/climb_v2_h1/manifest.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

WAVE = "climb_v2_h1"
BUCKET = f"s3://climb-s3-bucket/experiments/{WAVE}"
FRACTIONS = [("frac0p001", 0.001), ("frac0p01", 0.01), ("frac0p1", 0.1),
             ("frac0p3", 0.3), ("fracfull", None)]
AUGS = ["canonical", "enumerated"]
SEEDS = [0, 1, 2]
FP = 2_000_000


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=3)
    a = ap.parse_args()

    runs = []
    for seed in SEEDS:
        for aug in AUGS:
            for fk, fv in FRACTIONS:
                rid = f"scaling_{aug}_{fk}_s{seed}"
                sel = {
                    "objectives": {"mlm": 1.0},
                    "supervised_families": None,
                    "init_encoder_path": None,
                    "pretraining_seed": seed,
                    "total_forward_passes": FP,
                    "augmentation": aug,
                    "unsupervised_subset_fraction": fv,
                }
                runs.append({
                    "run_id": rid,
                    "run_type": "unsup_scaling",
                    "stage": "ladder",
                    "requires_pretrain": True,
                    "output_dir": f"experiments/{WAVE}/{rid}",
                    "backup_s3_uri": f"{BUCKET}/{rid}",
                    "evaluation_output_dir": f"experiments/{WAVE}/{rid}/moleculenet",
                    "selection": dict(sel),
                    "pretrain_config": {"run_id": rid, "selection": dict(sel)},
                })

    manifest = {
        "name": WAVE,
        "results_root": f"experiments/{WAVE}",
        "s3_backup_root": BUCKET,
        "tokenizer_path": "s3://climb-s3-bucket/tokenizer_10M",
        "runs": runs,
    }
    p = Path(a.out); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(manifest, indent=2))

    # Shard round-robin over fractions so no worker gets all the large-fraction (slower) runs.
    for i in range(a.workers):
        q = p.with_name(p.stem + f"_worker{i}.json")
        q.write_text(json.dumps({**manifest, "runs": runs[i::a.workers]}, indent=2))
        print(f"  worker{i}: {len(runs[i::a.workers])} runs -> {q}")

    total = len(runs) * FP
    print(f"\n{len(runs)} runs ({len(AUGS)} augmentations x {len(FRACTIONS)} fractions x "
          f"{len(SEEDS)} seeds) x {FP/1e6:.0f}M FP = {total/1e6:.0f}M FP")
    print(f"~{total/755/3600:.1f} GPU-h total, ~{total/755/3600/a.workers:.1f}h on {a.workers} boxes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
