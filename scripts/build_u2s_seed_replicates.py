"""Build pretraining-seed replicates for the u2s (unsup->sup) arm at the matched budget.

A1's error bars now use pretraining-seed spread where replicates exist, but unsup2sup had only
one seed while sup_only had three -- so its bar meant something different from its neighbours.
This closes that gap.

Seed-MATCHED by construction: seed s replicates warm-start from unsup_8M_s{s}, not from the
seed-0 base. Reusing one base for all seeds would measure only SFT-stage variance and understate
the true pipeline noise, which is exactly the quantity the error bar is supposed to represent.

Anchors deliberately get no replicates: random_baseline_00/01/02 are already three independent
random inits (i.e. three seeds), and ecfp4/fp_desc are deterministic classical baselines with no
pretraining stage to reseed -- their spread is head-seed only and correctly labelled as such.

Usage:
    python scripts/build_u2s_seed_replicates.py --out experiments/climb_v2_phase2/manifests/u2s_seeds.json
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

SRC = "experiments/climb_v2_phase2/manifest_wave1.json"
SEEDS = (1, 2)
BUCKET = "s3://climb-s3-bucket/experiments/climb_v2_phase2"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()

    m = json.loads(Path(SRC).read_text())
    twins = [r for r in m["runs"] if r["run_id"].endswith("_from8M")]
    if not twins:
        print("FATAL: no *_from8M templates found"); return 1

    runs = []
    for seed in SEEDS:
        base_run = f"unsup_8M_s{seed}"
        probe = subprocess.run(["aws", "s3", "ls", f"{BUCKET}/{base_run}/encoder/"],
                               capture_output=True, text=True, check=False)
        if "model.safetensors" not in probe.stdout:
            print(f"FATAL: seed-matched base {base_run} has no encoder in S3"); return 1
        for t in twins:
            r = json.loads(json.dumps(t))
            rid = f"{t['run_id']}_s{seed}"
            r["run_id"] = rid
            r["output_dir"] = f"experiments/climb_v2_phase2/{rid}"
            r["backup_s3_uri"] = f"{BUCKET}/{rid}"
            r["evaluation_output_dir"] = f"experiments/climb_v2_phase2/{rid}/moleculenet"
            r["depends_on"] = base_run
            pc = r.setdefault("pretrain_config", {})
            pc["run_id"] = rid
            for sel in (r.get("selection"), pc.get("selection")):
                if isinstance(sel, dict):
                    sel["init_encoder_path"] = f"experiments/climb_v2_phase2/{base_run}/encoder"
                    sel["pretraining_seed"] = seed
            runs.append(r)

    out = {k: v for k, v in m.items() if k != "runs"}
    out["name"] = "climb_v2_phase2_u2s_seeds"
    out["runs"] = runs
    p = Path(a.out); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2))

    shards = [runs[i::a.workers] for i in range(a.workers)]
    for i, sh in enumerate(shards):
        q = p.with_name(p.stem + f"_worker{i}.json")
        q.write_text(json.dumps({**out, "runs": sh}, indent=2))
        print(f"  worker{i}: {len(sh)} runs -> {q}")
    fp = sum((r.get("selection") or {}).get("total_forward_passes") or 0 for r in runs)
    print(f"\n{len(runs)} runs, {fp:,} FP = {fp/755/3600:.2f} GPU-h "
          f"(~{fp/755/3600/a.workers:.2f}h on {a.workers} boxes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
