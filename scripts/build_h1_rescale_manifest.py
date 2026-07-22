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

The config is CLONED from a generated phase-2 run rather than hand-written. A hand-written
`pretrain_config` containing only {run_id, selection} is accepted by the manifest loader and then
dies inside pretrain_v2 on `cfg["tokenizer_path"]`, because the real config also carries the data
paths, model, training and evaluation blocks. Cloning a known-good run and overriding only the
three fields that define this sweep keeps everything else identical to the rest of the paper.

Usage:
    python scripts/build_h1_rescale_manifest.py --out experiments/climb_v2_h1/manifest.json
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WAVE = "climb_v2_h1"
BUCKET = f"s3://climb-s3-bucket/experiments/{WAVE}"
TEMPLATE = "unsup_2M"          # 2M-FP canonical MLM: the exact shape this sweep varies
FRACTIONS = [("frac0p001", 0.001), ("frac0p01", 0.01), ("frac0p1", 0.1),
             ("frac0p3", 0.3), ("fracfull", None)]
AUGS = ["canonical", "enumerated"]
SEEDS = [0, 1, 2]
FP = 2_000_000


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", default="configs/v2_phase2.yaml")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=3)
    a = ap.parse_args()

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as t:
        full_path = Path(t.name)
    subprocess.run([sys.executable, str(ROOT / "experiment_v2.py"),
                    "--spec", str(ROOT / a.spec), "--output", str(full_path)],
                   check=True, cwd=ROOT, stdout=subprocess.DEVNULL)
    full = json.loads(full_path.read_text())
    by = {r["output_dir"].split("/")[-1]: r for r in full["runs"]}
    if TEMPLATE not in by:
        print(f"FATAL: template run {TEMPLATE!r} not in the generated manifest"); return 1
    tmpl = by[TEMPLATE]

    runs = []
    for seed in SEEDS:
        for aug in AUGS:
            for fk, fv in FRACTIONS:
                rid = f"scaling_{aug}_{fk}_s{seed}"
                r = json.loads(json.dumps(tmpl))          # deep copy of a KNOWN-GOOD config
                r["run_id"] = rid
                r["run_type"] = "unsup_scaling"
                r["output_dir"] = f"experiments/{WAVE}/{rid}"
                r["backup_s3_uri"] = f"{BUCKET}/{rid}"
                r["evaluation_output_dir"] = f"experiments/{WAVE}/{rid}/moleculenet"
                r["pretrain_config"]["run_id"] = rid
                # the three fields that define this sweep; everything else stays as phase 2
                for sel in (r.get("selection"), r["pretrain_config"].get("selection")):
                    if isinstance(sel, dict):
                        sel["augmentation"] = aug
                        sel["unsupervised_subset_fraction"] = fv
                        sel["pretraining_seed"] = seed
                        sel["total_forward_passes"] = FP
                        sel["init_encoder_path"] = None
                        sel["objectives"] = {"mlm": 1.0}
                # top-level mirror that pretrain_v2 reads for the subset fraction
                r["pretrain_config"]["unsupervised_subset_fraction"] = fv
                runs.append(r)

    manifest = {k: full[k] for k in full if k != "runs"}
    manifest["name"] = WAVE
    manifest["results_root"] = f"experiments/{WAVE}"
    manifest["s3_backup_root"] = BUCKET
    manifest["runs"] = runs

    p = Path(a.out); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(manifest, indent=2))
    full_path.unlink(missing_ok=True)

    # Assert the cloned config really is complete: this is the exact failure that wasted a launch.
    need = ["tokenizer_path", "unsupervised_data_paths", "model", "training", "evaluation"]
    miss = [k for k in need if k not in runs[0]["pretrain_config"]]
    if miss:
        print(f"FATAL: cloned pretrain_config is missing {miss}"); return 1
    print(f"cloned config from {TEMPLATE}: all of {need} present")

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
