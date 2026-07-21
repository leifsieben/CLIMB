"""Build the missing u2s_*_from48M rung of the unsup->sup ladder.

The ladder currently has from2M/from8M/from24M for all five recipes but no from48M rung, because
those runs warm-start off `unsup_48M`, which is still training. Derives each entry from the
existing from24M twin so the only differences are the ones that should differ: run id, output
paths, and the warm-start base.

Intended to be chained on the box already training unsup_48M, so the encoder is local and fresh
the moment the wave starts -- and so the box does useful work overnight instead of self-stopping
the minute unsup_48M verifies.

Usage:
    python scripts/build_u2s_from48M_manifest.py --out experiments/climb_v2_phase2/manifests/u2s_from48M.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

SRC = "experiments/climb_v2_phase2/manifest_wave1.json"
BASE_RUN = "unsup_48M"
BASE_ENC = f"experiments/climb_v2_phase2/{BASE_RUN}/encoder"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    m = json.loads(Path(SRC).read_text())
    twins = [r for r in m["runs"] if r["run_id"].endswith("_from24M")]
    if not twins:
        print("FATAL: no *_from24M template runs found"); return 1

    runs = []
    for t in twins:
        r = json.loads(json.dumps(t))                 # deep copy
        rid = r["run_id"].replace("_from24M", "_from48M")
        r["run_id"] = rid
        r["output_dir"] = f"experiments/climb_v2_phase2/{rid}"
        r["backup_s3_uri"] = f"s3://climb-s3-bucket/experiments/climb_v2_phase2/{rid}"
        r["evaluation_output_dir"] = f"experiments/climb_v2_phase2/{rid}/moleculenet"
        r["depends_on"] = BASE_RUN
        pc = r.setdefault("pretrain_config", {})
        pc["run_id"] = rid
        # the one substantive change: warm-start from the 48M MLM base instead of the 24M one
        for sel in (r.get("selection"), pc.get("selection")):
            if isinstance(sel, dict):
                sel["init_encoder_path"] = BASE_ENC
        runs.append(r)

    out = {k: v for k, v in m.items() if k != "runs"}
    out["name"] = "climb_v2_phase2_u2s_from48M"
    out["runs"] = runs

    p = Path(a.out); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2))
    fp = sum((r.get("selection") or {}).get("total_forward_passes") or 0 for r in runs)
    print(f"wrote {p}: {len(runs)} runs, {fp:,} FP = {fp/755/3600:.2f} GPU-h")
    for r in runs:
        print(f"   {r['run_id']:<32} base={r['selection']['init_encoder_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
