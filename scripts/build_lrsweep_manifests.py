"""Rebuild the SFT-LR sweep (E3 / Fig E2) manifests so the wave can actually run.

All 8 original lrsweep runs died in 13-990s having written only config.yaml and
run_status.json -- no metrics, no encoder. Cause: every run warm-starts from
`init_encoder_path: experiments/climb_v2/unsup_only_seed0/encoder`, and NO climb_v2 round-1 run
has a surviving encoder anywhere (that wave kept metrics and evals but never its weights). A
missing warm-start base kills the run at startup, which is why the failure looked instant and
uniform. The runs were never "trained but unevaluated" -- there is nothing to evaluate.

Two repairs, both required:

  1. Re-point the warm-start base at `climb_v2_phase2/unsup_2M`. That run is the closest
     available match to the original intent: MLM-only, and 1,999,872 achieved forward passes
     against the original base's 1,999,872 -- an identical budget. It is also leakage-deduped,
     which the round-1 wave was not.

  2. Hand the result to finalize_manifest.py, which injects `descriptor_precompute_dir` into
     every MTR arm. The 4 "dense" arms are pure MTR and the manifest carries no precompute dir,
     so they would compute descriptors on the fly at ~6x slowdown -- the exact condition that
     combined with a wall-clock cap to truncate runs in the earlier incident.

Output: two worker manifests of 4 runs each, balanced 2 dense + 2 sparse so both boxes exercise
both objectives (a config fault in one objective then shows up on both boxes within the hour,
rather than being hidden on a single box until the end).

Usage:
    python scripts/build_lrsweep_manifests.py <in_manifest.json> --outdir <dir>
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

NEW_BASE = "experiments/climb_v2_phase2/unsup_2M/encoder"
BUCKET = "s3://climb-s3-bucket/"


def _set_init_encoder(run: dict, path: str) -> None:
    """The path is duplicated in `selection` and `pretrain_config.selection`; the trainer reads
    the latter, the launcher reports the former. Setting only one leaves a manifest that looks
    corrected but still dies at startup."""
    for sel in (run.get("selection"), (run.get("pretrain_config") or {}).get("selection")):
        if isinstance(sel, dict) and "init_encoder_path" in sel:
            sel["init_encoder_path"] = path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("manifest")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--base", default=NEW_BASE)
    a = ap.parse_args()

    # Refuse to build a manifest around a base that is not actually there -- that is the whole
    # bug being fixed.
    probe = subprocess.run(["aws", "s3", "ls", f"{BUCKET}{a.base}/"],
                           capture_output=True, text=True, check=False)
    if probe.returncode != 0 or "model.safetensors" not in probe.stdout:
        print(f"FATAL: warm-start base {BUCKET}{a.base}/ has no model.safetensors", file=sys.stderr)
        return 1
    print(f"warm-start base OK: {BUCKET}{a.base}/")

    m = json.loads(Path(a.manifest).read_text())
    runs = m["runs"]
    for r in runs:
        _set_init_encoder(r, a.base)

    outdir = Path(a.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    repaired = outdir / "lrsweep_repaired.json"
    repaired.write_text(json.dumps(m, indent=2))

    # finalize_manifest.py is the single choke point that injects the precompute dir and orders
    # short-first; it exits non-zero if anything is off.
    fin = outdir / "lrsweep_final.json"
    r = subprocess.run([sys.executable, "scripts/finalize_manifest.py", str(repaired),
                        "--out", str(fin)], check=False)
    if r.returncode != 0:
        print("FATAL: finalize_manifest.py rejected the manifest", file=sys.stderr)
        return 1

    final = json.loads(fin.read_text())
    runs = final["runs"]

    # Balance 2 dense + 2 sparse per worker rather than splitting the ordered list in half.
    dense = [r for r in runs if "dense" in r["run_id"]]
    sparse = [r for r in runs if "dense" not in r["run_id"]]
    workers = [dense[0::2] + sparse[0::2], dense[1::2] + sparse[1::2]]

    owned = set()
    for i, w in enumerate(workers):
        for r in w:
            rid = r["output_dir"]
            if rid in owned:                      # one owner per run, always
                print(f"FATAL: {rid} claimed twice", file=sys.stderr)
                return 1
            owned.add(rid)
        out = outdir / f"lrsweep_worker{i}.json"
        out.write_text(json.dumps({**final, "runs": w}, indent=2))
        ids = [r["run_id"] for r in w]
        print(f"  worker{i}: {len(w)} runs -> {out}")
        for x in ids:
            print(f"      {x}")

    print(f"\ntotal {len(owned)} runs across {len(workers)} workers, no overlap")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
