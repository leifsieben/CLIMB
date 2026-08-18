"""Rebuild the SFT-source ablation (H3 / Fig C1 + transfer matrix J1) as a DEDUPED wave.

Why re-run rather than reuse: the original `climb_v2_ablation` was run pre-dedup. Its configs
carry no `eval_blocklist_path`, so the assay arms (seq_pcba, seq_l1000, seq_sparse_all,
seq_dense_plus_sparse) trained on molecules that reappear in the evaluation sets -- they are
carried in the figures today with a leakage dagger. Its encoders are also gone (no
`encoder/model.safetensors` survives anywhere under climb_v2_ablation), so the arms cannot be
re-evaluated in place: single-split numbers frozen at a protocol the rest of the paper no longer
uses, with a known leak. Re-running under the current setup fixes both at once.

Three repairs, all mandatory:

  1. `init_encoder_path` points at `experiments/climb_v2/unsup_only_seed0/encoder`, which does
     not exist -- the same dead warm-start base that killed all 8 lrsweep runs at startup. It is
     re-pointed at `climb_v2_phase2/unsup_2M/encoder`: MLM-only, 1,999,872 achieved fp against
     the original base's 1,999,872 (identical budget), and itself deduped.
  2. `eval_blocklist_path` is injected into every arm, which is what actually removes the leak.
     Omitting it would faithfully reproduce the problem being fixed.
  3. `descriptor_precompute_dir` via finalize_manifest.py, or the MTR-bearing arms (seq_mtr,
     seq_dense_plus_sparse) recompute descriptors on the fly at ~6x slowdown.

Writes to a NEW prefix (`climb_v2_ablation_dedup`) rather than over the original, so the
pre-dedup numbers stay available for a before/after leakage comparison -- which is itself
evidence for H6 -- instead of being silently replaced.

Usage:
    python scripts/build_ablation_dedup_manifests.py --outdir experiments/climb_v2_ablation_dedup/manifests
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

SRC = "experiments/climb_v2_ablation/manifest.json"  # AUDIT-OK: superseded-root — this IS the pre-dedup wave, read as the input to rebuild it
NEW_WAVE = "climb_v2_ablation_dedup"
NEW_BASE = "experiments/climb_v2_phase2/unsup_2M/encoder"
BLOCKLIST = "s3://climb-s3-bucket/configs/eval_blocklist.json"
BUCKET = "s3://climb-s3-bucket/"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--pretrain-only", action="store_true",
                    help="drop the eval-only anchors (phase-2 already has equivalents)")
    a = ap.parse_args()

    probe = subprocess.run(["aws", "s3", "ls", f"{BUCKET}{NEW_BASE}/"],
                           capture_output=True, text=True, check=False)
    if probe.returncode != 0 or "model.safetensors" not in probe.stdout:
        print(f"FATAL: warm-start base {BUCKET}{NEW_BASE}/ has no weights", file=sys.stderr)
        return 1
    print(f"warm-start base OK: {BUCKET}{NEW_BASE}/")

    m = json.loads(Path(SRC).read_text())
    m["name"] = NEW_WAVE
    m["results_root"] = f"experiments/{NEW_WAVE}"
    m["s3_backup_root"] = f"{BUCKET}experiments/{NEW_WAVE}"

    runs = []
    for r in m["runs"]:
        if a.pretrain_only and not r.get("requires_pretrain"):
            continue
        rid = r["run_id"]
        r["output_dir"] = f"experiments/{NEW_WAVE}/{rid}"
        r["backup_s3_uri"] = f"{BUCKET}experiments/{NEW_WAVE}/{rid}"
        r["evaluation_output_dir"] = f"experiments/{NEW_WAVE}/{rid}/moleculenet"
        pc = r.setdefault("pretrain_config", {})
        pc["run_id"] = rid
        if r.get("requires_pretrain"):
            # (1) live warm-start base, in BOTH copies of selection (trainer reads pretrain_config)
            for sel in (r.get("selection"), pc.get("selection")):
                if isinstance(sel, dict) and "init_encoder_path" in sel:
                    sel["init_encoder_path"] = NEW_BASE
            # (2) the actual de-leaking step
            pc["eval_blocklist_path"] = BLOCKLIST
        runs.append(r)
    m["runs"] = runs

    outdir = Path(a.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    staged = outdir / "ablation_dedup_repaired.json"
    staged.write_text(json.dumps(m, indent=2))

    # (3) precompute wiring + short-first ordering; refuses on any inconsistency
    final = outdir / "ablation_dedup_final.json"
    if subprocess.run([sys.executable, "scripts/finalize_manifest.py", str(staged),
                       "--out", str(final)], check=False).returncode != 0:
        print("FATAL: finalize_manifest.py rejected the manifest", file=sys.stderr)
        return 1

    fm = json.loads(final.read_text())
    rs = fm["runs"]
    shards = [rs[i::a.workers] for i in range(a.workers)]
    owned = set()
    for i, sh in enumerate(shards):
        for r in sh:
            if r["output_dir"] in owned:
                print(f"FATAL: {r['output_dir']} claimed twice", file=sys.stderr)
                return 1
            owned.add(r["output_dir"])
        p = outdir / f"ablation_dedup_worker{i}.json"
        p.write_text(json.dumps({**fm, "runs": sh}, indent=2))
        print(f"  worker{i}: {len(sh)} runs -> {p}")
        for r in sh:
            pre = "pretrain" if r.get("requires_pretrain") else "eval-only"
            print(f"      {r['run_id']:<24} {pre}")

    fp = sum((r.get("selection") or {}).get("total_forward_passes") or 0
             for r in rs if r.get("requires_pretrain"))
    print(f"\n{len(owned)} runs, no overlap | {fp:,} pretrain FP "
          f"= {fp/755/3600:.2f} GPU-h total, ~{fp/755/3600/a.workers:.2f}h wall on {a.workers} boxes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
