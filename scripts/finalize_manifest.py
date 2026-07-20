"""Finalize a launch manifest so it CANNOT reproduce the two failures that cost us:

  1. Dense/MTR runs missing `descriptor_precompute_dir` -> descriptors computed on
     the fly -> ~6x slowdown -> the (old) wall-clock cap truncated them.
  2. No deterministic ordering -> long runs ran before short ones, so a truncation
     wiped out whole experiments before any complete short datapoint existed.

This is the single choke point every manifest passes through before launch. It:
  - injects the precompute dir into every run whose objectives include MTR,
  - orders runs SHORT-FIRST (eval-only anchors, then ascending FP budget), so
    complete, trustworthy data lands as early as possible,
  - validates and prints a table; refuses (non-zero exit) if anything is off.

Usage:
  python scripts/finalize_manifest.py <in_manifest.json> [--out <out.json>]
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

PRECOMPUTE_DIR = "s3://climb-s3-bucket/tokenized_sources/pubchem_descriptors/"


def _budget(run: dict) -> int:
    sel = run.get("selection", {}) or {}
    pc_sel = (run.get("pretrain_config", {}) or {}).get("selection", {}) or {}
    return int(sel.get("total_forward_passes") or pc_sel.get("total_forward_passes") or 0)


def _uses_mtr(run: dict) -> bool:
    pc = run.get("pretrain_config", {}) or {}
    sel = pc.get("selection", {}) or {}
    objs = sel.get("objectives") or {}
    if isinstance(objs, dict) and float(objs.get("mtr", 0) or 0) > 0:
        return True
    # fallback heuristic on the id (dense = pure MTR, mixed = descriptors + minimol)
    rid = run.get("run_id", "").lower()
    return ("dense" in rid) or ("mixed" in rid) or ("mtr" in rid)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("manifest")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    m = json.load(open(args.manifest))
    runs = m["runs"]

    wired, problems = 0, []
    for r in runs:
        pc = r.setdefault("pretrain_config", {})
        if _uses_mtr(r):
            pc["descriptor_precompute_dir"] = PRECOMPUTE_DIR
            wired += 1
            # sanity: an MTR run with no descriptor stats path can't normalize targets
            if not pc.get("descriptor_stats_path"):
                problems.append(f"{r['run_id']}: MTR run missing descriptor_stats_path")

    # order: eval-only anchors first (fast complete data), then ascending FP budget
    def _key(r):
        rt = r.get("run_type", "")
        anchor = 0 if rt in ("ecfp4_anchor", "random_baseline") else 1
        return (anchor, _budget(r), r.get("run_id", ""))
    runs.sort(key=_key)
    m["runs"] = runs

    out = args.out or args.manifest
    json.dump(m, open(out, "w"), indent=2)

    print(f"[finalize] {len(runs)} runs | MTR runs wired with precompute: {wired}")
    print(f"[finalize] {'-'*66}")
    print(f"[finalize] {'ORDER':<5} {'RUN_ID':<34} {'BUDGET_FP':>12} {'MTR':>4}")
    for i, r in enumerate(runs):
        print(f"[finalize] {i:<5} {r['run_id']:<34} {_budget(r):>12,} {'yes' if _uses_mtr(r) else '-':>4}")

    if problems:
        print("\n[finalize] PROBLEMS:")
        for p in problems:
            print("  - " + p)
        return 1
    print(f"[finalize] wrote {out} — OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
