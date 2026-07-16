"""Split a phase-2 manifest into N per-worker manifests, balanced by compute.

Greedy longest-processing-time bin-packing on each run's total_forward_passes (a good
proxy for wall-clock). Dependencies are honored by CO-LOCATION: a `u2s` run
(depends_on a `unsup_*` ladder run) is pinned to the same worker as its base and
ordered after it, so warm-start encoder paths always exist locally when the run starts.

Non-pretrain anchors (ecfp4, random_baseline) are cheap and replicated onto worker 0.

Usage:
    # stage 1 (ladder + skip): everything with no cross-worker dependency
    python scripts/split_manifest.py --manifest m.json --workers 4 \
        --stages ladder skip --out_dir manifests/stage1
    # stage 2 (u2s): after the ladder encoders exist
    python scripts/split_manifest.py --manifest m.json --workers 4 \
        --stages u2s --out_dir manifests/stage2 --pin_deps
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _fp(run: dict) -> int:
    return int(run.get("pretrain_config", {}).get("selection", {})
               .get("total_forward_passes", 0) or 0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--workers", type=int, required=True)
    p.add_argument("--stages", nargs="+", default=None,
                   help="only include runs whose stage is in this set (anchors always on w0)")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--pin_deps", action="store_true",
                   help="co-locate a u2s run with its depends_on base (same worker, after it)")
    p.add_argument("--with_anchors", action="store_true",
                   help="include ecfp4/random anchors on worker0 (stage 1 only)")
    args = p.parse_args()

    man = json.loads(Path(args.manifest).read_text())
    runs = man["runs"]
    stages = set(args.stages) if args.stages else None

    anchors = [r for r in runs if not r.get("requires_pretrain", False)]
    pre = [r for r in runs if r.get("requires_pretrain", False)]
    if stages is not None:
        pre = [r for r in pre if r.get("stage") in stages]

    # buckets
    buckets = [[] for _ in range(args.workers)]
    load = [0] * args.workers
    worker_of = {}  # run_id -> worker idx (for dep co-location)

    # place independent (non-u2s) runs first, heaviest-first (LPT)
    indep = sorted([r for r in pre if r.get("stage") != "u2s"], key=_fp, reverse=True)
    for r in indep:
        w = load.index(min(load))
        buckets[w].append(r)
        load[w] += _fp(r)
        worker_of[r["run_id"]] = w

    # place u2s runs: pin to the base's worker if present + requested, else LPT
    u2s = sorted([r for r in pre if r.get("stage") == "u2s"], key=_fp, reverse=True)
    for r in u2s:
        dep = r.get("depends_on")
        if args.pin_deps and dep in worker_of:
            w = worker_of[dep]
        else:
            w = load.index(min(load))
        buckets[w].append(r)
        load[w] += _fp(r)
        worker_of[r["run_id"]] = w

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for i in range(args.workers):
        # anchors only on worker 0, and only when explicitly requested (stage 1)
        wruns = (anchors if (i == 0 and args.with_anchors) else []) + buckets[i]
        sub = {k: man[k] for k in ("name", "results_root", "s3_backup_root", "tokenizer_path")}
        sub["runs"] = wruns
        path = out_dir / f"worker{i}.json"
        path.write_text(json.dumps(sub, indent=2))
        gfp = sum(_fp(r) for r in wruns)
        print(f"worker{i}: {len(wruns)} runs, {gfp/1e6:.0f}M forward-passes "
              f"(~{gfp/700/3600:.1f} GPU-h @700 seq/s) -> {path}")


if __name__ == "__main__":
    main()
