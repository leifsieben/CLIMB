"""Backfill `verified.json` markers for runs that ARE complete but were never marked.

Why this exists: the worker's shutdown gate (older copies of `phase2_worker.sh`, including the
ones currently executing on live boxes) accepts ONLY a local `verified.json`. The launcher's
completion test is broader — it also accepts an S3 marker, an anchor exemption, and
achieved-FP >= 98% of budget. Anything complete via those broader routes is reported "missing"
by the old gate forever, so the box never self-terminates and bills indefinitely while emitting
a plausible-looking INCOMPLETE alert.

`launch_v2_wave.py` has since been fixed to share one completion test, but a bash script cannot
be safely swapped underneath a running process. So for boxes already mid-wave, this backfills the
markers their old gate is looking for — using the SAME `_is_complete` logic, so it can only mark
runs that genuinely are complete.

Usage (on the box):
    python scripts/backfill_verified.py --manifest <worker_manifest.json>        # dry run
    python scripts/backfill_verified.py --manifest <worker_manifest.json> --write
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from launch_v2_wave import _budget_fp, _final_fp_local, _final_fp_s3, _is_complete  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--write", action="store_true", help="actually write markers (default: dry run)")
    a = ap.parse_args()

    runs = json.loads(Path(a.manifest).read_text())["runs"]
    wrote = skipped = incomplete = 0

    for run in runs:
        rid = run["run_id"]
        run_dir = Path(run["output_dir"])
        marker = run_dir / "verified.json"
        if marker.exists():
            skipped += 1
            continue
        if not _is_complete(run):
            print(f"  INCOMPLETE  {rid} — leaving unmarked (this is a real gap, not a marker bug)")
            incomplete += 1
            continue

        # Record how completion was established so the marker is auditable rather than a bare flag.
        budget = _budget_fp(run)
        fp = max(_final_fp_local(run_dir), _final_fp_s3(run))
        if budget:
            route, frac = "achieved_fp", (fp / budget if budget else 0.0)
        else:
            route, frac = "anchor_or_marker", 1.0
        if not run_dir.exists():
            print(f"  NO LOCAL DIR {rid} — complete per S3 but nothing to mark locally")
            continue

        payload = {"achieved": fp, "budget": budget, "fraction": round(frac, 4),
                   "verified": True, "backfilled": True, "route": route}
        if a.write:
            marker.write_text(json.dumps(payload, indent=2))
            print(f"  WROTE       {rid}  ({route}, {frac:.1%})")
        else:
            print(f"  would write {rid}  ({route}, {frac:.1%})")
        wrote += 1

    verb = "wrote" if a.write else "would write"
    print(f"\n{verb} {wrote} marker(s); {skipped} already marked; {incomplete} genuinely incomplete")
    if not a.write and wrote:
        print("re-run with --write to apply")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
