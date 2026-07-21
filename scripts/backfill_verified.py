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

Three properties this file is careful about:

1. **Same schema as the real marker.** `_write_verified_marker` emits run_id/budget_fp/final_fp/
   fraction/verified_at_utc. A backfilled marker that invented its own key names would be read
   as a malformed marker by anything that inspects the fields (only `_is_complete` gets away
   with checking mere existence). Provenance keys are ADDED, not substituted.

2. **The achieved-FP number must not come from a corrupted source.** metrics.jsonl in S3 was
   demonstrably reverted to truncated copies by the cross-box sync bug (see phase2_worker.sh),
   so a run holding a perfectly good S3 `verified.json` could otherwise be re-marked from
   clobbered metrics as "verified, 40%" — a self-contradictory record. An existing marker is
   therefore the highest authority; metrics are only consulted when no marker exists anywhere.

3. **It refuses to mark truncated work.** Writing a completion marker for a run that never
   reached its budget is the original sin this whole harness exists to prevent, so a marker
   below the completion threshold is never written even if `_is_complete` somehow returned true.

Usage:
    python scripts/backfill_verified.py --manifest <worker_manifest.json>            # dry run
    python scripts/backfill_verified.py --manifest <worker_manifest.json> --write
    python scripts/backfill_verified.py --manifest <m.json> --write --s3   # also upload marker
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from launch_v2_wave import (  # noqa: E402
    _budget_fp, _final_fp_local, _final_fp_s3, _is_complete, _path_exists_s3,
)

COMPLETE_FRAC = 0.98
ANCHOR_TYPES = ("ecfp4_anchor", "random_baseline")


def _load_runs(path: str) -> list:
    m = json.loads(Path(path).read_text())
    return m["runs"] if isinstance(m, dict) and "runs" in m else m


def _existing_marker(run: dict, run_dir: Path) -> dict | None:
    """The authoritative completion record, if one already exists locally or in S3."""
    local = run_dir / "verified.json"
    if local.exists():
        try:
            return json.loads(local.read_text())
        except Exception:
            pass
    uri = run["backup_s3_uri"].rstrip("/") + "/verified.json"
    try:
        out = subprocess.run(["aws", "s3", "cp", uri, "-"], capture_output=True,
                             text=True, timeout=60, check=False)
        if out.returncode == 0:
            return json.loads(out.stdout)
    except Exception:
        pass
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--write", action="store_true", help="actually write markers (default: dry run)")
    ap.add_argument("--s3", action="store_true", help="also upload the marker to the run's S3 prefix")
    a = ap.parse_args()

    runs = _load_runs(a.manifest)
    wrote = skipped = incomplete = refused = 0

    for run in runs:
        # The run's identity on disk/S3 is its output_dir, not run_id: seed manifests reuse the
        # base run_id ("unsup_8M") for a run that actually lives at "unsup_8M_s1".
        run_dir = Path(run["output_dir"])
        rid = os.path.basename(str(run_dir).rstrip("/"))

        s3_marker_uri = run["backup_s3_uri"].rstrip("/") + "/verified.json"
        need_local = run_dir.exists() and not (run_dir / "verified.json").exists()
        need_s3 = a.s3 and not _path_exists_s3(s3_marker_uri)
        if not need_local and not need_s3:
            # Already marked everywhere we would write. Never rewrite an existing marker: it
            # would replace a genuine completion record (and its original verified_at_utc)
            # with a backfilled one, discarding the stronger provenance.
            skipped += 1
            continue
        if not _is_complete(run):
            print(f"  INCOMPLETE  {rid} — leaving unmarked (this is a real gap, not a marker bug)")
            incomplete += 1
            continue

        is_anchor = run.get("run_type") in ANCHOR_TYPES
        prior = _existing_marker(run, run_dir)

        if is_anchor:
            # Anchors carry no `selection` block and no FP budget at all; they are complete by
            # the anchor route (an evaluation suite exists), not by reaching a training budget.
            budget, fp, frac, route = 0, 0, 1.0, "anchor"
        elif prior and prior.get("final_fp"):
            # Highest authority: a genuine marker already recorded the achieved work. Never
            # recompute it from metrics.jsonl, which the sync bug could have reverted.
            budget = int(prior.get("budget_fp") or _budget_fp(run))
            fp = int(prior["final_fp"])
            frac = fp / budget if budget else 0.0
            route = "existing_marker"
        else:
            try:
                budget = _budget_fp(run)
            except (KeyError, TypeError):
                print(f"  SKIP        {rid} — no FP budget resolvable and not an anchor")
                continue
            fp = max(_final_fp_local(run_dir), _final_fp_s3(run))
            frac = fp / budget if budget else 0.0
            route = "achieved_fp"

        # Never certify truncated work — the failure mode this harness exists to prevent.
        if not is_anchor and frac < COMPLETE_FRAC:
            print(f"  REFUSED     {rid} — _is_complete said done but achieved only "
                  f"{fp}/{budget} ({frac:.1%}); NOT marking. Investigate (likely clobbered "
                  f"metrics.jsonl, or a genuinely truncated run).")
            refused += 1
            continue

        payload = {
            "run_id": run["run_id"],
            "budget_fp": budget,
            "final_fp": fp,
            "fraction": round(frac, 4),
            "verified_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "backfilled": True,
            "route": route,
        }

        # A run whose box has since been torn down (or whose dir was moved out of the synced
        # tree) has no local dir, but its completion is still established and still worth
        # publishing — the S3 marker is what makes completion immune to a corrupted
        # metrics.jsonl, which is precisely how the false "INCOMPLETE" alerts arose.
        targets = []
        if need_local:
            targets.append("local")
        if need_s3:
            targets.append("s3")
        if not targets:
            print(f"  NO LOCAL DIR {rid} — complete per S3; re-run with --s3 to publish the marker")
            continue

        if a.write:
            note = []
            if "local" in targets:
                (run_dir / "verified.json").write_text(json.dumps(payload, indent=2))
                note.append("local")
            if "s3" in targets:
                uri = run["backup_s3_uri"].rstrip("/") + "/verified.json"
                tmp = Path(os.environ.get("TMPDIR", "/tmp")) / f"verified_{rid}.json"
                tmp.write_text(json.dumps(payload, indent=2))
                r = subprocess.run(["aws", "s3", "cp", str(tmp), uri],
                                   capture_output=True, text=True, check=False)
                tmp.unlink(missing_ok=True)
                note.append("s3" if r.returncode == 0 else
                            f"S3 FAILED: {r.stderr.strip()[:80]}")
            print(f"  WROTE       {rid}  ({route}, {frac:.1%}) [{'+'.join(note)}]")
        else:
            print(f"  would write {rid}  ({route}, {frac:.1%}) [{'+'.join(targets)}]")
        wrote += 1

    verb = "wrote" if a.write else "would write"
    print(f"\n{verb} {wrote} marker(s); {skipped} already marked; "
          f"{incomplete} genuinely incomplete; {refused} refused (complete-but-truncated)")
    if not a.write and wrote:
        print("re-run with --write to apply")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
