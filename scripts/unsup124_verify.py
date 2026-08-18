#!/usr/bin/env python3
"""Prove the unsup124 runs are complete. Exit 0 only if EVERY run passes EVERY check.

Completion is derived from ACHIEVED FORWARD PASSES, never from a file existing. This project
has already been burned by the opposite: jobs killed early still wrote their summary file, so
truncated encoders looked finished and downstream analysis was built on them.

Also reads the artifacts BACK FROM S3 and parses them, because "aws s3 sync exited 0" is not
evidence that a readable encoder is in the bucket.
"""
import io
import json
import subprocess
import sys
import tempfile
from pathlib import Path

TOL = 0.98
FAIL = []


def check(rid, name, ok, detail=""):
    print(f"  {'ok    ' if ok else 'FAIL  '} [{rid}] {name}{(' :: ' + detail) if detail else ''}")
    if not ok:
        FAIL.append(f"{rid}:{name}")
    return ok


def s3_text(uri):
    r = subprocess.run(["aws", "s3", "cp", uri, "-"], capture_output=True, text=True)
    return r.stdout if r.returncode == 0 else None


def main():
    man = json.load(open(sys.argv[1]))
    for run in man["runs"]:
        rid = run["run_id"]
        s3 = run["backup_s3_uri"]
        budget = run["selection"]["total_forward_passes"]
        print(f"--- {rid} (budget {budget:,} FP) ---")

        # ---- 1. achieved forward passes, read back FROM S3
        mtxt = s3_text(f"{s3}/metrics.jsonl")
        if not check(rid, "metrics.jsonl readable from S3", bool(mtxt)):
            continue
        last = None
        for line in mtxt.splitlines():
            line = line.strip()
            if line:
                try:
                    last = json.loads(line)
                except Exception:
                    pass
        fp = int((last or {}).get("forward_passes_seen", 0))
        frac = fp / budget if budget else 0
        check(rid, f"achieved FP >= {TOL:.0%} of budget", frac >= TOL,
              f"{fp:,}/{budget:,} = {frac:.4f}")

        # ---- 2. verified.json agrees (and is not a stale/hand-written marker)
        vtxt = s3_text(f"{s3}/verified.json")
        if check(rid, "verified.json readable from S3", bool(vtxt)):
            try:
                v = json.loads(vtxt)
                check(rid, "verified.json agrees with metrics",
                      int(v.get("final_fp", -1)) == fp and int(v.get("budget_fp", -1)) == budget,
                      f"marker final_fp={v.get('final_fp')} budget_fp={v.get('budget_fp')}")
            except Exception as e:
                check(rid, "verified.json parses", False, repr(e)[:80])

        # ---- 3. encoder weights readable back from S3 and structurally sound
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "model.safetensors"
            r = subprocess.run(["aws", "s3", "cp", f"{s3}/encoder/model.safetensors", str(p),
                                "--only-show-errors"], capture_output=True, text=True)
            if check(rid, "encoder downloadable from S3", r.returncode == 0, r.stderr[:100]):
                try:
                    from safetensors.torch import load_file
                    sd = load_file(str(p))
                    n = sum(t.numel() for t in sd.values())
                    finite = all(bool(t.float().isfinite().all()) for t in list(sd.values())[:40])
                    check(rid, "encoder loads, params finite", n > 1e6 and finite,
                          f"{len(sd)} tensors, {n/1e6:.1f}M params")
                except Exception as e:
                    check(rid, "encoder loads", False, repr(e)[:100])
            cfg = s3_text(f"{s3}/encoder/config.json")
            check(rid, "encoder config.json readable", bool(cfg))

        # ---- 4. both eval schemes present, non-trivial, and covering all 7 tasks
        for scheme, minrows in (("moleculenet", 100), ("moleculenet_cv", 100)):
            stxt = s3_text(f"{s3}/{scheme}/suite_summary.json")
            if not check(rid, f"{scheme}/suite_summary.json readable", bool(stxt)):
                continue
            ptxt = s3_text(f"{s3}/{scheme}/test_predictions.csv")
            if check(rid, f"{scheme}/test_predictions.csv readable", bool(ptxt)):
                import csv
                rows = list(csv.DictReader(io.StringIO(ptxt)))
                ds = sorted({r["dataset"] for r in rows})
                check(rid, f"{scheme} has rows", len(rows) >= minrows, f"{len(rows):,} rows")
                check(rid, f"{scheme} covers 7 tasks", len(ds) == 7, ",".join(ds))

    print()
    if FAIL:
        print(f"VERIFY FAIL ({len(FAIL)}): {FAIL}")
        return 1
    print("VERIFY PASS - all runs complete and artifacts readable from S3")
    return 0


if __name__ == "__main__":
    sys.exit(main())
