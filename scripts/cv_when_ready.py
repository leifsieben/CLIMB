"""Wait for wave runs to verify complete, then run scaffold 5-fold CV on each as it lands.

The box-side pass emits only the single-split `moleculenet/` eval; the DEFAULT panel is scaffold
5-fold CV, which has to be produced locally. This watcher removes the "come back in three hours
and remember to start it" step.

Design notes:

  * **Pipelined, not barriered.** Each run is CV'd the moment IT verifies, rather than waiting
    for all of them, so CV overlaps the remaining training instead of adding an hour after it.
  * **Completion means `verified.json`**, the same marker the launcher and shutdown gate use.
    Never `path.exists()` on an encoder or a summary -- a truncated run produces those too, and
    trusting them is what silently poisoned an earlier wave.
  * **Idempotent.** A run whose CV summary already exists is skipped, so the watcher can be
    killed and restarted without redoing work or double-writing.
  * **Bounded.** It gives up after --max-hours and reports exactly which runs never verified,
    rather than polling forever against runs that died.

Usage:
    python scripts/cv_when_ready.py --wave climb_v2_lrsweep --runs a,b,c [--max-hours 8]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

BUCKET = "s3://climb-s3-bucket/experiments"
PY = ".venv_sanity/bin/python"
TOK = "figure_data/_tokenizer"
# Same 7-task core as cv_eval_local.py -- the panels must be directly comparable.
CORE = ["ESOL", "Lipophilicity", "QM7", "BBBP", "BACE", "Tox21", "HIV"]


def _s3_has(uri: str) -> bool:
    r = subprocess.run(["aws", "s3", "ls", uri], capture_output=True, text=True, check=False)
    return r.returncode == 0 and bool(r.stdout.strip())


def _verified(wave: str, run: str) -> bool:
    return _s3_has(f"{BUCKET}/{wave}/{run}/verified.json")


def _cv_done(fd: Path) -> bool:
    return (fd / "moleculenet_cv" / "suite_summary.json").exists()


def _log(msg: str) -> None:
    print(f"[cv-watch {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _run_cv(wave: str, run: str) -> str:
    """-> 'ok' | 'retry' | 'failed'."""
    fd = Path("figure_data") / wave / run
    enc = fd / "encoder"
    enc.mkdir(parents=True, exist_ok=True)
    subprocess.run(["aws", "s3", "sync", f"{BUCKET}/{wave}/{run}/encoder", str(enc)],
                   check=False, capture_output=True)
    if not (enc / "model.safetensors").exists() and not (enc / "pytorch_model.bin").exists():
        # NOT terminal. verified.json and the encoder are pushed by the same periodic sync, so a
        # run can be observed verified moments before its weights land in S3. Treating that race
        # as failure silently drops the run from CV forever -- which is exactly what happened to
        # two lrsweep runs. Retry instead, and only give up after RETRY_LIMIT attempts.
        _log(f"{run}: verified but weights not in S3 yet — will retry")
        return "retry"

    # DeepChem's featurization cache collides across concurrent/repeat evals.
    for d in glob.glob(os.path.join(tempfile.gettempdir(), "*-featurized")):
        shutil.rmtree(d, ignore_errors=True)

    cmd = [PY, "eval_v2.py", "--encoder", str(enc), "--tokenizer", TOK,
           "--output_dir", str(fd / "moleculenet_cv"),
           "--cv_folds", "5", "--head_seeds", "0", "1", "2",
           "--pool", "mean", "--standardize", "zscore", "--head", "mlp",
           "--max_length", "256", "--datasets", *CORE]
    _log(f"{run}: starting CV")
    r = subprocess.run(cmd, capture_output=True, text=True)
    ok = _cv_done(fd)
    _log(f"{run}: CV {'OK' if ok else 'FAILED'}")
    if not ok:
        _log(f"  stdout tail: {r.stdout[-500:]}")
        _log(f"  stderr tail: {r.stderr[-500:]}")
    return "ok" if ok else "failed"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wave", required=True)
    ap.add_argument("--runs", required=True, help="comma-separated run ids")
    ap.add_argument("--poll-seconds", type=int, default=300)
    ap.add_argument("--max-hours", type=float, default=8.0)
    a = ap.parse_args()

    pending = [r.strip() for r in a.runs.split(",") if r.strip()]
    done, failed = [], []
    retries: dict[str, int] = {}
    RETRY_LIMIT = 12          # ~1h at the default 300s poll, far beyond a 10-min sync cycle
    deadline = time.time() + a.max_hours * 3600
    _log(f"watching {len(pending)} runs in {a.wave}; poll {a.poll_seconds}s, "
         f"giving up after {a.max_hours}h")

    while pending and time.time() < deadline:
        ready = []
        for run in list(pending):
            fd = Path("figure_data") / a.wave / run
            if _cv_done(fd):                      # idempotent restart
                _log(f"{run}: CV already present — skipping")
                pending.remove(run)
                done.append(run)
            elif _verified(a.wave, run):
                ready.append(run)

        for run in ready:
            outcome = _run_cv(a.wave, run)
            if outcome == "retry":
                retries[run] = retries.get(run, 0) + 1
                if retries[run] >= RETRY_LIMIT:
                    _log(f"{run}: weights still absent after {RETRY_LIMIT} tries — giving up")
                    pending.remove(run); failed.append(run)
                continue          # stays pending; re-checked next poll
            pending.remove(run)
            (done if outcome == "ok" else failed).append(run)

        if pending:
            _log(f"waiting on {len(pending)}: {', '.join(pending)}")
            time.sleep(a.poll_seconds)

    _log(f"FINISHED — cv_ok={len(done)} cv_failed={len(failed)} never_verified={len(pending)}")
    if failed:
        _log(f"  CV failed: {', '.join(failed)}")
    if pending:
        _log(f"  never verified within {a.max_hours}h: {', '.join(pending)}")
    return 0 if not failed and not pending else 1


if __name__ == "__main__":
    raise SystemExit(main())
