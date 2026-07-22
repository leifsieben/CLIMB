"""Re-evaluate every local phase-2 encoder on the full 7-task core, both eval schemes.

Two gaps this closes:
  * HIV and Lipophilicity are absent from most runs' evals -- 24/37 single-split and 41/43 CV
    summaries are 5-task. HIV is 41k molecules, an order of magnitude larger than BBBP/BACE, so
    it is the single best-resolved task available and it is missing from nearly every panel.
  * The DEFAULT panel is now the DeepChem scaffold split, so the 7-task coverage is needed on
    the SINGLE-SPLIT evals, not only on CV.

Order is headline-first: the A1 matched-budget panel (anchors + the 8M rung) is produced before
the rest of the ladder, so the main figure is complete early rather than after every run.

Idempotent: a run whose summary already contains HIV for a given scheme is skipped, so this can
be interrupted and restarted freely.

Usage:
    python scripts/reeval_7task.py [--schemes single cv] [--limit N] [--dry-run]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

PY = ".venv_sanity/bin/python"
TOK = "figure_data/_tokenizer"
CORE = ["ESOL", "Lipophilicity", "QM7", "BBBP", "BACE", "Tox21", "HIV"]
WAVE = Path("figure_data/climb_v2_phase2")
S3_WAVE = "s3://climb-s3-bucket/experiments/climb_v2_phase2"
MATCHED = "8M"


def log(m):
    print(f"[reeval {time.strftime('%H:%M:%S')}] {m}", flush=True)


def has_hiv(run: Path, sub: str) -> bool:
    """Does this summary carry the HEADLINE HIV metric?

    Not `startswith("HIV")`: 21 runs scored by an older eval_v2 have HIV_MEAN (plain ROC-AUC)
    but no HIV_nef1_MEAN, and the loose check called them done -- so they were never re-queued
    and their HIV bars stayed absent from every figure. HIV is reported as NEF1% (top-1% early
    enrichment) everywhere, so that is the key that decides.
    """
    p = run / sub / "suite_summary.json"
    if not p.exists():
        return False
    try:
        return "HIV_nef1_MEAN" in json.loads(p.read_text())
    except Exception:
        return False


def priority(run: Path) -> tuple:
    """Headline first: anchors, then the matched-budget rung, then everything else."""
    n = run.name
    if "anchor" in n or n.startswith("random_baseline"):
        return (0, n)
    if f"_{MATCHED}" in n or n.endswith(f"from{MATCHED}"):
        return (1, n)
    return (2, n)


def run_eval(run: Path, scheme: str) -> bool:
    enc = run / "encoder"
    sub = "moleculenet" if scheme == "single" else "moleculenet_cv"
    is_anchor = "anchor" in run.name
    if not is_anchor and not (enc / "model.safetensors").exists():
        log(f"  {run.name}: no encoder — skip")
        return False
    # DeepChem's featurized cache collides across evals, so this used to wipe every *-featurized
    # dir in tmp. That is only safe when this script is the ONLY eval on the machine: a parallel
    # session running eval_v2 has its cache deleted mid-run. Skip the wipe when another eval_v2 is
    # alive -- a stale cache is a slow eval, deleting someone else's is a corrupted one.
    others = subprocess.run(["pgrep", "-f", "eval_v2.py"], capture_output=True, text=True)
    alive = [p for p in others.stdout.split() if p.strip() and int(p) != os.getpid()]
    if alive:
        log(f"  another eval_v2 is running (pid {' '.join(alive)}) - NOT clearing the shared "
            f"DeepChem cache")
    else:
        for d in glob.glob(os.path.join(tempfile.gettempdir(), "*-featurized")):
            shutil.rmtree(d, ignore_errors=True)

    cmd = [PY, "eval_v2.py", "--output_dir", str(run / sub),
           "--head_seeds", "0", "1", "2", "--datasets", *CORE]
    if scheme == "cv":
        cmd += ["--cv_folds", "5"]
    if is_anchor:
        # classical baselines have no encoder: featurizer/head come from the anchor's own recipe
        cmd += ["--featurizer", "fp_desc" if "fp_desc" in run.name else "ecfp4", "--head", "xgb"]
    else:
        cmd += ["--encoder", str(enc), "--tokenizer", TOK, "--pool", "mean",
                "--standardize", "zscore", "--head", "mlp", "--max_length", "256"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    ok = has_hiv(run, sub)
    log(f"  {run.name} [{scheme}]: {'OK' if ok else 'FAILED'}")
    if not ok:
        log(f"      stderr: {r.stderr[-300:]}")
        return False
    # Push straight back to S3. The FIRST pass of this script completed 40 single-split evals
    # (~5.5h) and every one was silently reverted hours later by a routine
    # `aws s3 sync s3://.../climb_v2_phase2 figure_data/climb_v2_phase2`: the fresh 7-task
    # summary differed in size from the stale 5-task object, so sync pulled the OLD file down on
    # top of it. Local-only results are not durable while any download sync can run -- upload as
    # we go so the two sides never disagree.
    up = subprocess.run(["aws", "s3", "sync", str(run / sub), f"{S3_WAVE}/{run.name}/{sub}",
                         "--only-show-errors"], capture_output=True, text=True)
    if up.returncode != 0:
        log(f"      WARNING: S3 upload failed ({up.stderr[-200:].strip()}) - result is local-only "
            f"and a later download sync can revert it")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--schemes", nargs="+", default=["single", "cv"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    runs = sorted([d for d in WAVE.glob("*/") if d.is_dir()], key=priority)
    todo = []
    for scheme in a.schemes:                      # all of scheme 1 before scheme 2
        sub = "moleculenet" if scheme == "single" else "moleculenet_cv"
        for r in runs:
            if not has_hiv(r, sub):
                todo.append((r, scheme))
    if a.limit:
        todo = todo[:a.limit]

    log(f"{len(todo)} (run, scheme) pairs need the 7-task core")
    for r, s in todo[:10]:
        log(f"   queued: {r.name} [{s}]")
    if len(todo) > 10:
        log(f"   ... and {len(todo)-10} more")
    if a.dry_run:
        return 0

    ok = fail = 0
    for r, s in todo:
        (ok := ok + 1) if run_eval(r, s) else (fail := fail + 1)
    log(f"DONE ok={ok} failed={fail}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
