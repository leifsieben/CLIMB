"""Does each polaris_scores.csv describe the predictions sitting NEXT TO IT?

WHY THIS EXISTS. On 2026-08-21 seven Polaris anchor cells were scored from a local mirror in which
one directory's test_predictions.csv was a stale copy — same byte SIZE as the file on S3, so
`aws s3 sync` compared size and mtime, judged them equal, and skipped the download. The scorer then
did its job perfectly on the wrong input. The resulting polaris_scores.csv was:

    complete   249 rows, 28 tasks, 0 NaN
    fresh      newer than every file in the directory
    valid      a correct scoring of a real prediction set

...just not the prediction set it was sitting next to. 245 of its 249 rows were wrong, and the arm
it fed leads the Ames panel.

Completeness cannot catch that — a stale file is complete. Timestamps cannot catch it — the write
really happened, after the predictions. The ONLY thing that catches it is recomputing from the
predictions actually present and comparing, which is what this does.

It scores into a TEMPORARY copy and never touches the directory, so it is safe to run against a
tree someone else is working in.

Run:  .venv_polaris/bin/python scripts/verify_polaris_scores.py <dir> [<dir> ...]
"""
from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SCORER = ROOT / "scripts" / "chemeleon_suite_score_polaris.py"
KEY = ["task", "seed", "metric"]


def verify(d: Path) -> bool:
    preds, scores = d / "test_predictions.csv", d / "polaris_scores.csv"
    if not preds.exists() or not scores.exists():
        print(f"  SKIP  {d.name}: missing {'predictions' if not preds.exists() else 'scores'}")
        return True
    with tempfile.TemporaryDirectory() as tmp:
        t = Path(tmp) / d.name
        t.mkdir()
        shutil.copy2(preds, t / "test_predictions.csv")
        # The scorer raises httpx.ConnectTimeout AFTER writing a complete file, so a non-zero exit
        # says nothing about whether the work was done. Judge the OUTPUT, not the return code --
        # which is the same rule this whole script exists to apply one level up.
        subprocess.run([sys.executable, str(SCORER), str(t)], capture_output=True)
        fresh = t / "polaris_scores.csv"
        if not fresh.exists():
            print(f"  FAIL  {d.name}: re-scoring produced nothing")
            return False
        a = pd.read_csv(scores).set_index(KEY).value
        b = pd.read_csv(fresh).set_index(KEY).value
        if set(a.index) != set(b.index):
            print(f"  FAIL  {d.name}: {len(set(a.index) ^ set(b.index))} row key(s) differ")
            return False
        delta = (a - b.reindex(a.index)).abs()
        bad = int((delta > 1e-9).sum())
        if bad:
            print(f"  FAIL  {d.name}: {bad} of {len(a)} rows disagree with their own predictions "
                  f"(max |delta| {delta.max():.6g}) -- the scores file describes a DIFFERENT run")
            return False
        print(f"  OK    {d.name}: all {len(a)} rows reproduce from the predictions beside them")
        return True


def main(dirs: list[str]) -> int:
    print("POLARIS SCORES vs THE PREDICTIONS BESIDE THEM")
    bad = sum(0 if verify(Path(d)) else 1 for d in dirs)
    print(f"\n{'CLEAN' if not bad else str(bad) + ' DIR(S) WHOSE SCORES DESCRIBE ANOTHER RUN'}")
    return bad


if __name__ == "__main__":
    raise SystemExit(1 if main(sys.argv[1:]) else 0)
