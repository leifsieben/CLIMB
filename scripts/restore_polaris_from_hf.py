"""Restore the 5 Polaris arms clobbered by the hERG top-up, from the pre-clobber HF revision.

On 2026-08-16 the hERG driver ran chemeleon_suite_run.py with polaris_tasks.txt reduced to a single
line, against prefixes that already held the full 28-task Polaris run. That runner REWRITES rather
than merges, so results.csv / test_predictions.csv / polaris_scores.csv were replaced by hERG-only
copies -- locally AND on S3 (and later on HF, when today's push propagated them).

Recovery: HF is git-backed, and revision fb14c83cc9 (2026-08-15) predates the clobber. It still has
28 tasks x 3 seeds including the 28,248-row per-seed test_predictions.csv -- so this is a FULL
recovery, not just the means.

Restores local, then MERGES the post-clobber hERG rows back in (they are the same quantity -- the
hERG mean matches the old summary to 7 decimals -- but keeping them means the restore cannot lose
the newer run either). Re-uploads to S3. Idempotent.

Run: python3 scripts/restore_polaris_from_hf.py [--execute]
"""
from __future__ import annotations
import argparse, csv, shutil, subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPO = "lsieben/climb-results"
REV = "fb14c83cc9"          # 2026-08-15, last revision before the clobber
ARMS = ["skip_dense_8M", "skip_dense_plus_sparse_8M", "skip_sparse_all_8M",
        "unsup_8M", "u2s_dense_from8M"]
FILES = ["polaris_scores.csv", "results.csv", "test_predictions.csv", "verified.json"]
DEST = ROOT / "figure_data" / "chemeleon_suite" / "polaris"
S3 = "s3://climb-s3-bucket/experiments/chemeleon_suite/polaris"


def merge_csv(restored: Path, current: Path, out: Path):
    """Union of the restored (28-task) rows and whatever the current file holds, de-duplicated.

    Keyed on the whole row, so re-running is a no-op and the hERG rows from the newer run survive
    alongside the 27 recovered tasks.
    """
    rows, seen, header = [], set(), None
    for src in (restored, current):
        if not src.exists():
            continue
        with src.open() as fh:
            rd = csv.reader(fh)
            try:
                h = next(rd)
            except StopIteration:
                continue
            header = header or h
            if h != header:
                continue
            for r in rd:
                k = tuple(r)
                if k not in seen:
                    seen.add(k)
                    rows.append(r)
    if header is None:
        return 0
    with out.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)
    return len(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--execute", action="store_true")
    a = ap.parse_args()
    from huggingface_hub import hf_hub_download

    for arm in ARMS:
        d = DEST / arm
        d.mkdir(parents=True, exist_ok=True)
        for fn in FILES:
            p = f"chemeleon_suite/polaris/{arm}/{fn}"
            try:
                got = Path(hf_hub_download(REPO, p, repo_type="dataset", revision=REV))
            except Exception as e:
                print(f"  {arm}/{fn}: HF fetch failed ({str(e)[:60]}) — skipped")
                continue
            cur = d / fn
            if fn.endswith(".csv"):
                if not a.execute:
                    n_new = len(list(csv.reader(got.open()))) - 1
                    n_cur = (len(list(csv.reader(cur.open()))) - 1) if cur.exists() else 0
                    print(f"  [dry] {arm}/{fn}: restored={n_new} current={n_cur}")
                    continue
                n = merge_csv(got, cur, cur)
                tasks = len({r["task"] for r in csv.DictReader(cur.open())}) if "task" in (cur.read_text()[:200]) else "?"
                print(f"  {arm}/{fn}: {n} rows, {tasks} tasks")
            else:
                if a.execute:
                    shutil.copy(got, cur)
                    print(f"  {arm}/{fn}: copied")
        if a.execute:
            subprocess.run(["aws", "s3", "sync", str(d), f"{S3}/{arm}", "--only-show-errors"], check=False)
    if not a.execute:
        print("\n[dry-run] pass --execute to restore.")
    else:
        print("\nrestored locally + synced to S3. Re-push to HF to fix the truncated copies there.")


if __name__ == "__main__":
    main()
