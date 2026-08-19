"""Merge one dataset's rows from a fresh summary into an existing one, keeping everything else.

eval_v2 writes moleculenet_summary.csv with open("w"), so evaluating a single extra dataset into
a run's existing dir DELETES every other dataset it had. Merging is the only safe way to top one
up -- the same trap that made an earlier top-up need re-running.
"""
from __future__ import annotations
import csv, sys
from pathlib import Path


def main(src: str, dest: str, *datasets: str) -> int:
    ds = set(datasets)
    rows_new = [r for r in csv.DictReader(Path(src).open()) if r["dataset"] in ds]
    if not rows_new:
        print(f"no rows for {ds} in {src}", file=sys.stderr)
        return 1
    d = Path(dest)
    keep = [r for r in csv.DictReader(d.open())] if d.exists() else []
    keep = [r for r in keep if r["dataset"] not in ds]
    fields = list(rows_new[0].keys())
    d.parent.mkdir(parents=True, exist_ok=True)
    with d.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in keep + rows_new:
            w.writerow({k: r.get(k, "") for k in fields})
    print(f"merged {len(rows_new)} {'/'.join(sorted(ds))} rows into {dest} (kept {len(keep)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(*sys.argv[1:]))
