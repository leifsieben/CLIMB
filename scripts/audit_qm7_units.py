"""Audit QM7 unit conventions across every run directory.

The hazard: moleculenet_cv/ can hold EITHER z-scored or native-unit QM7, the `standardize`
column says "zscore" in both cases, and the directory name is identical -- so every guard that
keys on a path or a column header is blind to it. Only the VALUE separates them.

A QM7 RMSE below 10 is z-scored; above 50 is native; nothing legitimately lands between, so a
value in the gap means seeds of different conventions were pooled and the result is meaningless.
Exits 1 if any arm mixes conventions across its own seeds, or if any value is in the gap.
"""
from __future__ import annotations
import csv, re, sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ZMAX, NMIN = 10.0, 50.0
SEED_SUFFIX = re.compile(r"_(s\d+|part\d+)$")


def convention(v: float) -> str:
    return "zscore" if v < ZMAX else "native" if v > NMIN else "GAP"


def main() -> int:
    arms = defaultdict(list)
    for f in sorted(ROOT.glob("figure_data/*/*/moleculenet_cv/moleculenet_summary.csv")):
        run = f.parent.parent
        with f.open() as fh:
            v = next((float(r[9]) for r in csv.reader(fh)
                      if len(r) > 9 and r[0] == "QM7" and r[7] == "MEAN"), None)
        if v is None:
            continue
        arms[(run.parent.name, SEED_SUFFIX.sub("", run.name))].append((run.name, v))

    bad = 0
    for (wave, arm), seeds in sorted(arms.items()):
        convs = {convention(v) for _, v in seeds}
        if "GAP" in convs or len(convs) > 1:
            bad += 1
            print(f"MIXED  {wave}/{arm}")
            for name, v in seeds:
                print(f"         {name:34} {v:10.4f}  {convention(v)}")
    print(f"\n{len(arms)} arms audited, {bad} with mixed or out-of-range QM7 units")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
