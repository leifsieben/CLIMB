#!/usr/bin/env python
"""Is this replicate cell complete AND built on the same molecules as its base?

Completeness alone does not answer that. A box whose deepchem parses Tox21 differently emits a
perfectly well-formed 220-row summary over a DIFFERENT molecule set -- 7,831 parsed where the
reference parses 7,823 -- and the ~0.008 ROC-AUC drift that follows is what cost us the fig_C2 and
fig_D Tox21 column. Nothing about the artifact looks wrong. Row counts, headers and value ranges
all pass. It is only detectable by comparing against something that already knows the answer.

n_train per (dataset, head_seed) is that something: it is a direct count of parsed, split
molecules, it is already recorded in every summary, and the base is the reference by definition.
If a replicate's fingerprint differs anywhere, its environment is not its base's environment, and
a seed spread taken across the two would be measuring the environment as much as the seed.

Usage:  verify_replicate_parse.py <cell.csv> <base.csv> <expected_rows>
Prints one line beginning OK or REJECT and exits 0 or 1.
"""
import csv, sys
from collections import Counter


def fingerprint(path: str) -> dict:
    """(dataset, head_seed) -> n_train, over the per-cell rows only.

    MEAN/STD and the bare foldN aggregate rows carry no n_train of their own, so they are skipped:
    including them would compare blanks and call every pair equal.
    """
    fp = {}
    with open(path) as fh:
        for r in csv.DictReader(fh):
            tag = r["head_seed"]
            if "_fold" not in tag:
                continue
            fp[(r["dataset"], tag)] = r["n_train"]
    return fp


def main() -> int:
    cell, base, want = sys.argv[1], sys.argv[2], int(sys.argv[3])
    try:
        with open(cell) as fh:
            n = sum(1 for _ in csv.DictReader(fh))
    except FileNotFoundError:
        print("REJECT cell absent")
        return 1
    if n != want:
        print(f"REJECT {n} rows, expected {want}")
        return 1

    try:
        a, b = fingerprint(cell), fingerprint(base)
    except FileNotFoundError:
        print("REJECT base summary absent -- cannot verify parse identity, refusing to pass it")
        return 1
    if not b:
        print("REJECT base fingerprint empty -- refusing to pass an unverifiable cell")
        return 1

    if set(a) != set(b):
        only = sorted(set(a) ^ set(b))[:4]
        print(f"REJECT cell/base cover different (dataset, seed_fold) keys, e.g. {only}")
        return 1

    bad = {k: (a[k], b[k]) for k in a if a[k] != b[k]}
    if bad:
        ds = Counter(k[0] for k in bad)
        k0 = sorted(bad)[0]
        print(f"REJECT parse differs from base in {len(bad)} cells across {sorted(ds)}; "
              f"e.g. {k0[0]} {k0[1]}: cell n_train={bad[k0][0]} base n_train={bad[k0][1]}")
        return 1

    print(f"OK {n} rows, parse identical to base across {len(a)} cells")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
