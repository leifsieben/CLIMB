#!/usr/bin/env python3
"""Verify the new runs landed on the SAME 5-fold CV partition as the existing ladder.

A different partition silently breaks every paired significance test between the new rungs and
the published ones, and nothing else in the pipeline would notice. The check merges
test_predictions.csv on (dataset, mol_index, output_index) against a reference run and reports
matched row counts plus, as a stronger assertion, that the molecule sitting at each mol_index is
literally the same SMILES in both files.

  Usage: unsup124_fold_check.py [--refs unsup_8M unsup_48M] [--new unsup_50M unsup_100M ...]
"""
import argparse
import csv
import io
import subprocess
import sys
from collections import defaultdict

S3 = "s3://climb-s3-bucket/experiments/climb_v2_phase2"


def load(run, scheme="moleculenet_cv"):
    uri = f"{S3}/{run}/{scheme}/test_predictions.csv"
    r = subprocess.run(["aws", "s3", "cp", uri, "-"], capture_output=True, text=True)
    if r.returncode != 0:
        return None
    rows = list(csv.DictReader(io.StringIO(r.stdout)))
    out = {}
    for x in rows:
        out[(x["dataset"], int(x["mol_index"]), int(x["output_index"]))] = x["raw_smiles"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refs", nargs="+", default=["unsup_8M", "unsup_48M"])
    ap.add_argument("--new", nargs="+", default=["unsup_8M_c124", "unsup_50M", "unsup_100M"])
    ap.add_argument("--scheme", default="moleculenet_cv")
    a = ap.parse_args()

    refs = {}
    for r in a.refs:
        d = load(r, a.scheme)
        if d is None:
            print(f"  !! reference {r} has no {a.scheme}/test_predictions.csv")
        else:
            refs[r] = d
            ds = sorted({k[0] for k in d})
            print(f"reference {r:14s} {len(d):>7,} rows  tasks={len(ds)} {ds}")
    print()

    bad = 0
    for n in a.new:
        d = load(n, a.scheme)
        if d is None:
            print(f"{n:16s} MISSING {a.scheme}/test_predictions.csv")
            bad += 1
            continue
        ds = sorted({k[0] for k in d})
        print(f"{n:16s} {len(d):>7,} rows  tasks={len(ds)} {ds}")
        for rname, rd in refs.items():
            shared_ds = {k[0] for k in d} & {k[0] for k in rd}
            keys_n = {k for k in d if k[0] in shared_ds}
            keys_r = {k for k in rd if k[0] in shared_ds}
            inter = keys_n & keys_r
            mism = [k for k in inter if d[k] != rd[k]]
            per = defaultdict(int)
            for k in inter:
                per[k[0]] += 1
            cov_n = 100 * len(inter) / max(len(keys_n), 1)
            cov_r = 100 * len(inter) / max(len(keys_r), 1)
            flag = "OK " if (not mism and len(inter) == len(keys_n) == len(keys_r)) else "!! "
            if mism or len(inter) != len(keys_r):
                bad += 1
            print(f"   {flag}vs {rname:12s} matched {len(inter):>7,} keys "
                  f"({cov_n:.2f}% of new, {cov_r:.2f}% of ref) on {len(shared_ds)} shared tasks; "
                  f"SMILES mismatches at same mol_index: {len(mism)}")
            print(f"        per-task matched: {dict(sorted(per.items()))}")
    print()
    print("FOLD_CHECK:", "PASS" if bad == 0 else f"ATTENTION ({bad} issues)")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
