#!/usr/bin/env python
"""Distinct COMPLETED tasks in one suite cell dir -- the completion metric, per track.

The two tracks are not symmetric and cannot share a counter. MoleculeACE scores in-process, so
results.csv carries the cells and counting its tasks is right. Polaris withholds its test labels,
so results.csv is HEADER-ONLY BY DESIGN; the deliverable is finite predictions, scored off-box
afterwards. Counting results.csv on Polaris therefore returns 0 forever -- it withholds complete
dirs from upload and hangs any shutdown gate behind them. That is not hypothetical: the first
launch of the xgb seed job reported the published, complete polaris/unsup_8M__xgb base as
"MISSING (0/28)", and the same shape left chemeleon_e2e_s1 silently unscored earlier.

A Polaris task counts only if EVERY one of its predictions is finite. A single NaN means the head
degenerated on that task (one NaN feature through an MLP yields an all-NaN output), and a dir that
looks complete but is quietly NaN is exactly what the counted-work rule exists to catch.
"""
import csv, math, sys

def count(d: str, track: str) -> int:
    if track == "polaris":
        ok: dict[str, bool] = {}
        with open(f"{d}/test_predictions.csv") as fh:
            for r in csv.DictReader(fh):
                try:
                    v = float(r["y_pred"])
                except (TypeError, ValueError):
                    v = float("nan")
                ok[r["task"]] = ok.get(r["task"], True) and math.isfinite(v)
        return sum(1 for good in ok.values() if good)
    with open(f"{d}/results.csv") as fh:
        return len({r["task"] for r in csv.DictReader(fh)})

if __name__ == "__main__":
    try:
        print(count(sys.argv[1], sys.argv[2]))
    except Exception:
        print(0)
