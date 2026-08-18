"""Score Fig F's Ames predictions, which use a different dump schema from every other Polaris run.

concat_redundancy_panels.py writes test_predictions.csv with the FEATURE-SET NAME in the `seed`
column ("fp+desc", "CLM", "desc+CLM", "fp+desc+CLM") -- there are no eval seeds, the four rows per
molecule are four feature combinations. chemeleon_suite_score_polaris.py does int(r["seed"]) and
dies with `invalid literal for int(): 'fp+desc'`, which is why Fig F sat at 2/3 panels with the
Ames predictions already written and unscored.

This scores each feature set separately through the benchmark's own evaluate() (Ames test labels
are withheld, so that is the only way), then APPENDS the Ames rows to concat_panels_climb.csv.
Append, not rewrite: the file already holds the MoleculeACE and CBS panels and a rewrite would
drop them -- the same trap that made this job need re-running in the first place.

Run with .venv_polaris/bin/python.
"""
from __future__ import annotations
import csv, json, sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import polaris as po

ROOT = Path(__file__).resolve().parent.parent
TASK = "tdcommons/ames"
PANEL = "Ames"


def main(model_dir: str, out_csv: str) -> int:
    d = Path(model_dir)
    preds = defaultdict(dict)
    for r in csv.DictReader((d / "test_predictions.csv").open()):
        if r["task"] != TASK:
            continue
        preds[r["seed"]][int(r["test_index"])] = float(r["y_pred"])   # seed col = feature set
    if not preds:
        print(f"no {TASK} rows in {d}/test_predictions.csv", file=sys.stderr)
        return 1

    bench = po.load_benchmark(TASK)
    rows = []
    for feat, dd in sorted(preds.items()):
        yp = np.array([dd[i] for i in range(len(dd))], dtype=np.float64)
        res = bench.evaluate(y_prob=yp, y_pred=(yp >= 0.5).astype(int))
        tbl = res.results if hasattr(res, "results") else res
        auc = None
        for _, rr in tbl.iterrows():
            if str(rr.get("Metric", "")).lower().replace("-", "_") in ("roc_auc", "rocauc"):
                auc = float(rr["Score"]); break
        if auc is None:
            print(f"  {feat}: no roc_auc in benchmark result", file=sys.stderr); continue
        rows.append({"task": PANEL, "features": feat, "metric": "roc_auc",
                     "mean": round(auc, 4), "std": ""})
        print(f"  Ames {feat:18} roc_auc={auc:.4f}", flush=True)

    out = Path(out_csv)
    existing, fields = [], ["task", "features", "metric", "mean", "std"]
    if out.exists():
        with out.open() as f:
            rd = csv.DictReader(f)
            fields = rd.fieldnames or fields
            existing = [r for r in rd if r.get("task") != PANEL]   # drop stale Ames, keep the rest
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in existing:
            w.writerow(r)
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})
    print(f"wrote {out}: {len(existing)} kept + {len(rows)} Ames rows")
    return 0 if rows else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1], sys.argv[2]))
