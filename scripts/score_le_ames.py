"""Score the Ames label-efficiency sweep (le_ames_*) from its prediction dumps.

Why this script exists: `results.csv` is EMPTY for the le_ames_* dirs -- but it is empty for
EVERY dir under figure_data/chemeleon_suite/polaris/, including the ones already in fig_A.
Polaris withholds test labels, so the suite runner cannot score in-process; scoring always
happens after the fact through the benchmark's own evaluate(). Nothing failed in these runs and
no int()-on-a-string bug is involved -- the sweep simply had no post-hoc scorer yet.

Emits one row per (arm, fraction, seed) plus the seed mean/std, so the figure session can draw
error bars from three independent eval seeds rather than a single point.

Run with .venv_polaris/bin/python.
"""
from __future__ import annotations
import csv, re, sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import polaris as po

ROOT = Path(__file__).resolve().parent.parent
POLARIS_DIR = ROOT / "figure_data" / "chemeleon_suite" / "polaris"
TASK = "tdcommons/ames"
OUT = ROOT / "figure_data" / "label_eff_ames.csv"
PAT = re.compile(r"^le_ames_(?P<arm>.+)_f(?P<frac>[0-9.]+)$")


def score_dir(bench, d: Path) -> dict[int, float]:
    preds: dict[int, dict[int, float]] = defaultdict(dict)
    with (d / "test_predictions.csv").open() as f:
        for r in csv.DictReader(f):
            if r["task"] != TASK:
                continue
            preds[int(r["seed"])][int(r["test_index"])] = float(r["y_pred"])
    out = {}
    for seed, dd in sorted(preds.items()):
        yp = np.array([dd[i] for i in range(len(dd))], dtype=np.float64)
        res = bench.evaluate(y_prob=yp, y_pred=(yp >= 0.5).astype(int))
        tbl = res.results if hasattr(res, "results") else res
        for _, rr in tbl.iterrows():
            if str(rr.get("Metric", "")).lower().replace("-", "_") in ("roc_auc", "rocauc"):
                out[seed] = float(rr["Score"])
                break
    return out


def main() -> int:
    dirs = sorted(p for p in POLARIS_DIR.glob("le_ames_*") if PAT.match(p.name))
    if not dirs:
        print("no le_ames_* dirs found", file=sys.stderr)
        return 1
    bench = po.load_benchmark(TASK)
    rows = []
    for d in dirs:
        m = PAT.match(d.name)
        arm, frac = m["arm"], float(m["frac"])
        scores = score_dir(bench, d)
        if not scores:
            print(f"  {d.name}: NO {TASK} rows -- skipped", file=sys.stderr)
            continue
        vals = np.array(list(scores.values()), dtype=np.float64)
        for seed, v in scores.items():
            rows.append({"arm": arm, "fraction": frac, "seed": seed, "metric": "roc_auc",
                         "value": round(v, 4), "mean": "", "std": "", "n_seeds": ""})
        rows.append({"arm": arm, "fraction": frac, "seed": "mean", "metric": "roc_auc",
                     "value": round(float(vals.mean()), 4), "mean": round(float(vals.mean()), 4),
                     "std": round(float(vals.std(ddof=1)) if len(vals) > 1 else 0.0, 4),
                     "n_seeds": len(vals)})
        print(f"  {arm:10} f={frac:<5} roc_auc={vals.mean():.4f} +/- "
              f"{(vals.std(ddof=1) if len(vals) > 1 else 0.0):.4f}  (n={len(vals)})", flush=True)
    fields = ["arm", "fraction", "seed", "metric", "value", "mean", "std", "n_seeds"]
    with OUT.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {OUT}: {len(rows)} rows from {len(dirs)} dirs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
