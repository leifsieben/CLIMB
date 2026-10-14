import csv
import json
import sys
import warnings
from collections import defaultdict
from pathlib import Path

warnings.filterwarnings("ignore")
import numpy as np
import polaris as po

ROOT = Path(__file__).resolve().parent.parent
MAN = json.loads((ROOT / "benchmarks" / "data" / "polaris" / "polaris_manifest.json").read_text())


def main(model_dir):
    model_dir = Path(model_dir)
    preds = defaultdict(dict)
    for r in csv.DictReader((model_dir / "test_predictions.csv").open()):
        preds[(r["task"], int(r["seed"]))][int(r["test_index"])] = float(r["y_pred"])

    out, bench_cache = [], {}
    for (task, seed), d in sorted(preds.items()):
        yp = np.array([d[i] for i in range(len(d))], dtype=np.float64)
        b = bench_cache.get(task) or bench_cache.setdefault(task, po.load_benchmark(task))
        ttype = MAN[task]["type"]
        try:


            if ttype == "classification":
                res = b.evaluate(y_pred=(yp > 0.5).astype(int), y_prob=yp)
            else:
                res = b.evaluate(y_pred=yp)
            for _, row in res.results.iterrows():
                out.append([task, seed, row["Metric"], float(row["Score"])])
        except Exception as exc:
            print(f"[score] FAIL {task} seed{seed}: {type(exc).__name__}: {exc}", file=sys.stderr)
    dst = model_dir / "polaris_scores.csv"
    with dst.open("w", newline="") as f:
        w = csv.writer(f); w.writerow(["task", "seed", "metric", "value"]); w.writerows(out)
    ntask = len({r[0] for r in out})
    print(f"[score] wrote {dst}: {len(out)} rows across {ntask} tasks")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: score_polaris.py <figure_data/chemeleon_suite/polaris/MODEL_DIR>")
    main(sys.argv[1])
