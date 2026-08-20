"""Score the fig_F concat panels' Ames predictions against the hidden Polaris labels.

WHY A SEPARATE SCRIPT. scripts/chemeleon_suite_score_polaris.py does `int(r["seed"])`, which is
right for every normal arm: there the column holds a fine-tune seed. The concat panels reuse the
same file format to carry SEVEN FEATURE BLOCKS instead -- the column holds "fp", "fp+desc",
"fp+desc+CLM" and so on -- so the shared scorer raises ValueError on the first row. Rather than
loosen the shared scorer (and lose the guarantee that a real seed column is numeric), this reads
the group key as an opaque string.

Output: polaris_scores.csv in the same dir, with `features` in place of `seed`, so nothing
downstream can mistake a feature block for a replicate and average across them.

Usage: .venv_polaris/bin/python scripts/score_polaris_concat.py figure_data/chemeleon_suite/polaris/concat_climb
"""
import csv, json, sys, warnings
from collections import defaultdict
from pathlib import Path

warnings.filterwarnings("ignore")
import numpy as np
import polaris as po

ROOT = Path(__file__).resolve().parent.parent
MAN = json.loads((ROOT / "chemeleon_suite" / "data" / "polaris" / "polaris_manifest.json").read_text())


def main(model_dir):
    model_dir = Path(model_dir)
    preds = defaultdict(dict)
    for r in csv.DictReader((model_dir / "test_predictions.csv").open()):
        preds[(r["task"], r["features"] if "features" in r else r["seed"])][int(r["test_index"])] = float(r["y_pred"])

    out, cache = [], {}
    for (task, feat), d in sorted(preds.items()):
        yp = np.array([d[i] for i in range(len(d))], dtype=np.float64)
        if not np.isfinite(yp).all():
            print(f"[score] SKIP {task} {feat}: non-finite predictions", file=sys.stderr); continue
        b = cache.get(task) or cache.setdefault(task, po.load_benchmark(task))
        try:
            if MAN[task]["type"] == "classification":
                res = b.evaluate(y_pred=(yp > 0.5).astype(int), y_prob=yp)
            else:
                res = b.evaluate(y_pred=yp)
            for _, row in res.results.iterrows():
                out.append([task, feat, row["Metric"], float(row["Score"])])
        except Exception as exc:
            print(f"[score] FAIL {task} {feat}: {type(exc).__name__}: {exc}", file=sys.stderr)

    dst = model_dir / "polaris_scores.csv"
    with dst.open("w", newline="") as f:
        w = csv.writer(f); w.writerow(["task", "features", "metric", "value"]); w.writerows(out)
    print(f"[score] wrote {dst}: {len(out)} rows, "
          f"{len({r[0] for r in out})} task(s), {len({r[1] for r in out})} feature block(s)")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: score_polaris_concat.py <figure_data/chemeleon_suite/polaris/concat_DIR>")
    main(sys.argv[1])
