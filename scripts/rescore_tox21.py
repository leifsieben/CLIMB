"""Re-score Tox21 from each run's OWN per-molecule predictions, NON-DESTRUCTIVELY.

WHY THIS EXISTS
---------------
On 2026-08-18 the new `bar vs error-bar centre` audit check (scripts/audit_figure_consistency.py,
check 8) failed on all 8 arms of fig_A's Tox21 panel: the bar, read from
`moleculenet_cv/moleculenet_summary.csv`, sat 2-4% BELOW the whisker, which is bootstrapped from
that same run's `moleculenet_cv/test_predictions.csv`. Two independent paths to one number,
disagreeing inside a single run directory.

The predictions are the sound artefact, and the summaries are partly stale:

  * The predictions are correctly masked. Tox21's OOF holds 77,864 rows = 5,858 positive +
    72,006 negative, which is exactly 93,876 - 16,012, and 16,012 is the count of w==0 cells in
    DeepChem's Tox21 weight matrix (verified directly against `dc.molnet.load_tox21`). Those are
    the same counts notes/corrections-tox21-regression-2026-08-05.md cites.
  * That note records the effect of the fix: "Tox21 AUC RISES ~+0.015...0.020 per arm". The two
    candidate values differ by +0.0184 -- so the HIGHER, prediction-derived number is the corrected
    one. (This is the direction both sessions initially got backwards, by reasoning from which tree
    a file came from instead of from the documented effect size.)
  * Decisive: re-scoring run-by-run and FOLD-BY-FOLD shows fold0 agreeing to +0.0013 while folds
    1-4 are each off by +0.019...+0.026. A masking bug would move every fold. One fold corrected
    and four not is an INTERRUPTED re-run -- and `.corrected_v2.json` was written anyway, which is
    why the marker sits next to stale rows and is worthless as evidence.

THE ESTIMATOR -- this is the summary's own estimator, not a new one
------------------------------------------------------------------
  for each of the 5 CV folds:
      for each of the 12 output_index assays:  metric on that (fold, assay) slice
      -> mean over the 12 assays            = that fold's value, ONE `foldK` row
Folds are scored SEPARATELY and averaged after, because moleculenet_summary.csv holds one row per
fold and the aggregator's point estimate is the mean of those rows; pooling the folds first would
make the per-fold rows -- and the SD across them -- mean something else. (Pooling gives 0.7726 for
the random_baseline trio where the correct order gives 0.7701.)

Fold assignment is computed on UNIQUE MOLECULES. Tox21's OOF carries 12 rows per molecule, and
feeding that duplicated SMILES list to `eval_v2._scaffold_kfold_indices(seed=0)` yields a partition
that only approximates the real one (~1.3% error).

NON-DESTRUCTIVE BY CONSTRUCTION
-------------------------------
Writes a NEW `moleculenet_cv_tox21fixed/` beside `moleculenet_cv/` and never touches the original.
Both readings stay inspectable side by side -- which is the only reason this was catchable at all --
and a wrong call costs nothing. Readers resolve it exactly like the QM7 native fix:
`figures.sixpanel.NATIVE_SUBDIRS["Tox21"]` and `six_panel_aggregate.TOX21_SUBDIRS`.

Run:  python3 scripts/rescore_tox21.py [--roots a,b] [--limit N] [--dry-run]
"""
from __future__ import annotations
import argparse, csv, json, sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from compare_models import _metric_over_cols                       # noqa: E402
from a2_bootstrap_errorbars import fold_ids                        # noqa: E402

FD = ROOT / "figure_data"
SRC_SUB, OUT_SUB = "moleculenet_cv", "moleculenet_cv_tox21fixed"
DEFAULT_ROOTS = ["climb_v2_phase2", "climb_v2_ablation_dedup", "climb_v2_h1", "climb_v2_vocab"]
METRICS = ["roc_auc", "nef1"]          # both are carried in moleculenet_summary.csv
KIND = {"roc_auc": "auc", "nef1": "nef1"}
NFOLD = 5


def rescore_run(root: str, run_dir: Path):
    """[(metric, fold, value)] recomputed from this run's own predictions, or None."""
    pred = run_dir / SRC_SUB / "test_predictions.csv"
    if not pred.exists():
        return None
    d = pd.read_csv(pred)
    if "dataset" not in d.columns:
        return None
    d = d[d.dataset == "Tox21"]
    if d.empty:
        return None
    m = d.rename(columns={"y_true": "y_true_a", "y_pred": "y_pred_a", "raw_smiles": "raw_smiles_a"})
    key = "mol_index" if "mol_index" in m.columns else "raw_smiles_a"
    uniq = m.drop_duplicates(subset=[key]).sort_values(key)
    fu = fold_ids(root, uniq["raw_smiles_a"].tolist(), uniq["y_true_a"].to_numpy())
    fmap = dict(zip(uniq["raw_smiles_a"], fu))
    folds = np.array([fmap.get(s, -1) for s in m["raw_smiles_a"]])
    out = []
    for metric in METRICS:
        for f in range(NFOLD):
            sub = m[folds == f]
            if sub.empty:
                continue
            v = _metric_over_cols(sub, "y_pred_a", KIND[metric])
            if np.isfinite(v):
                out.append((metric, f, float(v)))
    return out or None


def template_row(run_dir: Path):
    """Carry the non-metric columns (featurizer/pool/head/...) over from the original Tox21 rows,
    so the rewritten file is schema-identical and nothing downstream has to special-case it."""
    src = run_dir / SRC_SUB / "moleculenet_summary.csv"
    if not src.exists():
        return None, None
    rows = list(csv.DictReader(src.open()))
    if not rows:
        return None, None
    tox = [r for r in rows if r["dataset"] == "Tox21"]
    # e2e-style runs write no per-fold CSV at all, and some runs carry no Tox21 rows; the schema
    # columns are per-run, not per-dataset, so any row serves as the template.
    return (tox[0] if tox else rows[0]), list(rows[0].keys())


def write_fixed(run_dir: Path, vals, tmpl, cols):
    out = run_dir / OUT_SUB
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    if tmpl is None:                     # no per-fold CSV in this run: write suite_summary.json only
        cols, tmpl = [], None
    for metric in (METRICS if tmpl is not None else []):
        per = {f: v for mm, f, v in vals if mm == metric}
        if not per:
            continue
        for f in sorted(per):
            r = dict(tmpl)
            r.update(dataset="Tox21", main_metric=metric, head_seed=f"fold{f}",
                     main_value=repr(per[f]))
            rows.append(r)
        arr = np.array([per[f] for f in sorted(per)], dtype=float)
        for tag, v in (("MEAN", arr.mean()), ("STD", arr.std(ddof=0))):
            r = dict(tmpl)
            r.update(dataset="Tox21", main_metric=metric, head_seed=tag, main_value=repr(float(v)))
            rows.append(r)
    if rows and cols:
        with (out / "moleculenet_summary.csv").open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            w.writerows(rows)

    auc = np.array([v for mm, _, v in vals if mm == "roc_auc"], dtype=float)
    nef = np.array([v for mm, _, v in vals if mm == "nef1"], dtype=float)
    j = {"Tox21_MEAN": float(auc.mean()), "Tox21_STD": float(auc.std(ddof=0))}
    if len(nef):
        j.update(Tox21_nef1_MEAN=float(nef.mean()), Tox21_nef1_STD=float(nef.std(ddof=0)))
    j["_provenance"] = ("Tox21 re-scored from this run's own moleculenet_cv/test_predictions.csv "
                        "by scripts/rescore_tox21.py; per-fold, per-assay, folds averaged after. "
                        "The moleculenet_cv/ copy is left untouched.")
    (out / "suite_summary.json").write_text(json.dumps(j, indent=2))
    return float(auc.mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", default=",".join(DEFAULT_ROOTS))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    done = changed = 0
    for root in [r for r in a.roots.split(",") if r]:
        base = FD / root
        if not base.exists():
            continue
        print(f"\n=== {root} ===", flush=True)
        for run_dir in sorted(p for p in base.iterdir() if p.is_dir()):
            vals = rescore_run(root, run_dir)
            if not vals:
                continue
            tmpl, cols = template_row(run_dir)
            old = []
            summ = run_dir / SRC_SUB / "moleculenet_summary.csv"
            if summ.exists():
                old = [float(r["main_value"]) for r in csv.DictReader(summ.open())
                       if r["dataset"] == "Tox21" and r["main_metric"] == "roc_auc"
                       and r["head_seed"].startswith("fold")]
            if not old:
                # e2e-style runs: the old value lives in suite_summary.json instead
                sj = run_dir / SRC_SUB / "suite_summary.json"
                if sj.exists():
                    v = {k.lower(): vv for k, vv in json.load(sj.open()).items()}.get("tox21_mean")
                    if v is not None:
                        old = [float(v)]
            new_mean = float(np.mean([v for mm, _, v in vals if mm == "roc_auc"]))
            old_mean = float(np.mean(old)) if old else float("nan")
            delta = new_mean - old_mean
            if not a.dry_run:
                write_fixed(run_dir, vals, tmpl, cols)   # tmpl=None -> json only, no CSV
            done += 1
            if abs(delta) > 0.002:
                changed += 1
            print(f"  {run_dir.name:<44} {old_mean:.4f} -> {new_mean:.4f}  ({delta:+.4f})",
                  flush=True)
            if a.limit and done >= a.limit:
                break
        if a.limit and done >= a.limit:
            break
    print(f"\n{'DRY RUN: ' if a.dry_run else ''}re-scored {done} run(s); "
          f"{changed} moved by more than 0.002 ROC-AUC")


if __name__ == "__main__":
    main()
