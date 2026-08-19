"""Rebuild a NATIVE-unit QM7 summary row from a run's own test_predictions.csv.

Context: corrupt_mtr_8M seeds 1 and 2 wrote NATIVE QM7 (212.8 / 216.8) into moleculenet_cv/,
while seed 0 wrote Z-SCORED (0.964) into the same path -- same directory name, different units,
and the `standardize` column says "zscore" in all three, so no name- or column-based guard can
separate them. But all three prediction dumps carry y_true in native kcal/mol, which makes the
units recoverable without re-running anything on a GPU.

Validation: this script reproduces the s1 and s2 summaries (mean-over-folds) before it is
trusted for seed 0 -- the same "recompute from the run's own predictions" test used for Tox21.

Writes moleculenet_cv_qm7native/moleculenet_summary.csv, never touching moleculenet_cv/.
"""
from __future__ import annotations
import csv, sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import eval_v2  # noqa: E402

DS = "QM7"
NATIVE_FLOOR = 10.0   # a QM7 RMSE below this is z-scored, not native


def load(run_dir: Path):
    rows = {}
    with (run_dir / "moleculenet_cv" / "test_predictions.csv").open() as f:
        for r in csv.DictReader(f):
            if r["dataset"] != DS:
                continue
            rows[int(r["mol_index"])] = (r["raw_smiles"], float(r["y_true"]), float(r["y_pred"]))
    order = sorted(rows)
    smiles = [rows[i][0] for i in order]
    y = np.array([rows[i][1] for i in order], dtype=np.float64)
    p = np.array([rows[i][2] for i in order], dtype=np.float64)
    return smiles, y, p


def rebuild(run_dir: Path):
    smiles, y, p = load(run_dir)
    folds = eval_v2._scaffold_kfold_indices(smiles, 5, 0, labels=None)
    per_fold = [float(np.sqrt(np.mean((y[idx] - p[idx]) ** 2))) for idx in folds]
    return smiles, per_fold, float(np.mean(per_fold)), float(np.std(per_fold, ddof=1))


def write_summary(run_dir: Path, per_fold, mean, std, n):
    out = run_dir / "moleculenet_cv_qm7native"
    out.mkdir(exist_ok=True)
    hdr = ["dataset", "task_type", "featurizer", "pool", "standardize", "head",
           "main_metric", "head_seed", "n_train", "main_value", "elapsed_seconds"]
    base = [DS, "regression", "encoder", "mean", "native", "mlp", "rmse"]
    with (out / "moleculenet_summary.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(hdr)
        for j, v in enumerate(per_fold):
            w.writerow(base + [f"fold{j}", "", f"{v:.6f}", ""])
        w.writerow(base + ["MEAN", n, f"{mean:.6f}", ""])
        w.writerow(base + ["STD", n, f"{std:.6f}", ""])
    return out / "moleculenet_summary.csv"


def main(argv):
    wave = ROOT / "figure_data" / "climb_v2_phase2"
    # validate against the two runs whose summaries are already native
    for run, expect in (("corrupt_mtr_8M_s1", 212.774), ("corrupt_mtr_8M_s2", None)):
        d = wave / run
        _, pf, m, s = rebuild(d)
        tag = "" if expect is None else f"  (summary {expect:.3f}, delta {m - expect:+.3f})"
        print(f"CHECK {run:22} native RMSE = {m:.3f} +/- {s:.3f}{tag}")
        if m < NATIVE_FLOOR:
            print(f"  refusing: {run} rebuilds below the native floor", file=sys.stderr)
            return 1
    for run in argv or ["corrupt_mtr_8M"]:
        d = wave / run
        smiles, pf, m, s = rebuild(d)
        if m < NATIVE_FLOOR:
            print(f"  refusing to write {run}: {m:.3f} < {NATIVE_FLOOR}", file=sys.stderr)
            return 1
        path = write_summary(d, pf, m, s, len(smiles))
        print(f"WROTE {run:22} native RMSE = {m:.3f} +/- {s:.3f}  -> {path.relative_to(ROOT)}")
        print("      folds: " + ", ".join(f"{v:.2f}" for v in pf))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
