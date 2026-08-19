"""Rebuild ESOL / Lipophilicity summaries from each run's OWN prediction dump.

The bug: 78 ESOL and 42 Lipophilicity summaries in climb_v2_phase2 report a STANDARDIZED RMSE
while the prediction dump beside them is in native units. ESOL's ratio is ~2.1 (its label SD is
2.0955) and Lipophilicity's is ~1.2 -- which is why a magnitude band cannot find it and the QM7
guard's below-10/above-50 rule would miss Lipophilicity entirely.

The test used here needs no knowledge of either convention: the summary must agree with the RMSE
recomputed from that directory's own dump. Disagreement is the selector -- NOT the value looking
small. Two whole roots (climb_v2_ablation_dedup, climb_v2_lrsweep) are internally consistent in
z-scored space, summary AND dump; those are a different convention rather than a corruption, they
agree with themselves, and rebuilding them would be a no-op that made them look repaired.

Writes moleculenet_cv_regnative/, which figures.sixpanel.NATIVE_SUBDIRS already prefers for both
datasets, so nothing published is overwritten and the readers pick it up unchanged.

NOT applied here: eval_v2._bound_ood, the OOD prediction clamp that moleculenet_cv_qm7clamped uses.
This rebuild reproduces what the summary SHOULD have said from the predictions as dumped, and
nothing else -- mixing a units fix with a clamp would make neither checkable.
"""
from __future__ import annotations
import csv, sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import eval_v2  # noqa: E402

DATASETS = ("ESOL", "Lipophilicity")
TOL = 0.05                      # 5% -- fold-mean vs pooled RMSE differ by ~1-4% legitimately
OUTSUB = "moleculenet_cv_regnative"


def dump_rows(p: Path, ds: str):
    rows = {}
    with p.open() as f:
        for r in csv.DictReader(f):
            if r["dataset"] != ds:
                continue
            rows[int(r["mol_index"])] = (r["raw_smiles"], float(r["y_true"]), float(r["y_pred"]))
    if not rows:
        return None
    order = sorted(rows)
    return ([rows[i][0] for i in order],
            np.array([rows[i][1] for i in order]), np.array([rows[i][2] for i in order]))


def published(p: Path, ds: str):
    with p.open() as f:
        for r in csv.DictReader(f):
            if r["dataset"] == ds and r["head_seed"] == "MEAN" and r["main_metric"] == "rmse":
                try:
                    return float(r["main_value"])
                except ValueError:
                    return None
    return None


def rebuild(smiles, y, p):
    folds = eval_v2._scaffold_kfold_indices(smiles, 5, 0, labels=None)
    per = [float(np.sqrt(np.mean((y[i] - p[i]) ** 2))) for i in folds]
    return per, float(np.mean(per)), float(np.std(per, ddof=1))


def write(run_dir: Path, ds: str, per, mean, sd, n):
    out = run_dir / OUTSUB
    out.mkdir(exist_ok=True)
    f = out / "moleculenet_summary.csv"
    hdr = ["dataset", "task_type", "featurizer", "pool", "standardize", "head",
           "main_metric", "head_seed", "n_train", "main_value", "elapsed_seconds"]
    keep = []
    if f.exists():
        with f.open() as fh:
            rd = csv.DictReader(fh)
            keep = [r for r in rd if r["dataset"] != ds]
    base = [ds, "regression", "", "-", "native", "", "rmse"]
    with f.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(hdr)
        for r in keep:
            w.writerow([r.get(k, "") for k in hdr])
        for j, v in enumerate(per):
            w.writerow(base + [f"fold{j}", "", f"{v:.6f}", ""])
        w.writerow(base + ["MEAN", n, f"{mean:.6f}", ""])
        w.writerow(base + ["STD", n, f"{sd:.6f}", ""])


def main(argv) -> int:
    apply = "--apply" in argv
    agree, disagree, rebuilt, checked, same_root_ok = 0, [], 0, [], []
    for cv in sorted(ROOT.glob("figure_data/*/*/moleculenet_cv")):
        summ, preds = cv / "moleculenet_summary.csv", cv / "test_predictions.csv"
        if not (summ.exists() and preds.exists()):
            continue
        for ds in DATASETS:
            pub = published(summ, ds)
            got = dump_rows(preds, ds)
            if pub is None or got is None:
                continue
            smiles, y, p = got
            per, mean, sd = rebuild(smiles, y, p)
            rel = abs(mean - pub) / max(pub, 1e-9)
            if rel <= TOL:
                agree += 1
                checked.append((f"{cv.parent.parent.name}/{cv.parent.name}", ds, pub, mean))
                if cv.parent.parent.name == "climb_v2_phase2":
                    same_root_ok.append((cv, cv.parent.parent.name, cv.parent.name, ds, pub,
                                         mean, per, sd, len(smiles)))
            else:
                disagree.append((cv, cv.parent.parent.name, cv.parent.name, ds, pub, mean, per, sd,
                                 len(smiles)))

    print(f"VALIDATION: {agree} dirs where summary and dump already agree -- the rebuild reproduces")
    for name, ds, pub, mean in checked[:4]:
        print(f"    {name:26} {ds:14} published {pub:.4f}  rebuilt {mean:.4f}")
    roots = {}
    for _, root, *_ in disagree:
        roots[root] = roots.get(root, 0) + 1
    per_ds = {}
    for *_, ds, _, _, _, _, _ in [(d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7], d[8]) for d in disagree]:
        pass
    per_ds = {}
    for d in disagree:
        per_ds[d[3]] = per_ds.get(d[3], 0) + 1
    print(f"\nDISAGREE: {len(disagree)} (dir, dataset) pairs, by root: {roots}, by dataset: {per_ds}")
    for _, _, name, ds, pub, mean, *_ in disagree[:5]:
        print(f"    {name:26} {ds:14} published {pub:.4f}  ->  {mean:.4f}   ratio {mean/pub:.2f}")

    if not apply:
        print("\ndry run -- pass --apply to write moleculenet_cv_regnative/")
        return 0
    # Write the subdir for EVERY phase2 dir that has the dataset, not only the broken ones.
    # figures.sixpanel.suite_subdir picks the first candidate ANY run in the arm has, and
    # _pick_subdir then DROPS runs lacking it -- so leaving corrupt_mtr_8M with a regnative/ while
    # its _s1/_s2 have none would silently take that arm from 3 seeds to 1. For an already-correct
    # dir the rebuilt value reproduces its published one, which the validation pass above proves;
    # the point is subdir completeness, not a second correction.
    for cv, root, name, ds, pub, mean, per, sd, n in disagree + same_root_ok:
        write(cv.parent, ds, per, mean, sd, n)
        rebuilt += 1
    print(f"\nwrote {rebuilt} (dir, dataset) pairs into {OUTSUB}/  "
          f"({len(disagree)} corrected, {len(same_root_ok)} already-correct, for subdir completeness)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
