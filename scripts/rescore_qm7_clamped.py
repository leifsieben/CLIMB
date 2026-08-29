"""Re-score QM7 from each run's OWN predictions with the OOD clamp, NON-DESTRUCTIVELY.

WHY THIS EXISTS
---------------
Two scoring protocols disagreed. scripts/chemeleon_suite_run.py clips regression predictions to
the train target range +-25% ("Bound OOD extrapolation ... uniform across arms, a no-op for
well-behaved features"); eval_v2.py, which scores MoleculeNet and CBS, did not. So the SAME model
was bounded on MoleculeACE and Polaris and unbounded on QM7.

The difference is not academic. CheMeleon-frozen's QM7 predictions include -15,012 kcal/mol for a
molecule whose true value lies in [-2190, -693]. That one molecule is 75.5% of its fold's squared
error; the fold reads 427.7 RMSE where every other fold reads ~205, and the arm's published QM7
value is 268.8 against ~200 for everything else. It reads as "CheMeleon frozen is catastrophic on
QM7" when what happened is that an unbounded MLP extrapolated on a handful of molecules and one
protocol caught it while the other did not.

eval_v2._bound_ood now applies the clamp for future runs. This script applies it to predictions
already on disk, so nothing has to be re-run: the clamp is a scoring-time operation and every
run's per-molecule OOF dump is already there.

MEASURED EFFECT -- this is what makes it fair rather than a thumb on the scale:
    chemeleon_frozen  268.8 -> 208.8   (-60.0)
    random_encoder    205.3 -> 201.0    (-4.3)
    unsup             199.3 -> 197.4    (-1.9)
    e2e_no_pretrain   194.3 -> 194.1    (-0.2)
    sup_dense, ecfp_desc, chemeleon_e2e   unchanged to 0.1
The only arms it moves are the two whose encoders are not trained to represent chemistry, which is
exactly where OOD extrapolation is expected.

THE ESTIMATOR is the summary's own: per fold, score that fold, then average the folds. Folds are
reconstructed exactly as scripts/rescore_tox21.py does, and the clamp band is fit on each fold's
TRAIN targets (everything outside that fold), never on the test fold -- fitting it on all of y
would leak the test range into the bound.

NON-DESTRUCTIVE: writes a NEW moleculenet_cv_qm7clamped/ beside the source dir and never touches
it, so both readings stay inspectable side by side -- the same rule rescore_tox21.py follows.

Run:  python3 scripts/rescore_qm7_clamped.py [--roots a,b] [--dry-run]
"""
from __future__ import annotations
import argparse, csv, json, sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from a2_bootstrap_errorbars import fold_ids            # noqa: E402
import eval_v2                                          # noqa: E402

FD = ROOT / "figure_data"
SRC_SUBS = ("moleculenet_cv_qm7native", "moleculenet_cv")
OUT_SUB = "moleculenet_cv_qm7clamped"
DEFAULT_ROOTS = ["climb_v2_phase2", "climb_v2_ablation_dedup", "climb_v2_h1", "climb_v2_vocab"]
NFOLD = 5
# QM7 in NATIVE kcal/mol only. A z-scored dump (values ~0.85) would be clamped against a z-scored
# band and produce a number that is internally consistent and incomparable to every other arm --
# the mixed-unit failure this repo has already had once. Checked by VALUE, not by path.
NATIVE_MIN = -100.0


def rescore(root: str, run_dir: Path):
    for sub in SRC_SUBS:
        p = run_dir / sub / "test_predictions.csv"
        if not p.exists():
            continue
        d = pd.read_csv(p)
        if "dataset" not in d.columns:
            continue
        d = d[d.dataset == "QM7"]
        if d.empty:
            continue
        if d.y_true.min() > NATIVE_MIN:
            return ("ZSCORED", sub, float(d.y_true.min()))
        folds = fold_ids(root, d.raw_smiles.tolist(), d.y_true.to_numpy())
        raw, clamped = [], []
        for k in range(NFOLD):
            te, tr = d[folds == k], d[(folds != k) & (folds >= 0)]
            if te.empty or tr.empty:
                continue
            pred = eval_v2._bound_ood(te.y_pred.to_numpy(float), tr.y_true.to_numpy(float),
                                      "regression")
            raw.append(float(np.sqrt(((te.y_pred - te.y_true) ** 2).mean())))
            clamped.append(float(np.sqrt(((pred - te.y_true) ** 2).mean())))
        if clamped:
            return (sub, raw, clamped)
    return None


def write(run_dir: Path, sub: str, vals):
    out = run_dir / OUT_SUB
    out.mkdir(parents=True, exist_ok=True)
    src = run_dir / sub / "moleculenet_summary.csv"
    rows, cols = [], None
    if src.exists():
        allrows = list(csv.DictReader(src.open()))
        if allrows:
            cols = list(allrows[0].keys())
            tmpl = next((r for r in allrows if r["dataset"] == "QM7"), allrows[0])
            for k, v in enumerate(vals):
                r = dict(tmpl); r.update(dataset="QM7", main_metric="rmse",
                                         head_seed=f"fold{k}", main_value=repr(v))
                rows.append(r)
            a = np.array(vals, dtype=float)
            for tag, v in (("MEAN", a.mean()), ("STD", a.std(ddof=0))):
                r = dict(tmpl); r.update(dataset="QM7", main_metric="rmse", head_seed=tag,
                                         main_value=repr(float(v)))
                rows.append(r)
    if rows and cols:
        with (out / "moleculenet_summary.csv").open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols); w.writeheader(); w.writerows(rows)
    a = np.array(vals, dtype=float)
    (out / "suite_summary.json").write_text(json.dumps(
        {"QM7_MEAN": float(a.mean()), "QM7_STD": float(a.std(ddof=0)),
         "_provenance": (f"QM7 re-scored from {sub}/test_predictions.csv by "
                         "scripts/rescore_qm7_clamped.py: predictions clipped to each fold's TRAIN "
                         "target range +-25% (eval_v2._bound_ood), the same bound "
                         "scripts/chemeleon_suite_run.py has always applied on the suite tracks. "
                         "The source dir is untouched.")}, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", default=",".join(DEFAULT_ROOTS))
    ap.add_argument("--dry-run", action="store_true")
    # SCOPE IT TO THE RUNS THAT NEED IT. This is non-destructive and idempotent, so a
    # full sweep rewrites ~200 correct directories with identical content -- harmless
    # except that it moves their mtimes, and audit_figure_consistency then correctly
    # reports mainline_8M.csv and a2_errorbars.csv as OLDER than data they contain.
    # That happened on 2026-08-28 and cost a full aggregate + cluster-bootstrap rebuild
    # to clear a warning about files nothing had actually changed.
    ap.add_argument("--only", default="",
                    help="comma-separated run-dir names; default is every run")
    a = ap.parse_args()
    only = {x for x in a.only.split(',') if x}
    done = moved = skipped = 0
    for root in [r for r in a.roots.split(",") if r]:
        base = FD / root
        if not base.exists():
            continue
        print(f"\n=== {root} ===", flush=True)
        for run_dir in sorted(p for p in base.iterdir() if p.is_dir()):
            if only and run_dir.name not in only:
                continue
            r = rescore(root, run_dir)
            if r is None:
                continue
            if r[0] == "ZSCORED":
                print(f"  SKIP  {run_dir.name:<40} QM7 is z-scored in {r[1]} "
                      f"(min={r[2]:.3f}) -- clamping a z-scored dump would be incomparable")
                skipped += 1
                continue
            sub, raw, clamped = r
            if not a.dry_run:
                write(run_dir, sub, clamped)
            done += 1
            delta = float(np.mean(clamped) - np.mean(raw))
            if abs(delta) > 0.5:
                moved += 1
            print(f"  {run_dir.name:<44} {np.mean(raw):8.2f} -> {np.mean(clamped):8.2f} "
                  f"({delta:+7.2f})", flush=True)
    print(f"\n{'DRY RUN: ' if a.dry_run else ''}re-scored {done} run(s); {moved} moved by more "
          f"than 0.5 kcal/mol; {skipped} skipped (z-scored)")


if __name__ == "__main__":
    main()
