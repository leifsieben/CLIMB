"""SI Fig h — how much of a benchmark score is the SPLIT? Random vs scaffold, same everything else.

THE QUESTION (Leif 2026-08-29). Every number in this paper is a scaffold-split number, which is the
honest default: test scaffolds are unseen in training, so the score estimates generalisation to new
chemistry rather than interpolation within a series. A random split does not do that -- close
analogues land on both sides -- and it is what a great deal of published work reports. So: for one
model, on one dataset, with everything else held identical, how far apart are the two answers, and
does the RANKING of models survive the change?

Drawn as a slope plot, random on the left and scaffold on the right, one line per arm per panel.
The slope is the split penalty; a crossing is a ranking that depends on the split.

FOUR ARMS, chosen so the comparison is between MODEL FAMILIES rather than between budgets:
    CLIMB 100M supervised    skip_dense_100M_c124   the two halves of the 100M pair -- same corpus,
    CLIMB 100M unsupervised  unsup_100M             same budget, objective the only difference
    ECFP4 + XGBoost          the anchor the CLMs are actually measured against
    ECFP4+desc + XGBoost     the stronger anchor, which fig_B draws as its reference line

BOTH HALVES ARE RUN FRESH IN THIS WAVE. We already hold scaffold numbers for all four arms, and
they are NOT reused: this repository has measured waves disagreeing by up to 8% on the same model
through the same code (scripts/label_eff_fractions.py records Tox21 0.7356 against 0.7961). A slope
whose left end came from this wave and right end from another would draw the wave as much as the
split -- and the slope IS the entire message. Same code path, same seeds, same folds count, one
flag different.

WHAT IS AND IS NOT COVERED
    BACE, Tox21, QM7, HIV   run here, both schemes
    MoleculeACE             run here too (Leif 2026-08-29), via eval_v2's custom-task path: 30
                            separate small regressions, one run per (target, arm, scheme), and the
                            panel value is the macro RMSE over the 30. Its own `split` column is
                            IGNORED -- the whole question is what a random vs a scaffold partition
                            does, so using the provided split would answer neither.
    Ames                    run here on its 5,821 LABELLED TRAINING molecules (Leif 2026-08-29),
                            NOT on the Polaris benchmark split. Polaris withholds the 1,457 test
                            labels ("test labels intentionally absent"), so its own evaluation set
                            is fixed and hidden and cannot be re-partitioned -- the number reported
                            for Ames everywhere else in this paper genuinely cannot be produced
                            under two splits.
                            THIS PANEL IS THEREFORE NOT COMPARABLE IN ABSOLUTE VALUE to the Ames
                            elsewhere, and the caption must say so. It is included because the
                            SLOPE is this figure's message, not the level: a self-contained
                            random-vs-scaffold CV over 5,821 molecules (3,104 positives) measures
                            the split penalty exactly as the other five panels do. Refusing the
                            panel would have withheld a valid measurement because a DIFFERENT
                            measurement was impossible.

Writes: figure_data/SI_fig_h/<arm>__<scheme>/  (one eval_v2 output tree per cell)
Run:  python3 scripts/si_fig_h_split_sensitivity.py
"""
from __future__ import annotations
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "figure_data" / "SI_fig_h"
TOK = "figure_data/_tokenizer"

DATASETS = ["BACE", "Tox21", "QM7", "HIV"]
# Ames goes through the custom-task path, like MoleculeACE: its labelled training portion written
# out as smiles,y. Built here rather than by hand so the panel cannot silently drift from source.
AMES_SRC = ROOT / "chemeleon_suite" / "data" / "polaris" / "tdcommons__ames.csv"
AMES_CSV = ROOT / "figure_data" / "SI_fig_h" / "_ames_train.csv"
SCHEMES = ["random", "scaffold"]

MACE_DIR = ROOT / "chemeleon_suite" / "data" / "moleculeace"
# eval_v2's custom-task loader defaults to the column literally named `y`, while the mainline
# MoleculeACE runner uses "y [pEC50/pKi]" (scripts/chemeleon_suite_run.py:MACE_TARGET). CHECKED
# RATHER THAN ASSUMED: on every target the two differ by a constant offset with slope exactly
# 1.0 and identical SD, so the RMSE this produces is the same number the mainline protocol
# produces. Left on the default deliberately -- switching would require an eval_v2 CLI change for
# no effect. Do not "fix" this without re-checking that relationship.
MACE_TASKS = sorted(p.stem for p in MACE_DIR.glob("*.csv"))

# arm key -> the eval_v2 flags that define it. The two CLM arms differ from the anchors only in
# featurizer/head, so the split comparison is not confounded by the evaluation protocol.
ARMS = {
    "sup_dense_100M": ["--encoder", "figure_data/climb_v2_phase2/skip_dense_100M_c124/encoder",
                       "--tokenizer", TOK, "--featurizer", "encoder", "--pool", "mean",
                       "--head", "mlp"],
    "unsup_100M":     ["--encoder", "figure_data/climb_v2_phase2/unsup_100M/encoder",
                       "--tokenizer", TOK, "--featurizer", "encoder", "--pool", "mean",
                       "--head", "mlp"],
    "ecfp":           ["--featurizer", "ecfp4", "--head", "xgb"],
    "ecfp_desc":      ["--featurizer", "fp_desc", "--head", "xgb"],
}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    todo = [(a, s) for a in ARMS for s in SCHEMES]
    print(f"{len(todo)} cells: {len(ARMS)} arms x {len(SCHEMES)} schemes, "
          f"{len(DATASETS)} datasets each\n", flush=True)
    failed = []
    for i, (arm, scheme) in enumerate(todo, 1):
        dst = OUT / f"{arm}__{scheme}"
        summ = dst / "moleculenet_summary.csv"
        # Completion is judged on the SUMMARY EXISTING AND CARRYING ROWS, not on the directory --
        # eval_v2 creates the directory before it does any work, so `dst.exists()` would skip a
        # cell that died in its first minute.
        if summ.exists() and sum(1 for _ in summ.open()) > 1:
            print(f"[{i}/{len(todo)}] SKIP {arm} / {scheme}: already complete", flush=True)
            continue
        cmd = [sys.executable, "eval_v2.py", "--output_dir", str(dst),
               "--datasets", *DATASETS, "--cv_folds", "5", "--cv_scheme", scheme,
               "--head_seeds", "0", "1", "2", *ARMS[arm]]
        print(f"[{i}/{len(todo)}] {arm} / {scheme}", flush=True)
        r = subprocess.run(cmd, cwd=ROOT)
        if r.returncode != 0:
            print(f"    FAILED rc={r.returncode}", flush=True)
            failed.append((arm, scheme))
    # ---- Ames: the labelled training portion, as one custom task --------------------------
    import pandas as _pd
    if not AMES_CSV.exists():
        _d = _pd.read_csv(AMES_SRC)
        _tr = _d[_d.split == "train"][["smiles", "y"]]
        assert _tr.y.notna().all(), "Ames train rows must all be labelled"
        AMES_CSV.parent.mkdir(parents=True, exist_ok=True)
        _tr.to_csv(AMES_CSV, index=False)
        print(f"  wrote {AMES_CSV.name}: {len(_tr)} labelled molecules", flush=True)
    for arm in ARMS:
        for scheme in SCHEMES:
            dst = OUT / f"{arm}__{scheme}__ames"
            summ = dst / "moleculenet_summary.csv"
            if summ.exists() and sum(1 for _ in summ.open()) > 1:
                continue
            cmd = [sys.executable, "eval_v2.py", "--output_dir", str(dst),
                   "--task_csv", str(AMES_CSV), "--task_name", "Ames",
                   "--task_type", "classification", "--cv_folds", "5", "--cv_scheme", scheme,
                   "--head_seeds", "0", "1", "2", *ARMS[arm]]
            print(f"  Ames {arm} / {scheme}", flush=True)
            r = subprocess.run(cmd, cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if r.returncode != 0:
                failed.append((arm, scheme, "Ames"))

    # ---- MoleculeACE: 30 small regressions per (arm, scheme) ----------------------------
    for arm in ARMS:
        for scheme in SCHEMES:
            for t in MACE_TASKS:
                dst = OUT / f"{arm}__{scheme}__mace" / t
                summ = dst / "moleculenet_summary.csv"
                if summ.exists() and sum(1 for _ in summ.open()) > 1:
                    continue
                cmd = [sys.executable, "eval_v2.py", "--output_dir", str(dst),
                       "--task_csv", str(MACE_DIR / f"{t}.csv"), "--task_name", t,
                       "--task_type", "regression", "--cv_folds", "5", "--cv_scheme", scheme,
                       "--head_seeds", "0", "1", "2", *ARMS[arm]]
                r = subprocess.run(cmd, cwd=ROOT,
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                if r.returncode != 0:
                    failed.append((arm, scheme, t))
            done = sum(1 for t in MACE_TASKS
                       if (OUT / f"{arm}__{scheme}__mace" / t / "moleculenet_summary.csv").exists())
            print(f"  MoleculeACE {arm} / {scheme}: {done}/{len(MACE_TASKS)} targets", flush=True)

    if failed:
        print(f"\n{len(failed)} cell(s) FAILED: {failed}")
        return 1
    print("\nall cells complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
