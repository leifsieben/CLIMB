"""Analytic native-unit rescale of the END-TO-END regression outputs (ESOL/QM7/Lipo).

The e2e arms (no_pretrain_e2e = e2e_random_0*, and the eval-ceiling runs) were trained on
DeepChem-normalized regression targets, so their stored RMSE is in normalized units and their
per-molecule dumps are in normalized space. Both fixes for regression are exactly recoverable
without re-running the (expensive) full-encoder fine-tune, because the transform is a known affine:

    native_value = sigma * normalized_value + mu           (per task, DeepChem train sigma/mu)
    RMSE_native  = sigma * RMSE_normalized                 (mu cancels in the residual)

So we scale the summary RMSE rows by sigma and map the dump's y_true/y_pred by the full affine
(keeps them native-consistent for the I1 per-molecule analysis, whose lift is a ratio and is
therefore invariant, but the units must still match the corrected model arms). Classification
(Tox21) is NOT touched here -- its missing-label fix requires a real re-train.

Idempotent: writes `.rescaled_e2e_reg.json` per split dir; skips if present (avoids double-scaling).

Usage: python scripts/rescale_e2e_regression.py [--dry-run]
"""
from __future__ import annotations
import argparse, json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "figure_data"
# DeepChem scaffold-train sigma / mu (native units), measured from the raw targets.
STATS = {"ESOL": (2.066724, -2.866876),
         "QM7": (228.656034, -1531.135265),
         "Lipophilicity": (1.210993, 2.162905)}
REG = set(STATS)
# e2e arms in the standard per-run layout (eval_ceiling.csv is handled separately, see bottom).
E2E_RUNS = [("climb_v2_phase2", "e2e_random_00"),
            ("climb_v2_phase2", "e2e_random_01"),
            ("climb_v2_phase2", "e2e_random_02")]


def rescale_split(sd: Path, dry: bool) -> str:
    summ = sd / "moleculenet_summary.csv"
    if not summ.exists():
        return "no-summary"
    marker = sd / ".rescaled_e2e_reg.json"
    if marker.exists():
        return "cached"
    d = pd.read_csv(summ)
    reg_here = sorted(set(d.dataset) & REG)
    if not reg_here:
        return "no-regression"
    if dry:
        return f"would-rescale {reg_here}"
    # 1) summary RMSE rows -> x sigma (MEAN/STD/fold/seed and the *_train rows are all RMSE-scaled)
    for t in reg_here:
        sig = STATS[t][0]
        m = (d.dataset == t) & (d.main_metric.astype(str).str.startswith("rmse"))
        d.loc[m, "main_value"] = d.loc[m, "main_value"].astype(float) * sig
    d.to_csv(summ, index=False)
    # 2) suite_summary.json -> ESOL_MEAN/STD etc. x sigma
    js = sd / "suite_summary.json"
    if js.exists():
        j = json.loads(js.read_text())
        for k in list(j):
            t = k.split("_")[0]
            if t in reg_here and (k.endswith("_MEAN") or k.endswith("_STD")):
                j[k] = float(j[k]) * STATS[t][0]
        js.write_text(json.dumps(j, indent=2))
    # 3) per-molecule dump -> full affine on y_true / y_pred
    tp = sd / "test_predictions.csv"
    if tp.exists():
        p = pd.read_csv(tp)
        for t in reg_here:
            sig, mu = STATS[t]
            m = p.dataset == t
            for col in ("y_true", "y_pred"):
                p.loc[m, col] = p.loc[m, col].astype(float) * sig + mu
        p.to_csv(tp, index=False)
    marker.write_text(json.dumps({"rescaled": reg_here}))
    return f"rescaled {reg_here}"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    for wave, run in E2E_RUNS:
        for split in ("moleculenet_cv", "moleculenet"):
            sd = DATA / wave / run / split
            if sd.exists():
                print(f"[{wave}/{run}/{split}] {rescale_split(sd, args.dry_run)}", flush=True)


if __name__ == "__main__":
    main()
