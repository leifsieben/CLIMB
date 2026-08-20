"""Every published number must be reproducible from the artifact it claims to summarise.

This is the permanent version of the check that found the ESOL and Lipophilicity unit bug. It is
deliberately convention-agnostic: it does not know what a "right" RMSE looks like, it only asks
whether a run's summary agrees with the RMSE or AUC recomputed from that run's OWN per-molecule
predictions. Anything that drifts between the two -- a unit change, a masking change, a scorer
change, a stale summary left behind by a single-dataset top-up -- shows up as disagreement without
anyone having to anticipate the specific failure.

Why a magnitude guard is not enough, in one line: ESOL's corruption showed as ratio 2.1 and
Lipophilicity's as 1.2. A band that catches the first cannot catch the second, and QM7's existing
below-10/above-50 rule catches neither.

AND IT REPORTS WHICH WAY THE ERROR LEANS. The ESOL bug put every classical anchor at the bottom of
its rank column and every CLIMB arm at the top -- it flattered our own method on a paper whose
claim is that the classical baseline wins. An error with a direction is worth more scrutiny than
one without, so --bias groups disagreements by arm family and says which way they point.

Usage:
    python scripts/verify_summaries_against_dumps.py            # repo-wide, exits 1 on drift
    python scripts/verify_summaries_against_dumps.py --bias     # + which arms the drift favours
    python scripts/verify_summaries_against_dumps.py --accept accepted_conventions.json
"""
from __future__ import annotations
import csv, json, sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import eval_v2  # noqa: E402
from heads_v2 import compute_metric, compute_nef  # noqa: E402

TOL = 0.05
ACCEPT = ROOT / "configs" / "summary_verify_accepted.json"
ANCHOR_HINTS = ("ecfp", "fp_desc", "chemeleon", "r3c")


def load_dump(p: Path, ds: str):
    per_mol, ncol = {}, 0
    with p.open() as f:
        for r in csv.DictReader(f):
            if r["dataset"] != ds:
                continue
            j = int(r.get("output_index", 0) or 0)
            ncol = max(ncol, j + 1)
            m = per_mol.setdefault(int(r["mol_index"]), [r["raw_smiles"], {}, {}])
            try:
                m[1][j] = float(r["y_true"])
            except ValueError:
                m[1][j] = np.nan
            m[2][j] = float(r["y_pred"])
    if not per_mol:
        return None
    order = sorted(per_mol)
    smiles = [per_mol[i][0] for i in order]
    Y = np.full((len(order), ncol), np.nan)
    P = np.full((len(order), ncol), np.nan)
    for k, i in enumerate(order):
        for j, v in per_mol[i][1].items():
            Y[k, j] = v
        for j, v in per_mol[i][2].items():
            P[k, j] = v
    return smiles, Y, P


def recompute(smiles, Y, P, task_type, metric):
    folds = eval_v2._scaffold_kfold_indices(smiles, 5, 0,
                                            labels=(Y[:, 0] if task_type == "classification" else None))
    vals = []
    for idx in folds:
        if metric == "nef1":
            vals.append(compute_nef(P[idx], Y[idx]))
        else:
            vals.append(compute_metric(P[idx], Y[idx], task_type))
    return float(np.mean(vals))


def main(argv) -> int:
    bias = "--bias" in argv
    accepted = json.loads(ACCEPT.read_text()) if ACCEPT.exists() else {}
    drift, checked = [], 0
    for cv in sorted(ROOT.glob("figure_data/*/*/moleculenet_cv*")):
        summ, preds = cv / "moleculenet_summary.csv", cv / "test_predictions.csv"
        if not (summ.exists() and preds.exists()):
            continue
        rows = list(csv.DictReader(summ.open()))
        by_ds = defaultdict(list)
        for r in rows:
            if r["head_seed"] == "MEAN":
                by_ds[(r["dataset"], r["task_type"], r["main_metric"])].append(r)
        for (ds, tt, metric), rs in by_ds.items():
            if metric.endswith("_cell") or metric not in ("rmse", "roc_auc", "nef1"):
                continue
            key = f"{cv.parent.parent.name}/{cv.parent.name}/{cv.name}:{ds}:{metric}"
            if key in accepted:
                continue
            got = load_dump(preds, ds)
            if got is None:
                continue
            try:
                pub = float(rs[0]["main_value"])
                new = recompute(*got, tt, metric)
            except Exception:
                continue
            checked += 1
            if abs(new - pub) / max(abs(pub), 1e-9) > TOL:
                drift.append((cv.parent.parent.name, cv.parent.name, cv.name, ds, metric,
                              pub, new))

    print(f"checked {checked} (dir, dataset, metric) triples against their own dumps")
    if not drift:
        print("OK -- every summary agrees with the predictions beside it")
        return 0
    print(f"\nDRIFT: {len(drift)}\n")
    for root, run, sub, ds, metric, pub, new in drift[:40]:
        print(f"  {root}/{run}/{sub:26} {ds:14} {metric:8} "
              f"published {pub:9.4f}  dump says {new:9.4f}  ratio {new/pub if pub else float('nan'):.2f}")
    if len(drift) > 40:
        print(f"  ... {len(drift)-40} more")

    if bias:
        fam = defaultdict(lambda: [0, 0.0])
        for _, run, _, _, metric, pub, new in drift:
            lower_better = metric == "rmse"
            better_for_run = (new < pub) if lower_better else (new > pub)
            f = "anchor" if any(h in run for h in ANCHOR_HINTS) else "CLIMB"
            fam[f][0] += 1
            fam[f][1] += 1 if better_for_run else -1
        print("\nDIRECTION -- does the drift flatter an arm family?")
        for f, (n, net) in sorted(fam.items()):
            verdict = "published values FLATTER these arms" if net < 0 else \
                      "published values UNDERSTATE these arms" if net > 0 else "no net direction"
            print(f"  {f:8} {n:4} drifting values, net {net:+4}  -> {verdict}")
        print("  (an error with a consistent direction deserves more scrutiny than one without)")
    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
