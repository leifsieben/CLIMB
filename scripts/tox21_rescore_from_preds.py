"""Recompute Tox21 summary rows from each run's OWN test_predictions.csv.

WHY THIS EXISTS. The 2026-08-05 fix (79a0dfb) masked Tox21's 16,012 missing multi-assay cells,
which DeepChem encodes as y=0,w=0 rather than NaN; before it, those were scored as true inactives.
Masking them RAISES Tox21 AUC by ~+0.015..0.020 (notes/corrections-tox21-regression-2026-08-05.md).
The re-score regenerated each run's test_predictions.csv but did NOT rewrite the Tox21 rows of
moleculenet_summary.csv everywhere, so within one directory the predictions are POST-fix and the
summary is PRE-fix -- and the .corrected_v2.json markers sit next to pre-fix numbers, so they prove
nothing. Symptom: a CI centre computed from predictions disagrees with the bar computed from the
summary, by ~+0.02-0.03, on every arm.

WHAT IT CAN AND CANNOT REBUILD. test_predictions.csv stores the SEED-ENSEMBLED out-of-fold
prediction (one y_pred per molecule x assay), so this reproduces the `roc_auc`/`nef1` fold rows and
their MEAN/STD exactly. It CANNOT reproduce `roc_auc_cell`/`nef1_cell`, which are per-(seed,fold)
and need the individual seeds' predictions -- those are not in the dump. Cell rows are left
untouched and flagged, never silently recomputed from the wrong quantity.

Fold membership is not in the dump (`split` is always "test"), so it is regenerated deterministically
with eval_v2._scaffold_kfold_indices(smiles, 5, 0, labels=y) -- the same reconstruction
a2_bootstrap_errorbars.py uses, and the same call eval_v2 made when the run was scored.

Default is DRY-RUN and it writes to a NEW file unless --in-place is given.
"""
from __future__ import annotations
import argparse, csv, io, sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import eval_v2  # noqa: E402
from heads_v2 import compute_metric, compute_nef  # noqa: E402

DS = "Tox21"


def rebuild(run_dir: Path):
    """Return {(metric, head_seed): value} for the Tox21 rows this dump can rebuild."""
    pf = run_dir / "test_predictions.csv"
    rows = [r for r in csv.DictReader(pf.open()) if r["dataset"] == DS]
    if not rows:
        return None, "no Tox21 prediction rows"
    ncol = max(int(r["output_index"]) for r in rows) + 1
    mols, order = {}, []
    for r in rows:
        mi = int(r["mol_index"])
        if mi not in mols:
            mols[mi] = {"smi": r["raw_smiles"], "y": np.full(ncol, np.nan), "p": np.full(ncol, np.nan)}
            order.append(mi)
        j = int(r["output_index"])
        yt = r["y_true"]
        mols[mi]["y"][j] = float(yt) if yt not in ("", "nan", "None") else np.nan
        mols[mi]["p"][j] = float(r["y_pred"])
    smiles = [mols[m]["smi"] for m in order]
    Y = np.vstack([mols[m]["y"] for m in order])
    P = np.vstack([mols[m]["p"] for m in order])

    folds = eval_v2._scaffold_kfold_indices(smiles, 5, 0, labels=Y)
    out, aucs, nefs = {}, [], []
    for j, idx in enumerate(folds):
        idx = np.asarray(idx, dtype=int)
        a = compute_metric(P[idx], Y[idx], "classification")
        n = compute_nef(P[idx], Y[idx])
        out[("roc_auc", f"fold{j}")] = a
        out[("nef1", f"fold{j}")] = n
        aucs.append(a); nefs.append(n)
    out[("roc_auc", "MEAN")] = float(np.mean(aucs)); out[("roc_auc", "STD")] = float(np.std(aucs))
    out[("nef1", "MEAN")] = float(np.mean(nefs));   out[("nef1", "STD")] = float(np.std(nefs))
    return out, None


def merge(run_dir: Path, new: dict, in_place: bool):
    """Rewrite ONLY the Tox21 rows this dump can rebuild; every other row byte-identical."""
    sp = run_dir / "moleculenet_summary.csv"
    text = sp.read_text()
    rd = csv.reader(io.StringIO(text))
    all_rows = list(rd)
    header, body = all_rows[0], all_rows[1:]
    i_ds, i_met, i_seed, i_val = 0, 6, 7, 9
    changed = 0
    for r in body:
        if len(r) <= i_val or r[i_ds] != DS:
            continue
        k = (r[i_met], r[i_seed])
        if k in new:
            if abs(float(r[i_val]) - new[k]) > 1e-12:
                r[i_val] = repr(new[k]); changed += 1
    dst = sp if in_place else run_dir / "moleculenet_summary.tox21fixed.csv"
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(header); w.writerows(body)
    dst.write_text(buf.getvalue())
    return changed, dst


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dirs", nargs="+")
    ap.add_argument("--in-place", action="store_true")
    ap.add_argument("--apply", action="store_true", help="without this, nothing is written")
    a = ap.parse_args()
    for d in a.run_dirs:
        rd = Path(d)
        new, err = rebuild(rd)
        if err:
            print(f"  SKIP {rd}: {err}"); continue
        cur = None
        sp = rd / "moleculenet_summary.csv"
        if sp.exists():
            for r in csv.reader(sp.open()):
                if len(r) > 9 and r[0] == DS and r[6] == "roc_auc" and r[7] == "MEAN":
                    cur = float(r[9]); break
        print(f"  {rd}: summary={cur if cur is None else round(cur,4)} "
              f"-> from_predictions={round(new[('roc_auc','MEAN')],4)}")
        if a.apply:
            n, dst = merge(rd, new, a.in_place)
            print(f"      wrote {dst.name} ({n} Tox21 values updated)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
