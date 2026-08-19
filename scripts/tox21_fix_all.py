"""Categorise every run by Tox21 state and rebuild the fixable ones into moleculenet_cv_tox21fixed/.

Three states, distinguished by evidence rather than by which tree a file came from:
  A) predictions MASKED (77,864 Tox21 rows) but summary disagrees  -> STALE SUMMARY, rebuildable
     here from the run's own predictions. No GPU.
  B) predictions MASKED and summary agrees                          -> already correct, untouched.
  C) predictions UNMASKED (93,876 rows)                             -> the PREDICTIONS predate the
     2026-08-05 masking fix, so nothing in the directory can produce the corrected number. These
     need a re-eval from the checkpoint and are only REPORTED here, never guessed at.

Non-destructive by construction: writes moleculenet_cv_tox21fixed/moleculenet_summary.csv beside
the original and never modifies moleculenet_cv/. Only Tox21 roc_auc/nef1 fold + MEAN/STD rows are
recomputed; *_cell rows are per-(seed,fold) and cannot come from a seed-ensembled dump, so they are
carried through unchanged and reported.
"""
from __future__ import annotations
import csv, io, json, sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from tox21_rescore_from_preds import rebuild, DS  # noqa: E402

MASKED_ROWS = 77864          # 93,876 cells - 16,012 missing (locked by tests/test_moleculenet_labels.py)


def summary_tox21(p: Path):
    try:
        for r in csv.reader(p.open()):
            if len(r) > 9 and r[0] == DS and r[6] == "roc_auc" and r[7] == "MEAN":
                return float(r[9])
    except Exception:
        pass
    return None


def main() -> int:
    dirs = [Path(x) for x in Path("/tmp/tox21_dirs.txt").read_text().split()]
    rep = {"A_rebuilt": [], "B_already_ok": [], "C_needs_reeval": [], "no_tox21": []}
    for d in dirs:
        pf, sf = d / "test_predictions.csv", d / "moleculenet_summary.csv"
        if not pf.exists() or not sf.exists():
            continue
        nrows = sum(1 for line in pf.open() if line.startswith("Tox21,"))
        if nrows == 0:
            rep["no_tox21"].append(str(d)); continue
        cur = summary_tox21(sf)
        key = f"{d.parts[1]}/{d.parts[2]}"
        if nrows != MASKED_ROWS:
            rep["C_needs_reeval"].append({"run": key, "pred_rows": nrows, "summary": cur})
            continue
        new, err = rebuild(d)
        if err:
            rep["no_tox21"].append(str(d)); continue
        newv = new[("roc_auc", "MEAN")]
        if cur is not None and abs(cur - newv) <= 1e-4:
            rep["B_already_ok"].append({"run": key, "value": round(newv, 4)}); continue
        # rebuild into a NEW subdir
        out = d.parent / "moleculenet_cv_tox21fixed"
        out.mkdir(exist_ok=True)
        rows = list(csv.reader(sf.open()))
        header, body = rows[0], rows[1:]
        changed = 0
        for r in body:
            if len(r) > 9 and r[0] == DS and (r[6], r[7]) in new:
                r[9] = repr(new[(r[6], r[7])]); changed += 1
        buf = io.StringIO(); w = csv.writer(buf, lineterminator="\n")
        w.writerow(header); w.writerows(body)
        (out / "moleculenet_summary.csv").write_text(buf.getvalue())
        rep["A_rebuilt"].append({"run": key, "was": round(cur, 4) if cur else None,
                                 "now": round(newv, 4), "rows_updated": changed})
    Path("analysis/tox21_fix_report.json").write_text(json.dumps(rep, indent=2))
    print(f"A rebuilt (stale summary, fixed)   : {len(rep['A_rebuilt'])}")
    print(f"B already correct (untouched)      : {len(rep['B_already_ok'])}")
    print(f"C predictions pre-fix -> NEED RE-EVAL: {len(rep['C_needs_reeval'])}")
    print(f"no Tox21 rows                      : {len(rep['no_tox21'])}")
    if rep["A_rebuilt"]:
        d = [abs(x["now"] - x["was"]) for x in rep["A_rebuilt"] if x["was"]]
        print(f"  delta on rebuilt: min={min(d):.4f} max={max(d):.4f} mean={sum(d)/len(d):.4f}")
    import collections
    print("  C by wave:", dict(collections.Counter(x['run'].split('/')[0] for x in rep['C_needs_reeval'])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
