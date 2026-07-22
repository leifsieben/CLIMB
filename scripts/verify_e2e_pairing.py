"""Prove that a new run's per-molecule predictions PAIR with an existing arm's.

The §8.1 significance tests merge two arms' `test_predictions.csv` on
`(dataset, mol_index, output_index)`. If the two runs disagree about what molecule
index 7 of BACE is — different split, different loader, different fold seed — the merge
still succeeds, silently, on whatever rows happen to collide, and every p-value below it
is meaningless. Nothing else in the pipeline catches that.

So this checks the merge explicitly and on identity, not just on size: rows must line up
one-to-one AND `canonical_key` must agree on every merged row.

    python scripts/verify_e2e_pairing.py --new e2e_random_00 --ref random_baseline_00
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

KEYS = ["dataset", "mol_index", "output_index"]


def _load(run_dir: Path, scheme: str):
    p = run_dir / scheme / "test_predictions.csv"
    if not p.exists():
        return None
    d = pd.read_csv(p)
    return d.drop_duplicates(KEYS)


def check(root: Path, new_run: str, ref_run: str, schemes=("moleculenet", "moleculenet_cv")):
    report = {"new": new_run, "ref": ref_run, "schemes": {}, "ok": True}
    for scheme in schemes:
        a, b = _load(root / new_run, scheme), _load(root / ref_run, scheme)
        s = {"tasks": {}}
        if a is None or b is None:
            s["fatal"] = f"missing test_predictions.csv ({'new' if a is None else 'ref'})"
            report["schemes"][scheme] = s
            report["ok"] = False
            continue
        # A task one side never ran is out of scope, not misaligned — report it separately
        # so a deliberate scope decision cannot masquerade as a pairing failure (and, more
        # importantly, so a real misalignment is never excused as "probably out of scope").
        s["absent_from_new"] = sorted(set(b.dataset) - set(a.dataset))
        s["absent_from_ref"] = sorted(set(a.dataset) - set(b.dataset))
        for task in sorted(set(a.dataset) & set(b.dataset)):
            ta, tb = a[a.dataset == task], b[b.dataset == task]
            m = ta.merge(tb, on=KEYS, suffixes=("_new", "_ref"))
            key_mismatch = int((m.canonical_key_new != m.canonical_key_ref).sum()) if len(m) else 0
            # y_true must also agree: same molecule, same label, same (already-normalized)
            # target scale. A mismatch means the two runs saw different DeepChem transforms.
            ytol = float((m.y_true_new - m.y_true_ref).abs().max()) if len(m) else float("nan")
            row = {"n_new": len(ta), "n_ref": len(tb), "n_matched": len(m),
                   "canonical_key_mismatches": key_mismatch,
                   "max_abs_y_true_diff": ytol}
            ok = (len(ta) == len(tb) == len(m)) and key_mismatch == 0 and (
                not (ytol == ytol) or ytol < 1e-4)
            row["ok"] = bool(ok)
            if not ok:
                report["ok"] = False
            s["tasks"][task] = row
        report["schemes"][scheme] = s
    return report


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="figure_data/climb_v2_phase2")
    p.add_argument("--new", required=True)
    p.add_argument("--ref", default="random_baseline_00")
    args = p.parse_args()
    rep = check(Path(args.root), args.new, args.ref)
    print(json.dumps(rep, indent=2))
    raise SystemExit(0 if rep["ok"] else 3)


if __name__ == "__main__":
    main()
