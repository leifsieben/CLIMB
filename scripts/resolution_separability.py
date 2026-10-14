from __future__ import annotations
import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "figure_data" / "embedding_resolution"
EMB = OUT / "embeddings"


INPUT_OF = {"A": "_canonical", "B": ""}


SHORT = {"ECFP4+stereo": "ECFP", "ECFP4+desc": "ECFP+d",
         "CLIMB unsup (enum-aug)": "uns-ENUM", "CLIMB unsup (canon-ctrl)": "uns-CANON",
         "CLIMB sup": "sup", "random encoder": "rand",
         "notation (char n-gram)": "notation"}
MODES = {"A": ("stereo_flip", "ez_flip", "c_to_n", "add_methyl", "ch2_homologue",
               "halogen_swap", "add_fluorine", "isotope_13c", "ring_size", "regioisomer"),
         "B": ("smiles_enumeration", "kekule")}


DIRECTIONAL = {"c_to_n", "add_methyl", "add_fluorine", "isotope_13c", "ch2_homologue",
               "regioisomer",
               "smiles_enumeration", "kekule"}


SEEDS = [0, 1, 2, 3, 4]
TEST_FRAC = 0.2
PARAMS = dict(n_estimators=200, max_depth=4, learning_rate=0.3, n_jobs=8,
              tree_method="hist", verbosity=0)


def emb_file(name: str, suffix: str) -> Path:
    return EMB / f"{name.replace(' ', '_').replace('+', '_')}{suffix}.npz"


def load_lookup(name: str, suffix: str):
    """{smiles: vector} for one arm on one input convention."""
    f = emb_file(name, suffix)
    if not f.exists():
        return None
    z = np.load(f, allow_pickle=True)
    smi, X = z["smiles"], z["X"]
    assert len(smi) == len(X), f"{f.name}: {len(smi)} smiles vs {len(X)} vectors"
    return {s: i for i, s in enumerate(smi)}, X


def _groups(a, b, keep):

    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[rx] = ry

    for k in keep:
        union(("m", a[k]), ("m", b[k]))
        union(("p", k), ("m", a[k]))
    comps = {}
    for k in keep:
        comps.setdefault(find(("p", k)), []).append(k)
    return list(comps.values())


def run_cell(X_lookup, pairs, seed):

    idx, X = X_lookup
    a = [idx.get(s) for s in pairs.smiles_a]
    b = [idx.get(s) for s in pairs.smiles_b]
    keep = [k for k in range(len(pairs)) if a[k] is not None and b[k] is not None]


    degenerate = int((~(X[[a[k] for k in keep]] != X[[b[k] for k in keep]]).any(axis=1)).sum())
    if len(keep) < 50:
        return None, len(keep), 0, degenerate, 0

    comps = _groups(a, b, keep)


    largest = max(len(c) for c in comps)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(comps))
    target = max(1, int(round(TEST_FRAC * len(keep))))
    test_pairs, got = set(), 0
    for ci in order:
        if got >= target:
            break
        test_pairs.update(comps[ci])
        got += len(comps[ci])

    rows_tr, y_tr, rows_te, y_te = [], [], [], []
    for k in keep:
        dst_rows, dst_y = ((rows_te, y_te) if k in test_pairs else (rows_tr, y_tr))
        dst_rows.append(a[k]); dst_y.append(0)
        dst_rows.append(b[k]); dst_y.append(1)


    assert not (set(rows_tr) & set(rows_te)), "pair split leaked a molecule across the boundary"

    if len(set(y_te)) < 2 or not rows_tr:
        return None, len(keep), len(comps), degenerate, largest
    clf = xgb.XGBClassifier(**PARAMS, random_state=seed)
    clf.fit(X[rows_tr], np.array(y_tr))
    p = clf.predict_proba(X[rows_te])[:, 1]
    return float(roc_auc_score(y_te, p)), len(keep), len(comps), degenerate, largest


def main() -> int:
    pairs_by_input = {s: pd.read_csv(OUT / f"pairs{'_canonical' if s else ''}.csv")
                      for s in set(INPUT_OF.values())}
    rows = []
    for name in SHORT:
        for klass, suffix in INPUT_OF.items():
            look = load_lookup(name, suffix)
            if look is None:
                continue
            pt = pairs_by_input[suffix]
            for mode in MODES[klass]:
                sub = pt[(pt["mode"] == mode) & (pt.klass == klass)]
                aucs, n_used, n_grp, n_deg, n_big = [], 0, 0, 0, 0
                for seed in SEEDS:
                    auc, n_used, n_grp, n_deg, n_big = run_cell(look, sub, seed)
                    if auc is not None:
                        aucs.append(auc)
                if not aucs:
                    print(f"  SKIP {name} / {mode}: only {n_used} usable pairs")
                    continue
                rows.append(dict(
                    embedding=name, short=SHORT.get(name, ""), klass=klass, mode=mode,
                    input="canonical" if suffix else "as_written",
                    auc_mean=round(float(np.mean(aucs)), 4),
                    auc_sd=round(float(np.std(aucs, ddof=1)), 4) if len(aucs) > 1 else 0.0,
                    auc_min=round(float(np.min(aucs)), 4),
                    auc_max=round(float(np.max(aucs)), 4),
                    n_seeds=len(aucs), n_pairs=n_used, n_groups=n_grp,
                    n_degenerate=n_deg, largest_group=n_big,
                    directional=int(mode in DIRECTIONAL)))
                r = rows[-1]
                print(f"  {name:26s} {mode:20s} AUC {r['auc_mean']:.3f} "
                      f"+/- {r['auc_sd']:.3f}  (n={n_used}, groups={n_grp}"
                      f"{f', DEGENERATE {n_deg}' if n_deg else ''})")

    dst = OUT / "separability_auc.csv"
    fields = ["embedding", "short", "klass", "mode", "input", "auc_mean", "auc_sd",
              "auc_min", "auc_max", "n_seeds", "n_pairs", "n_groups", "n_degenerate",
              "largest_group", "directional"]
    with dst.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {dst}  {len(rows)} rows  (xgboost {xgb.__version__}, "
          f"{len(SEEDS)} split seeds, {int(TEST_FRAC * 100)}% held out by PAIR)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
