"""Wong S. aureus antibacterial screen -- 5-fold scaffold CV, NEF1% + ROC-AUC, any fig_A arm.

WHY ITS OWN RUNNER. Wong is not in either suite track and ships NO split column, so it gets our
scaffold folds like every other dataset in the universe. 39,121 molecules, 503 positives (1.29%).
At that prevalence NEF1% is the informative metric and plain accuracy is meaningless; ROC-AUC is
reported beside it. Compare CBS at 0.41% -- the same regime -- and BBBP at 76.5%, where nef1 pins
at exactly 1.0 for every feature block including plain fingerprints and discriminates nothing.

THE LABELS ARE PRE-BINARISED AND ARE NOT RE-DERIVED. y == (mean_relative < 0.2) was verified
row-by-row across all 39,121 rows: zero violations. The three sibling wong_* directories are human
cell-line cytotoxicity counter-screens, not antibacterial targets, and are deliberately untouched.

PER-FOLD VALUES ARE EMITTED. The folds are real here (unlike the suite tracks' fixed splits), and
they are free at emission and impossible to recover afterwards -- which is exactly what forced the
fig_F re-run.
"""
from __future__ import annotations
import argparse, csv, json, os, sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); os.chdir(ROOT)

K_FOLDS = 5
CSV = ROOT / "chemeleon_suite" / "data" / "wong_saureus.csv"


def load() -> tuple[list, np.ndarray]:
    rows = list(csv.DictReader(CSV.open()))
    smiles = [r["smiles"] for r in rows]
    y = np.array([[float(r["y"])] for r in rows], dtype=np.float64)
    # ASSERT THE LABEL RULE RATHER THAN TRUST THE COLUMN -- cheap, and it caught nothing here
    # (0 violations of y == mean_relative < 0.2), which is the point: now it is known, not assumed.
    bad = sum(1 for r in rows if int(r["y"]) != int(float(r["mean_relative"]) < 0.2))
    if bad:
        raise SystemExit(f"{bad} rows violate y == (mean_relative < 0.2); refusing to score a "
                         f"label column that does not match its own definition")
    print(f"[wong] {len(smiles)} molecules, {int(y.sum())} positives "
          f"({100 * y.mean():.2f}%), label rule verified on every row", flush=True)
    return smiles, y


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--featurizer", required=True,
                    choices=["ecfp4", "fp_desc", "chemeleon", "encoder", "hf_encoder", "npz"])
    ap.add_argument("--encoder", default=None)
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--hf_model", default=None)
    ap.add_argument("--hf_revision", default=None)
    ap.add_argument("--head", default="mlp", choices=["mlp", "linear", "xgb"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 117, 709])
    a = ap.parse_args()

    import eval_v2 as E
    from heads_v2 import make_head, compute_metric, compute_nef
    from scripts.chemeleon_suite_run import make_featurizer, NPZ_META

    smiles, y = load()
    feat, std = make_featurizer(a.featurizer, a.encoder, a.tokenizer,
                                hf_model=a.hf_model, hf_revision=a.hf_revision)
    X = np.asarray(feat(smiles), dtype=np.float32)
    print(f"[wong] features {X.shape} (standardize={std})", flush=True)

    folds = E._scaffold_kfold_indices(smiles, K_FOLDS, 0)
    out = ROOT / "figure_data" / "wong_saureus" / a.model
    out.mkdir(parents=True, exist_ok=True)
    fold_rows = []
    for seed in a.seeds:
        for j in range(K_FOLDS):
            te = np.array(folds[j])
            pool = np.array([i for f in range(K_FOLDS) if f != j for i in folds[f]])
            rng = np.random.default_rng(seed); perm = rng.permutation(len(pool))
            nv = max(1, int(0.1 * len(pool)))
            va, tr = pool[perm[:nv]], pool[perm[nv:]]
            Z = X
            if std == "zscore":
                mu, sd = np.nanmean(X[tr], 0), np.nanstd(X[tr], 0)
                sd[sd == 0] = 1.0
                Z = (X - mu) / sd
            h = make_head(a.head, "classification", 1, seed).fit(Z[tr], y[tr], Z[va], y[va])
            p = np.asarray(h.predict(Z[te]), dtype=np.float64).reshape(-1, 1)
            nef = float(compute_nef(p, y[te]))
            auc = float(compute_metric(p, y[te], "classification"))
            npos = int(y[te].sum())
            fold_rows.append(dict(task="Wong", model=a.model, seed=seed, fold=j,
                                  metric="nef1", value=round(nef, 6), n_test=len(te), n_pos=npos))
            fold_rows.append(dict(task="Wong", model=a.model, seed=seed, fold=j,
                                  metric="roc_auc", value=round(auc, 6), n_test=len(te), n_pos=npos))
            print(f"  seed {seed} fold {j}: nef1={nef:.4f} roc_auc={auc:.4f} "
                  f"({npos} positives of {len(te)})", flush=True)
    rows = []
    for m in ("nef1", "roc_auc"):
        v = [r["value"] for r in fold_rows if r["metric"] == m]
        rows.append(dict(task="Wong", model=a.model, metric=m,
                         mean=round(float(np.mean(v)), 4), std=round(float(np.std(v)), 4), n=len(v)))
    with (out / "results.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["task", "model", "metric", "mean", "std", "n"])
        w.writeheader(); w.writerows(rows)
    with (out / "fold_values.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["task", "model", "seed", "fold", "metric", "value",
                                          "n_test", "n_pos"])
        w.writeheader(); w.writerows(fold_rows)
    (out / "verified.json").write_text(json.dumps({
        "task": "Wong", "model": a.model, "featurizer": a.featurizer, "head": a.head,
        "seeds": a.seeds, "n_folds": K_FOLDS, "n_cells": len(a.seeds) * K_FOLDS,
        "metric_primary": "nef1", "positive_rate": round(float(y.mean()), 5),
        **({"hf_model": a.hf_model} if a.hf_model else {}),
        **({"hf_revision": a.hf_revision} if a.hf_revision else {}),
        **({"features_npz": a.encoder, "npz_provenance": NPZ_META} if a.featurizer == "npz" else {}),
    }, indent=2))
    for r in rows:
        print(f"[wong] {a.model} {r['metric']}={r['mean']:.4f} +/- {r['std']:.4f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
