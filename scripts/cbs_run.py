"""CBS inhibitor virtual screen -- the dataset's OWN provided folds, NEF1% + ROC-AUC, any fig_A arm.

WHY NOT SCAFFOLD FOLDS HERE, unlike Wong. CBS ships a `fold` column derived from UMAP clustering
(the `cluster` column is its basis), and that split is part of the benchmark's definition. Replacing
it with our scaffold folds would make our CBS numbers incomparable to every CBS number already in
the paper. Wong is the opposite case: it ships no split at all, so it gets ours.

METRIC. 0.41% positives -- the lowest-prevalence set in the universe -- so NEF1% is the informative
choice and ROC-AUC is reported beside it. For contrast, BBBP sits at 76.5% positives and nef1 pins
at exactly 1.0 there for every feature block including plain fingerprints, discriminating nothing.

PER-FOLD VALUES ARE EMITTED: the folds are real and given, so pairing is available downstream, and
they cannot be recovered after the fact.
"""
from __future__ import annotations
import argparse, csv, json, os, sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); os.chdir(ROOT)

CSV = ROOT / "data" / "cbs.csv"


def load():
    rows = list(csv.DictReader(CSV.open()))
    smiles = [r["smiles"] for r in rows]
    y = np.array([[float(r["y"])] for r in rows], dtype=np.float64)
    fold = np.array([int(r["fold"]) for r in rows])
    print(f"[cbs] {len(smiles)} molecules, {int(y.sum())} positives ({100 * y.mean():.2f}%), "
          f"{len(set(fold.tolist()))} provided folds", flush=True)
    return smiles, y, fold


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

    from heads_v2 import make_head, compute_metric, compute_nef
    from scripts.chemeleon_suite_run import make_featurizer, prepare_fold, NPZ_META

    smiles, y, fold = load()
    feat, std = make_featurizer(a.featurizer, a.encoder, a.tokenizer,
                                hf_model=a.hf_model, hf_revision=a.hf_revision)
    X = np.asarray(feat(smiles), dtype=np.float32)
    print(f"[cbs] features {X.shape} (standardize={std})", flush=True)

    out = ROOT / "figure_data" / "cbs" / a.model
    out.mkdir(parents=True, exist_ok=True)
    fold_rows = []
    for seed in a.seeds:
        for f in sorted(set(fold.tolist())):
            te = np.where(fold == f)[0]
            pool = np.where(fold != f)[0]
            rng = np.random.default_rng(seed); perm = rng.permutation(len(pool))
            nv = max(1, int(0.1 * len(pool)))
            va, tr = pool[perm[:nv]], pool[perm[nv:]]
            Z = prepare_fold(X, tr, a.head, std)
            h = make_head(a.head, "classification", 1, seed).fit(Z[tr], y[tr], Z[va], y[va])
            p = np.asarray(h.predict(Z[te]), dtype=np.float64).reshape(-1, 1)
            nef = float(compute_nef(p, y[te]))
            auc = float(compute_metric(p, y[te], "classification"))
            npos = int(y[te].sum())
            for m, v in (("nef1", nef), ("roc_auc", auc)):
                fold_rows.append(dict(task="CBS", model=a.model, seed=seed, fold=int(f),
                                      metric=m, value=round(v, 6), n_test=len(te), n_pos=npos))
            print(f"  seed {seed} fold {f}: nef1={nef:.4f} roc_auc={auc:.4f} "
                  f"({npos} positives of {len(te)})", flush=True)
    rows = []
    for m in ("nef1", "roc_auc"):
        v = [r["value"] for r in fold_rows if r["metric"] == m]
        rows.append(dict(task="CBS", model=a.model, metric=m,
                         mean=round(float(np.mean(v)), 4), std=round(float(np.std(v)), 4), n=len(v)))
    with (out / "results.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["task", "model", "metric", "mean", "std", "n"])
        w.writeheader(); w.writerows(rows)
    with (out / "fold_values.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["task", "model", "seed", "fold", "metric", "value",
                                           "n_test", "n_pos"])
        w.writeheader(); w.writerows(fold_rows)
    (out / "verified.json").write_text(json.dumps({
        "task": "CBS", "model": a.model, "featurizer": a.featurizer, "head": a.head,
        "seeds": a.seeds, "cv_scheme": "provided", "metric_primary": "nef1",
        "n_cells": len(fold_rows) // 2, "positive_rate": round(float(y.mean()), 5),
        **({"hf_model": a.hf_model} if a.hf_model else {}),
        **({"hf_revision": a.hf_revision} if a.hf_revision else {}),
        **({"features_npz": a.encoder, "npz_provenance": NPZ_META} if a.featurizer == "npz" else {}),
    }, indent=2))
    for r in rows:
        print(f"[cbs] {a.model} {r['metric']}={r['mean']:.4f} +/- {r['std']:.4f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
