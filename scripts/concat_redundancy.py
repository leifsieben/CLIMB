from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")
import os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.chdir(Path(__file__).resolve().parent.parent)
import numpy as np, pandas as pd, torch
from rdkit import RDLogger; RDLogger.DisableLog("rdApp.*")

import evaluate as E
from evaluate import ecfp4_features
from descriptors import rdkit_descriptors
from heads import make_head, compute_metric, compute_nef

TAG = os.environ.get("CONCAT_TAG") or "CLM"
OUTFILE = os.environ.get("CONCAT_OUT", "concat_redundancy.csv")


ENC = os.environ.get("CONCAT_ENC", "figure_data/climb_v2_phase2/unsup_8M/encoder")
TOK = "figure_data/_tokenizer"


_TASK_SEL = os.environ.get("CONCAT_TASKS", "").split()
TASKS = [("QM7", "regression"), ("BACE", "classification"),
         ("Tox21", "classification"), ("HIV", "classification")]
if _TASK_SEL:
    TASKS = [t for t in TASKS if t[0] in _TASK_SEL]
    if not TASKS:
        raise SystemExit(f"CONCAT_TASKS={_TASK_SEL} matched no dataset")
SEEDS = [0]
K, ML = 5, 256
OUT = Path("analysis/rigor"); OUT.mkdir(parents=True, exist_ok=True)

device = torch.device("mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
                      else "cuda" if torch.cuda.is_available() else "cpu")
print("device:", device, flush=True)
from transformers import ModernBertModel, PreTrainedTokenizerFast
encoder = ModernBertModel.from_pretrained(ENC, attn_implementation="sdpa", reference_compile=False).to(device)
encoder.eval()
tok = PreTrainedTokenizerFast.from_pretrained(TOK)


def feature_sets(smiles):

    fp = np.asarray(ecfp4_features(smiles), dtype=np.float32)
    d = np.asarray(rdkit_descriptors(list(smiles)), dtype=np.float32)
    d = np.asarray(d, dtype=np.float32); d[~np.isfinite(d)] = np.nan


    if _BLOCK_SEL and not any(TAG in b for b in _BLOCK_SEL):
        emb = None
    else:
        emb = E._encoder_features(encoder, tok, list(smiles), device, "mean", ML).astype(np.float32)
    if emb is None:
        return {"fp": fp, "desc": d, "fp+desc": np.concatenate([fp, d], 1)}
    return {"fp": fp, "desc": d, "fp+desc": np.concatenate([fp, d], 1), TAG: emb,
            f"fp+{TAG}": np.concatenate([fp, emb], 1),
            f"desc+{TAG}": np.concatenate([d, emb], 1),
            f"fp+desc+{TAG}": np.concatenate([fp, d, emb], 1)}


_BLOCK_SEL = [b for b in os.environ.get("CONCAT_BLOCKS", "").split(",") if b]


def _select(F):
    if not _BLOCK_SEL:
        return F
    missing = [b for b in _BLOCK_SEL if b not in F]
    if missing:
        raise SystemExit(f"CONCAT_BLOCKS names unknown blocks {missing}; have {sorted(F)}")
    return {k: v for k, v in F.items() if k in _BLOCK_SEL}


rows = []
fold_rows = []
for ds, tt in TASKS:
    s_all, y_all = E._load_moleculenet_full(ds)
    y_all = np.asarray(y_all, dtype=np.float64)
    if y_all.ndim == 1:
        y_all = y_all[:, None]
    n_out = y_all.shape[1]
    print(f"\n[{ds}] {len(s_all)} molecules, {n_out} output(s) — featurizing…", flush=True)
    F = feature_sets(s_all)
    F = _select(F)
    folds = E._scaffold_kfold_indices(s_all, K, 0)
    for name, X in F.items():
        fm, fn = [], []
        for j in range(K):
            test = np.array(folds[j])
            pool_ = np.array([i for f in range(K) if f != j for i in folds[f]])
            rng = np.random.default_rng(0); perm = rng.permutation(len(pool_))
            nv = max(1, int(0.1 * len(pool_)))
            va, tr = pool_[perm[:nv]], pool_[perm[nv:]]
            preds = []
            for sd in SEEDS:
                h = make_head("xgb", tt, n_out, sd).fit(X[tr], y_all[tr], X[va], y_all[va])
                preds.append(np.asarray(h.predict(X[test]), dtype=np.float64))
            pred = np.mean(preds, axis=0)
            fm.append(compute_metric(pred, y_all[test], tt))
            if tt == "classification":
                fn.append(compute_nef(pred, y_all[test]))
        met = "rmse" if tt == "regression" else "roc_auc"
        rows.append(dict(task=ds, features=name, metric=met,
                         mean=round(float(np.mean(fm)), 4), std=round(float(np.std(fm)), 4)))
        if fn:
            rows.append(dict(task=ds, features=name, metric="nef1",
                             mean=round(float(np.mean(fn)), 4), std=round(float(np.std(fn)), 4)))


        for j, v in enumerate(fm):
            fold_rows.append(dict(task=ds, features=name, metric=met, fold=j, value=round(float(v), 6)))
        for j, v in enumerate(fn):
            fold_rows.append(dict(task=ds, features=name, metric="nef1", fold=j, value=round(float(v), 6)))
        print(f"  {name:12} {met}={np.mean(fm):.4f}±{np.std(fm):.4f}", flush=True)

pd.DataFrame(rows).to_csv(OUT / OUTFILE, index=False)
FOLDFILE = OUTFILE.replace(".csv", "_folds.csv")
pd.DataFrame(fold_rows).to_csv(OUT / FOLDFILE, index=False)
print(f"\nDONE -> analysis/rigor/{OUTFILE}  (+ {FOLDFILE}, {len(fold_rows)} per-fold rows)", flush=True)
