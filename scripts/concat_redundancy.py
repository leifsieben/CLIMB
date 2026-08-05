"""Redundancy test: does the CLIMB embedding add anything on top of fingerprints+descriptors?

Reviewer's highest-value experiment. Feed the SAME XGBoost four feature sets under the SAME 5-fold
scaffold CV, per task:
    fp+desc            (the classical baseline = the fp_desc anchor)
    CLM                (unsup_8M masked-mean embedding, the headline CLM)
    desc+CLM           (clean ablation: does the embedding beat descriptors alone, no fp dilution)
    fp+desc+CLM        (concat: does the embedding add anything to the full baseline)

If fp+desc+CLM does not beat fp+desc, the 512-d embedding is informationally redundant given
fingerprints+descriptors — a much stronger claim than "the CLM scores lower". XGBoost is
scale-invariant, so raw features are concatenated (no standardization). Native-unit regression
targets; Tox21 missing labels are already NaN-masked by the fixed loader.

Writes analysis/rigor/concat_redundancy.csv.
"""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np, pandas as pd, torch
from rdkit import RDLogger; RDLogger.DisableLog("rdApp.*")

import eval_v2 as E
from eval_v2 import ecfp4_features
from descriptors_v2 import rdkit_descriptors
from heads_v2 import make_head, compute_metric, compute_nef
from transformers import ModernBertModel, PreTrainedTokenizerFast

ENC = "figure_data/climb_v2_phase2/unsup_8M/encoder"
TOK = "figure_data/_tokenizer"
TASKS = [("ESOL", "regression"), ("QM7", "regression"), ("BBBP", "classification"),
         ("BACE", "classification"), ("Tox21", "classification"), ("HIV", "classification")]
SEEDS = [0]           # XGBoost seed(s) per fold; fold spread is the error bar
K, ML = 5, 256
OUT = Path("analysis/rigor"); OUT.mkdir(parents=True, exist_ok=True)

device = torch.device("mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
                      else "cuda" if torch.cuda.is_available() else "cpu")
print("device:", device, flush=True)
encoder = ModernBertModel.from_pretrained(ENC, attn_implementation="sdpa", reference_compile=False).to(device)
encoder.eval()
tok = PreTrainedTokenizerFast.from_pretrained(TOK)


def feature_sets(smiles):
    fp = np.asarray(ecfp4_features(smiles), dtype=np.float32)
    d = np.asarray(rdkit_descriptors(list(smiles)), dtype=np.float32); d[~np.isfinite(d)] = np.nan
    emb = E._encoder_features(encoder, tok, list(smiles), device, "mean", ML).astype(np.float32)
    return {"fp+desc": np.concatenate([fp, d], 1), "CLM": emb,
            "desc+CLM": np.concatenate([d, emb], 1), "fp+desc+CLM": np.concatenate([fp, d, emb], 1)}


rows = []
for ds, tt in TASKS:
    s_all, y_all = E._load_moleculenet_full(ds)
    y_all = np.asarray(y_all, dtype=np.float64)
    if y_all.ndim == 1:
        y_all = y_all[:, None]
    n_out = y_all.shape[1]
    print(f"\n[{ds}] {len(s_all)} molecules, {n_out} output(s) — featurizing…", flush=True)
    F = feature_sets(s_all)
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
        print(f"  {name:12} {met}={np.mean(fm):.4f}±{np.std(fm):.4f}", flush=True)

pd.DataFrame(rows).to_csv(OUT / "concat_redundancy.csv", index=False)
print("\nDONE -> analysis/rigor/concat_redundancy.csv", flush=True)
