from __future__ import annotations
from functools import lru_cache
import numpy as np, pandas as pd
from scipy import stats






@lru_cache(maxsize=300000)
def _scaffold(smiles):
    """Bemis–Murcko generic scaffold SMILES (falls back to the input on any RDKit failure)."""
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold
    try:
        m = Chem.MolFromSmiles(smiles)
        if m is None:
            return smiles
        return MurckoScaffold.MurckoScaffoldSmiles(mol=m, includeChirality=False) or smiles
    except Exception:
        return smiles

def _auc(y, p):
    y = np.asarray(y); n1 = int((y == 1).sum()); n0 = int((y == 0).sum())
    if n1 == 0 or n0 == 0:
        return np.nan
    r = stats.rankdata(p)
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)

def _nef1(y, p):
    y = np.asarray(y); N = len(y); A = int((y == 1).sum())
    if A == 0 or N == 0:
        return np.nan
    n = int(np.ceil(0.01 * N))
    top = np.argsort(-np.asarray(p, dtype=float))[:n]
    return int((y[top] == 1).sum()) / min(n, A)

def _metric_over_cols(sub, pred_col, kind):
    """Metric on a set of rows, averaged over output columns (Tox21=12), NaN-safe."""
    vals = []
    for _, g in sub.groupby("output_index"):
        yt = g["y_true_a"].to_numpy(); pr = g[pred_col].to_numpy(float)
        ok = np.isfinite(yt)
        if ok.sum() == 0:
            continue
        yt, pr = yt[ok], pr[ok]
        v = (np.sqrt(np.mean((pr - yt) ** 2)) if kind == "rmse"
             else _auc(yt, pr) if kind == "auc" else _nef1(yt, pr))
        if np.isfinite(v):
            vals.append(v)
    return float(np.mean(vals)) if vals else np.nan
