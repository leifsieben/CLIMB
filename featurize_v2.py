"""v2 featurization: pooling, embedding standardization, and the ECFP4 anchor.

Single source of truth for turning a frozen encoder's token hidden states into a
fixed-length molecule vector, and for the classical fingerprint baseline. All three
featurizers (encoder / random-encoder / ECFP4) feed the *same* downstream head
pipeline in heads_v2, so comparisons are apples-to-apples.

Design notes (these fix the v1 "more pretraining = worse" pathology):
- We pool with a masked MEAN by default, not the CLS/<s> token. MLM encoders never
  train position-0 as a pooled representation, and CLS embeddings are strongly
  anisotropic (Gao et al. 2019), so raw CLS linear-separability degrades with more
  pretraining. Masked mean is the standard strong readout.
- We standardize (z-score, optionally PCA-whiten) the pooled embeddings, fit on the
  TRAIN split only, to further counter anisotropy before the head sees them.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import torch


# ---------- pooling ----------

def pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor, mode: str = "mean") -> torch.Tensor:
    """Pool token hidden states → one vector per sequence.

    Args:
        last_hidden_state: [B, L, H]
        attention_mask: [B, L] (1 = real token, 0 = pad)
        mode: "cls" (position 0), "mean" (masked mean over real tokens),
              or "cls_mean" (concat of both → [B, 2H]).
    """
    if mode == "cls":
        return last_hidden_state[:, 0, :]
    mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)  # [B, L, 1]
    summed = (last_hidden_state * mask).sum(dim=1)                   # [B, H]
    counts = mask.sum(dim=1).clamp(min=1.0)                          # [B, 1]
    mean = summed / counts
    if mode == "mean":
        return mean
    if mode == "cls_mean":
        return torch.cat([last_hidden_state[:, 0, :], mean], dim=-1)
    raise ValueError(f"Unknown pool mode: {mode!r} (expected cls|mean|cls_mean)")


# ---------- standardization (fit on train only) ----------

def fit_standardizer(train_feats: np.ndarray, method: str = "zscore") -> Dict:
    """Fit a feature standardizer on the TRAIN features. Returns a params dict to
    pass to apply_standardizer. method ∈ {none, zscore, pca_whiten}."""
    x = np.asarray(train_feats, dtype=np.float64)
    if method == "none":
        return {"method": "none"}
    if method == "zscore":
        mean = x.mean(axis=0)
        std = x.std(axis=0)
        std = np.where(std < 1e-8, 1.0, std)  # guard constant dims
        return {"method": "zscore", "mean": mean, "std": std}
    if method == "pca_whiten":
        from sklearn.decomposition import PCA
        n_comp = int(min(x.shape[0], x.shape[1]))
        # Guard against degenerate tiny-N cases.
        n_comp = max(1, min(n_comp, x.shape[1]))
        pca = PCA(n_components=n_comp, whiten=True, svd_solver="auto", random_state=0)
        pca.fit(x)
        return {"method": "pca_whiten", "pca": pca}
    raise ValueError(f"Unknown standardize method: {method!r}")


def apply_standardizer(feats: np.ndarray, params: Dict) -> np.ndarray:
    x = np.asarray(feats, dtype=np.float64)
    method = params.get("method", "none")
    if method == "none":
        return x.astype(np.float32)
    if method == "zscore":
        return ((x - params["mean"]) / params["std"]).astype(np.float32)
    if method == "pca_whiten":
        return params["pca"].transform(x).astype(np.float32)
    raise ValueError(f"Unknown standardize method: {method!r}")


# ---------- ECFP4 / Morgan fingerprint anchor ----------

def ecfp4_features(smiles: List[str], n_bits: int = 2048, radius: int = 2) -> np.ndarray:
    """Dense ECFP4 (Morgan, radius 2) bit vectors → [N, n_bits] float32. Invalid
    SMILES yield an all-zero row (logged count). Flows through the same head as the
    encoder features so ECFP4+XGBoost is a fair 'how good is a classical model' anchor.
    """
    from rdkit import Chem
    from rdkit.Chem import rdFingerprintGenerator

    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    out = np.zeros((len(smiles), n_bits), dtype=np.float32)
    n_bad = 0
    for i, smi in enumerate(smiles):
        mol = Chem.MolFromSmiles(smi) if smi else None
        if mol is None:
            n_bad += 1
            continue
        out[i] = gen.GetFingerprintAsNumPy(mol).astype(np.float32)
    if n_bad:
        print(f"[ecfp4_features] {n_bad}/{len(smiles)} SMILES failed to parse (zero-vector rows)")
    return out
