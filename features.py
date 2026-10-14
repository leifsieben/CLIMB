from __future__ import annotations

from typing import Dict, List

import numpy as np
import torch


def pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor, mode: str = "mean") -> torch.Tensor:

    if mode == "cls":
        return last_hidden_state[:, 0, :]
    mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
    summed = (last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1.0)
    mean = summed / counts
    if mode == "mean":
        return mean
    if mode == "cls_mean":
        return torch.cat([last_hidden_state[:, 0, :], mean], dim=-1)
    raise ValueError(f"Unknown pool mode: {mode!r} (expected cls|mean|cls_mean)")


def fit_standardizer(train_feats: np.ndarray, method: str = "zscore") -> Dict:

    x = np.asarray(train_feats, dtype=np.float64)
    if method == "none":
        return {"method": "none"}
    if method == "zscore":
        mean = x.mean(axis=0)
        std = x.std(axis=0)
        std = np.where(std < 1e-8, 1.0, std)
        return {"method": "zscore", "mean": mean, "std": std}
    if method == "pca_whiten":
        from sklearn.decomposition import PCA
        n_comp = int(min(x.shape[0], x.shape[1]))

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


FP_VARIANTS = {
    "ecfp4_stereo":     dict(radius=2, counts=False, include_chirality=True),
}
FP_VARIANT_DEFAULT = "ecfp4_stereo"


def _fp_settings(radius=None, include_chirality=None, counts=None):
    """Resolve explicit kwargs over FP_VARIANT over the default, and say which won."""
    import os
    name = os.environ.get("FP_VARIANT", FP_VARIANT_DEFAULT)
    if name not in FP_VARIANTS:
        raise ValueError(f"FP_VARIANT={name!r} unknown; expected one of {sorted(FP_VARIANTS)}")
    cfg = dict(FP_VARIANTS[name])
    for k, v in (("radius", radius), ("include_chirality", include_chirality), ("counts", counts)):
        if v is not None:
            cfg[k] = v
            name = f"{name}+explicit"
    return name, cfg

def ecfp4_features(smiles: List[str], n_bits: int = 2048, radius: int = None,
                   include_chirality: bool = None, counts: bool = None) -> np.ndarray:

    from rdkit import Chem
    from rdkit.Chem import rdFingerprintGenerator

    name, cfg = _fp_settings(radius, include_chirality, counts)
    radius, include_chirality, counts = cfg["radius"], cfg["include_chirality"], cfg["counts"]
    print(f"[ecfp4_features] variant={name} radius={radius} counts={counts} "
          f"chirality={include_chirality} n_bits={n_bits}", flush=True)
    gen = rdFingerprintGenerator.GetMorganGenerator(
        radius=radius, fpSize=n_bits, includeChirality=include_chirality)
    out = np.zeros((len(smiles), n_bits), dtype=np.float32)
    n_bad = 0
    for i, smi in enumerate(smiles):
        mol = Chem.MolFromSmiles(smi) if smi else None
        if mol is None:
            n_bad += 1
            continue
        fp = gen.GetCountFingerprintAsNumPy(mol) if counts else gen.GetFingerprintAsNumPy(mol)
        out[i] = fp.astype(np.float32)
    if n_bad:
        print(f"[ecfp4_features] {n_bad}/{len(smiles)} SMILES failed to parse (zero-vector rows)")
    return out
