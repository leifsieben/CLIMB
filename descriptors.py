from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


def _descriptor_functions():
    from rdkit.Chem import Descriptors
    return list(Descriptors.descList)


_DESC_FNS = None


def descriptor_names() -> List[str]:
    global _DESC_FNS
    if _DESC_FNS is None:
        _DESC_FNS = _descriptor_functions()
    return [name for name, _ in _DESC_FNS]


def rdkit_descriptors(smiles: List[str]) -> np.ndarray:

    from rdkit import Chem
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")

    global _DESC_FNS
    if _DESC_FNS is None:
        _DESC_FNS = _descriptor_functions()
    D = len(_DESC_FNS)
    out = np.full((len(smiles), D), np.nan, dtype=np.float32)
    for i, smi in enumerate(smiles):
        mol = Chem.MolFromSmiles(smi) if smi else None
        if mol is None:
            continue
        for j, (_, fn) in enumerate(_DESC_FNS):
            try:
                v = fn(mol)
            except Exception:
                continue
            if v is not None and np.isfinite(v):
                out[i, j] = v
    return out


def fit_descriptor_stats(smiles_sample: List[str]) -> Dict:
    """Fit per-descriptor mean/std over a sample (finite entries only)."""
    feats = rdkit_descriptors(smiles_sample)
    mean = np.nanmean(feats, axis=0)
    std = np.nanstd(feats, axis=0)
    std = np.where(~np.isfinite(std) | (std < 1e-8), 1.0, std)
    mean = np.where(np.isfinite(mean), mean, 0.0)
    return {"names": descriptor_names(), "mean": mean.tolist(), "std": std.tolist()}


def save_stats(stats: Dict, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(stats))


def load_stats(path: str) -> Dict:
    s = json.loads(Path(path).read_text())
    s["mean"] = np.asarray(s["mean"], dtype=np.float64)
    s["std"] = np.asarray(s["std"], dtype=np.float64)
    return s


def normalize(feats: np.ndarray, stats: Dict, clip: float = 10.0) -> np.ndarray:
    """(x-mean)/std, clipped; non-finite stays NaN so the MTR loss can mask it."""
    x = np.asarray(feats, dtype=np.float64)
    z = (x - stats["mean"]) / stats["std"]
    z = np.clip(z, -clip, clip)
    z[~np.isfinite(x)] = np.nan
    return z.astype(np.float32)


def n_descriptors() -> int:
    return len(descriptor_names())
