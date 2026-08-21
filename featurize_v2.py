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


# ---------- Morgan fingerprint anchor ----------
#
# The classical anchor must be the STRONGEST classical model we can build, not a nominal one:
# it is the comparator the paper's central claim rests on, so handicapping it would flatter the
# CLMs. Until 2026-08-19 it was handicapped in a way nobody had checked.
#
# WHAT WAS WRONG. GetMorganGenerator defaults includeChirality=False, so the anchor was
# STEREO-BLIND while every CLM arm sees stereochemistry (the tokenizer carries 13 '@' tokens plus
# the E/Z slash tokens, and 20.8% of the pre-training corpus is stereo-bearing). L- and D-alanine
# produced byte-identical fingerprints. The 217 RDKit descriptors are stereo-blind too, so
# fp_desc recovered nothing. On BACE, 16 groups of molecules had CONFLICTING LABELS and identical
# fingerprints; on Tox21, 82.
#
# WHAT THE SETTINGS BUY, measured as molecules sharing an identical vector across
# BACE+Tox21+QM7+10 MoleculeACE targets (29,918 unique molecules) -- an information measure, so
# nothing here was chosen by looking at downstream scores:
#
#   r=2 bits  2048 stereo OFF  (the old anchor)   572 collided   1.912%
#   r=2 bits  2048 stereo ON                      375 collided   1.253%
#   r=2 COUNTS 2048 stereo ON                      37 collided   0.124%
#   r=3 counts 2048 stereo ON                      12 collided   0.040%
#   r=4 counts 2048 stereo ON                       4 collided   0.013%
#
# Two results worth keeping, because both are counter-intuitive:
#   - COUNTS, not size, is the big win. Going bits -> counts cuts collisions 10x. Raising fpSize
#     from 2048 to 4096 or 8192 changes the count by ZERO at every radius: the collisions are
#     distinct molecules with identical substructure ENVIRONMENTS, not hash collisions, so more
#     bits cannot help. Do not "improve" this by widening the vector.
#   - includeRedundantEnvironments changes nothing either, at any radius.
#
# TWO VARIANTS ARE BUILT AND REPORTED, selected by the FP_VARIANT env var:
#
#   "ecfp4_stereo"  (DEFAULT)  radius 2, BITS, chirality on   -- the headline anchor
#   "morgan_r3_counts"         radius 3, counts, chirality on -- the max-information variant
#   "ecfp4_legacy"             radius 2, bits, chirality OFF  -- reproduces every pre-2026-08-19 number
#
# The default is deliberately the ORTHODOX one. "ECFP4" denotes a specific object -- Morgan
# radius 2, binary -- and a radius-3 count vector is not an ECFP4 with extra settings, it is a
# different descriptor. Since the anchor exists to be the classical baseline a reader recognises,
# the headline number is the textbook descriptor with the bug fixed, and the more informative
# variant is reported alongside it rather than substituted for it. Radius 4 was measured and
# rejected: it buys 8 molecules in 29,918 while adding environments that appear in a handful of
# training rows, a bad trade on the 600-molecule MoleculeACE targets.
#
# The variant is resolved ONCE per call and PRINTED, because this function is invoked across
# subprocess boundaries where a silently defaulted setting would produce plausible, unlabelled
# numbers -- the failure mode this repo keeps re-finding.

FP_VARIANTS = {
    "ecfp4_stereo":     dict(radius=2, counts=False, include_chirality=True),
    "morgan_r3_counts": dict(radius=3, counts=True,  include_chirality=True),
    "ecfp4_legacy":     dict(radius=2, counts=False, include_chirality=False),
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
    """Dense Morgan fingerprints -> [N, n_bits] float32, variant chosen by FP_VARIANT.

    Invalid SMILES yield an all-zero row (logged count). Flows through the same head as the
    encoder features, so Morgan+XGBoost stays a like-for-like 'how good is a classical model'
    anchor. Explicit kwargs override the variant; passing none uses FP_VARIANT, defaulting to
    the orthodox stereo-aware ECFP4.
    """
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
