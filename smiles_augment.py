"""SMILES enumeration (data augmentation).

Randomized non-canonical SMILES for the same molecule. Used on-the-fly during MLM
pretraining (see data_v2.RawSmilesMLMCollator) so a molecule re-visited across the
token budget is presented as a *different* valid SMILES each time — the enumeration
regularization that fixed-canonical pretraining cannot provide (Bjerrum 2017; MolFormer).

A pre-registered lever: augmentation ∈ {"canonical", "enumerated"}.
"""

from __future__ import annotations

import random
from typing import Optional

from rdkit import Chem
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")  # silence per-molecule parse warnings in the hot loop


def randomize_smiles(smiles: str, rng: Optional[random.Random] = None) -> str:
    """Return a random valid SMILES for the same molecule. Falls back to the input
    (canonicalized if possible) on parse failure so the training stream never breaks.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return smiles
    try:
        # RDKit's doRandom draws a random atom ordering from its own RNG each call,
        # giving a fresh valid SMILES per visit. `rng` is accepted for API symmetry.
        return Chem.MolToSmiles(mol, doRandom=True, canonical=False)
    except Exception:
        try:
            return Chem.MolToSmiles(mol)  # canonical fallback
        except Exception:
            return smiles
