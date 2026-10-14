from __future__ import annotations

import random
from typing import Optional

from rdkit import Chem
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")


def randomize_smiles(smiles: str, rng: Optional[random.Random] = None) -> str:

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return smiles
    try:


        return Chem.MolToSmiles(mol, doRandom=True, canonical=False)
    except Exception:
        try:
            return Chem.MolToSmiles(mol)
        except Exception:
            return smiles
