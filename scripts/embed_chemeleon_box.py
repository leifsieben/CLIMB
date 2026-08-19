"""Turn an exported SMILES list into CheMeleon vectors. Runs in a chemprop>=2.2 environment.

Deliberately dumb: it reads strings and writes vectors. No dataset loader, no folds, no scoring
-- those all stay in the reference environment, which is the whole point of splitting the step.
"""
from __future__ import annotations
import json, sys
from pathlib import Path

import numpy as np

BATCH = 512


def main(inp: str, out: str) -> int:
    payload = json.loads(Path(inp).read_text())
    smiles = payload["_all_unique"]
    from chemeleon_fingerprint import CheMeleonFingerprint
    from rdkit.Chem import MolFromSmiles
    fp = CheMeleonFingerprint(device="cpu")

    ok = [s for s in smiles if MolFromSmiles(s) is not None]
    bad = len(smiles) - len(ok)
    print(f"{len(smiles)} SMILES, {bad} RDKit-unparseable (dropped -- the caller RAISES on a "
          f"missing lookup rather than silently mean-filling)", flush=True)

    vecs = []
    for i in range(0, len(ok), BATCH):
        vecs.append(np.asarray(fp(ok[i:i + BATCH]), dtype=np.float32))
        print(f"  {min(i + BATCH, len(ok))}/{len(ok)}", flush=True)
    X = np.concatenate(vecs, axis=0)
    np.savez_compressed(out, smiles=np.array(ok, dtype=object), X=X)
    print(f"wrote {out}: {X.shape}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1], sys.argv[2]))
