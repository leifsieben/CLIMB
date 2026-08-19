"""Measure each representation's NUMERICAL noise floor, so "resolved" needs no chosen threshold.

The clean definition of resolving is binary: are the two vectors different at all? For a
fingerprint that is exactly bit-equality. For a continuous embedding, bit-equality is too strict
for a reason that has nothing to do with chemistry -- summing messages or attention in a different
order changes the last bits, so CheMeleon reads as "different" on two SMILES for the same molecule
even though its real answer is zero.

So the floor is MEASURED rather than picked: embed the SAME SMILES twice, in different batch
positions, and take the largest distance observed. Any difference at or below that is
floating-point ordering, not representation. Anything above it is information the model actually
carries.

For a deterministic featurizer the floor comes out at exactly 0, and the definition collapses to
"the vectors are not bit-identical" -- the simple version, recovered as a special case rather than
assumed.
"""
from __future__ import annotations
import json, os, sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT); sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
OUT = Path("figure_data/embedding_resolution")
N_PROBE = 128


def cos(A, B):
    na = np.linalg.norm(A, axis=1, keepdims=True); na[na == 0] = 1
    nb = np.linalg.norm(B, axis=1, keepdims=True); nb[nb == 0] = 1
    return 1.0 - np.sum((A / na) * (B / nb), axis=1)


def main() -> int:
    from embed_resolution_pairs import EMBEDDINGS, featurize
    mols = (OUT / "molecules.txt").read_text().split("\n")
    probe = mols[:N_PROBE]
    shuffled = list(reversed(probe))          # same molecules, different batch positions
    floors = {}
    for name, kind, spec in EMBEDDINGS:
        X1 = featurize(name, kind, spec, probe)
        X2 = featurize(name, kind, spec, shuffled)
        if X1 is None or X2 is None:
            continue
        X2 = X2[::-1]                          # undo the permutation, align row-for-row
        d = cos(X1, X2)
        exact = int(np.sum(np.all(X1 == X2, axis=1)))
        floors[name] = dict(max_cosine=float(np.max(d)), median=float(np.median(d)),
                            bit_identical=f"{exact}/{len(probe)}")
        print(f"{name:22} floor(max cosine over {len(probe)} re-embeddings) = "
              f"{np.max(d):.3e}   bit-identical {exact}/{len(probe)}", flush=True)
    (OUT / "noise_floor.json").write_text(json.dumps(floors, indent=1))
    print(f"\nwrote {OUT/'noise_floor.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
