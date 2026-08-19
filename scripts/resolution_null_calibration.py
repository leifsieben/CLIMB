"""Calibrate ONE threshold per representation: how far apart does it put a molecule from ITSELF?

The definition we want is "farther apart than could be explained by noise". That needs a null, and
the null has to be a difference that carries NO chemical information at all. The one available for
every representation is re-writing the same molecule: a random SMILES ordering denotes exactly the
same compound, so any distance a representation produces between two writings of it is that
representation's own noise.

For each of 200 anchor molecules we generate 10 random re-writings, embed all of them, and pool
the anchor-to-rewriting separations. The 95th percentile of that pooled distribution is the
representation's threshold: a pair further apart than this has under a 5% chance of being noise.

This is a MEASURED threshold, not a chosen one, and it is different for each representation --
which is the point. A fingerprint is exactly invariant, so its threshold is 0 and any difference
counts. A SMILES model is not, so it has to clear its own instability before a chemical difference
counts as resolved. That comparison is the experiment.

One honest overlap to state rather than hide: the class B mode `smiles_enumeration` is the same
KIND of transformation as the null, so it scores near 5% by construction. `kekule` and
`symmetry_equivalent` are different transformations and stay informative. The molecules used here
are disjoint from the reported pairs.
"""
from __future__ import annotations
import json, os, random, sys
from pathlib import Path

import numpy as np
from rdkit import Chem
from rdkit import RDLogger; RDLogger.DisableLog("rdApp.*")

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT); sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
OUT = Path("figure_data/embedding_resolution")
N_ANCHOR, N_REWRITE, SEED = 200, 10, 0


def build():
    rng = random.Random(SEED)
    import csv
    anchors = sorted({r["smiles_a"] for r in csv.DictReader((OUT / "pairs.csv").open())})
    rng.shuffle(anchors)
    picked, groups = [], {}
    for s in anchors:
        if len(picked) >= N_ANCHOR:
            break
        m = Chem.MolFromSmiles(s)
        alts = set()
        for _ in range(N_REWRITE * 4):
            w = Chem.MolToSmiles(m, canonical=False, doRandom=True)
            if Chem.MolFromSmiles(w) is not None and w != s:
                alts.add(w)
            if len(alts) >= N_REWRITE:
                break
        if len(alts) >= 3:
            picked.append(s); groups[s] = sorted(alts)
    return picked, groups


def main() -> int:
    picked, groups = build()
    mols = sorted(set(picked) | {w for v in groups.values() for w in v})
    (OUT / "null_molecules.txt").write_text("\n".join(mols))
    json.dump({"_all_unique": mols}, (OUT / "_null_smiles.json").open("w"))
    json.dump(groups, (OUT / "null_groups.json").open("w"))
    print(f"{len(picked)} anchors, {len(mols)} molecules total -> null_molecules.txt")
    print("Embed CheMeleon for these on a chemprop>=2.2 host, then re-run with --score")
    return 0


def score() -> int:
    from embed_resolution_pairs import EMBEDDINGS, featurize, cosine
    groups = json.loads((OUT / "null_groups.json").read_text())
    mols = (OUT / "null_molecules.txt").read_text().split("\n")
    idx = {s: i for i, s in enumerate(mols)}
    out = {}
    for name, kind, spec in EMBEDDINGS:
        if kind == "npz":
            spec = str(OUT / "chemeleon_null.npz")
        X = featurize(name, kind, spec, mols)
        if X is None:
            continue
        # background scale, so the null is on the SAME axis as the reported separations
        bg = X[[idx[s] for s in list(groups)[:200]]]
        nb = np.linalg.norm(bg, axis=1); nb[nb == 0] = 1
        BGn = bg / nb[:, None]
        seps = []
        for a, ws in groups.items():
            A = X[[idx[a]] * len(ws)]
            W = X[[idx[w] for w in ws]]
            d = cosine(A, W)
            na = np.linalg.norm(X[idx[a]]); na = na if na else 1.0
            med_bg = float(np.median(1.0 - (X[idx[a]] / na) @ BGn.T))
            seps += list(d / (med_bg if med_bg else 1.0))
        seps = np.array(seps)
        out[name] = dict(n=len(seps), median=float(np.median(seps)),
                         p95=float(np.percentile(seps, 95)), max=float(seps.max()))
        print(f"{name:22} null median {np.median(seps):.3e}   THRESHOLD p95 "
              f"{np.percentile(seps, 95):.3e}", flush=True)
    (OUT / "null_threshold.json").write_text(json.dumps(out, indent=1))
    print(f"\nwrote {OUT/'null_threshold.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(score() if "--score" in sys.argv else main())
