"""Dump frozen encoder embeddings + RDKit descriptors for the representation analysis.

Feeds two analyses that ask what the SUPERVISED (descriptor-regression) objective actually learns,
given that fig_E already shows its value IS the molecule->label correspondence (permuting the
targets preserves p(y) exactly and lands BELOW the untrained floor on all six panels):

  1. PCA of the embedding coloured by molecular properties      -> scripts/si_representation.py
  2. Descriptor-residual probing (is it more than a descriptor calculator?)

PROTOCOL -- matched to the paper's frozen probe so these embeddings are the SAME vectors fig_A
scores. `moleculenet_summary.csv` records featurizer=encoder, pool=mean, standardize=zscore, so
embeddings are MEAN-POOLED over non-pad tokens (not CLS, despite the checkpoint's
classifier_pooling="cls" -- that field is unused by our eval path).

IN-SET vs OUT-OF-SET. The MTR objective regressed RDKit's FULL `Descriptors.descList` (217
descriptors), so MolWt / MolLogP / TPSA are all IN the training set: finding them in the supervised
embedding tests that the training loss converged, not a hypothesis. Genuinely out-of-set properties
must come from outside that list, so we add:
    - Asphericity, RadiusOfGyration  (rdkit.Chem.Descriptors3D -- need an ETKDG conformer, so they
      are not a deterministic function of any 2D descriptor)
    - the benchmark's own label       (activity / atomization energy -- never seen in pretraining)

Writes: figure_data/_repr/<dataset>_embeddings.npz  (Z per arm, descriptor matrix, labels, smiles)
Run:    python3 scripts/embed_probe_dump.py --dataset BACE
"""
from __future__ import annotations
import argparse, sys, warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

FD = ROOT / "figure_data"
OUT = FD / "_repr"
TOKENIZER = ROOT / "paper_artifacts" / "derived" / "tokenizer"
# arm key -> the phase-2 run dirs holding its frozen encoder, ONE PER PRETRAINING SEED.
# Every conclusion here is about an objective, not a checkpoint, so each arm must be replicated
# across the same 3 pretraining seeds the rest of the paper uses.
ARMS = {"unsup": ["unsup_8M", "unsup_8M_s1", "unsup_8M_s2"],
        "sup_desc": ["skip_dense_8M", "skip_dense_8M_s1", "skip_dense_8M_s2"]}
# ECFP4 is not an encoder -- it is the classical baseline the paper anchors on (fig_A1 #2 overall).
# Included so "how descriptor-like is this representation?" has a reference point that is, by
# construction, pure structure and no learned property knowledge.
ECFP_BITS, ECFP_RADIUS = 2048, 2
PRED_RUN = "random_baseline_00"          # any run: we only need its raw_smiles / y_true columns


def load_molecules(dataset: str):
    """Unique (smiles, label) for one benchmark, from any run's OOF dump."""
    p = FD / "climb_v2_phase2" / PRED_RUN / "moleculenet_cv" / "test_predictions.csv"
    d = pd.read_csv(p)
    d = d[d.dataset == dataset]
    if "output_index" in d.columns:
        d = d[d.output_index == 0]
    d = d.drop_duplicates(subset=["raw_smiles"])
    return d.raw_smiles.tolist(), d.y_true.to_numpy(float)


def embed(smiles, run_dir: Path, batch=128):
    """Mean-pooled frozen embeddings [N, H]."""
    import torch
    from transformers import AutoTokenizer, AutoModel
    tk = AutoTokenizer.from_pretrained(str(TOKENIZER))
    model = AutoModel.from_pretrained(str(run_dir / "encoder"))
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(smiles), batch):
            enc = tk(smiles[i:i + batch], padding=True, truncation=True,
                     max_length=256, return_tensors="pt")
            # ModernBERT has no segment embeddings; the tokenizer emits token_type_ids anyway
            enc.pop("token_type_ids", None)
            h = model(**enc).last_hidden_state                     # [B, T, H]
            mask = enc["attention_mask"].unsqueeze(-1).float()     # mask-aware mean pool
            out.append(((h * mask).sum(1) / mask.sum(1).clamp(min=1)).cpu().numpy())
            print(f"    {min(i + batch, len(smiles))}/{len(smiles)}", end="\r", flush=True)
    print()
    return np.vstack(out).astype(np.float32)


def ecfp(smiles):
    """ECFP4 bit matrix [N, 2048] -- the classical structural baseline."""
    from rdkit import Chem, RDLogger
    from rdkit.Chem import rdFingerprintGenerator
    RDLogger.DisableLog("rdApp.*")
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=ECFP_RADIUS, fpSize=ECFP_BITS)
    out = np.zeros((len(smiles), ECFP_BITS), dtype=np.float32)
    for i, smi in enumerate(smiles):
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            out[i] = np.asarray(gen.GetFingerprintAsNumPy(mol), dtype=np.float32)
    return out


def descriptors_2d(smiles):
    """The 217 TRAINED descriptors, plus their names."""
    from descriptors_v2 import rdkit_descriptors, descriptor_names
    return rdkit_descriptors(smiles), descriptor_names()


def descriptors_3d(smiles):
    """Asphericity + RadiusOfGyration from an ETKDG conformer -- NOT in Descriptors.descList,
    so not a deterministic function of anything the model was trained to predict."""
    from rdkit import Chem, RDLogger
    from rdkit.Chem import AllChem, Descriptors3D
    RDLogger.DisableLog("rdApp.*")
    names = ["Asphericity", "RadiusOfGyration"]
    out = np.full((len(smiles), len(names)), np.nan, dtype=np.float32)
    for i, smi in enumerate(smiles):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        mol = Chem.AddHs(mol)
        try:
            if AllChem.EmbedMolecule(mol, randomSeed=0, maxAttempts=5) != 0:
                continue
            AllChem.MMFFOptimizeMolecule(mol, maxIters=200)
            out[i] = [Descriptors3D.Asphericity(mol), Descriptors3D.RadiusOfGyration(mol)]
        except Exception:
            pass
        if i % 200 == 0:
            print(f"    3D {i}/{len(smiles)}", end="\r", flush=True)
    print()
    return out, names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="BACE")
    ap.add_argument("--max-mols", type=int, default=4000)
    ap.add_argument("--skip-3d", action="store_true")
    a = ap.parse_args()

    smiles, y = load_molecules(a.dataset)
    if len(smiles) > a.max_mols:                       # deterministic subsample, keeps PCA readable
        idx = np.random.default_rng(0).choice(len(smiles), a.max_mols, replace=False)
        idx.sort()
        smiles = [smiles[i] for i in idx]
        y = y[idx]
    print(f"{a.dataset}: {len(smiles)} unique molecules")

    Z = {}
    for arm, runs in ARMS.items():
        for k, run in enumerate(runs):
            d = FD / "climb_v2_phase2" / run
            if not (d / "encoder" / "model.safetensors").exists():
                print(f"  SKIP {arm} seed{k}: no encoder at {run}")
                continue
            print(f"  embedding {arm} seed{k} ({run})")
            Z[f"{arm}_s{k}"] = embed(smiles, d)
    print("  ECFP4 (classical structural baseline, no encoder)")
    Z["ecfp_s0"] = ecfp(smiles)

    print("  RDKit 2D descriptors (the 217 trained targets)")
    D, dnames = descriptors_2d(smiles)
    if a.skip_3d:
        D3, d3names = np.zeros((len(smiles), 0), np.float32), []
    else:
        print("  RDKit 3D descriptors (ETKDG conformers; NOT trained on)")
        D3, d3names = descriptors_3d(smiles)

    OUT.mkdir(parents=True, exist_ok=True)
    f = OUT / f"{a.dataset}_embeddings.npz"
    np.savez_compressed(f, smiles=np.array(smiles, dtype=object), y=y,
                        D=D, dnames=np.array(dnames, dtype=object),
                        D3=D3, d3names=np.array(d3names, dtype=object),
                        **{f"Z_{k}": v for k, v in Z.items()})
    print(f"\nwrote {f.relative_to(ROOT)}")
    for k, v in Z.items():
        print(f"  Z_{k}: {v.shape}")
    print(f"  D: {D.shape} (trained)   D3: {D3.shape} (not trained)")


if __name__ == "__main__":
    main()
