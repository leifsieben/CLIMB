from __future__ import annotations
import csv, json, os, random, sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

OUT = Path("figure_data/embedding_resolution")
EMB_DIR = OUT / "embeddings"; EMB_DIR.mkdir(parents=True, exist_ok=True)


_CANON = os.environ.get("RESOLUTION_INPUT") == "canonical"
PAIRS = OUT / ("pairs_canonical.csv" if _CANON else "pairs.csv")
MOLS = OUT / ("molecules_canonical.txt" if _CANON else "molecules.txt")
SUFFIX = "_canonical" if _CANON else ""


BACKGROUND_N = 10000
SEED = 0


EMBEDDINGS = [
    ("ECFP4+stereo",     "fp",      dict(variant="ecfp4_stereo", desc=False)),
    ("ECFP4+desc",       "fp",      dict(variant="ecfp4_stereo", desc=True)),
    ("CLIMB sup",        "encoder", "figure_data/climb_v2_phase2/skip_dense_8M/encoder"),


    ("CLIMB unsup (enum-aug)", "encoder",
     "figure_data/climb_v2_h1/scaling_enumerated_fracfull_s0/encoder"),

    ("CLIMB unsup (canon-ctrl)", "encoder",
     "figure_data/climb_v2_h1/scaling_canonical_fracfull_s0/encoder"),
    ("random encoder",   "encoder", "figure_data/climb_v2_phase2/random_baseline_00/encoder"),


    ("notation (char n-gram)", "notation", None),
]


def load_pairs():
    return list(csv.DictReader(PAIRS.open()))


def background_smiles(rng, n, exclude):

    import build_resolution_pairs as B
    pool = [s for s in B.load_pool() if s not in exclude]
    if len(pool) < n:
        print(f"  [background] pool has {len(pool)} unpaired molecules, wanted {n}", flush=True)
    return rng.sample(pool, min(n, len(pool)))


def featurize(name, kind, spec, smiles):
    if kind == "notation":
        from sklearn.feature_extraction.text import CountVectorizer


        cv = CountVectorizer(analyzer="char", ngram_range=(1, 2), lowercase=False, min_df=2)
        return np.asarray(cv.fit_transform(list(smiles)).todense(), dtype=np.float32)
    if kind == "fp":
        os.environ["FP_VARIANT"] = spec["variant"]
        from features import ecfp4_features
        X = np.asarray(ecfp4_features(smiles), dtype=np.float32)
        if spec["desc"]:
            from descriptors import rdkit_descriptors
            D = np.asarray(rdkit_descriptors(list(smiles)), dtype=np.float32)


            D = np.where(np.isfinite(D), D, np.nan)


            mu = np.nanmean(D, axis=0); sd = np.nanstd(D, axis=0)
            mu = np.where(np.isfinite(mu), mu, 0.0)
            sd = np.where(np.isfinite(sd) & (sd > 1e-8), sd, 1.0)
            Z = np.nan_to_num(np.clip((D - mu) / sd, -10.0, 10.0), nan=0.0)
            X = np.concatenate([X, Z], axis=1)
        return X
    if kind == "encoder":
        import torch
        from transformers import AutoTokenizer, ModernBertModel
        from evaluate import _encoder_features
        tok = AutoTokenizer.from_pretrained("figure_data/_tokenizer")
        enc = ModernBertModel.from_pretrained(spec, attn_implementation="sdpa",
                                              reference_compile=False).eval()
        return np.asarray(_encoder_features(enc, tok, smiles, torch.device("cpu"), "mean", 256),
                          dtype=np.float32)
    raise ValueError(kind)






def main() -> int:
    rng = random.Random(SEED)
    pairs = load_pairs()
    paired = {r["smiles_a"] for r in pairs} | {r["smiles_b"] for r in pairs}
    bg = background_smiles(rng, BACKGROUND_N, paired)


    uniq = sorted(paired | set(bg))
    print(f"{len(pairs)} pairs, {len(uniq)} unique molecules "
          f"(incl. {len(bg)} background)\n")
    MOLS.write_text("\n".join(uniq))

    for name, kind, spec in EMBEDDINGS:
        print(f"=== {name} ===", flush=True)
        X = featurize(name, kind, spec, uniq)
        if X is None:
            continue


        np.savez_compressed(EMB_DIR / f"{name.replace(' ', '_').replace('+', '_')}{SUFFIX}.npz",
                            smiles=np.array(uniq, dtype=object), X=X)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
