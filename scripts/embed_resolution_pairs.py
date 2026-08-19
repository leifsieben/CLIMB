"""Compute every embedding for the resolution pairs and score how often each one RESOLVES a pair.

Scoring, deliberately in two parts because they answer different questions:

  COLLISION (metric-free, airtight). Are the two vectors bit-for-bit identical? If so the
  representation has destroyed the difference outright -- no head, however good, can recover it.
  This is the only claim that needs no threshold and no choice of distance, and it is the one that
  caught the stereo bug.

  SEPARATION (graded, comparable across spaces). cosine distance between A and B, divided by the
  MEDIAN cosine distance from A to 1,000 random background molecules. Raw distances are not
  comparable across a 2048-bit binary vector, a 768-d CLM space and CheMeleon's 2048-d, so every
  distance is expressed relative to that embedding's own typical scale. A ratio of 0.001 means the
  pair sits a thousand times closer than a random molecule; 1.0 means the perturbation moved the
  molecule as far as picking a different molecule entirely.

Class A pairs are DIFFERENT molecules: a collision is a failure.
Class B pairs are the SAME molecule: any nonzero distance is a failure, because it is an artefact
of how the molecule was written rather than of chemistry.

Cosine is used for every embedding so the numbers sit on one axis; Tanimoto is reported alongside
for the binary fingerprints, where it is the conventional choice, as a check that the ranking does
not depend on that decision.
"""
from __future__ import annotations
import csv, json, os, random, sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

OUT = Path("figure_data/embedding_resolution")
EMB_DIR = OUT / "embeddings"; EMB_DIR.mkdir(parents=True, exist_ok=True)
PAIRS = OUT / "pairs.csv"
BACKGROUND_N = 1000
SEED = 0

# (name, kind, spec) -- kind decides how the vectors are produced
EMBEDDINGS = [
    ("ECFP4+stereo",     "fp",      dict(variant="ecfp4_stereo", desc=False)),
    ("ECFP4+desc",       "fp",      dict(variant="ecfp4_stereo", desc=True)),
    ("CLIMB sup",        "encoder", "figure_data/climb_v2_phase2/skip_dense_8M/encoder"),
    ("CLIMB unsup",      "encoder", "figure_data/climb_v2_phase2/unsup_8M/encoder"),
    ("CheMeleon",        "npz",     "figure_data/embedding_resolution/chemeleon_pairs.npz"),
    # ---- controls ----
    ("random encoder",   "encoder", "figure_data/climb_v2_phase2/random_baseline_00/encoder"),
    ("ECFP4 stereo-blind", "fp",    dict(variant="ecfp4_legacy", desc=False)),
]


def load_pairs():
    return list(csv.DictReader(PAIRS.open()))


def background_smiles(rng, n):
    import build_resolution_pairs as B          # same pool, same construction
    pool = B.load_pool()
    return rng.sample(pool, min(n, len(pool)))


def featurize(name, kind, spec, smiles):
    if kind == "fp":
        os.environ["FP_VARIANT"] = spec["variant"]
        from featurize_v2 import ecfp4_features
        X = np.asarray(ecfp4_features(smiles), dtype=np.float32)
        if spec["desc"]:
            from descriptors_v2 import rdkit_descriptors
            D = np.asarray(rdkit_descriptors(list(smiles)), dtype=np.float32)
            D[~np.isfinite(D)] = 0.0
            # standardize the descriptor block before concatenating: raw descriptors span 20+
            # orders of magnitude, so an unstandardized concatenation is a descriptor-only
            # embedding wearing a fingerprint's name.
            mu, sd = D.mean(0), D.std(0); sd[sd == 0] = 1.0
            X = np.concatenate([X, (D - mu) / sd], axis=1)
        return X
    if kind == "encoder":
        import torch
        from transformers import AutoTokenizer, ModernBertModel
        from eval_v2 import _encoder_features
        tok = AutoTokenizer.from_pretrained("figure_data/_tokenizer")
        enc = ModernBertModel.from_pretrained(spec, attn_implementation="sdpa",
                                              reference_compile=False).eval()
        return np.asarray(_encoder_features(enc, tok, smiles, torch.device("cpu"), "mean", 256),
                          dtype=np.float32)
    if kind == "npz":
        p = Path(spec)
        if not p.exists():
            print(f"  [{name}] {p} absent -- run scripts/embed_chemeleon_box.py on a "
                  f"chemprop>=2.2 host and copy it here. Skipping.", flush=True)
            return None
        z = np.load(p, allow_pickle=True)
        table = {str(s): z["X"][i] for i, s in enumerate(z["smiles"])}
        miss = [s for s in smiles if s not in table]
        if miss:
            print(f"  [{name}] {len(miss)} SMILES absent from the table, e.g. {miss[:2]} -- "
                  f"skipping rather than mean-filling", flush=True)
            return None
        return np.asarray([table[s] for s in smiles], dtype=np.float32)
    raise ValueError(kind)


def cosine(A, B):
    na = np.linalg.norm(A, axis=1, keepdims=True); na[na == 0] = 1.0
    nb = np.linalg.norm(B, axis=1, keepdims=True); nb[nb == 0] = 1.0
    return 1.0 - np.sum((A / na) * (B / nb), axis=1)


def tanimoto(A, B):
    inter = np.sum(np.minimum(A, B), axis=1)
    union = np.sum(np.maximum(A, B), axis=1)
    union[union == 0] = 1.0
    return 1.0 - inter / union


def main() -> int:
    rng = random.Random(SEED)
    pairs = load_pairs()
    bg = background_smiles(rng, BACKGROUND_N)
    uniq = sorted({r["smiles_a"] for r in pairs} | {r["smiles_b"] for r in pairs} | set(bg))
    idx = {s: i for i, s in enumerate(uniq)}
    print(f"{len(pairs)} pairs, {len(uniq)} unique molecules "
          f"(incl. {len(bg)} background)\n")
    (OUT / "molecules.txt").write_text("\n".join(uniq))

    ia = np.array([idx[r["smiles_a"]] for r in pairs])
    ib = np.array([idx[r["smiles_b"]] for r in pairs])
    ibg = np.array([idx[s] for s in bg])

    rows = []
    for name, kind, spec in EMBEDDINGS:
        print(f"=== {name} ===", flush=True)
        X = featurize(name, kind, spec, uniq)
        if X is None:
            continue
        # Persist the raw vectors, not just the derived distances: any later question -- a
        # different metric, a probe, an adversarial split -- needs the embeddings themselves, and
        # recomputing them means re-standing-up a chemprop box for CheMeleon.
        np.savez_compressed(EMB_DIR / f"{name.replace(' ', '_').replace('+', '_')}.npz",
                            smiles=np.array(uniq, dtype=object), X=X)
        A, B = X[ia], X[ib]
        d_cos = cosine(A, B)
        d_tan = tanimoto(A, B) if kind == "fp" else np.full(len(pairs), np.nan)
        exact = np.all(A == B, axis=1)
        # background scale: median cosine from each anchor to 1,000 random molecules
        BG = X[ibg]
        nb = np.linalg.norm(BG, axis=1); nb[nb == 0] = 1.0
        BGn = BG / nb[:, None]
        na = np.linalg.norm(A, axis=1); na[na == 0] = 1.0
        med_bg = np.median(1.0 - (A / na[:, None]) @ BGn.T, axis=1)
        med_bg[med_bg == 0] = 1.0
        for r, dc, dt, ex, mb in zip(pairs, d_cos, d_tan, exact, med_bg):
            rows.append(dict(embedding=name, mode=r["mode"], klass=r["klass"],
                             pair_id=r["pair_id"], cosine=float(dc),
                             tanimoto=(None if np.isnan(dt) else float(dt)),
                             identical=bool(ex), bg_median=float(mb),
                             separation=float(dc / mb)))
        print(f"  identical pairs: {int(exact.sum())}/{len(pairs)}", flush=True)

    with (OUT / "distances.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["embedding", "mode", "klass", "pair_id", "cosine",
                                          "tanimoto", "identical", "bg_median", "separation"])
        w.writeheader(); w.writerows(rows)
    print(f"\nwrote {OUT/'distances.csv'}: {len(rows)} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
