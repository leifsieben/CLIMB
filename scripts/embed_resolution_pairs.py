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
# RESOLUTION_INPUT=canonical runs the whole thing on RDKit-canonical strings. That is the fair
# setup for the class A chemistry questions: the notation modes are a separate question, and
# leaving them mixed in charges a sequence model for a difference the pipeline could remove.
_CANON = os.environ.get("RESOLUTION_INPUT") == "canonical"
PAIRS = OUT / ("pairs_canonical.csv" if _CANON else "pairs.csv")
MOLS = OUT / ("molecules_canonical.txt" if _CANON else "molecules.txt")
SUFFIX = "_canonical" if _CANON else ""
# 10,000, up from 1,000. sigma_j is estimated once per representation on this set and then
# divides every reported number, so it is the cheapest precision in the whole measurement.
BACKGROUND_N = 10000
SEED = 0

# (name, kind, spec) -- kind decides how the vectors are produced
EMBEDDINGS = [
    ("ECFP4+stereo",     "fp",      dict(variant="ecfp4_stereo", desc=False)),
    ("ECFP4+desc",       "fp",      dict(variant="ecfp4_stereo", desc=True)),
    ("Morgan r3-counts", "fp",      dict(variant="morgan_r3_counts", desc=False)),
    ("Morgan r3-cnt+desc", "fp",    dict(variant="morgan_r3_counts", desc=True)),
    ("CLIMB sup",        "encoder", "figure_data/climb_v2_phase2/skip_dense_8M/encoder"),
    ("CLIMB unsup",      "encoder", "figure_data/climb_v2_phase2/unsup_8M/encoder"),
    # CheMeleon is RETIRED (figures.arms.RETIRED) and is not re-embedded here: its vectors need a
    # chemprop>=2.2 host, and the stored npz covers only the superseded 100-pair molecule set. The
    # old npz files are kept in figure_data/embedding_resolution/ rather than deleted.
    # CLIMB unsup pretrained WITH SMILES-enumeration augmentation. The mainline arms are
    # augmentation="canonical" (checked in their own config.yaml and metadata.json), so this is
    # the only encoder in the repo that saw randomized SMILES during pretraining -- it answers
    # directly whether augmentation buys notation-invariance, rather than leaving it assumed.
    ("CLIMB unsup (enum-aug)", "encoder",
     "figure_data/climb_v2_h1/scaling_enumerated_fracfull_s0/encoder"),
    # ---- controls ----
    ("CLIMB unsup (canon-ctrl)", "encoder",
     "figure_data/climb_v2_h1/scaling_canonical_fracfull_s0/encoder"),
    ("random encoder",   "encoder", "figure_data/climb_v2_phase2/random_baseline_00/encoder"),
    ("ECFP4 stereo-blind", "fp",    dict(variant="ecfp4_legacy", desc=False)),
]


def load_pairs():
    return list(csv.DictReader(PAIRS.open()))


def background_smiles(rng, n, exclude):
    """n molecules from the same pool, none of which appears in any reported pair.

    Excluding the paired molecules matters: sigma_j is the per-dimension spread this measurement
    divides by, so estimating it partly on the molecules under test would let an edit shrink its
    own yardstick.
    """
    import build_resolution_pairs as B          # same pool, same construction
    pool = [s for s in B.load_pool() if s not in exclude]
    if len(pool) < n:
        print(f"  [background] pool has {len(pool)} unpaired molecules, wanted {n}", flush=True)
    return rng.sample(pool, min(n, len(pool)))


def featurize(name, kind, spec, smiles):
    if kind == "fp":
        os.environ["FP_VARIANT"] = spec["variant"]
        from featurize_v2 import ecfp4_features
        X = np.asarray(ecfp4_features(smiles), dtype=np.float32)
        if spec["desc"]:
            from descriptors_v2 import rdkit_descriptors
            D = np.asarray(rdkit_descriptors(list(smiles)), dtype=np.float32)
            # Ipc overflows float32 for ~0.6% of drug-like molecules, and where it does not it
            # still reaches 1.2e32 -- 28 orders of magnitude above every other descriptor. Zeroing
            # the overflows and then standardizing on the raw column leaves sigma set by that one
            # value, so every other molecule lands at z ~ 0 and the column is dead. Standardize on
            # the finite entries and clip, matching descriptors_v2.normalize(clip=10).
            D = np.where(np.isfinite(D), D, np.nan)
            # standardize the descriptor block before concatenating: raw descriptors span 20+
            # orders of magnitude, so an unstandardized concatenation is a descriptor-only
            # embedding wearing a fingerprint's name.
            mu = np.nanmean(D, axis=0); sd = np.nanstd(D, axis=0)
            mu = np.where(np.isfinite(mu), mu, 0.0)
            sd = np.where(np.isfinite(sd) & (sd > 1e-8), sd, 1.0)
            Z = np.nan_to_num(np.clip((D - mu) / sd, -10.0, 10.0), nan=0.0)
            X = np.concatenate([X, Z], axis=1)
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
        X, S = z["X"], z["smiles"]   # hoisted: np.load on .npz is lazy, z["X"] re-decodes each time
        table = {str(s): X[i] for i, s in enumerate(S)}
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
    paired = {r["smiles_a"] for r in pairs} | {r["smiles_b"] for r in pairs}
    bg = background_smiles(rng, BACKGROUND_N, paired)
    # Both branches now build the molecule list from the pairs and the background. It used to be
    # read from a fixed file in the canonical branch so the rows lined up with the CheMeleon npz;
    # CheMeleon is retired, and that coupling made the background un-growable.
    uniq = sorted(paired | set(bg))
    idx = {s: i for i, s in enumerate(uniq)}
    print(f"{len(pairs)} pairs, {len(uniq)} unique molecules "
          f"(incl. {len(bg)} background)\n")
    MOLS.write_text("\n".join(uniq))

    ia = np.array([idx[r["smiles_a"]] for r in pairs])
    ib = np.array([idx[r["smiles_b"]] for r in pairs])
    ibg = np.array([idx[s] for s in bg])

    rows = []
    for name, kind, spec in EMBEDDINGS:
        print(f"=== {name} ===", flush=True)
        if kind == "npz" and _CANON:
            spec = "figure_data/embedding_resolution/chemeleon_canonical.npz"
        X = featurize(name, kind, spec, uniq)
        if X is None:
            continue
        # Persist the raw vectors, not just the derived distances: any later question -- a
        # different metric, a probe, an adversarial split -- needs the embeddings themselves, and
        # recomputing them means re-standing-up a chemprop box for CheMeleon.
        np.savez_compressed(EMB_DIR / f"{name.replace(' ', '_').replace('+', '_')}{SUFFIX}.npz",
                            smiles=np.array(uniq, dtype=object), X=X)
        A, B = X[ia], X[ib]
        d_cos = cosine(A, B)
        d_tan = tanimoto(A, B) if kind == "fp" else np.full(len(pairs), np.nan)
        exact = np.all(A == B, axis=1)
        # background scale: median cosine from each anchor to the BACKGROUND_N random molecules
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

    with (OUT / f"distances{SUFFIX}.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["embedding", "mode", "klass", "pair_id", "cosine",
                                          "tanimoto", "identical", "bg_median", "separation"])
        w.writeheader(); w.writerows(rows)
    print(f"\nwrote {OUT}/distances{SUFFIX}.csv: {len(rows)} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
