"""Fig I1 / H9 — per-molecule novelty: max ECFP4 Tanimoto to the pretraining corpus.

H9 asks whether downstream gains are representation-learning or partly memorization: do molecules
UNLIKE anything in the pretraining corpus benefit as much as molecules similar to it? Answering
it needs, per evaluated molecule, its nearest-neighbour similarity to the pretraining corpus.

The notebook lists this as blocked on "ECFP4 fingerprints absent" plus per-molecule dumps for
"only ~2 runs". Both are stale: eval_v2 now writes test_predictions.csv with `raw_smiles` for 97
runs, and fingerprints are derivable from those SMILES on CPU. No GPU and no retraining needed.

Method: ECFP4 (Morgan radius 2, 2048 bits) for every evaluated molecule and for a random sample
of the pretraining corpus; then max Tanimoto of each molecule against the sample, computed as
packed-uint64 popcounts (np.bitwise_count) so 60k x 500k stays minutes rather than hours.

The corpus is SAMPLED, so a reported max-similarity is a LOWER BOUND on similarity to the full
~12M corpus, and the bound loosens as the sample shrinks. The sample size is written into the
output so the figure caption can state it. Binning molecules by this score stays valid because
every molecule is scored against the identical reference set.

Usage:
    python scripts/compute_tanimoto_novelty.py --ref-n 500000 --out figure_data/_tanimoto
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator

RDLogger.DisableLog("rdApp.*")          # corpus SMILES include some RDKit-unparseable entries

CORPUS = "s3://climb-s3-bucket/tokenized_sources/pubchem_filtered/"
NBITS = 2048


def _log(m: str) -> None:
    print(f"[tanimoto {time.strftime('%H:%M:%S')}] {m}", flush=True)


def _fp_gen():
    # DELIBERATELY STEREO-BLIND, unlike featurize_v2.ecfp4_features (fixed 2026-08-19 to carry
    # chirality). Different question, different answer. This is a SIMILARITY axis: the "memorized"
    # group asks "did the model effectively already read this molecule", and a stereo-blind match
    # is the LOOSER test, so it OVER-counts memorization. The finding is that the memorized group
    # shows no advantage, so over-counting is the conservative direction -- turning chirality on
    # would shrink the group and weaken a null result in our own favour. Corpus Tanimoto is also
    # conventionally computed without chirality. Do not "fix" this to match the featurizer.
    return rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=NBITS)


def _pack(smiles: list[str]) -> tuple[np.ndarray, list[str]]:
    """SMILES -> packed uint64 bit matrix (n, NBITS/64), dropping unparseable entries."""
    gen = _fp_gen()
    rows, kept = [], []
    for s in smiles:
        m = Chem.MolFromSmiles(s)
        if m is None:
            continue
        arr = np.frombuffer(gen.GetFingerprintAsNumPy(m).astype(np.uint8).tobytes(), dtype=np.uint8)
        rows.append(np.packbits(arr).view(np.uint64))
        kept.append(s)
    if not rows:
        return np.zeros((0, NBITS // 64), dtype=np.uint64), []
    return np.vstack(rows), kept


def _load_corpus(ref_n: int, cache: Path) -> list[str]:
    cache.mkdir(parents=True, exist_ok=True)
    local = cache / "shard_00000.parquet"
    if not local.exists():
        _log("downloading one corpus shard")
        subprocess.run(["aws", "s3", "cp", f"{CORPUS}shard_00000.parquet", str(local)],
                       check=True, capture_output=True)
    import pandas as pd
    df = pd.read_parquet(local, columns=["SMILES_canonical"])
    smi = df["SMILES_canonical"].dropna().astype(str).tolist()
    _log(f"corpus shard holds {len(smi):,} molecules")
    if len(smi) > ref_n:
        rng = np.random.default_rng(0)                 # fixed seed: reference set is reproducible
        smi = [smi[i] for i in rng.choice(len(smi), ref_n, replace=False)]
    return smi


# The canonical six-panel suite uses a SECOND prediction schema. eval_v2 writes
# raw_smiles/dataset, but the chemeleon_suite runners (MoleculeACE, Polaris) write task/smiles --
# so a plain raw_smiles reader silently skipped every MoleculeACE and Ames molecule and this table
# stayed MoleculeNet-only. That is what left fig_C1/fig_C2 with no x-axis for the canonical panels.
# Both schemas are read here; unknown files are skipped as before rather than guessed at.
POLARIS_TASK = "tdcommons/ames"          # the Polaris panel; the track scores 28 tasks, we bin one


def _collect_eval_molecules() -> list[str]:
    seen = {}

    def add(smi, label):
        smi = (smi or "").strip()
        if smi and smi not in seen:
            seen[smi] = label

    for p in sorted(Path("figure_data").glob("**/test_predictions.csv")):
        parts = set(p.parts)
        try:
            with open(p) as fh:
                rd = csv.DictReader(fh)
                cols = set(rd.fieldnames or [])
                if "raw_smiles" in cols:                      # eval_v2 schema (MolNet + CBS)
                    for row in rd:
                        ds = (row.get("dataset") or "").strip()
                        add(row.get("raw_smiles"), "CBS" if ds.lower() == "cbs" else ds)
                elif "smiles" in cols:                        # chemeleon_suite schema
                    if "moleculeace" in parts:
                        # pooled over the 30 targets, matching the MoleculeACE PANEL the figures plot
                        for row in rd:
                            add(row.get("smiles"), "MoleculeACE")
                    elif "polaris" in parts:
                        for row in rd:
                            if (row.get("task") or "").strip() == POLARIS_TASK:
                                add(row.get("smiles"), "Ames")
        except Exception as e:
            _log(f"  skip {p}: {e}")
    return list(seen), seen


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref-n", type=int, default=500_000)
    ap.add_argument("--out", default="figure_data/_tanimoto")
    ap.add_argument("--query-limit", type=int, default=0, help="0 = all (debug aid)")
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    qs, ds_of = _collect_eval_molecules()
    if a.query_limit:
        qs = qs[:a.query_limit]
    _log(f"evaluated molecules: {len(qs):,}")

    ref_smi = _load_corpus(a.ref_n, out / "_cache")
    _log(f"fingerprinting {len(ref_smi):,} corpus molecules")
    ref, _ = _pack(ref_smi)
    ref_pc = np.bitwise_count(ref).sum(axis=1).astype(np.int32)
    _log(f"reference matrix {ref.shape}")

    _log("fingerprinting evaluated molecules")
    q, q_kept = _pack(qs)
    q_pc = np.bitwise_count(q).sum(axis=1).astype(np.int32)
    _log(f"query matrix {q.shape} ({len(qs) - len(q_kept)} unparseable dropped)")

    best = np.zeros(len(q_kept), dtype=np.float32)
    t0 = time.time()
    for i in range(len(q_kept)):
        inter = np.bitwise_count(ref & q[i]).sum(axis=1).astype(np.int32)
        union = ref_pc + q_pc[i] - inter
        np.maximum(best[i], (inter / np.maximum(union, 1)).max(), out=best[i:i + 1])
        if i and i % 5000 == 0:
            el = time.time() - t0
            _log(f"  {i:,}/{len(q_kept):,}  ({el:.0f}s, eta {el/i*(len(q_kept)-i):.0f}s)")

    csv_path = out / "corpus_similarity.csv"
    with open(csv_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["raw_smiles", "dataset", "max_tanimoto_to_corpus"])
        for s, b in zip(q_kept, best):
            w.writerow([s, ds_of.get(s, ""), f"{b:.4f}"])

    meta = {
        "reference": "pubchem_filtered shard_00000, random subsample, seed 0",
        "reference_n": int(ref.shape[0]),
        "fingerprint": f"ECFP4 (Morgan r=2, {NBITS} bits)",
        "n_molecules_scored": len(q_kept),
        "caveat": ("max similarity is a LOWER BOUND on similarity to the full ~12M corpus, "
                   "since the reference is a subsample; all molecules share one reference set, "
                   "so relative binning is unaffected"),
        "quantiles": {str(p): float(np.percentile(best, p)) for p in (5, 25, 50, 75, 95)},
    }
    (out / "corpus_similarity_meta.json").write_text(json.dumps(meta, indent=2))
    _log(f"wrote {csv_path}  median max-Tanimoto={np.median(best):.3f}")
    _log(json.dumps(meta["quantiles"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
