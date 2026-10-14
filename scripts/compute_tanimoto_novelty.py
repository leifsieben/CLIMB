from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from dedup_i1_reanalysis import _popcount_rows
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator

RDLogger.DisableLog("rdApp.*")

CORPUS = "raw_data/pubchem_filtered"
NBITS = 2048


def _log(m: str) -> None:
    print(f"[tanimoto {time.strftime('%H:%M:%S')}] {m}", flush=True)


def _fp_gen():


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
    local = Path(CORPUS) / "shard_00000.parquet"
    import pandas as pd
    df = pd.read_parquet(local, columns=["SMILES_canonical"])
    smi = df["SMILES_canonical"].dropna().astype(str).tolist()
    _log(f"corpus shard holds {len(smi):,} molecules")
    if len(smi) > ref_n:
        rng = np.random.default_rng(0)
        smi = [smi[i] for i in rng.choice(len(smi), ref_n, replace=False)]
    return smi


POLARIS_TASK = "tdcommons/ames"


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
                if "raw_smiles" in cols:
                    for row in rd:
                        ds = (row.get("dataset") or "").strip()
                        add(row.get("raw_smiles"), "CBS" if ds.lower() == "cbs" else ds)
                elif "smiles" in cols:
                    if "moleculeace" in parts:

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
    ref_pc = _popcount_rows(ref).sum(axis=1).astype(np.int32)
    _log(f"reference matrix {ref.shape}")

    _log("fingerprinting evaluated molecules")
    q, q_kept = _pack(qs)
    q_pc = _popcount_rows(q).sum(axis=1).astype(np.int32)
    _log(f"query matrix {q.shape} ({len(qs) - len(q_kept)} unparseable dropped)")

    best = np.zeros(len(q_kept), dtype=np.float32)
    t0 = time.time()
    for i in range(len(q_kept)):
        inter = _popcount_rows(ref & q[i]).sum(axis=1).astype(np.int32)
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
