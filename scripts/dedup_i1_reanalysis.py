"""Note C — Fig I1 / 5d-e memorization vs representation: de-duplication reanalysis.

The concern (reviewer / friend's Note C): Fig I1 reports downstream *lift* by max-ECFP4-
Tanimoto-to-corpus bin. Exact-match molecules have Tanimoto = 1.0 by construction, so they sit
inside the top-similarity bin. The top-quartile lift is therefore a *mixture* of interpolation and
outright memorization, and the two handles §2.5 treats as independent (the disclosed 0-7% overlap
group, and the Tanimoto bins) are partly the same molecules.

This script produces the DATA the notebook (notebook_cells/22.py, ipynb session's lane) needs to
re-plot I1 with exact matches excluded, plus the memorized group reported separately. It does NOT
plot anything.

Two things are computed, against the FULL ~12M `pubchem_filtered` corpus (all 12 shards), not the
500k single-shard subsample that `compute_tanimoto_novelty.py` uses (whose max-Tanimoto is a
documented LOWER bound — useless for an exact-match question):

  1. EXACT MATCH (cheap; --mode exact): is each eval molecule literally in the corpus? Under two
     canonicalization keys, because the corpus and the leakage audit disagree and both are defensible:
       * key_nosalt : RDKit isomeric canonical, NO salt-stripping. This is exactly how the corpus
                      stores `SMILES_canonical`, so membership is a literal string lookup against
                      the corpus — the most defensible "is this molecule in the corpus as stored".
       * key_salt   : largest-fragment (salt/solvent stripped) then isomeric canonical. Matches
                      scripts/leakage_audit._ikey — the key behind the disclosed 0-7% number. Catches
                      a molecule whose salt form is in the corpus.
     Computed for ALL eval datasets (cheap), so we get a complete leakage picture, not just I1.

  2. FULL-CORPUS MAX TANIMOTO (heavy; --mode full): true max ECFP4 Tanimoto of each I1 eval molecule
     (ESOL + QM7) to the full 12M corpus, plus a count of corpus neighbours at Tanimoto==1.0 and in
     the near-duplicate band [0.95,1.0). This corrects the binning (the current bins are quantiles of
     a lower-bound score) and answers Note C's near-dup question (stereoisomers / salt forms /
     tautomers that miss the exact key but are functionally memorized). Restricted to ESOL+QM7 (the
     only tasks I1 plots) to keep it ~2h of CPU rather than ~16h over all 63k eval molecules.

Fingerprint spec is identical to compute_tanimoto_novelty.py (ECFP4 = Morgan r=2, 2048 bits, packed
uint64 popcounts) so the two are directly comparable.

Usage:
    python scripts/dedup_i1_reanalysis.py --mode exact --out analysis/dedup_i1     # local, minutes
    python scripts/dedup_i1_reanalysis.py --mode full  --out analysis/dedup_i1     # box, ~2h
    python scripts/dedup_i1_reanalysis.py --mode both  --out analysis/dedup_i1
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import subprocess
import sys
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

# np.bitwise_count is numpy >= 2.0. The AWS boxes' climb venv is numpy 1.x, where this driver died
# with AttributeError four seconds in, which is why this job was stuck on a laptop. Portable
# byte-lookup popcount fallback: identical results, and every call site immediately does
# .sum(axis=1), so returning per-BYTE counts instead of per-uint64 counts gives the same row totals.
if hasattr(np, "bitwise_count"):
    _popcount_rows = np.bitwise_count
else:                                                    # numpy 1.x fallback
    _POPCNT8 = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint16)

    def _popcount_rows(x):
        b = np.ascontiguousarray(x).view(np.uint8)
        return _POPCNT8[b].reshape(x.shape[0], -1)

CORPUS_S3 = "s3://climb-s3-bucket/tokenized_sources/pubchem_filtered/"
CACHE = Path("figure_data/_tanimoto/_cache")
SIMILARITY_CSV = Path("figure_data/_tanimoto/corpus_similarity.csv")
NBITS = 2048
# The full-corpus (all 12 shards, TRUE max) pass is restricted to the tasks fig_C1 actually plots.
# It was ESOL+QM7 for the old MoleculeNet figure; the canonical six has only two regression tasks,
# MoleculeACE and QM7, so fig_C1's canonical form needs MoleculeACE added. This list can be
# overridden with I1_TASKS="MoleculeACE QM7" to avoid re-doing tasks already computed.
#
# No schema change is needed here even though MoleculeACE predictions come from the chemeleon_suite
# runner (task/smiles) rather than eval_v2 (dataset/raw_smiles): this script reads its molecules
# from corpus_similarity.csv, which already carries MoleculeACE rows in the raw_smiles/dataset
# form, so the join keys still match the figure exactly.
I1_TASKS = tuple(os.environ.get("I1_TASKS", "ESOL QM7 MoleculeACE").split())
NEAR_DUP_LO = 0.95                  # [0.95, 1.0) = near-duplicate band (non-identical fingerprint)


def _log(m: str) -> None:
    print(f"[dedup {time.strftime('%H:%M:%S')}] {m}", flush=True)


# ----------------------------------------------------------------------------- canonical keys
def canon_nosalt(smiles: str):
    """Isomeric RDKit canonical, no salt-stripping — matches corpus SMILES_canonical."""
    m = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(m) if m is not None else None


def canon_salt(smiles: str):
    """Largest-fragment then isomeric canonical — matches leakage_audit._ikey (the 0-7% key)."""
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return None
    frags = smiles.split(".")
    if len(frags) > 1:
        m = Chem.MolFromSmiles(max(frags, key=len))
        if m is None:
            return None
    return Chem.MolToSmiles(m)


# ----------------------------------------------------------------------------- corpus shards
def _shards() -> list[Path]:
    CACHE.mkdir(parents=True, exist_ok=True)
    paths = []
    for i in range(12):
        f = CACHE / f"shard_{i:05d}.parquet"
        if not f.exists():
            _log(f"downloading {f.name}")
            subprocess.run(["aws", "s3", "cp", f"{CORPUS_S3}{f.name}", str(f)],
                           check=True, capture_output=True)
        paths.append(f)
    return paths


def _shard_smiles(shard: Path) -> list[str]:
    df = pd.read_parquet(shard, columns=["SMILES_canonical"])
    return df["SMILES_canonical"].dropna().astype(str).tolist()


# ----------------------------------------------------------------------------- eval molecules
def _eval_rows(only_i1: bool):
    """Read eval molecules from the existing Tanimoto CSV so join keys (raw_smiles) match the
    figure exactly. Returns list of dicts {raw_smiles, dataset}."""
    rows = []
    with open(SIMILARITY_CSV) as fh:
        for r in csv.DictReader(fh):
            if only_i1 and r["dataset"] not in I1_TASKS:
                continue
            rows.append({"raw_smiles": r["raw_smiles"], "dataset": r["dataset"]})
    return rows


# ----------------------------------------------------------------------------- fingerprints
def _pack(smiles: list[str]):
    from rdkit.Chem import rdFingerprintGenerator
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=NBITS)
    rows, kept_idx = [], []
    for i, s in enumerate(smiles):
        m = Chem.MolFromSmiles(s)
        if m is None:
            continue
        arr = np.frombuffer(gen.GetFingerprintAsNumPy(m).astype(np.uint8).tobytes(), dtype=np.uint8)
        rows.append(np.packbits(arr).view(np.uint64))
        kept_idx.append(i)
    if not rows:
        return np.zeros((0, NBITS // 64), dtype=np.uint64), []
    return np.vstack(rows), kept_idx


# ----------------------------------------------------------------------------- mode: exact
def run_exact(out: Path):
    rows = _eval_rows(only_i1=False)
    _log(f"eval molecules (all datasets): {len(rows):,}")

    # Canonicalize eval side once.
    for r in rows:
        r["key_nosalt"] = canon_nosalt(r["raw_smiles"])
        r["key_salt"] = canon_salt(r["raw_smiles"])
    eval_nosalt = {r["key_nosalt"] for r in rows if r["key_nosalt"]}
    eval_salt = {r["key_salt"] for r in rows if r["key_salt"]}
    _log(f"unique eval keys: {len(eval_nosalt):,} nosalt, {len(eval_salt):,} salt")

    # Stream corpus. For key_nosalt: corpus stores exactly this key, so string membership needs no
    # RDKit on the corpus side. For key_salt: a single-fragment corpus entry's stored canonical IS
    # its salt-stripped key, so string membership in the stored set covers it; only multi-fragment
    # ('.') corpus entries need re-stripping, which are rare and cheap.
    matched_nosalt: set[str] = set()
    matched_salt: set[str] = set()
    n_corpus = 0
    n_multifrag = 0
    t0 = time.time()
    for shard in _shards():
        smis = _shard_smiles(shard)
        n_corpus += len(smis)
        sset = set(smis)
        matched_nosalt |= eval_nosalt & sset
        matched_salt |= eval_salt & sset            # single-frag corpus entries covered here
        for s in smis:
            if "." in s:
                n_multifrag += 1
                k = canon_salt(s)
                if k in eval_salt:
                    matched_salt.add(k)
        _log(f"  {shard.name}: corpus so far {n_corpus:,}  "
             f"matched nosalt={len(matched_nosalt)} salt={len(matched_salt)}  "
             f"({time.time()-t0:.0f}s)")

    # Per-molecule flags + per-dataset summary.
    for r in rows:
        r["exact_nosalt"] = int(r["key_nosalt"] in matched_nosalt) if r["key_nosalt"] else 0
        r["exact_salt"] = int(r["key_salt"] in matched_salt) if r["key_salt"] else 0

    out.mkdir(parents=True, exist_ok=True)
    per_mol = out / "exact_match_per_molecule.csv"
    with open(per_mol, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["raw_smiles", "dataset", "key_salt", "exact_nosalt", "exact_salt"])
        for r in rows:
            w.writerow([r["raw_smiles"], r["dataset"], r["key_salt"] or "",
                        r["exact_nosalt"], r["exact_salt"]])

    summary = {}
    datasets = sorted({r["dataset"] for r in rows})
    for d in datasets:
        dr = [r for r in rows if r["dataset"] == d]
        n = len(dr)
        summary[d] = {
            "n": n,
            "exact_nosalt": sum(r["exact_nosalt"] for r in dr),
            "exact_salt": sum(r["exact_salt"] for r in dr),
            "pct_nosalt": round(100 * sum(r["exact_nosalt"] for r in dr) / max(n, 1), 2),
            "pct_salt": round(100 * sum(r["exact_salt"] for r in dr) / max(n, 1), 2),
        }
    meta = {
        "corpus": "pubchem_filtered, ALL 12 shards (full ~12M)",
        "corpus_molecules": n_corpus,
        "corpus_multifragment_entries": n_multifrag,
        "keys": {
            "key_nosalt": "RDKit isomeric canonical, no salt-stripping (matches corpus SMILES_canonical)",
            "key_salt": "largest-fragment then isomeric canonical (matches leakage_audit._ikey / 0-7% key)",
        },
        "per_dataset": summary,
    }
    (out / "exact_match_summary.json").write_text(json.dumps(meta, indent=2))
    _log(f"wrote {per_mol} and exact_match_summary.json")
    _log("EXACT-MATCH per dataset (n / nosalt / salt):")
    for d in datasets:
        s = summary[d]
        _log(f"  {d:14} n={s['n']:6}  nosalt={s['exact_nosalt']:5} ({s['pct_nosalt']:.2f}%)  "
             f"salt={s['exact_salt']:5} ({s['pct_salt']:.2f}%)")
    return summary


# ----------------------------------------------------------------------------- mode: full
def run_full(out: Path):
    rows = _eval_rows(only_i1=True)
    qs = [r["raw_smiles"] for r in rows]
    ds_of = {r["raw_smiles"]: r["dataset"] for r in rows}
    _log(f"I1 eval molecules (ESOL+QM7): {len(qs):,}")

    q, kept_idx = _pack(qs)
    q_smiles = [qs[i] for i in kept_idx]
    q_pc = _popcount_rows(q).sum(axis=1).astype(np.int32)
    _log(f"query matrix {q.shape} ({len(qs)-len(q_smiles)} unparseable dropped)")

    best = np.zeros(len(q_smiles), dtype=np.float32)
    n_identical = np.zeros(len(q_smiles), dtype=np.int64)     # corpus neighbours at Tanimoto==1.0
    n_neardup = np.zeros(len(q_smiles), dtype=np.int64)       # in [0.95, 1.0)
    n_corpus = 0
    t0 = time.time()
    for shard in _shards():
        smis = _shard_smiles(shard)
        ref, _ = _pack(smis)
        ref_pc = _popcount_rows(ref).sum(axis=1).astype(np.int32)
        n_corpus += ref.shape[0]
        for i in range(len(q_smiles)):
            inter = _popcount_rows(ref & q[i]).sum(axis=1).astype(np.int32)
            union = ref_pc + q_pc[i] - inter
            tan = inter / np.maximum(union, 1)
            m = tan.max()
            if m > best[i]:
                best[i] = m
            n_identical[i] += int((tan >= 0.99999).sum())
            n_neardup[i] += int(((tan >= NEAR_DUP_LO) & (tan < 0.99999)).sum())
        _log(f"  {shard.name}: corpus {n_corpus:,}  ({time.time()-t0:.0f}s, "
             f"eta {(time.time()-t0)/ (list(_shards()).index(shard)+1) * (12-list(_shards()).index(shard)-1):.0f}s)")

    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / "full_corpus_similarity_i1.csv"
    with open(csv_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["raw_smiles", "dataset", "max_tanimoto_to_corpus_full",
                    "n_corpus_identical", "n_corpus_neardup_0p95"])
        for s, b, ni, nd in zip(q_smiles, best, n_identical, n_neardup):
            w.writerow([s, ds_of[s], f"{b:.4f}", int(ni), int(nd)])

    meta = {
        "corpus": "pubchem_filtered, ALL 12 shards (full)",
        "corpus_molecules": int(n_corpus),
        "fingerprint": f"ECFP4 (Morgan r=2, {NBITS} bits) — identical to compute_tanimoto_novelty.py",
        "tasks": list(I1_TASKS),
        "n_scored": len(q_smiles),
        "near_dup_band": [NEAR_DUP_LO, 1.0],
        "columns": {
            "max_tanimoto_to_corpus_full": "true max ECFP4 Tanimoto to the full 12M corpus (NOT a lower bound)",
            "n_corpus_identical": "count of corpus molecules with Tanimoto==1.0 (fingerprint-identical)",
            "n_corpus_neardup_0p95": "count of corpus molecules with Tanimoto in [0.95,1.0)",
        },
        "median_max_tanimoto": {t: float(np.median([b for s, b in zip(q_smiles, best)
                                                    if ds_of[s] == t])) for t in I1_TASKS},
    }
    (out / "full_corpus_similarity_i1_meta.json").write_text(json.dumps(meta, indent=2))
    _log(f"wrote {csv_path}")
    for t in I1_TASKS:
        _log(f"  {t}: median max-Tanimoto(full)={meta['median_max_tanimoto'][t]:.3f}")
    return meta


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["exact", "full", "both"], default="exact")
    ap.add_argument("--out", default="analysis/dedup_i1")
    a = ap.parse_args()
    out = Path(a.out)
    if a.mode in ("exact", "both"):
        run_exact(out)
    if a.mode in ("full", "both"):
        run_full(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
