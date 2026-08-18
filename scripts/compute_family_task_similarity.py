"""Fig J1 / H10 — chemical similarity between each SFT family and each evaluation task.

J1 shows WHICH SFT family transfers to which eval task. On its own that is a description, not a
test: H10 claims transfer is governed by molecular/domain content similarity, and testing it
needs an independent similarity measure to correlate the transfer pattern against.

For each (family, task) pair this computes the mean over eval-test molecules of their MAXIMUM
ECFP4 Tanimoto to the family's training molecules -- "how close is the typical test molecule to
something this family actually trained on". Correlating that against J1's lift is the H10 test:
a positive relation supports content-similarity-driven transfer, a flat one says transfer is
driven by something else.

Note this is now a cleaner test than it would have been pre-dedup: the deduped ablation removed
exact eval-test overlap, so any residual similarity signal is genuine near-neighbour transfer
rather than memorisation of the same molecule.

Families are read from the supervised wide parquet, where membership is indicated by non-null
values in that family's prefixed columns.

Usage:
    python scripts/compute_family_task_similarity.py [--family-n 20000] [--out figure_data/_tanimoto]
"""
from __future__ import annotations

import argparse, csv, json, subprocess, time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator

RDLogger.DisableLog("rdApp.*")
NBITS = 2048
FAMILIES = ["PCBA", "L1000_MCF7", "L1000_VCAP", "PCQM", "WONG"]
SUP = "s3://climb-s3-bucket/tokenized/supervised_wide_parquet/"


def log(m): print(f"[famsim {time.strftime('%H:%M:%S')}] {m}", flush=True)


def pack(smiles):
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=NBITS)
    rows = []
    for s in smiles:
        m = Chem.MolFromSmiles(s) if s else None
        if m is None:
            continue
        arr = np.frombuffer(gen.GetFingerprintAsNumPy(m).astype(np.uint8).tobytes(), dtype=np.uint8)
        rows.append(np.packbits(arr).view(np.uint64))
    return np.vstack(rows) if rows else np.zeros((0, NBITS // 64), dtype=np.uint64)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--family-n", type=int, default=20000, help="molecules sampled per family")
    ap.add_argument("--shards", type=int, default=12,
                    help="how many shards to SPREAD across; the wide parquet is family-ordered, "
                         "so reading the first N only ever finds the first family")
    ap.add_argument("--out", default="figure_data/_tanimoto")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    cache = out / "_cache"; cache.mkdir(exist_ok=True)

    # ---- eval-task molecules (already collected for I1) ----
    tasks = {}
    with open(out / "corpus_similarity.csv") as fh:
        for r in csv.DictReader(fh):
            tasks.setdefault(r["dataset"], []).append(r["raw_smiles"])
    log("eval tasks: " + ", ".join(f"{k}={len(v)}" for k, v in tasks.items()))

    # ---- family molecules from the supervised wide parquet ----
    fam_smiles = {f: [] for f in FAMILIES}
    # The wide parquet is family-ordered: shard 0 is entirely PCQM. Reading the first N
    # shards therefore finds ONE family. Spread the probes across the whole range instead.
    # Shard indices where each family actually HAS data, found by scanning the probe column of
    # all 55 shards. The parquet is family-ordered and the families are wildly unequal in size
    # (PCQM spans shards 0-30, PCBA 36-44, while L1000/WONG live in just 36 and 52), so neither
    # "first N" nor an even spread finds them -- an even spread over 55 shards misses 36/52
    # entirely, which is why L1000 and WONG came back empty.
    FAMILY_SHARDS = {"PCQM": [0, 2, 4], "PCBA": [36, 38, 40, 42],
                     "L1000_MCF7": [36, 52], "L1000_VCAP": [36, 52], "WONG": [31, 36, 52]}
    idxs = sorted({i for v in FAMILY_SHARDS.values() for i in v})
    for i in idxs:
        loc = cache / f"sup_{i:05d}.parquet"
        if not loc.exists():
            subprocess.run(["aws", "s3", "cp", f"{SUP}shard_{i:05d}.parquet", str(loc)],
                           check=True, capture_output=True)
        pf = pq.ParquetFile(loc)
        names = pf.schema_arrow.names
        for fam in FAMILIES:
            cols = [c for c in names if c.startswith(fam + "__")][:1]  # one probe column is enough
            if not cols:
                continue
            t = pf.read(columns=["smiles_canon"] + cols).to_pydict()
            probe = t[cols[0]]
            for smi, v in zip(t["smiles_canon"], probe):
                if smi and v is not None:
                    fam_smiles[fam].append(smi)
        log(f"shard {i}: " + ", ".join(f"{f}={len(fam_smiles[f])}" for f in FAMILIES))

    rng = np.random.default_rng(0)
    fam_fp = {}
    for f, smi in fam_smiles.items():
        if not smi:
            log(f"{f}: NO molecules found — skipping"); continue
        # SORTED, not just de-duplicated. `list({...})` iterates a set, and Python randomises str
        # hashing per process, so the iteration order -- and therefore which family_n molecules
        # rng.choice lands on -- differed run to run EVEN AT A FIXED SEED. That made this table
        # irreproducible across processes: nobody could re-derive these numbers, and appending new
        # rows would silently mix two different family samples. Sorting pins the candidate order so
        # a fixed seed now yields a fixed sample.
        s = sorted({x for x in smi})
        if len(s) > a.family_n:
            s = [s[i] for i in rng.choice(len(s), a.family_n, replace=False)]
        fam_fp[f] = pack(s)
        log(f"{f}: fingerprinted {fam_fp[f].shape[0]} unique molecules")

    rows = []
    for task, smi in tasks.items():
        q = pack(list(dict.fromkeys(smi)))
        qpc = np.bitwise_count(q).sum(axis=1).astype(np.int32)
        for fam, ref in fam_fp.items():
            rpc = np.bitwise_count(ref).sum(axis=1).astype(np.int32)
            best = np.zeros(len(q), dtype=np.float32)
            for i in range(len(q)):
                inter = np.bitwise_count(ref & q[i]).sum(axis=1).astype(np.int32)
                union = rpc + qpc[i] - inter
                best[i] = (inter / np.maximum(union, 1)).max()
            rows.append(dict(task=task, family=fam,
                             mean_max_tanimoto=float(best.mean()),
                             median_max_tanimoto=float(np.median(best)),
                             frac_above_0p4=float((best > 0.4).mean()),
                             n_task=int(len(q)), n_family=int(ref.shape[0])))
            log(f"  {task:<14} x {fam:<12} mean-max={best.mean():.3f}")

    p = out / "family_task_similarity.csv"
    with open(p, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    (out / "family_task_similarity_meta.json").write_text(json.dumps(
        {"fingerprint": f"ECFP4 (Morgan r=2, {NBITS} bits)",
         "statistic": "per eval-test molecule, MAX Tanimoto to the family's molecules; reported as the mean/median over test molecules",
         "family_sample_n": a.family_n, "sup_shards_read": a.shards,
         "caveat": "families are SAMPLED, so similarities are lower bounds; all pairs share the same sampling so relative comparison is unaffected",
         "sampling_is_deterministic": True,
         "regenerated": "2026-08-18: whole table regenerated in ONE run when the canonical panels "
                        "(MoleculeACE, CBS, Ames) were added, so every pair shares one family "
                        "sample. Values differ slightly from the pre-2026-08-18 table: that shift "
                        "is RESAMPLING NOISE, not a correction. Prior table kept as "
                        "family_task_similarity.PRE_CANONICAL.csv. Family sampling is now pinned "
                        "(candidates sorted before rng.choice), so a fixed seed reproduces this "
                        "table across processes; it previously did not, because Python randomises "
                        "str hashing per process."},
        indent=2))
    log(f"wrote {p} ({len(rows)} family x task pairs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
