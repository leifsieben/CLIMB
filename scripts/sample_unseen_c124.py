"""N molecules from pubchem_124m_full that are NOT in pubchem_filtered, deterministically.

For the descriptor linear-probe experiment. The arm the comparison is built around -- supervised
descriptor pretraining -- is the one a contaminated pool would flatter: a molecule seen during
pretraining was seen WITH its 217 descriptor targets, so part of its probe score would be
memorisation rather than encoding. ECFP4 and the random encoder have no pretraining and the
unsupervised arm never saw a descriptor, so the contamination lands on exactly one arm.

THE KEY. Both corpora store a `SMILES_canonical` column, so string equality is the membership test
-- but only if both were canonicalised by the same pipeline, which this script VERIFIES rather than
assumes: it re-canonicalises a sample of pubchem_filtered with the local RDKit and refuses to run if
the stored strings do not round-trip. A membership test on a key that is not canonical would report
a clean pool that is not one, which is the blocklist's failure exactly.

Sampling is seeded and the seed goes in the filename, so the file names the run that made it.

    python scripts/sample_unseen_c124.py --n 10000 --seed 0 --out_dir analysis/
"""
from __future__ import annotations
import argparse, csv, random, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
FILTERED = "s3://climb-s3-bucket/tokenized_sources/pubchem_filtered/"
BIG = "s3://climb-s3-bucket/tokenized_sources/pubchem_124m_full/"
COL = "SMILES_canonical"


def _shards(prefix):
    import subprocess
    out = subprocess.run(["aws", "s3", "ls", prefix], capture_output=True, text=True).stdout
    return sorted(prefix + l.split()[-1] for l in out.splitlines() if l.strip().endswith(".parquet"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--verify_sample", type=int, default=2000)
    ap.add_argument("--out_dir", default="analysis")
    a = ap.parse_args()

    import pyarrow.parquet as pq
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")
    from data_v2 import materialize_path

    # ---- 1. is the stored column actually canonical under OUR rdkit? ----------------------------
    f0 = _shards(FILTERED)[0]
    sample = pq.read_table(materialize_path(f0), columns=[COL]).column(0).to_pylist()[: a.verify_sample]
    ok = sum(1 for s in sample if (m := Chem.MolFromSmiles(s)) is not None and Chem.MolToSmiles(m) == s)
    frac = ok / max(len(sample), 1)
    print(f"[key] {ok}/{len(sample)} stored SMILES round-trip to themselves ({frac:.3%})", flush=True)
    if frac < 0.99:
        raise SystemExit("stored SMILES are not canonical under this rdkit -- string equality is "
                         "NOT a valid membership test; refusing to produce a pool whose overlap "
                         "would be unknown")

    # ---- 2. the 12M exclusion set ---------------------------------------------------------------
    seen = set()
    for p in _shards(FILTERED):
        seen.update(pq.read_table(materialize_path(p), columns=[COL]).column(0).to_pylist())
        print(f"[filtered] {p.split('/')[-1]}: {len(seen):,} unique so far", flush=True)

    # ---- 3. draw from the big corpus, shard order seeded ----------------------------------------
    big = _shards(BIG)
    rng = random.Random(a.seed)
    rng.shuffle(big)
    picked, scanned = [], 0
    for p in big:
        rows = pq.read_table(materialize_path(p), columns=[COL]).column(0).to_pylist()
        idx = list(range(len(rows)))
        random.Random(a.seed).shuffle(idx)
        for i in idx:
            s = rows[i]
            scanned += 1
            if s and s not in seen:
                picked.append(s)
                if len(picked) >= a.n:
                    break
        print(f"[big] {p.split('/')[-1]}: {len(picked):,}/{a.n} after scanning {scanned:,}", flush=True)
        if len(picked) >= a.n:
            break

    out = Path(a.out_dir) / f"unseen_c124_n{a.n}_seed{a.seed}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["smiles"]); w.writerows([[s] for s in picked])
    overlap = sum(1 for s in picked if s in seen)
    print(f"\n[unseen] wrote {out} with {len(picked):,} molecules")
    print(f"[unseen] overlap with pubchem_filtered: {overlap} of {len(picked)} (must be 0)")
    print(f"[unseen] scanned {scanned:,} molecules of the 124M corpus; seed {a.seed}")
    return 0 if overlap == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
