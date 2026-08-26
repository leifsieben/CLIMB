"""Is a corpus's precomputed descriptor directory COMPLETE -- every shard, at its own row count?

Separate from verify_descriptor_alignment.py on purpose. This asks "is all of it there", that one
asks "is it the right molecules". Neither implies the other: a complete directory can be
misaligned, and a perfectly aligned directory can be missing the shards a longer run would reach --
which fails 30 hours in, not at launch.

Sizes, not existence. A shard interrupted mid-upload leaves a short object, and an existence test
calls that finished forever.

    python scripts/verify_descriptor_dir.py --corpus pubchem_124m_full
"""
from __future__ import annotations
import argparse, os, subprocess, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from precompute_descriptors import CORPORA

WIDTH, ITEMSIZE, HEADER = 217, 2, 128


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", choices=sorted(CORPORA), required=True)
    ap.add_argument("--width", type=int, default=WIDTH)
    a = ap.parse_args()
    import pyarrow.parquet as pq

    shards_uri, desc_uri = CORPORA[a.corpus]
    out = subprocess.run(["aws", "s3", "ls", shards_uri], capture_output=True, text=True).stdout
    shards = sorted(l.split()[-1] for l in out.splitlines() if l.strip().endswith(".parquet"))
    if not shards:
        print(f"[verify-dir] no shards under {shards_uri} -- refusing to call that complete")
        return 1

    out = subprocess.run(["aws", "s3", "ls", desc_uri], capture_output=True, text=True).stdout
    have = {l.split()[-1]: int(l.split()[-2]) for l in out.splitlines()
            if l.strip().endswith(".npy")}

    bad = []
    for s in shards:
        stem = s[:-len(".parquet")]
        name = f"descriptors_{stem}.npy"
        if name not in have:
            bad.append((name, "ABSENT")); continue
        rows = pq.ParquetFile(shards_uri + s).metadata.num_rows
        want = rows * a.width * ITEMSIZE + HEADER
        if have[name] != want:
            bad.append((name, f"{have[name]} bytes, expected {want} for {rows} rows"))

    print(f"[verify-dir] {a.corpus}: {len(shards) - len(bad)} of {len(shards)} shards complete")
    for name, why in bad[:20]:
        print(f"[verify-dir] BAD {name}: {why}")
    if len(bad) > 20:
        print(f"[verify-dir] ... and {len(bad) - 20} more")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
