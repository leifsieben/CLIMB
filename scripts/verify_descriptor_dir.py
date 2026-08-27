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
from c124_priority_order import needed

WIDTH, ITEMSIZE, HEADER = 217, 2, 128


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", choices=sorted(CORPORA), required=True)
    ap.add_argument("--width", type=int, default=WIDTH)
    ap.add_argument("--shards", default=None,
                    help="comma-separated 5-digit ids -- check ONLY these. For a box that was "
                         "given an explicit list, whose job is done when its own list is done.")
    ap.add_argument("--budget", type=int, default=None,
                    help="check only the shards a run of this forward-pass budget will OPEN. "
                         "Without it, every shard in the corpus must be present.")
    a = ap.parse_args()
    import pyarrow.parquet as pq

    shards_uri, desc_uri = CORPORA[a.corpus]
    out = subprocess.run(["aws", "s3", "ls", shards_uri], capture_output=True, text=True).stdout
    shards = sorted(l.split()[-1] for l in out.splitlines() if l.strip().endswith(".parquet"))
    if a.shards:
        want = {x.strip() for x in a.shards.split(",") if x.strip()}
        shards = [s for s in shards if s.replace(".parquet", "").replace("shard_", "") in want]
        missing = want - {s.replace(".parquet", "").replace("shard_", "") for s in shards}
        if missing:
            print(f"[verify-dir] the corpus has no shard for requested id(s): {sorted(missing)}")
            return 1
        print(f"[verify-dir] checking the {len(shards)} requested shard(s)")
    if a.budget:
        # A rung does not read the whole corpus, and requiring shards it will never open would
        # block a launch on work that does not exist yet. Check exactly the set it opens --
        # including the prefetch margin, since a prefetched shard is opened whether or not its
        # rows are consumed.
        want = set(needed(a.budget, len(shards)))
        shards = [s for s in shards if s.replace(".parquet", "").replace("shard_", "") in want]
        print(f"[verify-dir] budget {a.budget:,} opens {len(shards)} of the corpus's shards")
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
