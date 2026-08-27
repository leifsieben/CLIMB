"""Emit one `--shard N --row_range lo-hi` line per part, tiling each shard exactly.

Written out as a job list rather than computed inside the xargs line because the tiling is the
thing the merger checks: if the plan and the merge disagree about how a shard is divided, the run
stalls at merge -- which is the safe direction -- but only if both read the same arithmetic. Row
counts come from the parquet itself, never from an assumed 1,000,000.
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from precompute_descriptors import CORPORA


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", choices=sorted(CORPORA), required=True)
    ap.add_argument("--shards", required=True, help="comma-separated 5-digit ids")
    ap.add_argument("--parts", type=int, required=True)
    a = ap.parse_args()

    import pyarrow.parquet as pq
    import pyarrow.dataset as ds

    shards_uri, _ = CORPORA[a.corpus]
    files = sorted(ds.dataset(shards_uri, format="parquet").files)
    index = {Path(f).stem.replace("shard_", ""): i for i, f in enumerate(files)}

    for sid in [x.strip() for x in a.shards.split(",") if x.strip()]:
        if sid not in index:
            raise SystemExit(f"no shard {sid} in {shards_uri}")
        i = index[sid]
        f = files[i] if files[i].startswith("s3://") else "s3://" + files[i]
        n = pq.ParquetFile(f).metadata.num_rows
        # ceil division, so the last part absorbs the remainder and the tiling ends exactly at n
        step = -(-n // a.parts)
        for lo in range(0, n, step):
            print(f"--shard {i} --row_range {lo}-{min(lo + step, n)}")


if __name__ == "__main__":
    main()
