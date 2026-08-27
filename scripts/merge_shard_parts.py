"""Concatenate row-range parts into the one descriptor file a shard is allowed to publish.

The only failure that matters here is a silent one: parts that overlap, leave a gap, or land out
of order all produce a plausible array of the right dtype whose rows no longer correspond to the
shard's molecules, and nothing downstream can detect it. So this refuses to publish unless the
parts TILE [0, n_rows) exactly -- sorted, contiguous, starting at 0, ending at n_rows -- and unless
the concatenated row count equals the parquet's own count. Presence of "enough files" is not a check.
"""
from __future__ import annotations
import argparse, os, re, subprocess, sys, tempfile

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from precompute_descriptors import CORPORA

PART_RE = re.compile(r"rows_(\d{9})_(\d{9})\.npy$")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--corpus", choices=sorted(CORPORA), required=True)
    p.add_argument("--shard", type=int, required=True)
    p.add_argument("--keep_parts", action="store_true")
    args = p.parse_args()

    shards_s3, out_s3 = CORPORA[args.corpus]
    import pyarrow.parquet as pq
    import pyarrow.dataset as ds

    files = sorted(ds.dataset(shards_s3, format="parquet").files)
    f = files[args.shard]
    f = f if f.startswith("s3://") else "s3://" + f
    name = Path(f).stem
    n_rows = pq.ParquetFile(f).metadata.num_rows

    prefix = f"{out_s3.rstrip('/')}/parts/{name}/"
    listing = subprocess.run(["aws", "s3", "ls", prefix], capture_output=True, text=True).stdout
    parts = []
    for line in listing.splitlines():
        cols = line.split()
        if len(cols) >= 4:
            m = PART_RE.search(cols[-1])
            if m:
                parts.append((int(m.group(1)), int(m.group(2)), cols[-1]))
    parts.sort()
    if not parts:
        raise SystemExit(f"no parts under {prefix}")

    cursor = 0
    for lo, hi, _ in parts:
        if lo != cursor:
            raise SystemExit(f"REFUSING {name}: parts do not tile -- expected row {cursor}, got {lo}")
        cursor = hi
    if cursor != n_rows:
        raise SystemExit(f"REFUSING {name}: parts cover {cursor} rows, shard has {n_rows}")

    tmp = Path(tempfile.mkdtemp())
    arrs = []
    for lo, hi, key in parts:
        loc = tmp / key
        subprocess.run(["aws", "s3", "cp", prefix + key, str(loc), "--only-show-errors"], check=True)
        a = np.load(loc)
        if a.shape[0] != hi - lo:
            raise SystemExit(f"REFUSING {name}: part {key} holds {a.shape[0]} rows, its name claims {hi - lo}")
        arrs.append(a)

    full = np.concatenate(arrs, axis=0)
    if full.shape[0] != n_rows:
        raise SystemExit(f"REFUSING {name}: merged {full.shape[0]} rows against {n_rows} in the parquet")

    local = str(tmp / f"descriptors_{name}.npy")
    np.save(local, full)
    dest = f"{out_s3.rstrip('/')}/descriptors_{name}.npy"
    subprocess.run(["aws", "s3", "cp", local, dest, "--only-show-errors"], check=True)
    # Print the exact line the existing completion gate greps for, so a merged shard verifies by the
    # same rule as a serially-computed one rather than needing a second, divergent code path.
    print(f"[precompute] DONE {name}: wrote {full.shape} -> {out_s3}", flush=True)
    if not args.keep_parts:
        subprocess.run(["aws", "s3", "rm", prefix, "--recursive", "--only-show-errors"])


if __name__ == "__main__":
    main()
