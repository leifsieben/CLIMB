"""Compute descriptors for a ROW RANGE of one shard -- the missing unit of parallelism.

precompute_descriptors.py parallelizes one-process-per-SHARD, so a shard is pinned to a single
core and takes ~2.9h no matter how many cores are idle next to it. When a rung is blocked on a
handful of named shards that floor IS the critical path, and a 16-core box with 9 shards on it
leaves 7 cores doing nothing. This splits a shard by rows instead, so N cores attack one shard.

Each part is an exact slice [lo, hi) of the shard's rows, written to a parts/ prefix. Merging is a
SEPARATE step (merge_shard_parts.py) that concatenates in row order and refuses to publish unless
the parts tile the shard exactly -- misaligned descriptors are the one error in this pipeline that
raises nothing and poisons every downstream number, so the check is on coverage, not on file count.
"""
from __future__ import annotations
import argparse, os, subprocess, sys, tempfile

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from precompute_descriptors import CORPORA  # one registry, so the corpus->dir pairing can't diverge


def part_key(out_s3: str, name: str, lo: int, hi: int) -> str:
    return f"{out_s3.rstrip('/')}/parts/{name}/rows_{lo:09d}_{hi:09d}.npy"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--corpus", choices=sorted(CORPORA), required=True)
    p.add_argument("--shard", type=int, required=True)
    p.add_argument("--row_range", required=True, help="lo-hi, half-open [lo, hi)")
    p.add_argument("--stats_path", default="configs/descriptor_stats.json")
    p.add_argument("--chunk", type=int, default=20000)
    args = p.parse_args()

    shards_s3, out_s3 = CORPORA[args.corpus]
    import pyarrow.parquet as pq
    import pyarrow.dataset as ds
    from descriptors_v2 import rdkit_descriptors, normalize, load_stats

    if not Path(args.stats_path).exists():
        subprocess.run(["aws", "s3", "cp", "s3://climb-s3-bucket/configs/descriptor_stats.json",
                        args.stats_path, "--only-show-errors"], check=True)
    stats = load_stats(args.stats_path)
    width = len(stats["mean"])

    files = sorted(ds.dataset(shards_s3, format="parquet").files)
    f = files[args.shard]
    f = f if f.startswith("s3://") else "s3://" + f
    name = Path(f).stem
    n_rows = pq.ParquetFile(f).metadata.num_rows

    lo, hi = (int(x) for x in args.row_range.split("-"))
    hi = min(hi, n_rows)
    if lo >= hi:
        raise SystemExit(f"empty range {lo}-{hi} for {name} ({n_rows} rows)")

    dest = part_key(out_s3, name, lo, hi)
    want_bytes = (hi - lo) * width * 2 + 128
    have = subprocess.run(["aws", "s3", "ls", dest], capture_output=True, text=True).stdout.split()
    if have and have[-2].isdigit() and int(have[-2]) == want_bytes:
        print(f"[rows] SKIP {name}[{lo}:{hi}] already on S3", flush=True)
        return

    # Read only the rows we own. The whole column is materialized by pyarrow either way, but the
    # descriptor call is what costs 3 hours, so slicing before rdkit is the entire point.
    smiles = pq.read_table(f, columns=["SMILES_canonical"]).column(0).to_pylist()[lo:hi]
    out = np.empty((len(smiles), width), dtype=np.float16)
    for j in range(0, len(smiles), args.chunk):
        c = smiles[j:j + args.chunk]
        out[j:j + len(c)] = normalize(rdkit_descriptors(c), stats).astype(np.float16)
        print(f"[rows] {name}[{lo}:{hi}] {j + len(c)}/{len(smiles)}", flush=True)

    local = str(Path(tempfile.gettempdir()) / f"part_{name}_{lo}_{hi}.npy")
    np.save(local, out)
    subprocess.run(["aws", "s3", "cp", local, dest, "--only-show-errors"], check=True)
    print(f"[rows] DONE {name}[{lo}:{hi}] wrote {out.shape} -> {dest}", flush=True)


if __name__ == "__main__":
    main()
