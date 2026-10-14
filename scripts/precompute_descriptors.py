from __future__ import annotations
import argparse, os, sys


for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
SMILES_COL = "SMILES_canonical"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input_dir", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--shard_range", default="0-11", help="inclusive, e.g. 0-3")
    p.add_argument("--stats_path", default="configs/descriptor_stats.json")
    p.add_argument("--chunk", type=int, default=20000)
    args = p.parse_args()

    import pyarrow.parquet as pq
    import pyarrow.dataset as ds
    from descriptors import rdkit_descriptors, normalize, load_stats, descriptor_names

    stats = load_stats(args.stats_path)
    if stats["names"] != descriptor_names():
        raise ValueError("RDKit descriptor names/order differ from the supplied statistics")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    lo, hi = (int(x) for x in args.shard_range.split("-"))
    dset = ds.dataset(args.input_dir, format="parquet")
    files = sorted(dset.files)
    for si in range(lo, hi + 1):
        source = files[si]
        name = Path(source).stem
        tbl = pq.read_table(source, columns=[SMILES_COL])
        smiles = tbl.column(0).to_pylist()
        n = len(smiles)
        out = np.empty((n, len(stats["mean"])), dtype=np.float16)
        for j in range(0, n, args.chunk):
            chunk = smiles[j:j + args.chunk]
            out[j:j + len(chunk)] = normalize(rdkit_descriptors(chunk), stats).astype(np.float16)
            if (j // args.chunk) % 5 == 0:
                print(f"[precompute] {name}: {j+len(chunk)}/{n}", flush=True)
        destination = output_dir / f"descriptors_{name}.npy"
        np.save(destination, out)
        print(f"[precompute] DONE {name}: wrote {out.shape} -> {destination}", flush=True)


if __name__ == "__main__":
    main()
