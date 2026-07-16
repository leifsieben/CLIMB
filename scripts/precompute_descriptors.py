"""Precompute normalized RDKit descriptors for the PubChem shards (MTR speedup).

On-the-fly descriptor computation starves the GPU (~120 vs 748 seq/s). Since the corpus
is only ~12M molecules (12 shards x 1M), we compute the 217 descriptors ONCE per shard,
z-normalize with the cached stats, and write a companion float16 array
`descriptors_shard_NNNNN.npy` (shape [n_rows, 217], row-aligned to the shard). The MTR
streaming path then reads these instead of calling RDKit (GPU-bound → full speed).

Parallelize across boxes with --shard_range (e.g. "0-3", "4-7", "8-11").

    python scripts/precompute_descriptors.py --shard_range 0-11 \
        --stats_path configs/descriptor_stats.json \
        --out_s3 s3://climb-s3-bucket/tokenized_sources/pubchem_descriptors/
"""
from __future__ import annotations
import argparse, subprocess, sys, tempfile
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
SHARDS = "s3://climb-s3-bucket/tokenized_sources/pubchem_filtered/"
SMILES_COL = "SMILES_canonical"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--shard_range", default="0-11", help="inclusive, e.g. 0-3")
    p.add_argument("--stats_path", default="configs/descriptor_stats.json")
    p.add_argument("--out_s3", required=True)
    p.add_argument("--chunk", type=int, default=20000)
    args = p.parse_args()

    import pyarrow.parquet as pq
    import pyarrow.dataset as ds
    from descriptors_v2 import rdkit_descriptors, normalize, fit_descriptor_stats, load_stats, save_stats
    from data_v2 import make_raw_smiles_dataset

    if Path(args.stats_path).exists():
        stats = load_stats(args.stats_path)
        print(f"[precompute] loaded stats ({len(stats['mean'])} descriptors)", flush=True)
    else:
        print("[precompute] fitting descriptor stats (20k sample) ...", flush=True)
        sample = [ex["smiles"] for _, ex in zip(range(20000), make_raw_smiles_dataset([SHARDS], subset_seed=0))]
        stats = fit_descriptor_stats(sample)
        save_stats(stats, args.stats_path)
        subprocess.run(["aws", "s3", "cp", args.stats_path,
                        "s3://climb-s3-bucket/configs/descriptor_stats.json"], check=True)

    lo, hi = (int(x) for x in args.shard_range.split("-"))
    dset = ds.dataset(SHARDS, format="parquet")
    files = sorted(dset.files)
    for si in range(lo, hi + 1):
        f = "s3://" + files[si] if not files[si].startswith("s3://") else files[si]
        name = Path(files[si]).stem  # shard_00000
        tbl = pq.read_table(f, columns=[SMILES_COL])
        smiles = tbl.column(0).to_pylist()
        n = len(smiles)
        out = np.empty((n, len(stats["mean"])), dtype=np.float16)
        for j in range(0, n, args.chunk):
            chunk = smiles[j:j + args.chunk]
            out[j:j + len(chunk)] = normalize(rdkit_descriptors(chunk), stats).astype(np.float16)
            if (j // args.chunk) % 5 == 0:
                print(f"[precompute] {name}: {j+len(chunk)}/{n}", flush=True)
        local = str(Path(tempfile.gettempdir()) / f"descriptors_{name}.npy")
        np.save(local, out)
        subprocess.run(["aws", "s3", "cp", local, f"{args.out_s3.rstrip('/')}/descriptors_{name}.npy"], check=True)
        print(f"[precompute] DONE {name}: wrote {out.shape} -> {args.out_s3}", flush=True)


if __name__ == "__main__":
    main()
