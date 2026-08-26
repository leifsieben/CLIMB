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

# Both corpora name their shards shard_NNNNN.parquet, so a precompute directory is only safe if it
# is CORPUS-SPECIFIC -- pointing two corpora at one directory attaches the wrong molecules'
# descriptors to every row and nothing raises. Keyed here rather than passed as a free-form string
# so the pairing cannot be got wrong at the call site.
CORPORA = {
    "pubchem_filtered": ("s3://climb-s3-bucket/tokenized_sources/pubchem_filtered/",
                         "s3://climb-s3-bucket/tokenized_sources/pubchem_descriptors/"),
    "pubchem_124m_full": ("s3://climb-s3-bucket/tokenized_sources/pubchem_124m_full/",
                          "s3://climb-s3-bucket/tokenized_sources/pubchem_124m_descriptors/"),
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--shard_range", default="0-11", help="inclusive, e.g. 0-3")
    p.add_argument("--stats_path", default="configs/descriptor_stats.json")
    p.add_argument("--corpus", choices=sorted(CORPORA), default="pubchem_filtered",
                   help="picks BOTH the shard source and its own descriptor directory")
    p.add_argument("--out_s3", default=None, help="override; defaults to the corpus's own directory")
    p.add_argument("--chunk", type=int, default=20000)
    args = p.parse_args()
    SHARDS, default_out = CORPORA[args.corpus]
    args.out_s3 = args.out_s3 or default_out
    if args.corpus not in args.out_s3:
        raise SystemExit(f"refusing: --out_s3 {args.out_s3} is not named for corpus {args.corpus}")
    print(f"[precompute] corpus {args.corpus}\n  from {SHARDS}\n  to   {args.out_s3}", flush=True)

    import pyarrow.parquet as pq
    import pyarrow.dataset as ds
    from descriptors_v2 import rdkit_descriptors, normalize, fit_descriptor_stats, load_stats, save_stats
    from data_v2 import make_raw_smiles_dataset

    # The canonical stats define the MTR target space for EVERY run in the project. A box that
    # happens not to have the file locally must fetch it, never refit it: refitting silently
    # renormalizes the targets and the old branch then uploaded the new fit OVER the canonical
    # file, which would have made every July run incomparable with everything after it, with no
    # error anywhere. Refit only when nothing canonical exists.
    if not Path(args.stats_path).exists():
        Path(args.stats_path).parent.mkdir(parents=True, exist_ok=True)
        got = subprocess.run(["aws", "s3", "cp", "s3://climb-s3-bucket/configs/descriptor_stats.json",
                              args.stats_path, "--only-show-errors"])
        if got.returncode == 0:
            print("[precompute] fetched the canonical stats from S3 (did NOT refit)", flush=True)

    if Path(args.stats_path).exists():
        stats = load_stats(args.stats_path)
        print(f"[precompute] loaded stats ({len(stats['mean'])} descriptors)", flush=True)
    else:
        print("[precompute] no canonical stats anywhere -- fitting (20k sample) ...", flush=True)
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
