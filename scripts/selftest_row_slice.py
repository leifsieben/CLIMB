"""Does the row-sliced path reproduce a shard that is ALREADY published, byte for byte?

The row-split writer is new code producing the array every MTR run trains against, and its
characteristic bug -- an off-by-N row offset -- yields a correctly shaped, correctly typed,
entirely wrong file that no downstream check can catch. A misaligned probe earlier in this project
scored corr 0.218 against corr 1.000 for the aligned one, so the signal is enormous; the problem
was never detecting it, it was remembering to look.

So: pick a shard already on S3, recompute a slice of it through the SAME code the box is about to
run, and require an exact match. Anything less and the box refuses to write.
"""
from __future__ import annotations
import argparse, os, subprocess, sys, tempfile

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from precompute_descriptors import CORPORA

WIDTH, ITEMSIZE, HEADER = 217, 2, 128


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", choices=sorted(CORPORA), required=True)
    ap.add_argument("--exclude", default="", help="ids this box owns -- never self-test against one")
    ap.add_argument("--lo", type=int, default=250_000, help="offset, deliberately not 0: an "
                    "off-by-N bug is invisible at the start of the file")
    ap.add_argument("--n", type=int, default=20_000)
    a = ap.parse_args()

    import pyarrow.parquet as pq
    import pyarrow.dataset as ds
    from descriptors_v2 import rdkit_descriptors, normalize, load_stats

    shards_uri, desc_uri = CORPORA[a.corpus]
    mine = {x.strip() for x in a.exclude.split(",") if x.strip()}

    listing = subprocess.run(["aws", "s3", "ls", desc_uri], capture_output=True, text=True).stdout
    published = {}
    for line in listing.splitlines():
        c = line.split()
        if len(c) >= 4 and c[-1].startswith("descriptors_shard_") and c[-1].endswith(".npy"):
            published[c[-1].replace("descriptors_shard_", "").replace(".npy", "")] = int(c[-2])
    files = sorted(ds.dataset(shards_uri, format="parquet").files)
    index = {Path(f).stem.replace("shard_", ""): i for i, f in enumerate(files)}

    # Only a shard whose published size matches its own row count is a valid reference -- testing
    # against a truncated object would compare new work to old garbage.
    ref = None
    for sid, size in sorted(published.items()):
        if sid in mine or sid not in index:
            continue
        f = files[index[sid]]
        f = f if f.startswith("s3://") else "s3://" + f
        rows = pq.ParquetFile(f).metadata.num_rows
        if size == rows * WIDTH * ITEMSIZE + HEADER and rows > a.lo + a.n:
            ref = (sid, f, rows); break
    if ref is None:
        print("[selftest] no complete published shard to test against -- refusing to assume")
        return 1
    sid, f, rows = ref
    print(f"[selftest] reference shard_{sid} ({rows} rows), comparing rows {a.lo}:{a.lo + a.n}")

    stats = load_stats("configs/descriptor_stats.json")
    smiles = pq.read_table(f, columns=["SMILES_canonical"]).column(0).to_pylist()[a.lo:a.lo + a.n]
    mine_arr = normalize(rdkit_descriptors(smiles), stats).astype(np.float16)

    tmp = Path(tempfile.mkdtemp()) / "ref.npy"
    subprocess.run(["aws", "s3", "cp", f"{desc_uri.rstrip('/')}/descriptors_shard_{sid}.npy",
                    str(tmp), "--only-show-errors"], check=True)
    theirs = np.load(tmp, mmap_mode="r")[a.lo:a.lo + a.n]

    if mine_arr.shape != theirs.shape:
        print(f"[selftest] SHAPE MISMATCH {mine_arr.shape} vs {theirs.shape}")
        return 1
    same = np.array_equal(np.asarray(mine_arr), np.asarray(theirs))
    if same:
        print(f"[selftest] PASS -- {a.n} rows identical to the published shard, bit for bit")
        return 0
    # Not equal: report how badly, so the log distinguishes "rdkit drifted a ulp" from "wrong rows".
    d = np.abs(np.asarray(mine_arr, np.float32) - np.asarray(theirs, np.float32))
    frac = float((d > 0).mean())
    print(f"[selftest] FAIL -- {frac:.4%} of cells differ, maxdiff {float(d.max()):.4f}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
