"""Verify precomputed descriptors are attached to the RIGHT MOLECULES, not merely in the right place.

A corpus-keyed path, a correct object size and a matching row count are all checks on bookkeeping.
The failure they cannot see is descriptors row-shifted or taken from the other corpus: every shard
is named shard_NNNNN.parquet in both corpora, so a wrong pairing produces a perfectly well-formed
directory of the right size. That failure is only visible by recomputing molecules and comparing.

Reads the stored rows by BYTE RANGE out of S3 -- row i of a [n, 217] float16 npy is 434 bytes at
offset 128 + i*434 -- so a probe costs one range GET rather than a 434 MB download, and indexes them
the way data_v2 does (batch_idx * 10_000 + i) so the arithmetic under test is the arithmetic in use.

    python scripts/verify_descriptor_alignment.py --corpus pubchem_124m_full --shards 0-61
"""
from __future__ import annotations
import argparse, json, random, subprocess, sys, tempfile
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))
from precompute_descriptors import CORPORA  # one definition of the corpus->dir pairing

WIDTH, ITEMSIZE, HEADER, BATCH = 217, 2, 128, 10_000


def stored_row(npy_uri: str, row: int) -> np.ndarray:
    lo = HEADER + row * WIDTH * ITEMSIZE
    hi = lo + WIDTH * ITEMSIZE - 1
    with tempfile.NamedTemporaryFile(suffix=".bin") as fh:
        bucket, key = npy_uri[5:].split("/", 1)
        r = subprocess.run(["aws", "s3api", "get-object", "--bucket", bucket, "--key", key,
                            "--range", f"bytes={lo}-{hi}", fh.name],
                           capture_output=True, text=True)
        if r.returncode != 0:
            raise SystemExit(f"range GET failed for {npy_uri} row {row}: {r.stderr.strip()}")
        return np.frombuffer(Path(fh.name).read_bytes(), dtype=np.float16).astype(np.float32)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", choices=sorted(CORPORA), required=True)
    ap.add_argument("--shards", required=True, help="inclusive range, e.g. 0-61")
    ap.add_argument("--n_probes", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    import pyarrow.dataset as pads
    from descriptors_v2 import rdkit_descriptors, normalize, load_stats

    shards_uri, desc_uri = CORPORA[a.corpus]
    stats = load_stats("configs/descriptor_stats.json")
    lo, hi = (int(x) for x in a.shards.split("-"))
    rng = random.Random(a.seed)

    ok = fail = 0
    for _ in range(a.n_probes):
        si = rng.randint(lo, hi)
        name = f"shard_{si:05d}"
        ds = pads.dataset(f"{shards_uri}{name}.parquet", format="parquet")
        col = next(c for c in ds.schema.names if "SMILES" in c or "smi" in c.lower())
        bi = rng.randint(0, 20)
        smis = None
        for k, b in enumerate(ds.to_batches(columns=[col], batch_size=BATCH)):
            if k == bi:
                smis = b.column(0).to_pylist(); break
        if not smis:
            print(f"  {name} batch {bi}: no such batch (short shard) -- skipped"); continue
        i = rng.randrange(len(smis))
        row = bi * BATCH + i                      # the loader's own arithmetic
        live = normalize(rdkit_descriptors([smis[i]]), stats)[0].astype(np.float32)
        got = stored_row(f"{desc_uri.rstrip('/')}/descriptors_{name}.npy", row)
        m = np.isfinite(live) & np.isfinite(got)
        corr = float(np.corrcoef(live[m], got[m])[0, 1]) if m.sum() > 2 else float("nan")
        diff = float(np.abs(live[m] - got[m]).max()) if m.any() else float("inf")
        good = corr > 0.999 and diff < 0.05
        ok, fail = ok + good, fail + (not good)
        print(f"  {name} row {row:>7}: corr {corr:.6f} maxdiff {diff:.4f} "
              f"{'ok' if good else 'MISMATCH -- descriptors are not this molecule'}")

    print(f"[verify] {ok} matched, {fail} mismatched over {ok + fail} probes in {a.corpus}")
    return 1 if fail or ok == 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
