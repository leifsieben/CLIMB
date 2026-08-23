"""Compute the 1,613 Mordred 2D descriptors for a SMILES list and cache them as an .npz.

WHY THIS IS ITS OWN STEP. CheMeleon is a D-MPNN pretrained to regress exactly these 1,613 Mordred
descriptors. So "does CheMeleon add anything beyond Mordred?" is the sharpest available test of
whether it learned structure or merely compressed its own training target -- but it is only sharp
if the descriptor set is the SAME one. Calculator(descriptors, ignore_3D=True) yields 1613, which
is that set exactly, not an approximation of it.

SPLIT FROM THE LOADERS, for the same reason the CheMeleon table is. mordredcommunity wants a
modern numpy; deepchem 2.8.0 defines our Tox21 parse and is pinned to an old one. The reference
environment exports the SMILES (scripts/export_task_smiles.py or the MolNet equivalent) and this
runs anywhere -- it reads strings and writes vectors, so no parse, fold or scoring decision can
drift between the two interpreters.

SHARDED BY SUBPROCESS, NOT BY mordred's nproc. Mordred's internal multiprocessing dies with an
EOFError under macOS spawn semantics. Independent workers over disjoint slices need no shared
state, so they work everywhere and a crashed shard costs only its own slice.

    python scripts/compute_mordred.py --smiles figure_data/_molnet_smiles.json \
        --out figure_data/_mordred_features.npz --shards 10

A failed descriptor becomes NaN rather than an exception: XGBoost consumes NaN natively and the
concat experiment already relies on that for the RDKit block. A molecule RDKit cannot parse is
DROPPED and reported -- the lookup on the other side raises on a miss rather than mean-filling,
so a silently absent molecule surfaces as a loud KeyError instead of a fabricated vector.
"""
from __future__ import annotations
import argparse, json, os, subprocess, sys, time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent


SHARD = "?"          # set by main() so progress lines identify their worker


def compute(smiles: list) -> tuple:
    """-> (kept_smiles, X) with non-finite entries as NaN."""
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")
    from mordred import Calculator, descriptors

    calc = Calculator(descriptors, ignore_3D=True)
    kept, rows = [], []
    # EMIT PROGRESS. The first version wrote nothing until its slice finished, so for 30 minutes
    # there was no way to tell a slow shard from a hung one without reading /proc CPU counters --
    # and "slow" and "stalled" need opposite responses, so that distinction has to be cheap.
    t0, n = time.time(), len(smiles)
    for i, s in enumerate(smiles):
        if i and i % 250 == 0:
            el = time.time() - t0
            rate = i / el
            print(f"[shard {SHARD}] {i}/{n} {rate:.1f} mol/s "
                  f"eta {(n - i) / max(rate, 1e-9) / 60:.1f} min", flush=True)
        m = Chem.MolFromSmiles(s)
        if m is None:
            continue
        v = calc(m)
        # Mordred returns error objects (not floats) for descriptors it cannot evaluate on a
        # given molecule. float() on those raises, so coerce per element.
        out = np.empty(len(v), dtype=np.float32)
        for j, x in enumerate(v):          # j, not i: i is the outer molecule counter
            try:
                f = float(x)
            except Exception:
                f = np.nan
            out[j] = f
        out[~np.isfinite(out)] = np.nan
        kept.append(s)
        rows.append(out)
    X = np.asarray(rows, dtype=np.float32) if rows else np.zeros((0, 1613), np.float32)
    return kept, X


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--smiles", required=True, help="JSON with _all_unique, or a .txt one per line")
    p.add_argument("--out", required=True)
    p.add_argument("--shards", type=int, default=1)
    p.add_argument("--shard", type=int, default=None, help="internal: run only this shard")
    a = p.parse_args()

    src = Path(a.smiles)
    if src.suffix == ".json":
        smiles = json.loads(src.read_text())["_all_unique"]
    else:
        smiles = [l.strip() for l in src.read_text().splitlines() if l.strip()]

    if a.shard is not None:
        global SHARD
        SHARD = a.shard
        sl = smiles[a.shard::a.shards]
        t = time.time()
        kept, X = compute(sl)
        np.savez_compressed(f"{a.out}.shard{a.shard}",
                            smiles=np.array(kept, dtype=object), X=X)
        print(f"[shard {a.shard}] {len(kept)}/{len(sl)} in {time.time()-t:.0f}s", flush=True)
        return 0

    # driver: fan out, then merge
    print(f"{len(smiles)} SMILES over {a.shards} shards", flush=True)
    procs = [subprocess.Popen([sys.executable, __file__, "--smiles", a.smiles, "--out", a.out,
                               "--shards", str(a.shards), "--shard", str(i)],
                              stdout=open(f"{a.out}.shard{i}.log", "w"),
                              stderr=subprocess.STDOUT)
             for i in range(a.shards)]
    rc = [p.wait() for p in procs]
    if any(rc):
        print(f"FATAL shard exit codes {rc} -- refusing to merge a partial table")
        return 1

    S, X = [], []
    for i in range(a.shards):
        f = f"{a.out}.shard{i}.npz"
        z = np.load(f, allow_pickle=True)
        # HOIST: npz members decode lazily, so touching z["X"] per element re-reads the member.
        s_i, x_i = z["smiles"], z["X"]
        S.extend(str(s) for s in s_i)
        X.append(x_i)
    X = np.concatenate(X, 0)
    if len(S) != len(X):
        print(f"FATAL {len(S)} smiles vs {len(X)} vectors")
        return 1
    np.savez_compressed(a.out, smiles=np.array(S, dtype=object), X=X)
    for i in range(a.shards):
        os.unlink(f"{a.out}.shard{i}.npz")
    dropped = len(smiles) - len(S)
    print(f"wrote {a.out}: {X.shape}, {dropped} RDKit-unparseable dropped", flush=True)
    nanfrac = float(np.mean(~np.isfinite(X)))
    print(f"non-finite entries: {nanfrac*100:.2f}% (kept as NaN; XGBoost consumes them natively)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
