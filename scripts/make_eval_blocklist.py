from __future__ import annotations
import argparse, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def canon_key(smiles: str):
    from rdkit import Chem
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")
    try:
        frags = smiles.split(".")
        s = max(frags, key=len) if len(frags) > 1 else smiles
        m = Chem.MolFromSmiles(s)
        return Chem.MolToSmiles(m) if m is not None else None
    except Exception:
        return None


def eval_keys():
    from evaluate import _load_moleculenet
    from config import MOLECULENET_TASKS
    tasks = [t[0] for t in MOLECULENET_TASKS] + ["HIV"]
    keys = set()
    for t in tasks:
        try:
            tr, _, va, _, te, _ = _load_moleculenet(t)
            for split in (tr, va, te):
                for s in split:
                    k = canon_key(str(s))
                    if k:
                        keys.add(k)
            print(f"[blocklist] {t}: folded all splits; running key total {len(keys)}", flush=True)
        except Exception as e:
            print(f"[blocklist] {t} FAILED: {e}", flush=True)
    return keys


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--parquet", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    keys = eval_keys()
    print(f"[blocklist] {len(keys)} unique eval canonical keys", flush=True)

    import pyarrow.dataset as ds
    dset = ds.dataset(args.parquet, format="parquet")
    blocked_smiles = set()
    seen = 0
    for batch in dset.to_batches(columns=["smiles_canon"], batch_size=16384):
        for s in batch.column(0).to_pylist():
            seen += 1
            if s and canon_key(s) in keys:
                blocked_smiles.add(s)
        if seen % 500000 == 0:
            print(f"[blocklist] scanned {seen} parquet rows, {len(blocked_smiles)} matched", flush=True)
    print(f"[blocklist] DONE: {len(blocked_smiles)} parquet smiles_canon strings blocked "
          f"(of {seen} rows)", flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(sorted(blocked_smiles)))


if __name__ == "__main__":
    main()
