"""Molecule-level leakage audit: eval TEST molecules vs pretraining + SFT data.

Task-disjointness (MoleculeNet tasks != PCBA/L1000/PCQM/WONG) does NOT rule out
MOLECULE overlap: the same compound can sit in an SFT assay and an eval test split.
This computes InChIKey (connectivity block, first 14 chars) overlap of each eval
task's TEST split against (a) a sample of the PubChem pretraining corpus and (b) each
SFT family's molecules (from the parquet's `smiles_canon`).

Output: JSON + printed table of overlap counts / % per (eval task, source).

Run on a box (needs RDKit + S3):
    python scripts/leakage_audit.py --pretrain_sample 2000000 --sft_sample 400000 \
        --out experiments/climb_v2_phase2/leakage_audit.json
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
from multiprocessing import Pool

sys.path.insert(0, str(Path(__file__).parent.parent))


def _ikey(smiles: str):
    """Canonical-SMILES dedup key — ~100x faster than InChIKey and adequate for
    overlap detection (same 2D structure -> same RDKit canonical SMILES). We strip
    salts/solvents by taking the largest fragment so a molecule and its salt match."""
    from rdkit import Chem
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")
    try:
        m = Chem.MolFromSmiles(smiles)
        if m is None:
            return None
        frags = smiles.split(".")
        if len(frags) > 1:  # keep largest fragment (drop counterions/solvent)
            m = Chem.MolFromSmiles(max(frags, key=len))
            if m is None:
                return None
        return Chem.MolToSmiles(m)  # RDKit canonical SMILES
    except Exception:
        return None


def ikey_set(smiles_list, procs=8):
    # single-threaded on purpose: RDKit + multiprocessing.fork can deadlock on this box,
    # and the canonical-SMILES key is fast enough (~0.2 ms/mol) to not need a Pool.
    return set(k for k in (_ikey(s) for s in smiles_list) if k)


def eval_test_keys():
    """InChIKey set of each MoleculeNet eval task's TEST split."""
    from eval_v2 import _load_moleculenet
    from config_v2 import MOLECULENET_TASKS_V2
    tasks = [t[0] for t in MOLECULENET_TASKS_V2] + ["HIV"]
    out = {}
    for t in tasks:
        try:
            _, _, _, _, te_s, _ = _load_moleculenet(t)
            out[t] = (list(te_s), ikey_set(list(te_s)))
            print(f"[eval] {t}: {len(te_s)} test molecules, {len(out[t][1])} unique InChIKeys", flush=True)
        except Exception as e:
            print(f"[eval] {t} FAILED: {e}", flush=True)
    return out


def pretrain_keys(n):
    from data_v2 import make_raw_smiles_dataset
    paths = ["s3://climb-s3-bucket/tokenized_sources/pubchem_filtered/"]
    sm = [ex["smiles"] for _, ex in zip(range(n), make_raw_smiles_dataset(paths, subset_seed=0))]
    print(f"[pretrain] sampled {len(sm)} SMILES", flush=True)
    return ikey_set(sm)


def sft_family_keys(sample_per_family):
    """Per-family key sets from the wide parquet's smiles_canon. Reads only a few
    REPRESENTATIVE label columns (not all ~3,300): L1000/PCQM/WONG are dense within
    their block so one column identifies the family exactly; PCBA is sparse per-assay
    so we OR ~40 PCBA columns (slight undercount, but PCBA is huge). Rows are routed to
    the first matching family in priority order (blocks are exclusive)."""
    import numpy as np, pyarrow.dataset as ds
    from data_v2 import _family_columns
    families = ["L1000_MCF7", "L1000_VCAP", "PCQM", "WONG", "PCBA"]  # priority order
    dset = ds.dataset("s3://climb-s3-bucket/tokenized/supervised_wide_parquet/", format="parquet")
    fam_cols = {f: _family_columns(dset.schema, [f]) for f in families}
    rep = {f: (fam_cols[f][:40] if f == "PCBA" else fam_cols[f][:1]) for f in families if fam_cols[f]}
    per_fam_smiles = {f: [] for f in families}
    counts = {f: 0 for f in families}          # total molecules seen per family (for L1000 diag)
    need = {f: sample_per_family for f in families}
    cols = ["smiles_canon"] + sorted({c for f in rep for c in rep[f]})
    for batch in dset.to_batches(columns=cols, batch_size=8192):
        d = batch.to_pydict()
        sm = np.array(d["smiles_canon"], dtype=object)
        n = len(sm)
        assigned = np.full(n, -1, dtype=np.int64)
        for fi, f in enumerate(families):      # priority order; first match wins
            if f not in rep:
                continue
            arr = np.array([[float(v) if v is not None else np.nan for v in d[c]] for c in rep[f]],
                           dtype=np.float64)     # (n_repcols, n)
            fin = np.isfinite(arr).any(axis=0) & (assigned < 0)
            assigned[fin] = fi
        for fi, f in enumerate(families):
            rows = np.where(assigned == fi)[0]
            counts[f] += len(rows)
            if need[f] > 0 and len(rows):
                take = [s for s in sm[rows[:need[f]]] if s]
                per_fam_smiles[f].extend(take); need[f] -= len(take)
    out = {}
    for f in families:
        out[f] = ikey_set(per_fam_smiles[f])
        print(f"[sft] {f}: {counts[f]} molecules seen (sampled {len(per_fam_smiles[f])}), "
              f"{len(out[f])} unique keys", flush=True)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pretrain_sample", type=int, default=2_000_000)
    p.add_argument("--sft_sample", type=int, default=400_000)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    ev = eval_test_keys()
    pre = pretrain_keys(args.pretrain_sample)
    sft = sft_family_keys(args.sft_sample)

    report = {}
    print("\n==== LEAKAGE: eval TEST molecules found in each source ====")
    hdr = f"{'eval task':8} {'|test|':>7} {'pretrain%':>10} " + " ".join(f"{f:>12}" for f in sft)
    print(hdr)
    for t, (te_s, keys) in ev.items():
        n = len(keys) or 1
        row = {"n_test_unique": len(keys),
               "pretrain_overlap": len(keys & pre), "pretrain_pct": round(100*len(keys & pre)/n, 1)}
        cells = f"{t:8} {len(keys):>7} {row['pretrain_pct']:>9.1f}% "
        for f in sft:
            ov = len(keys & sft[f]); row[f"{f}_overlap"] = ov; row[f"{f}_pct"] = round(100*ov/n, 1)
            cells += f"{ov:>6}/{100*ov/n:>4.1f}%"
        report[t] = row
        print(cells)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(f"\n[leakage] wrote {args.out}")
    print("NOTE: pretrain% is a lower bound (sampled). SFT% is over the sampled cap per family.")


if __name__ == "__main__":
    main()
