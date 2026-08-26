"""Every eval set added AFTER configs/eval_blocklist.json was built, against every SFT family.

The blocklist holds 34,301 canonical SMILES and was built from the six MoleculeNet tasks of July.
Wong, CBS, Ames, MoleculeACE, Polaris and FartDB joined the suite afterwards and were never added,
so nothing was ever excluded on their behalf. A blocklist cannot report what it was never asked
about: it does not fail, it just answers a smaller question than the one being asked of it.

Wong is the one that got noticed because a supervised family shares its NAME. That is not evidence
the others are clean -- it is evidence that name collisions are what humans notice.

TEST SET DEFINITION. For cross-validated suites (Wong, FartDB, MoleculeACE, CBS) every molecule is
in a held-out fold exactly once, so EVERY molecule is a test molecule. For split-column suites the
test rows are used where the column exists. Where in doubt this counts MORE molecules as test,
which can only overstate leakage -- the direction you want an audit to err in.

    python scripts/leakage_audit_v2.py --sft_sample 400000 --out analysis/leakage_v2.json
"""
from __future__ import annotations
import argparse, csv, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from leakage_audit import _ikey, ikey_set, sft_family_keys      # one definition of the key rule


def _csv_smiles(path: Path, col_candidates=("smiles", "SMILES", "canonical_smiles"), test_only=False):
    if not path.exists():
        return None
    rows = list(csv.DictReader(path.open()))
    if not rows:
        return None
    col = next((c for c in col_candidates if c in rows[0]), None)
    if col is None:
        return None
    if test_only and "split" in rows[0]:
        rows = [r for r in rows if str(r["split"]).lower() in ("test", "1", "true")] or rows
    return [r[col] for r in rows if r[col]]


def eval_sets(wong_csv: Path) -> dict:
    out = {}
    w = _csv_smiles(wong_csv)
    if w: out["Wong"] = w
    c = _csv_smiles(ROOT / "data" / "cbs.csv")
    if c: out["CBS"] = c
    mace = sorted((ROOT / "chemeleon_suite" / "data" / "moleculeace").glob("*.csv"))
    if mace:
        s = []
        for f in mace: s += (_csv_smiles(f) or [])
        out["MoleculeACE"] = s
    pol = sorted((ROOT / "chemeleon_suite" / "data" / "polaris").glob("*.csv"))
    for f in pol:
        name = f.stem.replace("polaris__", "").replace("tdcommons__", "")
        s = _csv_smiles(f)
        if s: out[f"Polaris:{name}" if "ames" not in name else "Ames"] = s
    try:
        from huggingface_hub import hf_hub_download
        import pyarrow.parquet as pq
        p = hf_hub_download("FartLabs/FartDB", "data/full-00000-of-00001.parquet", repo_type="dataset")
        t = pq.read_table(p)
        col = next(c for c in t.schema.names if "smi" in c.lower())
        out["FartDB"] = t.column(col).to_pylist()
    except Exception as e:
        print(f"[eval] FartDB unavailable: {e}", flush=True)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sft_sample", type=int, default=400_000)
    ap.add_argument("--wong_csv", default=str(ROOT / "chemeleon_suite" / "data" / "wong_saureus.csv"))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    evs = eval_sets(Path(a.wong_csv))
    print(f"[eval] {len(evs)} eval sets: " + ", ".join(f"{k}({len(v)})" for k, v in evs.items()), flush=True)
    sft = sft_family_keys(a.sft_sample)

    blocklist = set()
    bl = ROOT / "configs" / "eval_blocklist.json"
    if bl.exists():
        blocklist = set(json.loads(bl.read_text()))
        print(f"[blocklist] {len(blocklist)} canonical SMILES", flush=True)

    report, rows = {}, []
    for name, smiles in evs.items():
        keys = ikey_set(list(smiles))
        n = len(keys) or 1
        rec = {"n_molecules": len(smiles), "n_unique": len(keys),
               "in_blocklist": len(keys & blocklist),
               "in_blocklist_pct": round(100 * len(keys & blocklist) / n, 2)}
        for f, fk in sft.items():
            ov = keys & fk
            # OVERLAP is not LEAKAGE. The blocklist drops its molecules from the SFT data at
            # training time (pretrain_v2 -> data_v2's `blocklist` argument), so a molecule in both
            # the eval set and an SFT family was still excluded if it was on the list. What leaked
            # is the part the list did not cover -- which for the July-era list is everything an
            # eval set added later contributed.
            leaked = ov - blocklist
            rec[f"{f}_overlap"] = len(ov)
            rec[f"{f}_pct"] = round(100 * len(ov) / n, 2)
            rec[f"{f}_leaked"] = len(leaked)
            rec[f"{f}_leaked_pct"] = round(100 * len(leaked) / n, 2)
        report[name] = rec
        rows.append((name, rec))

    fams = list(sft)
    print("\n==== EVAL TEST MOLECULES THAT LEAKED (in an SFT family, NOT on the blocklist) ====")
    print(f"{'eval set':22} {'uniq':>7} {'blocked':>8} " + " ".join(f"{f:>14}" for f in fams))
    for name, rec in rows:
        line = f"{name:22} {rec['n_unique']:>7} {rec['in_blocklist_pct']:>7.2f}% "
        for f in fams:
            line += f"{rec[f'{f}_leaked']:>7}/{rec[f'{f}_leaked_pct']:>5.1f}%"
        print(line)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(report, indent=2))
    print(f"\n[leakage-v2] wrote {a.out}")
    print("SFT% is over the sampled cap per family, so it is a LOWER BOUND on overlap.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
