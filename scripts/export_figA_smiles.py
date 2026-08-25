"""Export every molecule fig_A needs, once, so the out-of-environment CLMs are featurized exactly
once each rather than per task.

Covers both suite tracks plus the two new datasets. The extractor keys its npz on these strings and
the consumer looks them up STRICTLY, so this list defines the contract: a molecule missing here
becomes a loud KeyError at probe time, not a quietly imputed row.
"""
from __future__ import annotations
import csv, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

def main() -> int:
    out, per = [], {}
    def add(name, smis):
        smis = [s for s in smis if s]
        per[name] = len(set(smis))
        out.extend(smis)

    for track, d in (("moleculeace", ROOT / "chemeleon_suite/data/moleculeace"),
                     ("polaris", ROOT / "chemeleon_suite/data/polaris")):
        n = []
        for f in sorted(d.glob("*.csv")):
            with f.open() as fh:
                for r in csv.DictReader(fh):
                    s = r.get("smiles") or r.get("SMILES")
                    if s: n.append(s)
        add(track, n)

    w = Path("/home/ec2-user/chempfn-data/eval/locked/wong_saureus/wong_saureus.csv")
    if not w.exists():
        w = ROOT / "chemeleon_suite/data/wong_saureus.csv"
    if w.exists():
        with w.open() as fh:
            add("wong", [r["smiles"] for r in csv.DictReader(fh)])
    else:
        print(f"WARNING: wong csv not found at {w} -- exporting without it", file=sys.stderr)

    try:
        import pandas as pd
        from huggingface_hub import hf_hub_download
        df = pd.read_parquet(hf_hub_download("FartLabs/FartDB",
                                             "data/full-00000-of-00001.parquet", repo_type="dataset"))
        add("fartdb", df["Standardized SMILES"].tolist())
    except Exception as e:
        print(f"WARNING: FartDB unavailable ({e}) -- exporting without it", file=sys.stderr)

    uniq = sorted(set(out))
    Path("figure_data").mkdir(exist_ok=True)
    Path("figure_data/_figA_smiles.json").write_text(json.dumps({"_all_unique": uniq, "per_source": per}))
    print(f"per source (unique): {per}")
    print(f"wrote figure_data/_figA_smiles.json: {len(uniq)} unique molecules from {len(out)} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
