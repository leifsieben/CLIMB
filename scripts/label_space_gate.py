"""Does any SFT label column measure the same thing as an eval target?

THIS IS THE CHECK THAT WOULD HAVE CAUGHT WONG, and it is not about molecules. skip_mixed_8M trained
on WONG__Wong_SA__Active, which IS the Wong eval target -- same assay, same molecules -- so it was
trained to predict the eval's own label on 88.5% of the eval set. Every other overlap we found was
gene expression or bioassay activity against ADMET and mutagenicity endpoints: different label
space, which is transfer learning rather than a failure of it.

Molecule overlap is the expensive check and it did not answer the question. 88.5% overlap meant
nothing until we looked at what the labels were, and bbb-martins' 23% overlap turned out to be 0%
after the blocklist. This compares two short lists of NAMES and runs in seconds.

WHAT IT CANNOT DO, stated because a gate that overstates its reach is worse than none: it catches
NAME coincidence. Two measurements of the same endpoint under unrelated names would pass. It is a
tripwire against the failure that actually occurred -- a dataset joining the eval suite whose name
already appears in the supervised label space -- not a proof of semantic disjointness.

    python scripts/label_space_gate.py
"""
from __future__ import annotations
import re, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

FAMILIES = ["L1000_MCF7", "L1000_VCAP", "PCQM", "WONG", "PCBA"]
WIDE = "s3://climb-s3-bucket/tokenized/supervised_wide_parquet/"

# Every dataset the eval suite scores, by the token a label column would carry if it were the same
# measurement. Derived from the suite rather than typed where possible; the ones with no on-disk
# registry are listed here and that list is the thing to update when a dataset joins.
EVAL_TOKENS = {
    "wong": "Wong S. aureus growth inhibition",
    "cbs": "CBS inhibitor virtual screen",
    "ames": "Ames mutagenicity",
    "fartdb": "FartDB five-class odour",
    "moleculeace": "MoleculeACE activity cliffs",
    "bace": "BACE", "bbbp": "BBBP", "tox21": "Tox21", "hiv": "HIV",
    "esol": "ESOL", "qm7": "QM7", "clintox": "ClinTox", "sider": "SIDER",
    "polaris": "Polaris ADMET / PKIS2",
}


def main() -> int:
    import pyarrow.dataset as ds
    from data_v2 import _family_columns

    schema = ds.dataset(WIDE, format="parquet").schema
    hits = []
    for fam in FAMILIES:
        cols = _family_columns(schema, [fam])
        if not cols:
            print(f"[label-gate] {fam}: no columns found -- schema changed?"); continue
        blob = " ".join(cols).lower()
        for tok, desc in EVAL_TOKENS.items():
            if re.search(rf"\b{re.escape(tok)}", blob) or tok in fam.lower():
                sample = [c for c in cols if tok in c.lower()][:4] or cols[:2]
                hits.append((fam, tok, desc, len(cols), sample))
        print(f"[label-gate] {fam}: {len(cols)} label columns")

    if not hits:
        print("\n[label-gate] PASS -- no SFT label column names an eval target")
        return 0
    print("\n[label-gate] FAIL -- an SFT label column names an eval target:")
    for fam, tok, desc, n, sample in hits:
        print(f"  {fam} ({n} columns) matches eval '{tok}' ({desc})")
        for c in sample:
            print(f"      {c}")
    print("\nAny arm trained on that family has seen the eval's own label. Either drop the family "
          "from those runs or drop the dataset from the eval suite -- a blocklist does not fix "
          "this, because the molecules are supposed to be there.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
