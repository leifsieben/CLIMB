"""Export the exact SMILES eval_v2 would featurize, so featurization can happen elsewhere.

Run in the REFERENCE environment (the one whose Tox21 parse gives 77,864 prediction rows). The
box that computes CheMeleon embeddings then never touches a dataset loader -- it receives a plain
list of strings and returns vectors, so no parsing, fold-assignment or scoring decision can drift
between the two machines.
"""
from __future__ import annotations
import json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import eval_v2  # noqa: E402

OUT = ROOT / "figure_data" / "_task_smiles.json"
# The canonical-panel default. Overridable on the command line because the SI head comparison
# needs the full MolNet CV list (ESOL QM7 BBBP BACE Tox21 HIV) and a second hardcoded list is
# exactly the drift this file exists to prevent:
#   python3 scripts/export_task_smiles.py ESOL QM7 BBBP BACE Tox21 HIV  [--out path.json]
DATASETS = ["BACE", "Tox21", "HIV"]
CBS_CSV = "data/cbs.csv"


def main(argv=()) -> int:
    argv = list(argv)
    out = OUT
    if "--out" in argv:
        i = argv.index("--out")
        out = Path(argv[i + 1]); del argv[i:i + 2]
    datasets = argv or DATASETS
    payload = {}
    for ds in datasets:
        smiles, _ = eval_v2._load_moleculenet_full(ds)
        payload[ds] = [str(s) for s in smiles]
        print(f"{ds:8} {len(smiles)} molecules", flush=True)
    eval_v2.register_custom_task("cbs", CBS_CSV)
    smiles, _ = eval_v2._load_moleculenet_full("cbs")
    payload["cbs"] = [str(s) for s in smiles]
    print(f"{'cbs':8} {len(smiles)} molecules", flush=True)

    uniq = sorted({s for v in payload.values() for s in v})
    payload["_all_unique"] = uniq
    out.write_text(json.dumps(payload))
    print(f"wrote {out}: {len(uniq)} unique SMILES across {len(datasets)+1} tasks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
