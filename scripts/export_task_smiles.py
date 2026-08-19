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
DATASETS = ["BACE", "Tox21", "HIV"]
CBS_CSV = "data/cbs.csv"


def main() -> int:
    payload = {}
    for ds in DATASETS:
        smiles, _ = eval_v2._load_moleculenet_full(ds)
        payload[ds] = [str(s) for s in smiles]
        print(f"{ds:8} {len(smiles)} molecules", flush=True)
    eval_v2.register_custom_task("cbs", CBS_CSV)
    smiles, _ = eval_v2._load_moleculenet_full("cbs")
    payload["cbs"] = [str(s) for s in smiles]
    print(f"{'cbs':8} {len(smiles)} molecules", flush=True)

    uniq = sorted({s for v in payload.values() for s in v})
    payload["_all_unique"] = uniq
    OUT.write_text(json.dumps(payload))
    print(f"wrote {OUT}: {len(uniq)} unique SMILES across {len(DATASETS)+1} tasks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
