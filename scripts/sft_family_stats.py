"""Exact per-family statistics for the SI methods table, measured off the parquet.

Every number here has a denominator, because the ones that matter are ratios and the reader will
not be in the room. PCBA is mostly UNMEASURED: "1.2% active" over all (molecule, assay) cells and
over MEASURED cells differ by orders of magnitude, and only the second describes the assay.

WONG's four readouts are reported per column, never pooled. Wong_SA is antibacterial activity and
the other three are cytotoxicity counter-screens; one pooled rate over four unrelated assays is the
same error as a macro that cancels.

    python scripts/sft_family_stats.py --families PCBA WONG --out analysis/sft_family_stats.json
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
WIDE = "s3://climb-s3-bucket/tokenized/supervised_wide_parquet/"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--families", nargs="+", default=["PCBA", "WONG"])
    ap.add_argument("--per_column", nargs="*", default=["WONG"],
                    help="families reported per label column instead of pooled")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    import numpy as np, pyarrow.compute as pc
    from data_v2 import _family_columns, parquet_dataset

    dset = parquet_dataset(WIDE)
    schema = dset.schema
    report = {}
    for fam in a.families:
        cols = _family_columns(schema, [fam])
        print(f"[{fam}] {len(cols)} label columns; streaming", flush=True)
        measured = np.zeros(len(cols), dtype=np.int64)
        positive = np.zeros(len(cols), dtype=np.int64)
        molecules = 0          # rows carrying at least one measurement in this family
        rows_total = 0
        for batch in dset.to_batches(columns=cols, batch_size=8192):
            rows_total += batch.num_rows
            any_measured = np.zeros(batch.num_rows, dtype=bool)
            for i, c in enumerate(batch.columns):
                arr = c.to_numpy(zero_copy_only=False).astype("float64")
                m = np.isfinite(arr)
                measured[i] += int(m.sum())
                positive[i] += int(np.sum(arr[m] > 0.5))
                any_measured |= m
            molecules += int(any_measured.sum())
            if rows_total % 819200 == 0:
                print(f"  [{fam}] {rows_total:,} rows", flush=True)

        cells = int(len(cols)) * int(molecules)
        rec = {
            "label_columns": len(cols),
            "molecules_with_any_measurement": int(molecules),
            "rows_in_parquet": int(rows_total),
            "measured_cells": int(measured.sum()),
            "possible_cells_over_family_molecules": cells,
            "measured_fraction_of_possible": round(float(measured.sum()) / cells, 6) if cells else None,
            "positive_cells": int(positive.sum()),
            "active_rate_over_measured": round(float(positive.sum()) / float(measured.sum()), 6)
                                          if measured.sum() else None,
            "active_rate_over_possible": round(float(positive.sum()) / cells, 8) if cells else None,
        }
        if fam in a.per_column:
            rec["per_column"] = {
                c: {"measured": int(measured[i]), "positive": int(positive[i]),
                    "active_rate_over_measured": round(float(positive[i]) / measured[i], 6)
                                                 if measured[i] else None}
                for i, c in enumerate(cols)}
        report[fam] = rec
        print(json.dumps({fam: rec}, indent=2), flush=True)

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(report, indent=2))
    print(f"[stats] wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
