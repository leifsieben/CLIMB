"""Build the combined label-efficiency table the Fig B1p1 notebook reads.

The two producer scripts emit their arms separately:
  scripts/label_eff_fractions.py      -> label_efficiency_fractions{,_summary}.csv       (4 frozen arms)
  scripts/label_eff_fractions_e2e.py  -> label_efficiency_fractions_e2e{,_summary}.csv   (e2e arm)

`notebook_cells/14.py` reads ONE file, `label_efficiency_fractions_all_summary.csv` (all 5 arms).
This script is the reproducible build step for that combined file (previously a manual concat),
so the figure's input has a scripted, re-runnable provenance.

Run after either producer changes:
    python scripts/build_label_eff_combined.py
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import pandas as pd

R = Path("analysis/rigor")
FROZEN_LONG, FROZEN_SUM = R / "label_efficiency_fractions.csv", R / "label_efficiency_fractions_summary.csv"
E2E_LONG, E2E_SUM = R / "label_efficiency_fractions_e2e.csv", R / "label_efficiency_fractions_e2e_summary.csv"
ALL_LONG, ALL_SUM = R / "label_efficiency_fractions_all.csv", R / "label_efficiency_fractions_all_summary.csv"

EXPECT_ARMS = {"random", "unsup", "sup", "unsup2sup", "e2e"}


def main():
    for f in (FROZEN_LONG, FROZEN_SUM, E2E_LONG, E2E_SUM):
        if not f.exists():
            raise SystemExit(f"missing input: {f} (run the frozen and e2e producers first)")

    long_ = pd.concat([pd.read_csv(FROZEN_LONG), pd.read_csv(E2E_LONG)], ignore_index=True)
    summ = pd.concat([pd.read_csv(FROZEN_SUM), pd.read_csv(E2E_SUM)], ignore_index=True)
    long_.to_csv(ALL_LONG, index=False)
    summ.sort_values(["arm", "task", "metric", "split", "fraction"]).to_csv(ALL_SUM, index=False)

    # sanity: all 5 arms, and every (arm,task) test-primary series has 5 distinct budgets
    arms = set(summ.arm.unique())
    prim = summ[(summ.split == "test") & (summ.metric.isin(["roc_auc", "rmse"]))]
    npts = prim.groupby(["arm", "task"]).n_train.nunique()
    print(f"arms: {sorted(arms)}")
    print(f"test-primary points: {len(prim)} (expect 5 arms x 7 tasks x 5 = 175)")
    assert arms == EXPECT_ARMS, f"arm set {sorted(arms)} != {sorted(EXPECT_ARMS)}"
    bad = npts[npts != 5]
    assert bad.empty, f"non-5-point series:\n{bad}"
    print(f"OK -> {ALL_LONG} ({len(long_)} rows), {ALL_SUM} ({len(summ)} points)")


if __name__ == "__main__":
    main()
