"""Add the Ames panel's analytic SE to figure_data/six_panel/a2_errorbars.csv.

WHY THIS EXISTS. The six-panel suite swapped hERG for Ames on 2026-08-17, but a2_errorbars.csv
still keyed its Polaris row as `hERG`. Fig A2 looks the interval up by panel name, found nothing
for `Ames`, and silently fell back to the LEGACY sd_total — the SD across 3 eval seeds, which is
run-to-run jitter on a fixed split, not sampling uncertainty. The result was an Ames whisker of
+-0.0025 next to +-0.03 sampling CIs on every other panel: an order of magnitude too small, and a
different estimand from its neighbours. That is the exact inconsistency the panel swap was supposed
to remove, so it is fixed here rather than left to the next full bootstrap run.

METHOD — the same one the hERG row used, and for the same reason. Polaris withholds test labels
(chemeleon_suite/data/polaris/tdcommons__ames.csv has `y` populated for train only), so the test
set CANNOT be resampled and a bootstrap is impossible. The Hanley-McNeil analytic SE of an AUC is
therefore the honest estimate, and the panel must be captioned as DERIVED, not resampled.

Class balance: labels are withheld for the test split, so the positive rate is taken from TRAIN
(3104 / 5821 = 53.32%) and applied to the 1457 test molecules -> 777 positive, 680 negative. Ames
is close to balanced, which is the best case for AUC SE; hERG was 67.7% positive at n=132.

This script ONLY touches the Ames rows. It deliberately does not recompute the scaffold- or
target-cluster bootstraps: those are being rebuilt by the compute session against the correct
pooled estimator, and re-running them here would overwrite that work with the known-broken
global-ranking version.

Run:  python3 scripts/a2_ames_se.py
"""
from __future__ import annotations

import csv
import math
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FD = ROOT / "figure_data"
OUT = FD / "six_panel" / "a2_errorbars.csv"
TASK, METRIC = "tdcommons/ames", "roc_auc"
N_POS, N_NEG = 777, 680                      # 1457 test at the 53.32% train active rate


def hanley_mcneil(a: float, n1: int, n0: int) -> float:
    q1, q2 = a / (2 - a), 2 * a * a / (1 + a)
    return math.sqrt((a * (1 - a) + (n1 - 1) * (q1 - a * a) + (n0 - 1) * (q2 - a * a)) / (n1 * n0))


def main() -> None:
    import sys
    sys.path.insert(0, str(ROOT))
    from figures.arms import ARMS
    import figures.fig_A2 as A2

    rows = list(csv.DictReader(open(OUT)))
    fields = rows[0].keys() if rows else []
    kept = [r for r in rows if r["panel"] not in ("hERG", "Ames")]   # drop the stale key
    dropped = len(rows) - len(kept)

    added = 0
    for arm in A2.MODELS:
        base = ARMS[arm]["src"].get("mace")
        if not base:
            continue
        vals = []
        for d in (base, f"{base}_s1", f"{base}_s2"):
            f = FD / "chemeleon_suite" / "polaris" / d / "polaris_scores.csv"
            if not f.exists():
                continue
            for r in csv.DictReader(open(f)):
                if r["task"] == TASK and r["metric"] == METRIC and r["value"] not in ("", "nan"):
                    vals.append(float(r["value"]))
        if not vals:
            print(f"  {arm:18s} no Ames score -> no interval")
            continue
        a = st.mean(vals)
        se = hanley_mcneil(a, N_POS, N_NEG)
        kept.append(dict(arm=arm, panel="Ames", metric="roc_auc", value=round(a, 4),
                         ci_lo=round(a - 1.96 * se, 4), ci_hi=round(a + 1.96 * se, 4),
                         se=round(se, 4), method="analytic_hanley_mcneil_DERIVED",
                         n_units=N_POS + N_NEG))
        added += 1
        print(f"  {arm:18s} AUC {a:.4f}  SE {se:.4f}  95% CI "
              f"[{a - 1.96 * se:.4f}, {a + 1.96 * se:.4f}]  (n={len(vals)} eval scores)")

    with open(OUT, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(fields))
        w.writeheader()
        w.writerows(kept)
    print(f"\ndropped {dropped} stale hERG/Ames rows, wrote {added} Ames rows -> "
          f"{OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
