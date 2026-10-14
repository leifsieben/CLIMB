from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
FD = ROOT / "figure_data"
OUT = FD / "SI_fig_a" / "SI_fig_a_e2e_need.csv"
CI = FD / "six_panel" / "a2_errorbars.csv"
MAIN = FD / "six_panel" / "mainline_8M.csv"

sys.path.insert(0, str(ROOT))
from figures.arms import ARMS, PANELS, PANEL_ORDER, E2E_PAIRS, series_label


PAIRS = E2E_PAIRS
PROTOCOL = "mainline"


HIGHER = {p: int(PANELS[p]["higher_better"]) for p in PANEL_ORDER}


def _sd(extra, key="sd_total"):
    m = re.search(rf"{key}=([-\d.eE]+)", str(extra))
    return float(m.group(1)) if m else np.nan


def main() -> None:
    if not CI.exists():
        raise SystemExit(f"{CI} absent -- run scripts/a2_bootstrap_errorbars.py first")
    ci = pd.read_csv(CI)
    main_tbl = pd.read_csv(MAIN)
    rows, missing, drift = [], [], []

    for frozen_arm, e2e_arm in PAIRS:
        label = series_label(frozen_arm)
        for panel in PANEL_ORDER:
            for probe, arm in (("frozen", frozen_arm), ("end2end", e2e_arm)):
                r = ci[(ci.arm == arm) & (ci.panel == panel)]
                assert len(r) <= 1, f"{arm}/{panel}: {len(r)} bootstrap rows, expected 1"


                if not len(r) or not np.isfinite(float(r.ci_lo.iloc[0])) \
                        or not np.isfinite(float(r.ci_hi.iloc[0])):
                    why = str(r.method.iloc[0]) if len(r) else "no row"
                    missing.append((panel, label, probe, arm, why))
                    continue
                r = r.iloc[0]


                m = main_tbl[(main_tbl.arm == arm) & (main_tbl.panel == panel)]
                if len(m):
                    a, b = float(r.value), float(m.value.iloc[0])
                    if abs(a - b) > 0.01 * max(abs(b), 1e-9):
                        drift.append((panel, arm, a, b))
                rows.append(dict(panel=panel, higher_better=HIGHER[panel], encoder=label,
                                 probe=probe, value=round(float(r.value), 6),
                                 lo=round(float(r.ci_lo), 6), hi=round(float(r.ci_hi), 6),
                                 n_units=int(r.n_units), method=str(r.method),
                                 protocol=PROTOCOL))

    if drift:
        for panel, arm, a, b in drift:
            print(f"  DRIFT {arm}/{panel}: bootstrap {a:.4f} vs mainline {b:.4f}")
        raise SystemExit("bootstrap and aggregate disagree by >1% -- different vintages of a run; "
                         "re-run scripts/six_panel_aggregate.py and the bootstrap together")
    if missing:
        for panel, label, probe, arm, why in missing:
            print(f"  MISSING {panel:12s} {label:18s} {probe:8s} (arm {arm}; {why})")
        raise SystemExit(f"{len(missing)} point(s) have no cluster-bootstrap interval. This table "
                         f"takes EVERY bar from one estimator, so a missing interval is a failure "
                         f"rather than a point with a different kind of bar. Add the arm to "
                         f"A2_ARMS in scripts/a2_bootstrap_errorbars.py and re-run it.")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    cols = ["panel", "higher_better", "encoder", "probe", "value", "lo", "hi",
            "n_units", "method", "protocol"]
    with open(OUT, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    d = pd.DataFrame(rows)
    assert len(d) == len(PAIRS) * len(PANEL_ORDER) * 2, (
        f"expected {len(PAIRS)*len(PANEL_ORDER)*2} rows, got {len(d)} -- a panel or probe is missing")


    for _p, _g in d.groupby("panel"):
        assert _g.method.nunique() == 1, (
            f"panel {_p} mixes interval methods {sorted(set(_g.method))} -- its arms' bars would "
            f"not be comparable with each other, which is the one comparison this figure needs")
    print(f"wrote {OUT.relative_to(ROOT)}  {len(rows)} rows "
          f"({len(PAIRS)} encoders x {len(PANEL_ORDER)} panels x 2 probes, no holes)")
    print(f"every point: {PROTOCOL} wave. interval method, one per panel:")
    for _p in PANEL_ORDER:
        print(f"    {_p:<12} {d[d.panel == _p].method.iloc[0]}")

    print("\ndoes end-to-end fine-tuning beat the frozen probe at full data?")
    print("  (delta signed so + = end2end better; * = the two intervals do not overlap)")
    for panel in PANEL_ORDER:
        g = d[d.panel == panel]
        sign = 1 if g.higher_better.iloc[0] else -1
        for frozen_arm, _ in PAIRS:
            lab = series_label(frozen_arm)
            fr = g[(g.encoder == lab) & (g.probe == "frozen")]
            ee = g[(g.encoder == lab) & (g.probe == "end2end")]
            if not len(fr) or not len(ee):
                continue
            fr, ee = fr.iloc[0], ee.iloc[0]
            delta = sign * (float(ee.value) - float(fr.value))
            sep = "*" if (fr.hi < ee.lo or ee.hi < fr.lo) else " "
            print(f"   {panel:<12}{lab:<20} frozen {fr.value:9.4f} [{fr.lo:.4f},{fr.hi:.4f}]   "
                  f"end2end {ee.value:9.4f} [{ee.lo:.4f},{ee.hi:.4f}]   delta {delta:+9.4f}{sep}")
    print("   * = the frozen and end2end intervals are disjoint")


if __name__ == "__main__":
    main()
