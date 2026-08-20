"""SI Fig a — build the "do you need end-to-end training?" table (the figure's ONLY input).

For each canonical panel, the SAME pretrained encoder used two ways at FULL downstream data:

  frozen    encoder frozen, a probe trained on the labels
  end2end   the whole network fine-tuned on the labels

Three encoders, and for each the configuration the paper actually reports:

  unsupervised        unsup            -> unsup_e2e
  supervised, desc    sup_dense        -> sup_dense_e2e
  CheMeleon           chemeleon_frozen_xgb -> chemeleon_e2e

ONE WAVE, ONE ESTIMATOR (rewritten 2026-08-20 on Leif's instruction that every point carry the
same error-bar definition and be genuinely comparable). Both are properties this table used to
lack, and neither was cosmetic:

WAVE. This was built from TWO waves -- mainline on MoleculeACE/Ames/HIV, label-efficiency at its
100% fraction on BACE/Tox21/QM7 -- which forced a per-panel protocol column, a per-panel anchor
resolver, and a per-panel decision about where the external comparator could be drawn at all. The
split was a historical accident: when it was chosen, the end-to-end CLIMB arms had no MolNet runs,
so BACE/Tox21/QM7 had to come from somewhere else. They have them now. mainline_8M.csv carries
BOTH probes for ALL THREE encoders on ALL SIX canonical panels, so the whole figure is one wave and
the protocol column is a constant. What that removes is not tidiness, it is three real holes:
CheMeleon was frozen-only on BACE and Tox21 and absent from QM7 entirely.

The wave difference was never small. The same ECFP4+desc anchor reads 0.8712 on BACE in the
mainline wave and 0.7836 under label-efficiency -- 8.8 points, more than the spread between arms on
that panel -- and the same CheMeleon representation reads 0.8712 against 0.8289. Everything drawn
here now shares one split construction, so those offsets cannot enter any comparison.

ESTIMATOR. Error bars were "±1 SD of that panel's replicate unit", and the replicate unit was not
the same thing for every arm: pretraining-seed spread where an arm has 3 pretrainings, head-seed
spread for CheMeleon, which has ONE pretraining by construction. Those are different estimands, and
putting them on one axis invites a comparison neither supports -- a wide CheMeleon bar and a narrow
CLIMB bar would look like a precision difference when they measure different sources of variation.

Every interval here now comes from figure_data/six_panel/a2_errorbars.csv, and every arm WITHIN a
panel shares one method -- which is the comparison this figure needs. The method is panel-shaped
by necessity, and that is asserted rather than assumed:

  MolNet panels   scaffold cluster bootstrap -- resample the test scaffolds, recompute, 2.5/97.5
  MoleculeACE     target cluster bootstrap -- resample the 30 targets; there is no single pooled
                  test set to resample, since the panel IS a macro-mean over separate tasks
  Ames            analytic, derived from the AUC and the class counts, because Polaris withholds
                  the test labels and no resample of them is possible

Across panels the metrics differ anyway, so a cross-panel comparison was never available and
nothing is lost. Intervals are ASYMMETRIC because the bootstrap distribution is, and averaging the
two sides would hide skew on exactly the small panels where skew is largest.

The drawn VALUE comes from the same file, so the point and its interval can never describe
different estimators -- the defect audit check 8 exists to catch. Each value is still cross-checked
against mainline_8M.csv here, so a disagreement is a loud failure rather than a silent redraw.

NO FALLBACK. A point with no bootstrap interval is a FAILURE, not a point with a different kind of
bar. That is the whole discipline this rewrite buys.

CBS is no longer emitted. It is not one of the six canonical panels, it is not drawn, and the
bootstrap does not cover it -- keeping it would mean carrying rows that violate the single-estimator
rule for no visible benefit.

Writes: figure_data/SI_fig_a/SI_fig_a_e2e_need.csv
Run:    python3 scripts/build_SI_fig_a_table.py
"""
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
from figures.arms import ARMS, PANELS, PANEL_ORDER, E2E_PAIRS, series_label    # noqa: E402

# (frozen arm key, end2end arm key). LABELS COME FROM arms.py, never from a literal here: the
# label is the join key figures/SI_fig_a.py matches on, and when the two were separate literals
# they drifted -- arms.py renamed sup_dense to "supervised, desc" while the figure still asked for
# "supervised, dense", the join returned nothing, and that encoder's line vanished from all six
# panels with nothing failing.
#
# CheMeleon's frozen arm is the XGBOOST PROBE, not the MLP probe (Leif 2026-08-20: the only two
# CheMeleon models the paper mentions are frozen+XGBoost and end-to-end-from-foundation). That is
# the same convention fig_A1 already uses -- each representation at the head that suits it, which
# SI fig f measures as a property in its own right -- so the two figures now agree on what
# "CheMeleon, frozen" means. It does mean the CheMeleon slope changes head between its ends while
# the CLIMB slopes do not; the caption says so, because the alternative is reporting a CheMeleon
# configuration the paper never otherwise uses.
PAIRS = E2E_PAIRS
PROTOCOL = "mainline"
# Direction from arms.py, not a literal dict. The literal omitted HIV once, and a dict that happens
# to contain a key with a stale value is the version of that bug that does not raise.
# PANEL_ORDER, not PANELS: PANELS carries CBS, which this figure does not draw and which the
# cluster bootstrap does not cover (it resamples MolNet/MoleculeACE/Polaris test scaffolds).
# Iterating PANELS made the build fail on six CBS points that were never wanted -- the right
# failure for a missing interval, on the wrong panel set.
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
                # A ROW IS NOT AN INTERVAL. The bootstrap emits a placeholder with method
                # MISSING_OOF and NaN bounds when an arm has a summary number but no per-molecule
                # dump for that panel, which is a different thing from "never run" and must not be
                # drawn as a bare point. Treat it exactly like an absent row.
                if not len(r) or not np.isfinite(float(r.ci_lo.iloc[0])) \
                        or not np.isfinite(float(r.ci_hi.iloc[0])):
                    why = str(r.method.iloc[0]) if len(r) else "no row"
                    missing.append((panel, label, probe, arm, why))
                    continue
                r = r.iloc[0]
                # CROSS-CHECK AGAINST THE AGGREGATE, so a stale bootstrap cannot redraw a point
                # quietly. The bootstrap's observed value IS the bar's estimator by construction
                # (pooled over the same seed dirs, per-fold then averaged), so any real gap means
                # one of the two files is a different vintage of the run.
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
    # ONE METHOD PER PANEL, not one method for the whole table. The estimator is necessarily
    # panel-shaped: MolNet panels resample test SCAFFOLDS, MoleculeACE resamples its 30 TARGETS
    # (there is no single pooled test set to resample), and Ames has withheld labels so its
    # interval is derived analytically from the AUC and the class counts. What comparability
    # requires is that every arm WITHIN a panel share one method -- comparing across panels was
    # never valid anyway, since they carry different metrics. Asserting one method globally was
    # the wrong invariant and would have blocked a correct table.
    for _p, _g in d.groupby("panel"):
        assert _g.method.nunique() == 1, (
            f"panel {_p} mixes interval methods {sorted(set(_g.method))} -- its arms' bars would "
            f"not be comparable with each other, which is the one comparison this figure needs")
    print(f"wrote {OUT.relative_to(ROOT)}  {len(rows)} rows "
          f"({len(PAIRS)} encoders x {len(PANEL_ORDER)} panels x 2 probes, no holes)")
    print(f"every interval: {d.method.iloc[0]}; every point: {PROTOCOL} wave")

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
