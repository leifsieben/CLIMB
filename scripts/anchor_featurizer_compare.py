"""Before/after table for the anchor featurizer change, over all three variants.

  stereo-blind   the published pre-2026-08-19 anchor, archived under <run>/_pre_stereo_fix/
  ECFP4+stereo   the headline: same descriptor, chirality flag on
  Morgan r3-cnt  the max-information variant, in <run>_r3c/

Reports mean +/- sd over the three seed dirs, and the ECFP-vs-ECFP+desc GAP per panel, because
if counts or chirality mostly recover what the descriptor block was already supplying then that
gap should narrow -- which says more about what the descriptors contribute than either number.
"""
from __future__ import annotations
import csv, statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
P2 = ROOT / "figure_data/climb_v2_phase2"
CBS = ROOT / "figure_data/cbs_benchmark"

PANELS = [("BACE", "moleculenet_cv", "roc_auc", False), ("Tox21", "moleculenet_cv_tox21fixed", "roc_auc", False),
          ("HIV", "moleculenet_cv", "nef1", False), ("QM7", "moleculenet_cv_qm7native", "rmse", True),
          ("BBBP", "moleculenet_cv", "roc_auc", False), ("ESOL", "moleculenet_cv", "rmse", True),
          ("cbs", "moleculenet_cv", "nef1", False)]
ARMS = {"ECFP": "ecfp4_anchor", "ECFP+desc": "fp_desc_anchor"}
VARIANTS = [("stereo-blind", "{b}{s}/_pre_stereo_fix/{sub}"),
            ("ECFP4+stereo", "{b}{s}/{sub}"),
            ("Morgan r3-cnt", "{b}{s}_r3c/{sub}")]


def read(path: Path, ds, metric):
    if not path.exists():
        return None
    for r in csv.DictReader(path.open()):
        if r["dataset"] == ds and r["head_seed"] == "MEAN" and r["main_metric"] == metric:
            try:
                return float(r["main_value"])
            except ValueError:
                return None
    return None


def collect(base, tmpl, ds, sub, metric):
    root = CBS if ds == "cbs" else P2
    vals = []
    for suf in ("", "_s1", "_s2"):
        p = root / tmpl.format(b=base, s=suf, sub=sub) / "moleculenet_summary.csv"
        v = read(p, ds, metric)
        if v is not None:
            vals.append(v)
    return vals


def main() -> int:
    print(f"{'panel':7} {'arm':10}" + "".join(f"{n:>22}" for n, _ in VARIANTS) + "     n")
    print("-" * 96)
    gaps = {}
    for ds, sub, metric, lower_better in PANELS:
        for arm, base in ARMS.items():
            line = f"{ds:7} {arm:10}"
            got = {}
            for vname, tmpl in VARIANTS:
                vals = collect(base, tmpl, ds, sub, metric)
                got[vname] = vals
                if vals:
                    sd = st.stdev(vals) if len(vals) > 1 else 0.0
                    line += f"{st.mean(vals):>15.4f} +/-{sd:>5.4f}" if metric != "rmse" \
                        else f"{st.mean(vals):>14.2f} +/-{sd:>6.2f}"
                else:
                    line += f"{'--':>22}"
            n = max(len(v) for v in got.values()) if got else 0
            print(line + f"{n:>6}")
            gaps[(ds, arm)] = {k: (st.mean(v) if v else None) for k, v in got.items()}
        # ECFP -> ECFP+desc gap per variant
        g = "        gap       "
        for vname, _ in VARIANTS:
            a, b = gaps[(ds, "ECFP")][vname], gaps[(ds, "ECFP+desc")][vname]
            g += f"{(b - a):>22.4f}" if (a is not None and b is not None) else f"{'--':>22}"
        print(g + "   <- ECFP+desc minus ECFP")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
