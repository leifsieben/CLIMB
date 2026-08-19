"""SI Fig a — build the "do you need end-to-end training?" table (the figure's ONLY input).

For each canonical panel, the SAME pretrained encoder used two ways at FULL downstream data:

  frozen    encoder frozen, probe trained on the labels
  end2end   the whole network fine-tuned on the labels

Two encoders: `unsupervised` (unsup_8M) and `supervised, dense` (skip_dense_8M) — the best-two arms.

PROTOCOL: each PANEL is internally consistent (its frozen and end2end numbers come from the same
wave, split and seed grid), which is what the within-panel frozen-vs-end2end comparison needs. The
protocol DIFFERS BETWEEN panels and the `protocol` column says which:

  MoleculeACE, Ames   mainline protocol. frozen from figure_data/six_panel/mainline_8M.csv;
                      end2end from chemeleon_suite/{moleculeace,polaris}/<arm>_e2e/.
  CBS                 both sides from cbs_benchmark/<arm>{,_e2e}/moleculenet_cv/ (the end2end
                      side landed 2026-08-18). The SAME 5 benchmark-provided UMAP folds, each fold
                      already ensembled over the 3 seeds, on both sides -- so frozen and end2end
                      carry the identical estimand (SD across 5 folds). NOTE: two dirs of the e2e
                      names briefly existed as smoke stubs (1 fold, 2 epochs, NEF1 0.0); they are
                      quarantined as *_SMOKE_STUB and this reader must never fall back to them.
  BACE, Tox21, QM7    label-efficiency protocol at the 100% fraction. frozen from
                      analysis/rigor/label_efficiency_fractions_all.csv; end2end from
                      figure_data/six_panel/six_panel_e2e.csv. Same pair SI Fig f uses.

NEVER compare a value in one panel to a value in another; compare frozen vs end2end WITHIN a panel.

Writes: figure_data/SI_fig_a/SI_fig_a_e2e_need.csv
Run:    python3 scripts/build_SI_fig_a_table.py
"""
from __future__ import annotations

import csv
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
FD = ROOT / "figure_data"
OUT = FD / "SI_fig_a" / "SI_fig_a_e2e_need.csv"

# arms.py key -> (mainline arm key, e2e run dir, label-efficiency frozen arm, wave-3 e2e arm)
# key, mainline arm key, FROZEN run dir, e2e run dir, label-eff frozen arm, wave-3 e2e arm, label
ENCODERS = [("unsup", "unsup", "unsup_8M", "unsup_8M_e2e", "unsup", "unsup_only", "unsupervised"),
            ("sup_dense", "sup_dense", "skip_dense_8M", "skip_dense_8M_e2e", "sup",
             "sup_only:dense", "supervised, desc")]
AMES = ("tdcommons/ames", "roc_auc")
HIGHER = {"MoleculeACE": 0, "CBS": 1, "BACE": 1, "Ames": 1, "Tox21": 1, "QM7": 0}
MOL_METRIC = {"BACE": "roc_auc", "Tox21": "roc_auc", "QM7": "rmse"}


def _sd(extra, key="sd_total"):
    m = re.search(rf"{key}=([-\d.eE]+)", str(extra))
    return float(m.group(1)) if m else np.nan


def main() -> None:
    main_tbl = pd.read_csv(FD / "six_panel" / "mainline_8M.csv")
    fro = pd.read_csv(ROOT / "analysis" / "rigor" / "label_efficiency_fractions_all.csv")
    e2e = pd.read_csv(FD / "six_panel" / "six_panel_e2e.csv")
    rows = []

    def add(panel, enc_label, probe, value, sd, n, protocol):
        if not np.isfinite(value):
            return
        rows.append(dict(panel=panel, higher_better=HIGHER[panel], encoder=enc_label, probe=probe,
                         value=round(float(value), 6),
                         sd=("" if not np.isfinite(sd) else round(float(sd), 6)),
                         n=n, protocol=protocol))

    for key, main_arm, main_dir, e2e_dir, le_arm, w3_arm, label in ENCODERS:
        # ---- MoleculeACE + hERG: mainline protocol ----
        for panel in ("MoleculeACE", "Ames"):
            r = main_tbl[(main_tbl.arm == main_arm) & (main_tbl.panel == panel)]
            if len(r):
                add(panel, label, "frozen", float(r.value.iloc[0]), _sd(r.extra.iloc[0]), 3,
                    "mainline")
        p = FD / "chemeleon_suite" / "moleculeace" / e2e_dir / "results.csv"
        if p.exists():
            o = pd.read_csv(p)
            o = o[(o.subset == "overall") & (o.metric == "rmse")]
            per = o.groupby("seed").value.mean()
            add("MoleculeACE", label, "end2end", per.mean(),
                per.std(ddof=1) if len(per) > 1 else np.nan, len(per), "mainline")
        p = FD / "chemeleon_suite" / "polaris" / e2e_dir / "polaris_scores.csv"
        if p.exists():
            v = pd.read_csv(p)
            v = v[(v.task == AMES[0]) & (v.metric == AMES[1])].value.astype(float)
            if len(v):
                add("Ames", label, "end2end", v.mean(),
                    v.std(ddof=1) if len(v) > 1 else np.nan, len(v), "mainline")

        # ---- CBS: both sides = the 5 benchmark-provided folds, each fold already the ensemble
        # over the 3 seeds. Read BOTH from cbs_benchmark/ rather than taking frozen from
        # mainline_8M.csv, so the two sides of the comparison are literally the same estimand
        # (SD across 5 folds) instead of frozen-sd_total vs e2e-fold-SD.
        fro_cbs = FD / "cbs_benchmark" / main_dir / "moleculenet_cv" / "moleculenet_summary.csv"
        if fro_cbs.exists():
            o = pd.read_csv(fro_cbs)
            v = o[(o.main_metric == "nef1")
                  & o.head_seed.astype(str).str.fullmatch(r"fold\d+")].main_value.astype(float)
            if len(v):
                add("CBS", label, "frozen", v.mean(),
                    v.std(ddof=1) if len(v) > 1 else np.nan, len(v), "CBS provided folds")
        # exact dir only: never glob, or the *_SMOKE_STUB dirs of the same stem could be picked up
        e2e_cbs = FD / "cbs_benchmark" / e2e_dir / "moleculenet_cv" / "per_fold.csv"
        if e2e_cbs.exists():
            v = pd.read_csv(e2e_cbs).nef1.astype(float)
            if len(v):
                add("CBS", label, "end2end", v.mean(),
                    v.std(ddof=1) if len(v) > 1 else np.nan, len(v), "CBS provided folds")

        # ---- BACE / Tox21 / QM7: label-efficiency protocol at 100% ----
        for panel, metric in MOL_METRIC.items():
            f = fro[(fro.arm == le_arm) & (fro.task == panel) & (fro.metric == metric)
                    & (fro.split == "test") & (fro.pct == 100)].value.astype(float)
            if len(f):
                add(panel, label, "frozen", f.mean(), f.std(ddof=1) if len(f) > 1 else np.nan,
                    len(f), "label-efficiency 100%")
            e = e2e[(e2e.arm == w3_arm) & (e2e.task == panel) & (e2e.metric == metric)
                    & (e2e.split == "test") & (e2e.pct == 100)].value.astype(float)
            if len(e):
                add(panel, label, "end2end", e.mean(), e.std(ddof=1) if len(e) > 1 else np.nan,
                    len(e), "label-efficiency 100%")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    cols = ["panel", "higher_better", "encoder", "probe", "value", "sd", "n", "protocol"]
    with open(OUT, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    d = pd.DataFrame(rows)
    empty = [p for p in HIGHER if p not in set(d.panel)]
    print(f"wrote {OUT.relative_to(ROOT)}  {len(rows)} rows")
    print(f"panels with no end2end run of a pretrained encoder (drawn empty): {', '.join(empty)}")

    print("\ndoes end-to-end fine-tuning beat the frozen probe at full data?")
    print("  (delta signed so + = end2end better; compared against the combined SD)")
    for panel in [p for p in HIGHER if p in set(d.panel)]:
        g = d[d.panel == panel]
        sign = 1 if g.higher_better.iloc[0] else -1
        for *_, label in ENCODERS:
            fr = g[(g.encoder == label) & (g.probe == "frozen")]
            ee = g[(g.encoder == label) & (g.probe == "end2end")]
            if not len(fr) or not len(ee):
                continue
            delta = sign * (float(ee.value.iloc[0]) - float(fr.value.iloc[0]))
            sd = np.hypot(pd.to_numeric(fr.sd, errors="coerce").iloc[0],
                          pd.to_numeric(ee.sd, errors="coerce").iloc[0])
            flag = "*" if np.isfinite(sd) and abs(delta) > sd else " "
            print(f"   {panel:<12}{label:<20} frozen {float(fr.value.iloc[0]):9.4f}   "
                  f"end2end {float(ee.value.iloc[0]):9.4f}   delta {delta:+9.4f}{flag}")
    print("   * = |delta| exceeds the combined SD")


if __name__ == "__main__":
    main()
