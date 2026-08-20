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
# The trailing LABEL is arms.py's, not a literal. It is the join key the figure matches on, and
# when the two were separate literals they drifted: arms.py renamed sup_dense to "supervised,
# desc" while figures/SI_fig_a.py still asked for "supervised, dense", so the figure's join
# returned nothing and that encoder's line silently vanished from all six panels. Nothing failed --
# the panels were still populated by the OTHER encoder, so no empty-panel check fired.
import sys as _sys
_sys.path.insert(0, str(ROOT))
from figures.arms import ARMS, PANELS

ENCODERS = [("unsup", "unsup", "unsup_8M", "unsup_8M_e2e", "unsup", "unsup_only",
             ARMS["unsup"]["label"]),
            ("sup_dense", "sup_dense", "skip_dense_8M", "skip_dense_8M_e2e", "sup",
             "sup_only:dense", ARMS["sup_dense"]["label"])]
AMES = ("tdcommons/ames", "roc_auc")
# Direction comes from arms.py, not a literal dict. The literal omitted HIV, so adding the HIV
# branch raised KeyError -- which is the good failure. The bad version of the same bug is a dict
# that happens to contain the key with a stale value.
HIGHER = {p: int(d["higher_better"]) for p, d in PANELS.items()}
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

        # ---- HIV: mainline protocol, BOTH sides in the same wave ----
        #
        # HIV was the one empty panel in this figure until 2026-08-20, and it was empty because
        # this builder had no branch for it, not because the data was missing: HIV's e2e runs are
        # 5-fold CV in climb_v2_phase2 (mainline), while the branch below covers only the three
        # label-efficiency panels. Both sides are now read from the mainline wave.
        #
        # SD IS THE PRETRAINING-SEED SPREAD ON BOTH SIDES, deliberately. mainline_8M.csv offers
        # sd_total for HIV, but that is over 15 cells (3 seeds x 5 folds) and reads 0.104 -- an
        # order of magnitude above the e2e side's 0.012, which is a spread over 3 seed means. One
        # slope drawn with a fold-spread bar at one end and a seed-spread bar at the other would
        # invite exactly the comparison it cannot support, so the frozen end uses sd_seeds.
        hiv_metric = PANELS["HIV"]["metric"]
        rf = main_tbl[(main_tbl.arm == main_arm) & (main_tbl.panel == "HIV")
                      & (main_tbl.metric == hiv_metric)]
        if len(rf):
            add("HIV", label, "frozen", float(rf.value.iloc[0]), _sd(rf.extra.iloc[0], "sd_seeds"),
                3, "mainline")
        vals = []
        for d in (e2e_dir, f"{e2e_dir}_s1", f"{e2e_dir}_s2"):
            f = FD / "climb_v2_phase2" / d / "moleculenet_cv" / "moleculenet_summary.csv"
            if not f.exists():
                continue
            o = pd.read_csv(f)
            o = o[(o.dataset == "HIV") & (o.main_metric == hiv_metric) & (o.head_seed == "MEAN")]
            if len(o):
                vals.append(float(o.main_value.iloc[0]))
        if vals:
            add("HIV", label, "end2end", float(np.mean(vals)),
                float(np.std(vals, ddof=1)) if len(vals) > 1 else np.nan, len(vals), "mainline")

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

    # ---- CheMeleon, frozen vs end2end, PROTOCOL-MATCHED PER PANEL (user 2026-08-20) ------------
    #
    # The external comparator answers the same question this figure asks, so it is drawn beside
    # the CLIMB encoders. Which SOURCE it comes from depends on the panel, for exactly the reason
    # the anchor is resolved per panel in figures/SI_fig_a.py: the two waves differ by more than
    # the arms do.
    #
    #   MoleculeACE, Ames, HIV   mainline wave -> mainline_8M.csv. Matches the CLIMB rows beside
    #                            it, and both probes exist, so the full frozen->end2end slope is
    #                            drawn.
    #   BACE, Tox21, QM7         the CLIMB rows here are label-efficiency at 100%, and CheMeleon
    #                            was NOT in that wave until 2026-08-20. It is now, for the FROZEN
    #                            probe: scripts/label_eff_fractions.py grew a PRECOMPUTED branch
    #                            that reads the CheMeleon vectors from an npz through the same
    #                            zscore and the same MLP head the CLIMB lines use, so the arm
    #                            differs from them in representation only.
    #
    # THE MAINLINE VALUE WAS NOT USABLE ON THOSE THREE PANELS, and this is now measured rather
    # than argued. Same arm, same panel, the two waves read:
    #
    #     BACE    mainline 0.8712   label-efficiency 0.8289    -0.042
    #     Tox21   mainline 0.8293   label-efficiency 0.7516    -0.078
    #
    # and the offset is not a CheMeleon property -- unsup moves -0.033/-0.061 and sup_dense
    # -0.033/-0.074 between the same two waves. It is the split construction (scaffold 5-fold CV
    # vs a single scaffold hold-out), which is harder, for every arm. Drawing the mainline level
    # against label-efficiency neighbours therefore placed CheMeleon 4 to 8 points too high on
    # panels where the whole point is where it sits relative to them.
    #
    # NO END2END ON THOSE PANELS, AND THAT IS DELIBERATE. CheMeleon end-to-end is a chemprop
    # D-MPNN fine-tune, which the label-efficiency driver cannot host (it has an encoder branch, a
    # classical branch and now a precomputed branch -- a D-MPNN fits none). Pairing the matched
    # frozen point with the MAINLINE end2end point would draw a slope whose rise is mostly the
    # protocol offset above, manufacturing a frozen->end2end gain out of a wave difference. So
    # those panels carry the frozen marker alone and the caption states why.
    #
    # QM7 CARRIES NO CHEMELEON POINT AT ALL. Its label-efficiency cell returned test RMSE 1818.9
    # against train 206.9 -- against a target SD of 228.7, worse than predicting the training mean,
    # while every other arm in that wave lands at 200-213. It is a broken cell, not a
    # representation result, and it is quarantined in
    # analysis/rigor/label_efficiency_chemeleon_frozen_summary.QM7_FAILED.csv rather than plotted.
    # The mainline QM7 value is not a substitute for it: unsup moves 197.9 -> 212.7 between the
    # same two waves, so the same offset applies here too.
    #
    # SD: sd_seeds where the arm has pretraining replicates, sd_total otherwise. CheMeleon has ONE
    # pretraining by construction on the two suite tracks (n_seeds=1), so there sd_total -- its
    # head/eval-seed spread -- is the only replicate spread it can have. On the label-efficiency
    # panels the SD is the head-seed spread over the 3 cells, the same estimand as the CLIMB rows.
    CHEMELEON = [("chemeleon_frozen", "frozen"), ("chemeleon_e2e", "end2end")]
    che_label = ARMS["chemeleon_frozen"]["label"].split(",")[0]          # "CheMeleon"
    CHE_LE = ROOT / "analysis" / "rigor" / "label_efficiency_chemeleon_frozen_summary.csv"
    che_le = pd.read_csv(CHE_LE) if CHE_LE.exists() else pd.DataFrame()
    for arm_key, probe in CHEMELEON:
        for panel in HIGHER:
            if panel in MOL_METRIC:
                # label-efficiency panel: matched source only, frozen only
                if probe != "frozen" or not len(che_le):
                    continue
                # metric matched EXPLICITLY -- the summary carries nef1 alongside roc_auc for the
                # classification panels, and a positional read would take whichever sorted first.
                f = che_le[(che_le.task == panel) & (che_le.metric == MOL_METRIC[panel])
                           & (che_le.split == "test") & (che_le.pct == 100)]
                if not len(f):
                    continue
                assert len(f) == 1, f"CheMeleon {panel}: {len(f)} label-efficiency rows, expected 1"
                add(panel, che_label, "frozen", float(f["mean"].iloc[0]),
                    float(f["std"].iloc[0]), int(f.n_cells.iloc[0]), "label-efficiency 100%")
                continue
            r = main_tbl[(main_tbl.arm == arm_key) & (main_tbl.panel == panel)]
            if not len(r):
                continue
            extra = r.extra.iloc[0]
            sd = _sd(extra, "sd_seeds")
            if not np.isfinite(sd):
                sd = _sd(extra, "sd_total")
            n = _sd(extra, "n_seeds")
            add(panel, che_label, probe, float(r.value.iloc[0]), sd,
                int(n) if np.isfinite(n) else 3, "mainline")

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
