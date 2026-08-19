"""SI Fig e — build the label-efficiency crossover table (the figure's ONLY input).

The question: end-to-end training on the downstream task uses NO pretraining at all. Pretrained
CLIMB encoders, used frozen, should beat it when labels are scarce — that is the whole case for
pretraining. But end2end has the whole network to fit, so it must catch up eventually. WHERE?

Three arms, identical hold-out split, identical label fractions, identical seed grid, so the only
difference is the model:

  e2e     end-to-end from a RANDOM init — no pretraining          (`e2e`   in the source)
  sup     supervised, dense  — pretrained, frozen + probe          (`sup`   in the source)
  unsup   unsupervised (MLM) — pretrained, frozen + probe          (`unsup` in the source)

Source: analysis/rigor/label_efficiency_fractions_all.csv (5 fractions x 3 subsample seeds x 3 head
seeds; 3 cells at 100%, where there is nothing to subsample). Values are the MEAN over those cells.

PANEL SCOPE — 4 of the canonical six. The label-fraction grid was built on MoleculeNet, so
MoleculeACE and Ames have no fraction sweep for any arm yet; both were launched 2026-08-18 and are
emitted as empty panels until they land, so the gap stays visible rather than being hidden by
silently reshaping the figure to the tasks that happen to have data.

CBS IS SUBSTITUTED BY HIV, and the caption must say so. CBS cannot be swept at all: it has 43
actives in 10,445 molecules, so an 80% train split at the sweep's fractions leaves ~2 actives at 5%
and ~3 at 10%. NEF1@1% computed from 2 actives is noise, and any crossover it produced would be an
artefact of losing the positive class rather than a label-budget effect — which is why cbs_e2e.py
was written without a fraction dimension in the first place. HIV is the right stand-in because it
is the same KIND of task: a rare-active virtual screen scored with the same NEF1@1% metric, so the
panel still asks the question CBS was there to ask. It has 32,901 training molecules, so the
positive class survives subsampling. (Decision: user, via the compute session, 2026-08-18.)

ARM MAPPING ON THE HIV PANEL is the same as every other panel: `e2e` / `sup` / `unsup`. The compute
session suggested mapping the source's `random` arm to "no pretrain", but `random` is the FROZEN
random-encoder floor, a different model from the end-to-end-from-random-init arm the other panels
draw. Using it would have put a different estimand on one panel of a six-panel figure.

PROTOCOL NOTE: single hold-out split, NOT the 5-fold scaffold CV of Figs A2/B — absolute values are
not comparable across those figures. Verified 2026-08-17: native units (QM7 ~200 kcal/mol).

No error bars are emitted for the figure (user decision 2026-08-17, matching Fig B); the per-point
SD across seed cells is kept in the `sd` column if a referee asks.

Writes: figure_data/SI_fig_e/SI_fig_e_crossover.csv
Run:    python3 scripts/build_SI_fig_e_table.py
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "analysis" / "rigor" / "label_efficiency_fractions_all.csv"
OUT = ROOT / "figure_data" / "SI_fig_e" / "SI_fig_e_crossover.csv"

# canonical panel -> the task name in the label-efficiency source (None = never run)
PANEL_TASK = {"MoleculeACE": "MoleculeACE", "HIV": "HIV", "BACE": "BACE",
              "Ames": "Ames", "Tox21": "Tox21", "QM7": "QM7"}
PRIMARY = {"BACE": "roc_auc", "Tox21": "roc_auc", "QM7": "rmse", "HIV": "nef1",
           "MoleculeACE": "macro_rmse"}
# panel -> the task actually drawn there, when it is not the panel's own task. The figure prints
# this so a reader can never mistake the HIV curve for a CBS curve; the caption explains why.
# No substitution any more: HIV took CBS's slot in the canonical six on 2026-08-19, for the same
# reason it stood in here -- CBS cannot be subsampled (43 actives) and, as it turned out, could not
# separate models either. This figure was the first place HIV replaced CBS; the panel set has now
# caught up with it.
SUBSTITUTED = {}
# arm key in the source -> the canonical arms.py key whose colour/label the figure must use
ARMS = [("e2e", "e2e_no_pretrain"), ("sup", "sup_dense"), ("unsup", "unsup")]


# ---------------------------------------------------------------------------------------------
# MoleculeACE's sweep is NOT in the label-efficiency CSV -- it was run later, per-target, and lands
# as figure_data/chemeleon_suite/moleculeace/le_mace_<arm>_f<frac>/results.csv. Same protocol as the
# MolNet panels (train indices subsampled per task, test untouched, 3 eval seeds), so a point here
# means what a point there means.
MACE_DIR = ROOT / "figure_data" / "chemeleon_suite" / "moleculeace"
MACE_DATA = ROOT / "chemeleon_suite" / "data" / "moleculeace"
MACE_ARMS = {"unsup": "unsup", "sup_dense": "sup", "e2e": "e2e"}   # dir token -> source-arm name

# Ames arrives pre-scored (Polaris withholds test labels, so scoring is always post-hoc through
# bench.evaluate() -- an EMPTY results.csv under chemeleon_suite/polaris/ is the normal state for
# every arm there, not a failed run).
AMES_SCORES = ROOT / "figure_data" / "label_eff_ames.csv"
AMES_SPLIT = ROOT / "chemeleon_suite" / "data" / "polaris" / "tdcommons__ames.csv"
AMES_ARM_KEY = {"unsup": "unsup", "sup_dense": "sup_dense", "e2e": "e2e_no_pretrain"}


def ames_train_total() -> int:
    """Labelled TRAINING molecules for Ames, from the benchmark's own split column (5,821)."""
    import csv as _csv
    with open(AMES_SPLIT) as fh:
        return sum(1 for r in _csv.DictReader(fh) if r.get("split") == "train")


def ames_rows():
    """[(arm_key, pct, n_train, roc_auc, sd, n_seeds)] -- mean over the 3 EVAL seeds."""
    if not AMES_SCORES.exists():
        return []
    total = ames_train_total()
    d = pd.read_csv(AMES_SCORES)
    d = d[d.metric == "roc_auc"]
    out = []
    for (arm, frac), g in d.groupby(["arm", "fraction"]):
        key = AMES_ARM_KEY.get(arm)
        if key is None:
            continue
        v = g.value.to_numpy(float)
        out.append((key, int(round(float(frac) * 100)), int(round(float(frac) * total)),
                    float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else float("nan"), len(v)))
    return out


def mace_train_total() -> int:
    """Exact labelled TRAINING molecules across the 30 targets, from the benchmark's own splits.

    The x-axis is 'labelled training molecules', so it must be the real count, not fraction x an
    assumed split ratio. Summing `split == "train"` over the 30 target CSVs gives 38,912.
    """
    import csv as _csv
    tot = 0
    for f in sorted(MACE_DATA.glob("*.csv")):
        with open(f) as fh:
            tot += sum(1 for r in _csv.DictReader(fh) if r.get("split") == "train")
    return tot


def mace_rows(panel: str):
    """[(arm_key, pct, n_train, macro_rmse, sd, n_cells)] -- macro-mean RMSE over the 30 targets,
    matching the MoleculeACE panel's metric everywhere else in the paper."""
    total = mace_train_total()
    out = []
    for dir_tok, src_arm in MACE_ARMS.items():
        arm_key = dict(unsup="unsup", sup_dense="sup_dense", e2e="e2e_no_pretrain")[dir_tok]
        for d in sorted(MACE_DIR.glob(f"le_mace_{dir_tok}_f*")):
            frac = float(d.name.rsplit("_f", 1)[1])
            f = d / "results.csv"
            if not f.exists():
                continue
            t = pd.read_csv(f)
            t = t[(t.metric == "rmse") & (t.subset == "overall")]
            if t.empty:
                continue
            # macro-mean over targets FIRST, then across eval seeds -- the same order as
            # scripts/six_panel_aggregate.mace_seed_macros, so this is the same estimand
            per_seed = t.groupby("seed").value.mean()
            out.append((arm_key, int(round(frac * 100)), int(round(frac * total)),
                        float(per_seed.mean()),
                        float(per_seed.std(ddof=1)) if len(per_seed) > 1 else float("nan"),
                        int(len(per_seed))))
    return out


def main() -> None:
    d = pd.read_csv(SRC)
    rows = []
    for panel, task in PANEL_TASK.items():
        if task is None:
            continue
        if panel == "Ames":
            for arm_key, pct, n_train, v, sd, n in ames_rows():
                rows.append(dict(panel=panel, task="Ames", metric="roc_auc", higher_better=1,
                                 arm=arm_key, substituted_for="", pct=pct, n_train=n_train,
                                 value=round(v, 6), sd=round(sd, 6) if sd == sd else "",
                                 n_cells=n))
            continue
        if panel == "MoleculeACE":
            for arm_key, pct, n_train, v, sd, n in mace_rows(panel):
                rows.append(dict(panel=panel, task="MoleculeACE", metric="macro_rmse",
                                 higher_better=0, arm=arm_key, substituted_for="",
                                 pct=pct, n_train=n_train, value=round(v, 6),
                                 sd=round(sd, 6) if sd == sd else "", n_cells=n))
            continue
        m = PRIMARY[task]
        for src_arm, arm_key in ARMS:
            g = d[(d.arm == src_arm) & (d.task == task) & (d.metric == m) & (d.split == "test")]
            for pct, cell in g.groupby("pct"):
                v = cell.value.to_numpy(float)
                rows.append(dict(panel=panel, task=task, metric=m,
                                 higher_better=int(m != "rmse"), arm=arm_key,
                                 pct=int(pct), n_train=int(cell.n_train.iloc[0]),
                                 substituted_for=panel if panel in SUBSTITUTED else "",
                                 value=round(float(np.mean(v)), 6),
                                 sd=round(float(np.std(v, ddof=1)), 6) if len(v) > 1 else "",
                                 n_cells=len(v)))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    cols = ["panel", "task", "metric", "higher_better", "arm", "substituted_for", "pct",
            "n_train", "value", "sd", "n_cells"]
    with open(OUT, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    missing = [p for p, t in PANEL_TASK.items() if t is None]
    print(f"wrote {OUT.relative_to(ROOT)}  {len(rows)} rows")
    print(f"panels with NO label-fraction sweep (drawn empty): {', '.join(missing)}")
    for panel, task in SUBSTITUTED.items():
        print(f"SUBSTITUTION: the {panel} panel draws {task} "
              f"({PRIMARY[task]}) — {panel} has too few actives to subsample; state in caption")

    f = pd.DataFrame(rows)
    for panel in [p for p, t in PANEL_TASK.items() if t]:
        p = f[f.panel == panel]
        print(f"\n{panel} ({p.metric.iloc[0]}):")
        print(p.pivot(index="arm", columns="n_train", values="value").round(4).to_string())


if __name__ == "__main__":
    main()
