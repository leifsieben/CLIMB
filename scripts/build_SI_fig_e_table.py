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
PANEL_TASK = {"MoleculeACE": None, "CBS": "HIV", "BACE": "BACE",
              "Ames": None, "Tox21": "Tox21", "QM7": "QM7"}
PRIMARY = {"BACE": "roc_auc", "Tox21": "roc_auc", "QM7": "rmse", "HIV": "nef1"}
# panel -> the task actually drawn there, when it is not the panel's own task. The figure prints
# this so a reader can never mistake the HIV curve for a CBS curve; the caption explains why.
SUBSTITUTED = {"CBS": "HIV"}
# arm key in the source -> the canonical arms.py key whose colour/label the figure must use
ARMS = [("e2e", "e2e_no_pretrain"), ("sup", "sup_dense"), ("unsup", "unsup")]


def main() -> None:
    d = pd.read_csv(SRC)
    rows = []
    for panel, task in PANEL_TASK.items():
        if task is None:
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
