"""Fig F — build the frozen-vs-end2end crossover table (the figure's ONLY input).

The question: a pretrained CLIMB encoder can be used two ways — freeze it and train a probe, or
fine-tune the whole thing end-to-end. Freezing wins in the small-data regime (the probe has few
parameters to overfit); end2end must win eventually. WHERE does it cross?

Both sides use the SAME two encoders, the SAME label fractions, the SAME single hold-out split and
the SAME seed grid, so the difference between them is the probe strategy and nothing else:

  frozen    analysis/rigor/label_efficiency_fractions_all.csv   arms `unsup`, `sup`
  end2end   figure_data/six_panel/six_panel_e2e.csv             arms `unsup_only`, `sup_only:dense`
  reference the same frozen file's `e2e` arm = end2end from a RANDOM init (no pretraining at all)

Scope: BACE, Tox21, QM7 — the panels where a crossover can exist at all. MoleculeACE (<=3.7k per
target) and hERG (132 test molecules) are entirely small-data, and CBS e2e is a separate
fine-tuning path; the Wave-3 driver deliberately did not run them. BBBP is dropped from the paper.

PROTOCOL NOTE: this whole figure is on the label-efficiency single hold-out split, NOT the 5-fold
scaffold CV of Figs A2/B, so its absolute values are NOT comparable to those figures (frozen unsup
BACE reads 0.825 here vs 0.858 there). It is internally consistent, which is what the crossover
needs. Verified 2026-08-17: both sources are in native units (QM7 ~200 kcal/mol).

Error bars are +-1 SD across the (subsample seed x head seed) cells: 9 at each of the 5/10/25/50%
fractions, 3 at 100% (nothing to subsample).

Writes: figure_data/figF/figF_crossover.csv
Run:    python3 scripts/build_figF_table.py
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
FROZEN = ROOT / "analysis" / "rigor" / "label_efficiency_fractions_all.csv"
E2E = ROOT / "figure_data" / "six_panel" / "six_panel_e2e.csv"
OUT = ROOT / "figure_data" / "figF" / "figF_crossover.csv"

TASKS = ["BACE", "Tox21", "QM7"]
PRIMARY = {"BACE": "roc_auc", "Tox21": "roc_auc", "QM7": "rmse"}
LOWER_BETTER = {"QM7"}

# encoder -> (arm name in the frozen file, arm name in the e2e file, display label)
ENCODERS = [("unsup", "unsup", "unsup_only", "unsupervised"),
            ("sup",   "sup",   "sup_only:dense", "supervised, dense")]
NOPRETRAIN = ("e2e", "no pretrain, end2end")


def cells(df, arm, task, metric):
    """{pct: [values]} for the test split of one (arm, task, metric)."""
    d = df[(df.arm == arm) & (df.task == task) & (df.metric == metric) & (df.split == "test")]
    return {int(p): g.value.to_numpy(float) for p, g in d.groupby("pct")}


def main() -> None:
    fro = pd.read_csv(FROZEN)
    e2e = pd.read_csv(E2E)
    rows = []

    for task in TASKS:
        m = PRIMARY[task]
        ntrain = (fro[(fro.task == task)].groupby("pct").n_train.first().to_dict())

        def emit(probe, label, enc_key, vals):
            for pct, v in sorted(vals.items()):
                if not len(v):
                    continue
                rows.append(dict(task=task, metric=m,
                                 direction="lower" if task in LOWER_BETTER else "higher",
                                 probe=probe, encoder=enc_key, label=label,
                                 pct=pct, n_train=int(ntrain.get(pct, 0)),
                                 mean=round(float(np.mean(v)), 6),
                                 sd=round(float(np.std(v, ddof=1)), 6) if len(v) > 1 else "",
                                 n_cells=len(v)))

        for key, farm, earm, label in ENCODERS:
            emit("frozen", label, key, cells(fro, farm, task, m))
            emit("end2end", label, key, cells(e2e, earm, task, m))
        emit("end2end", NOPRETRAIN[1], "none", cells(fro, NOPRETRAIN[0], task, m))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    cols = ["task", "metric", "direction", "probe", "encoder", "label", "pct", "n_train",
            "mean", "sd", "n_cells"]
    with open(OUT, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {OUT.relative_to(ROOT)}  {len(rows)} rows")

    # ---- where does end2end overtake frozen, for the same encoder? -------------------------
    d = pd.DataFrame(rows)
    print("\ncrossover (end2end - frozen, same encoder; + = end2end ahead):")
    for task in TASKS:
        sign = -1 if task in LOWER_BETTER else 1
        for key, _, _, label in ENCODERS:
            t = d[(d.task == task) & (d.encoder == key)]
            fr = t[t.probe == "frozen"].set_index("pct")
            ee = t[t.probe == "end2end"].set_index("pct")
            pcts = sorted(set(fr.index) & set(ee.index))
            delta = {p: sign * (ee.loc[p, "mean"] - fr.loc[p, "mean"]) for p in pcts}
            cross = next((f"{p}%" for p in pcts if delta[p] > 0), "never in range")
            s = "  ".join(f"{p}%:{delta[p]:+.4f}" for p in pcts)
            print(f"   {task:<6} {label:<20} first ahead at {cross:<14} | {s}")


if __name__ == "__main__":
    main()
