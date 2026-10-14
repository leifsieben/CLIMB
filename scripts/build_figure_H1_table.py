from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
import sys as _sys
_sys.path.insert(0, str(ROOT))
from figures.arms import ARMS as _ARMS_REGISTRY

RIGOR = ROOT / "analysis" / "rigor"
SOURCES = [RIGOR / "label_efficiency_fractions.csv",
           RIGOR / "label_efficiency_fractions_e2e.csv"]


SRC_FPDESC = RIGOR / "label_efficiency_fractions_fpdesc.csv"
OUT = ROOT / "figure_data" / "SI_fig_e" / "SI_fig_e_crossover.csv"


PANEL_TASK = {"MoleculeACE": "MoleculeACE", "HIV": "HIV", "BACE": "BACE",
              "Ames": "Ames", "Tox21": "Tox21", "QM7": "QM7"}
PRIMARY = {"BACE": "roc_auc", "Tox21": "roc_auc", "QM7": "rmse", "HIV": "nef1",
           "MoleculeACE": "macro_rmse"}


SUBSTITUTED = {}

ARMS = [("e2e", "e2e_no_pretrain"), ("sup", "sup_dense"), ("unsup", "unsup"),
        ("fp_desc", "ecfp_desc")]


MACE_DIR = ROOT / "figure_data" / "chemeleon_suite" / "moleculeace"
MACE_DATA = ROOT / "benchmarks" / "data" / "moleculeace"


MACE_ARMS = {"unsup": "unsup", "sup_dense": "sup_dense", "e2e": "e2e_no_pretrain",
             "fp_desc": "ecfp_desc"}
_bad = [v for v in MACE_ARMS.values() if v not in _ARMS_REGISTRY]
assert not _bad, f"MACE_ARMS maps to arm key(s) absent from arms.py: {_bad}"


AMES_SCORES = ROOT / "figure_data" / "label_eff_ames.csv"
AMES_SPLIT = ROOT / "benchmarks" / "data" / "polaris" / "tdcommons__ames.csv"
AMES_ARM_KEY = {"unsup": "unsup", "sup_dense": "sup_dense", "e2e": "e2e_no_pretrain",
                "fp_desc": "ecfp_desc"}


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

    import csv as _csv
    tot = 0
    for f in sorted(MACE_DATA.glob("*.csv")):
        with open(f) as fh:
            tot += sum(1 for r in _csv.DictReader(fh) if r.get("split") == "train")
    return tot


def mace_rows(panel: str):

    total = mace_train_total()
    out = []
    for dir_tok, arm_key in MACE_ARMS.items():
        for d in sorted(MACE_DIR.glob(f"le_mace_{dir_tok}_f*")):
            frac = float(d.name.rsplit("_f", 1)[1])
            f = d / "results.csv"
            if not f.exists():
                continue
            t = pd.read_csv(f)
            t = t[(t.metric == "rmse") & (t.subset == "overall")]
            if t.empty:
                continue


            per_seed = t.groupby("seed").value.mean()
            value = float(per_seed.mean())


            per_target = t.groupby("task").value.mean()
            sd = float(per_target.std(ddof=1)) if len(per_target) > 1 else float("nan")
            out.append((arm_key, int(round(frac * 100)), int(round(frac * total)),
                        value, sd, int(len(per_seed))))
    return out


def main() -> None:
    d = pd.concat([pd.read_csv(src) for src in SOURCES], ignore_index=True)
    if SRC_FPDESC.exists():
        extra = pd.read_csv(SRC_FPDESC)


        print(f"  + {len(extra)} rows from {SRC_FPDESC.name}: "
              f"{sorted(set(extra.arm))} on {sorted(set(extra.task))}")
        d = pd.concat([d, extra], ignore_index=True)
    else:
        print(f"  {SRC_FPDESC.name} ABSENT -- the XGBoost anchor will be missing from every panel")
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
