"""Fig E — build the corruption-ladder table (one tidy CSV, the figure's ONLY input).

CANONICAL SIX PANELS (migrated 2026-08-18). Previously this ran on the old MoleculeNet task set
(ESOL/BBBP/BACE/HIV/Tox21/QM7) because the corrupted encoders had MoleculeNet evals only. The
MoleculeACE / CBS / Polaris(Ames) evals of all 15 corrupted-arm runs landed 2026-08-18, so the
figure now runs on the paper's canonical six:

    MoleculeACE (macro RMSE, 30 targets) | CBS (NEF1%) | BACE (ROC-AUC)
    Ames (ROC-AUC)                       | Tox21 (mean ROC-AUC) | QM7 (RMSE)

Two independent corruption experiments, both scored with the SAME frozen probe, both expressed as
lift over the SAME kind of floor:

  panel "supervised"    real  = skip_dense_8M{,_s1,_s2}   (supervised, dense: descriptor regression)
                        garble= corrupt_mtr_8M{,_s1,_s2}  (descriptor targets permuted across batch)

  panel "unsupervised"  real  = unsup_8M{,_s1,_s2}        (MLM on real SMILES)
                        garble= corrupt_mlm_8M{,_s1,_s2}  -> shuffled (token order permuted in-seq)
                                bigram_8M{,_s1,_s2}       -> bigram   (resampled from bigram stats)
                                unigram_8M{,_s1,_s2}      -> unigram  (resampled from unigram stats)
                                wiki_real_8M{,_s1,_s2}    -> wiki     (Wikipedia; zero chemistry)

FLOOR CHOICE. Both panels lift over `no pretrain (frozen)` — a random-init encoder, frozen, same
probe. That isolates what the PRETRAINING OBJECTIVE contributes with architecture and probe held
fixed. (Figs C2/D lift over the end2end floor instead, because they ask a different question —
"is the frozen pipeline worth it at all". Do not mix the two.)

ONE FLOOR PER (PANEL, DATASET), FROM THAT DATASET'S OWN EVAL WAVE:
  MoleculeACE / CBS / Ames  -> random_baseline_0{0,1,2}, scored in the same benchmark wave as the
                               corrupted arms themselves. Shared by both sub-panels: it is
                               literally the same control encoder on the same benchmark, so
                               splitting it would invent a difference that does not exist.
  BACE / Tox21 / QM7        -> supervised sub-panel: random_baseline_0{0,1,2} [climb_v2_phase2]
                               unsupervised sub-panel: `no_pretrain (frozen)` [climb_v2_expA]
                               (the three baselines were scored in both waves; each sub-panel takes
                               the floor from the wave its own arms were scored in, so no cell ever
                               mixes waves).

UNITS -- AND WHY THIS FILE DELIBERATELY DOES NOT READ moleculenet_cv_qm7native/. Lift over a floor
is exactly scale-invariant — 100*(k*floor - k*arm)/|k*floor| does not
depend on k — so the z-scored-vs-native QM7 convention split that afflicts the ABSOLUTE panels
(fig_A, fig_B) cannot reach this figure, provided arm and floor come from the same wave, which the
rule above guarantees. This is why fig_E did not have to wait for the qm7native re-eval.

Concretely: the supervised panel reads QM7 from the ORDINARY moleculenet_cv/ for every run --
skip_dense_8M{,_s1,_s2}, corrupt_mtr_8M and the three random_baseline floors alike -- even though
a native copy now exists for some of them. Switching the ones that have it would MIX conventions
inside a single lift (corrupt_mtr_8M has no native copy), which is the one thing that would break
the scale-invariance argument. One convention per (panel, dataset), chosen so that every run in
that cell can supply it. Do not "upgrade" this to qm7native unless every run in SUP_ARMS and
FLOOR_RUNS has it.

ERROR BARS. ONE estimand everywhere: SD across the 3 PRETRAINING seeds, propagated through the
lift transform with the floor held fixed (sd_lift = 100*sd_arm/|floor|). Cells with fewer than 2
pretraining runs emit n_seeds<2 / blank lift_sd_pct and are drawn without a whisker rather than
borrowing a fold SD, which would not be the same estimand.

Writes: figure_data/fig_E/fig_E_lift.csv
Run:    python3 scripts/build_fig_E_table.py
"""
from __future__ import annotations

import csv
import json
import statistics as stat
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))
import six_panel_aggregate as A                                    # noqa: E402  (loaders)

PHASE2 = ROOT / "figure_data" / "climb_v2_phase2"
RIGOR = ROOT / "analysis" / "rigor"
OUT = ROOT / "figure_data" / "fig_E" / "fig_E_lift.csv"

TASKS = ["MoleculeACE", "CBS", "BACE", "Ames", "Tox21", "QM7"]
LOWER_BETTER = {"MoleculeACE", "QM7"}
METRIC = {"MoleculeACE": "rmse", "CBS": "nef1", "BACE": "roc_auc",
          "Ames": "roc_auc", "Tox21": "roc_auc", "QM7": "rmse"}
# The three panels that come from the benchmark trees rather than the MolNet suite summaries.
BENCH = {"MoleculeACE", "CBS", "Ames"}
MOLNET = [t for t in TASKS if t not in BENCH]                      # BACE, Tox21, QM7

def seeds(base: str) -> list[str]:
    return [base, f"{base}_s1", f"{base}_s2"]

# (arm key, display label, the three pretraining-run dirs)
SUP_ARMS = [("real",             "supervised, dense",                    seeds("skip_dense_8M")),
            ("targets_permuted", "supervised, dense, corrupted targets",  seeds("corrupt_mtr_8M"))]
LADDER = [("real",     "unsupervised",                       seeds("unsup_8M")),
          ("shuffled", "shuffled tokens",                    seeds("corrupt_mlm_8M")),
          ("bigram",   "bigram-resampled corpus",            seeds("bigram_8M")),
          ("unigram",  "unigram-resampled corpus",           seeds("unigram_8M")),
          ("wiki",     "English Wikipedia (zero chemistry)", seeds("wiki_real_8M"))]

FLOOR_RUNS = ["random_baseline_00", "random_baseline_01", "random_baseline_02"]
LADDER_MOLNET_FLOOR = "no_pretrain (frozen)"                       # expA's name for the same thing


# ---------------------------------------------------------------- per-run readers -------------
def bench_run_value(run: str, task: str) -> float:
    """One number for ONE pretraining run on one benchmark panel, pooling that run's eval seeds.

    Deliberately one dir at a time (never `mace_seed_macros(base)`, which pools _s1/_s2 for you):
    the declared estimand is the SD ACROSS pretraining runs, so the runs must stay separable.
    """
    if task == "MoleculeACE":
        m = A.mace_seed_macros([run])
        return float(stat.mean(m)) if m else np.nan
    if task == "Ames":
        c = A.polaris_cells([run], "tdcommons/ames", "roc_auc")
        return float(stat.mean([v for _, v in c])) if c else np.nan
    if task == "CBS":
        folds = A.mol_fold_values([run], "cbs", "nef1", root="cbs_benchmark")
        if folds:
            return float(stat.mean([v for _, v in folds]))
        summ = A.mol_dir_summaries([run], "cbs", "nef1", root="cbs_benchmark")
        return float(summ[0][1]) if summ else np.nan
    raise ValueError(task)


def molnet_run_value(run: str, task: str) -> float:
    """<task>_MEAN from climb_v2_phase2/<run>/moleculenet_cv/suite_summary.json (the phase-2 wave)."""
    p = PHASE2 / run / "moleculenet_cv" / "suite_summary.json"
    if not p.exists():
        return np.nan
    v = json.load(open(p)).get(f"{task}_MEAN")
    return float(v) if v is not None else np.nan


def agg(vals: list[float]) -> tuple[float, float, int]:
    v = [x for x in vals if np.isfinite(x)]
    if not v:
        return np.nan, np.nan, 0
    return float(np.mean(v)), (float(np.std(v, ddof=1)) if len(v) > 1 else np.nan), len(v)


def lift(arm: float, floor: float, task: str) -> float:
    if not (np.isfinite(arm) and np.isfinite(floor)) or floor == 0:
        return np.nan
    return 100 * (floor - arm) / abs(floor) if task in LOWER_BETTER else 100 * (arm - floor) / abs(floor)


def main() -> None:
    lad = pd.read_csv(RIGOR / "expA_ladder_summary.csv")
    wiki = pd.read_csv(RIGOR / "expB_wiki_summary.csv")
    # expA/expB name their arms differently from the run dirs; map ours -> theirs for MolNet only.
    LADDER_MOLNET_ARM = {"real": ("real (unsup_only)", lad), "shuffled": ("shuffle_tokens", lad),
                         "bigram": ("bigram_resample", lad), "unigram": ("unigram_resample", lad),
                         "wiki": ("wiki_real", wiki)}

    def rigor_cell(df, arm, task):
        r = df[(df.arm == arm) & (df.dataset == task)]
        if not len(r):
            return np.nan, np.nan, 0
        return float(r["mean"].iloc[0]), float(r["std"].iloc[0]), int(r["n_seeds"].iloc[0])

    # ---------------- floors ----------------
    # benchmark panels: one shared random-init frozen control, same wave as the corrupted arms
    floor_bench = {t: agg([bench_run_value(r, t) for r in FLOOR_RUNS])[0] for t in BENCH}
    floor_bench_n = {t: agg([bench_run_value(r, t) for r in FLOOR_RUNS])[2] for t in BENCH}
    # MolNet panels: per-wave floors
    floor_sup = {t: agg([molnet_run_value(r, t) for r in FLOOR_RUNS])[0] for t in MOLNET}
    floor_lad = {t: rigor_cell(lad, LADDER_MOLNET_FLOOR, t)[0] for t in MOLNET}

    rows = []

    def emit(panel, key, label, mean, sd, n, task, f, fsrc):
        if not np.isfinite(mean):
            return
        rows.append(dict(
            panel=panel, arm=key, label=label, dataset=task, metric=METRIC[task],
            direction="lower" if task in LOWER_BETTER else "higher",
            mean=round(mean, 6), sd=("" if not np.isfinite(sd) else round(sd, 6)), n_seeds=n,
            floor=("" if not np.isfinite(f) else round(f, 6)), floor_source=fsrc,
            lift_pct=round(lift(mean, f, task), 4),
            lift_sd_pct=("" if not (np.isfinite(sd) and np.isfinite(f)) else round(100 * sd / abs(f), 4))))

    BSRC = "random_baseline_0{0,1,2} [benchmark wave]"
    for panel, arms in (("supervised", SUP_ARMS), ("unsupervised", LADDER)):
        for key, label, runs in arms:
            # the three benchmark panels: always per-run, always the shared benchmark-wave floor
            for t in BENCH:
                mean, sd, n = agg([bench_run_value(r, t) for r in runs])
                emit(panel, key, label, mean, sd, n, t, floor_bench[t], BSRC)
            # BACE / Tox21 / QM7: supervised reads phase-2 dirs, the ladder reads the expA/expB summaries
            for t in MOLNET:
                if panel == "supervised":
                    mean, sd, n = agg([molnet_run_value(r, t) for r in runs])
                    emit(panel, key, label, mean, sd, n, t, floor_sup[t],
                         "random_baseline_0{0,1,2} [climb_v2_phase2]")
                else:
                    arm_name, df = LADDER_MOLNET_ARM[key]
                    mean, sd, n = rigor_cell(df, arm_name, t)
                    emit(panel, key, label, mean, sd, n, t, floor_lad[t],
                         f"{LADDER_MOLNET_FLOOR} [climb_v2_expA]")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    cols = ["panel", "arm", "label", "dataset", "metric", "direction", "mean", "sd", "n_seeds",
            "floor", "floor_source", "lift_pct", "lift_sd_pct"]
    with open(OUT, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {OUT.relative_to(ROOT)}  {len(rows)} rows")

    d = pd.DataFrame(rows)
    for panel in ("supervised", "unsupervised"):
        p = d[d.panel == panel]
        print(f"\n{panel} — lift % over no pretrain (frozen):")
        print(p.pivot(index="label", columns="dataset", values="lift_pct")
               .reindex(columns=TASKS).round(1).to_string())
        print("  n pretraining seeds per cell:")
        print(p.pivot(index="label", columns="dataset", values="n_seeds")
               .reindex(columns=TASKS).to_string())
    print("\nfloors:")
    for t in BENCH:
        print(f"  {t:12s} {floor_bench[t]:.4f}  (n={floor_bench_n[t]} control runs, benchmark wave)")
    for t in MOLNET:
        print(f"  {t:12s} sup={floor_sup[t]:.4f} [phase2]   ladder={floor_lad[t]:.4f} [expA]")
    miss = [f"{r['label']}/{r['dataset']}" for r in rows if r["n_seeds"] < 2]
    if miss:
        print("\nCELLS WITH <2 PRETRAINING RUNS (drawn without a whisker):", ", ".join(miss))


if __name__ == "__main__":
    main()
