"""Fig E — build the corruption-ladder table (one tidy CSV, the figure's ONLY input).

Two independent corruption experiments, both scored with the SAME frozen probe on the SAME
5-fold scaffold-CV protocol, both expressed as lift over the SAME kind of floor:

  panel "supervised"    real  = skip_dense_8M{,_s1,_s2}      (supervised, dense: descriptor regression)
                        garble= corrupt_mtr_8M               (descriptor targets permuted across batch)
                        floor = random_baseline_0{0,1,2}     [wave: climb_v2_phase2]

  panel "unsupervised"  real  = unsup_8M{,_s1,_s2}           (MLM on real SMILES)
                        garble= corrupt_mlm_8M     -> shuffled  (token order permuted in-sequence)
                                bigram_8M          -> bigram    (resampled from corpus bigram stats)
                                unigram_8M         -> unigram   (resampled from corpus unigram stats)
                                wiki_real_8M       -> wiki      (English Wikipedia; zero chemistry)
                        floor = random_baseline_0{0,1,2}     [wave: climb_v2_expA/_baselines + expB]

FLOOR CHOICE. Both panels lift over `no_pretrain (frozen)` — a random-init encoder, frozen, with
the same probe trained on it. That is the control that isolates what the PRETRAINING OBJECTIVE
contributes while holding architecture and probe protocol fixed, and it is the floor the earlier
synthetic-statistics ladder used. (Figs C2/D lift over the end2end floor instead, because they ask
a different question — "is the frozen pipeline worth it at all". Do not mix the two.)

ONE FLOOR PER PANEL, FROM THE PANEL'S OWN EVAL WAVE. The three random_baseline encoders were
scored twice: once in the phase-2 wave and once in the expA native re-eval. The values agree to
<=0.6pp of lift, but each panel uses the floor from the wave its own arms were scored in, so no
panel ever mixes waves. Verified 2026-08-17: all sources are in NATIVE regression units
(QM7 ~200 kcal/mol, not the normalized ~0.87) — the stale warning in build_expA_ladder_summary.py
about phase-2 being normalized does not apply to any run read here.

ERROR BARS. ONE estimand everywhere: SD across the 3 PRETRAINING seeds, propagated through the
lift transform with the floor held fixed (sd_lift = 100*sd_arm/|floor|). corrupt_mtr_8M exists as a
SINGLE pretraining run, so it has no seed SD and is emitted with n_seeds=1 / lift_sd empty — it is
drawn without a whisker rather than borrowing a fold SD, which would not be the same estimand.

Writes: figure_data/figE/figE_lift.csv
Run:    python3 scripts/build_figE_table.py
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PHASE2 = ROOT / "figure_data" / "climb_v2_phase2"
RIGOR = ROOT / "analysis" / "rigor"
OUT = ROOT / "figure_data" / "figE" / "figE_lift.csv"

TASKS = ["ESOL", "BBBP", "BACE", "Tox21", "QM7", "HIV"]
LOWER_BETTER = {"ESOL", "QM7"}                       # RMSE; the rest are ROC-AUC
METRIC = {t: ("rmse" if t in LOWER_BETTER else "roc_auc") for t in TASKS}

# ---- panel 1: supervised, read from the phase-2 suite summaries -----------------------------
SUP_ARMS = [("real",             "supervised, dense",
             ["skip_dense_8M", "skip_dense_8M_s1", "skip_dense_8M_s2"]),
            ("targets_permuted", "supervised, dense, corrupted targets",
             ["corrupt_mtr_8M"])]
SUP_FLOOR = ["random_baseline_00", "random_baseline_01", "random_baseline_02"]

# ---- panel 2: unsupervised ladder, read from the expA/expB rigor summaries -------------------
# (arm key, display label, the `arm` string in the source CSV, which source CSV)
LADDER = [("real",     "unsupervised",                       "real (unsup_only)", "expA"),
          ("shuffled", "shuffled tokens",                    "shuffle_tokens",    "expA"),
          ("bigram",   "bigram-resampled corpus",            "bigram_resample",   "expA"),
          ("unigram",  "unigram-resampled corpus",           "unigram_resample",  "expA"),
          ("wiki",     "English Wikipedia (zero chemistry)", "wiki_real",         "expB")]
LADDER_FLOOR = "no_pretrain (frozen)"


def suite_mean(run: str, task: str) -> float:
    """The run's CV score for one task = <task>_MEAN of moleculenet_cv/suite_summary.json."""
    p = PHASE2 / run / "moleculenet_cv" / "suite_summary.json"
    if not p.exists():
        return np.nan
    v = json.load(open(p)).get(f"{task}_MEAN")
    return float(v) if v is not None else np.nan


def lift(arm: float, floor: float, task: str) -> float:
    if not (np.isfinite(arm) and np.isfinite(floor)) or floor == 0:
        return np.nan
    return 100 * (floor - arm) / abs(floor) if task in LOWER_BETTER else 100 * (arm - floor) / abs(floor)


def main() -> None:
    lad = pd.read_csv(RIGOR / "expA_ladder_summary.csv")
    wiki = pd.read_csv(RIGOR / "expB_wiki_summary.csv")
    src = {"expA": lad, "expB": wiki}

    def cell(df: pd.DataFrame, arm: str, task: str) -> tuple[float, float, int]:
        r = df[(df.arm == arm) & (df.dataset == task)]
        if not len(r):
            return np.nan, np.nan, 0
        return float(r["mean"].iloc[0]), float(r["std"].iloc[0]), int(r["n_seeds"].iloc[0])

    rows = []

    # ---------------- supervised panel (phase-2 wave) ----------------
    floor_vals = {t: [suite_mean(r, t) for r in SUP_FLOOR] for t in TASKS}
    for t in TASKS:
        vs = [v for v in floor_vals[t] if np.isfinite(v)]
        floor_vals[t] = float(np.mean(vs)) if vs else np.nan
    for key, label, runs in SUP_ARMS:
        for t in TASKS:
            per_seed = [v for v in (suite_mean(r, t) for r in runs) if np.isfinite(v)]
            if not per_seed:
                continue
            mean = float(np.mean(per_seed))
            sd = float(np.std(per_seed, ddof=1)) if len(per_seed) >= 2 else np.nan
            f = floor_vals[t]
            rows.append(dict(panel="supervised", arm=key, label=label, dataset=t,
                             metric=METRIC[t], direction="lower" if t in LOWER_BETTER else "higher",
                             mean=round(mean, 6), sd=("" if not np.isfinite(sd) else round(sd, 6)),
                             n_seeds=len(per_seed), floor=round(f, 6),
                             floor_source="random_baseline_0{0,1,2} [climb_v2_phase2]",
                             lift_pct=round(lift(mean, f, t), 4),
                             lift_sd_pct=("" if not np.isfinite(sd) else round(100 * sd / abs(f), 4))))

    # ---------------- unsupervised ladder (expA/expB wave) ----------------
    for t in TASKS:
        fm, _, _ = cell(lad, LADDER_FLOOR, t)
        for key, label, arm, which in LADDER:
            mean, sd, n = cell(src[which], arm, t)
            if not np.isfinite(mean):
                continue
            rows.append(dict(panel="unsupervised", arm=key, label=label, dataset=t,
                             metric=METRIC[t], direction="lower" if t in LOWER_BETTER else "higher",
                             mean=round(mean, 6), sd=("" if not np.isfinite(sd) else round(sd, 6)),
                             n_seeds=n, floor=round(fm, 6),
                             floor_source=f"{LADDER_FLOOR} [climb_v2_expA/_baselines]",
                             lift_pct=round(lift(mean, fm, t), 4),
                             lift_sd_pct=("" if not np.isfinite(sd) else round(100 * sd / abs(fm), 4))))

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
        n = p.groupby("label").n_seeds.max()
        if (n < 3).any():
            print("  single-seed arms (no whisker):", ", ".join(n[n < 3].index))


if __name__ == "__main__":
    main()
