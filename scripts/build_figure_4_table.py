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
import six_panel_aggregate as A

PHASE2 = ROOT / "figure_data" / "climb_v2_phase2"
RIGOR = ROOT / "analysis" / "rigor"
FD = ROOT / "figure_data"
OUT = ROOT / "figure_data" / "fig_E" / "fig_E_lift.csv"

TASKS = ["MoleculeACE", "HIV", "BACE", "Ames", "Tox21", "QM7"]
LOWER_BETTER = {"MoleculeACE", "QM7"}
METRIC = {"MoleculeACE": "rmse", "HIV": "nef1", "BACE": "roc_auc",
          "Ames": "roc_auc", "Tox21": "roc_auc", "QM7": "rmse"}


BENCH = {"MoleculeACE", "Ames"}
MOLNET = [t for t in TASKS if t not in BENCH]

def seeds(base: str) -> list[str]:
    return [base, f"{base}_s1", f"{base}_s2"]


SUP_ARMS = [("real",             "supervised, dense",                    seeds("skip_dense_8M")),
            ("targets_permuted", "supervised, desc, corrupted targets",  seeds("corrupt_mtr_8M"))]
LADDER = [("real",     "unsupervised",                       seeds("unsup_8M")),
          ("shuffled", "shuffled tokens",                    seeds("corrupt_mlm_8M")),
          ("bigram",   "bigram-resampled corpus",            seeds("bigram_8M")),
          ("unigram",  "unigram-resampled corpus",           seeds("unigram_8M")),
          ("wiki",     "English Wikipedia (zero chemistry)", seeds("wiki_real_8M"))]

FLOOR_RUNS = ["random_baseline_00", "random_baseline_01", "random_baseline_02"]
LADDER_MOLNET_FLOOR = "no_pretrain (frozen)"


def bench_run_value(run: str, task: str) -> float:

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

    import json as _json


    subs = (("moleculenet_cv_tox21fixed", "moleculenet_cv") if task == "Tox21" else
            ("moleculenet_cv_qm7native", "moleculenet_cv") if task == "QM7" else
            ("moleculenet_cv",))
    for sub in subs:
        d = PHASE2 / run / sub
        p_ = d / "suite_summary.json"
        if p_.exists():


            m = METRIC[task]
            v = _json.load(open(p_)).get(
                f"{task}_MEAN" if m in ("roc_auc", "rmse") else f"{task}_{m}_MEAN")
            if v is not None:
                return float(v)


        c = d / "moleculenet_summary.csv"
        if c.exists():
            import csv as _csv
            vals = [float(r["main_value"]) for r in _csv.DictReader(open(c))
                    if r["dataset"] == task and r["main_metric"] == METRIC[task]
                    and r["head_seed"].startswith("fold") and r["main_value"] not in ("", "nan")]
            if vals:
                return float(np.mean(vals))
    return np.nan


QM7_ZSCORED_MAX = 10.0


def _qm7_is_native(v: float) -> bool:
    return np.isfinite(v) and v > QM7_ZSCORED_MAX


def qm7_consistent(vals: list[float], floor_val: float, label: str) -> list[float]:
    """Keep the arm values whose UNIT matches the floor's; report anything dropped."""
    want_native = _qm7_is_native(floor_val)
    keep = [v for v in vals if np.isfinite(v) and _qm7_is_native(v) == want_native]
    drop = [v for v in vals if np.isfinite(v) and _qm7_is_native(v) != want_native]
    if drop:
        conv = "native" if want_native else "z-scored"
        print(f"  QM7 UNIT SKIP  {label}: floor is {conv}; dropped {len(drop)} seed(s) in the other "
              f"convention ({', '.join(f'{v:.4g}' for v in drop)}). Same directory, different "
              f"units -- a name-based subdir check cannot see this.")
    return keep


_CV_ROOTS = ["climb_v2_expB", "climb_v2_expA", "climb_v2_phase2"]


def _cv_tree_cell(arm: str, task: str, metric: str):

    import csv as _csv
    import statistics as _st
    base = {"wiki_real": "wiki_real_8M"}.get(arm, arm)
    per_seed = []
    for d in seeds(base):
        for root in _CV_ROOTS:
            f = FD / root / d / "moleculenet_cv" / "moleculenet_summary.csv"
            if not f.exists():
                continue
            v = [float(r["main_value"]) for r in _csv.DictReader(f.open())
                 if r["dataset"] == task and r["main_metric"] == metric
                 and r["head_seed"] not in ("MEAN", "STD")
                 and r["main_value"] not in ("", "nan")]
            if v:
                per_seed.append(_st.mean(v))
            break
    if not per_seed:
        return None
    return (_st.mean(per_seed),
            _st.stdev(per_seed) if len(per_seed) > 1 else 0.0,
            len(per_seed))


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

    LADDER_MOLNET_ARM = {"real": ("real (unsup_only)", lad), "shuffled": ("shuffle_tokens", lad),
                         "bigram": ("bigram_resample", lad), "unigram": ("unigram_resample", lad),
                         "wiki": ("wiki_real", wiki)}

    def rigor_cell(df, arm, task):

        m = METRIC[task]
        r = df[(df.arm == arm) & (df.dataset == task) & (df.metric == m)]
        if not len(r):


            fb = _cv_tree_cell(arm, task, m)
            if fb is not None:
                print(f"  [rigor_cell] {arm} / {task}: {m!r} absent from the summary, "
                      f"read from moleculenet_cv/ instead -> {fb[0]:.4f} (n={fb[2]})")
                return fb
            print(f"  [rigor_cell] {arm} / {task}: no {m!r} row "
                  f"(has {sorted(df[(df.arm == arm) & (df.dataset == task)].metric.unique())}) "
                  f"-- cell dropped")
            return np.nan, np.nan, 0
        return float(r["mean"].iloc[0]), float(r["std"].iloc[0]), int(r["n_seeds"].iloc[0])


    floor_bench = {t: agg([bench_run_value(r, t) for r in FLOOR_RUNS])[0] for t in BENCH}
    floor_bench_n = {t: agg([bench_run_value(r, t) for r in FLOOR_RUNS])[2] for t in BENCH}

    def _floor_sup(t):
        vals = [molnet_run_value(r, t) for r in FLOOR_RUNS]
        if t == "QM7":
            ref = next((v for v in vals if np.isfinite(v)), float("nan"))
            vals = qm7_consistent(vals, ref, "floor/no_pretrain (frozen)")
        return agg(vals)[0]

    floor_sup = {t: _floor_sup(t) for t in MOLNET}
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

            for t in BENCH:
                mean, sd, n = agg([bench_run_value(r, t) for r in runs])
                emit(panel, key, label, mean, sd, n, t, floor_bench[t], BSRC)

            for t in MOLNET:
                if panel == "supervised":
                    vals = [molnet_run_value(r, t) for r in runs]
                    if t == "QM7":
                        vals = qm7_consistent(vals, floor_sup[t], f"{panel}/{label}")
                    mean, sd, n = agg(vals)
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
        print(f"\n{panel} — lift % over no pretrain, random:")
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
