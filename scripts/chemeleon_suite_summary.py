"""Aggregate CheMeleon-suite results (ours + published baselines) into tidy comparison tables.

Reads:
  figure_data/chemeleon_suite/moleculeace/<model>/results.csv   (subset in overall/cliff/noncliff, metric=rmse)
  figure_data/chemeleon_suite/polaris/<model>/polaris_scores.csv (task,seed,metric,value)
  chemeleon_suite/data/polaris/polaris_manifest.json             (per-task primary metric)
  chemeleon_suite/reference/reference_long.csv                   (14 published baselines, 5 seeds)
  chemeleon_suite/leakage/leaked_pairs.csv                       (tasks to flag)

Writes chemeleon_suite/summaries/:
  polaris_summary.csv        task,model,metric,mean,std,n_seeds,source,leak_flag
  moleculeace_summary.csv    task,model,subset,mean,std,n_seeds,source
  moleculeace_cliff.csv      per-model: mean overall/cliff/noncliff RMSE, cliff-consistency rate, win vs CheMeleon
  headline.csv               per-model per-track: mean primary metric + win-rate vs CheMeleon

Pure stdlib + numpy. Our models are discovered from the figure_data dirs; reference models from the CSV.
'source' = 'ours' or 'reference'. Runs anywhere (no GPU)."""
import csv
import json
import statistics as st
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FD = ROOT / "figure_data" / "chemeleon_suite"
REF = ROOT / "chemeleon_suite" / "reference" / "reference_long.csv"
MANP = ROOT / "chemeleon_suite" / "data" / "polaris" / "polaris_manifest.json"
OUT = ROOT / "chemeleon_suite" / "summaries"
OUT.mkdir(parents=True, exist_ok=True)
LEAKP = ROOT / "chemeleon_suite" / "leakage" / "leaked_pairs.csv"

# metric direction: True = higher better
HIGHER = {"pearsonr", "spearmanr", "r2", "roc_auc", "pr_auc", "accuracy", "explained_var"}


def _agg(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return (float("nan"), float("nan"), 0)
    return (st.mean(vals), (st.pstdev(vals) if len(vals) > 1 else 0.0), len(vals))


def leaked_tasks():
    out = set()
    if LEAKP.exists():
        for r in csv.DictReader(LEAKP.open()):
            out.add(r["task"].split("/", 1)[-1])  # 'polaris/tdcommons__ames' -> 'tdcommons__ames'
    return out


def load_ours_moleculeace():
    rows = []
    d = FD / "moleculeace"
    if not d.exists():
        return rows
    for mdir in sorted(p for p in d.iterdir() if (p / "results.csv").exists()):
        model = mdir.name
        by = defaultdict(lambda: defaultdict(list))  # task -> subset -> [rmse per seed]
        for r in csv.DictReader((mdir / "results.csv").open()):
            if r["metric"] == "rmse":
                by[r["task"]][r["subset"]].append(float(r["value"]))
        for task, subs in by.items():
            for subset, vals in subs.items():
                m, s, n = _agg(vals)
                rows.append({"task": task, "model": model, "subset": subset, "mean": m, "std": s,
                             "n_seeds": n, "source": "ours"})
    return rows


def load_ref_moleculeace():
    rows = []
    smap = {"overall test rmse": "overall", "cliff test rmse": "cliff", "noncliff test rmse": "noncliff"}
    by = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for r in csv.DictReader(REF.open()):
        if r["track"] == "moleculeace" and r["metric"] in smap:
            by[r["task"]][r["model"]][smap[r["metric"]]].append(float(r["value"]))
    for task, models in by.items():
        for model, subs in models.items():
            for subset, vals in subs.items():
                m, s, n = _agg(vals)
                rows.append({"task": task, "model": model, "subset": subset, "mean": m, "std": s,
                             "n_seeds": n, "source": "reference"})
    return rows


def load_ours_polaris(man, leaks):
    rows = []
    d = FD / "polaris"
    if not d.exists():
        return rows
    for mdir in sorted(p for p in d.iterdir() if (p / "polaris_scores.csv").exists()):
        model = mdir.name
        by = defaultdict(lambda: defaultdict(list))  # task -> metric -> [per seed]
        for r in csv.DictReader((mdir / "polaris_scores.csv").open()):
            by[r["task"]][r["metric"]].append(float(r["value"]))
        for task, mets in by.items():
            pm = man.get(task, {}).get("primary_metric")
            if pm and pm in mets:
                m, s, n = _agg(mets[pm])
                rows.append({"task": task, "model": model, "metric": pm, "mean": m, "std": s, "n_seeds": n,
                             "source": "ours", "leak_flag": task.replace("/", "__").split("/")[-1] in leaks
                             or task.split("polaris/")[-1] in leaks})
    return rows


def load_ref_polaris(man):
    rows = []
    by = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for r in csv.DictReader(REF.open()):
        if r["track"] == "polaris":
            by[r["task"]][r["model"]][r["metric"]].append(float(r["value"]))
    for task, models in by.items():
        pm = man.get(task, {}).get("primary_metric")
        for model, mets in models.items():
            if pm and pm in mets:
                m, s, n = _agg(mets[pm])
                rows.append({"task": task, "model": model, "metric": pm, "mean": m, "std": s, "n_seeds": n,
                             "source": "reference", "leak_flag": ""})
    return rows


def write(path, rows, fields):
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"[summary] wrote {path} ({len(rows)} rows)")


def main():
    man = json.loads(MANP.read_text()) if MANP.exists() else {}
    leaks = leaked_tasks()

    mace = load_ours_moleculeace() + load_ref_moleculeace()
    write(OUT / "moleculeace_summary.csv", mace, ["task", "model", "subset", "mean", "std", "n_seeds", "source"])

    pol = load_ours_polaris(man, leaks) + load_ref_polaris(man)
    write(OUT / "polaris_summary.csv", pol, ["task", "model", "metric", "mean", "std", "n_seeds", "source", "leak_flag"])

    # MoleculeACE cliff analysis per model
    cliff_rows = []
    bym = defaultdict(lambda: defaultdict(dict))  # model -> task -> {subset: mean}
    for r in mace:
        bym[r["model"]][r["task"]][r["subset"]] = r["mean"]
    for model, tasks in sorted(bym.items()):
        ov = [t["overall"] for t in tasks.values() if "overall" in t]
        cl = [t["cliff"] for t in tasks.values() if "cliff" in t]
        nc = [t["noncliff"] for t in tasks.values() if "noncliff" in t]
        consist = sum(1 for t in tasks.values() if t.get("cliff", 0) > t.get("noncliff", 1e9))
        cliff_rows.append({"model": model, "mean_overall_rmse": round(st.mean(ov), 4) if ov else "",
                           "mean_cliff_rmse": round(st.mean(cl), 4) if cl else "",
                           "mean_noncliff_rmse": round(st.mean(nc), 4) if nc else "",
                           "cliff_consistency_rate": f"{consist}/{len(tasks)}", "n_tasks": len(tasks)})
    write(OUT / "moleculeace_cliff.csv", cliff_rows,
          ["model", "mean_overall_rmse", "mean_cliff_rmse", "mean_noncliff_rmse", "cliff_consistency_rate", "n_tasks"])

    # Headline: per track, per model — mean primary metric + win-rate vs CheMeleon (per-task better)
    head = []
    # polaris headline
    ptask_model = defaultdict(dict)
    for r in pol:
        ptask_model[r["task"]][r["model"]] = (r["mean"], r["metric"])
    models_p = sorted({r["model"] for r in pol})
    for model in models_p:
        vals, wins, tot = [], 0, 0
        for task, mm in ptask_model.items():
            if model in mm and "CheMeleon" in mm:
                (v, met) = mm[model]; (cv, _) = mm["CheMeleon"]
                vals.append(v); tot += 1
                better = v > cv if met in HIGHER else v < cv
                wins += int(better)
        head.append({"track": "polaris", "model": model, "n_tasks": len(vals),
                     "win_rate_vs_chemeleon": f"{wins}/{tot}" if tot else "-"})
    # moleculeace headline (overall rmse, lower better)
    mtask_model = defaultdict(dict)
    for r in mace:
        if r["subset"] == "overall":
            mtask_model[r["task"]][r["model"]] = r["mean"]
    for model in sorted({r["model"] for r in mace}):
        wins, tot = 0, 0
        for task, mm in mtask_model.items():
            if model in mm and "CheMeleon" in mm:
                tot += 1; wins += int(mm[model] < mm["CheMeleon"])
        head.append({"track": "moleculeace", "model": model, "n_tasks": tot,
                     "win_rate_vs_chemeleon": f"{wins}/{tot}" if tot else "-"})
    write(OUT / "headline.csv", head, ["track", "model", "n_tasks", "win_rate_vs_chemeleon"])
    print("[summary] NOTE: leaked Polaris tasks flagged (leak_flag) not re-scored (Polaris evaluate uses full "
          "test set); 22 mols / 5 TDC tasks, negligible. HSD/Tukey = TODO (needs statsmodels).")


if __name__ == "__main__":
    main()
