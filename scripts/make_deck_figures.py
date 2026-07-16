"""Publication-quality figures for the CLIMB v2 slide deck, from the exploratory
combined_results.csv (per-task absolute metrics). Produces:

  fig1_core.png    — per-task bars: random / sup / unsup / mixed / ECFP4
  fig2_lift.png    — % improvement over the random floor (unsup vs sup vs mixed)
  fig3_scaling.png — per-task metric vs #unique molecules (canonical vs enumerated)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

TASKS = ["ESOL", "BBBP", "BACE", "Tox21", "QM7"]          # display order
METHODS = ["random", "sup", "unsup", "mixed", "ecfp4"]
LABELS = {"random": "random", "sup": "sup", "unsup": "unsup",
          "mixed": "mixed", "ecfp4": "ECFP4"}
COLORS = {"random": "#b0b0b0", "sup": "#e8843c", "unsup": "#2f6fb0",
          "mixed": "#4ba36b", "ecfp4": "#4a4a4a"}
CORPUS = 124_000_000

plt.rcParams.update({"font.size": 12, "axes.titlesize": 13,
                     "axes.spines.top": False, "axes.spines.right": False})


def _load(root: Path) -> pd.DataFrame:
    return pd.read_csv(root / "report" / "combined_results.csv")


def _method_value(df, dataset, method):
    """(mean, std_across_seeds_or_nan) for a method on a dataset."""
    d = df[df.dataset == dataset]
    if method == "random":
        v = d[d.run_type == "random_baseline"]["mean"]
        return float(v.mean()), float(v.std())
    run_id = {"unsup": "unsup_only_seed0", "sup": "sup_only_seed0",
              "mixed": "mixed_seed0", "ecfp4": "ecfp4_anchor"}[method]
    row = d[d.run_id == run_id]
    return (float(row["mean"].iloc[0]), float(row["std"].iloc[0])) if len(row) else (np.nan, np.nan)


def fig_core(df, out):
    fig, axes = plt.subplots(1, 5, figsize=(16, 3.6))
    for ax, ds in zip(axes, TASKS):
        metric = df[df.dataset == ds]["main_metric"].iloc[0]
        vals = [_method_value(df, ds, m) for m in METHODS]
        means = [v[0] for v in vals]
        errs = [0 if np.isnan(v[1]) else v[1] for v in vals]
        ax.bar(range(len(METHODS)), means, yerr=errs, capsize=3,
               color=[COLORS[m] for m in METHODS], edgecolor="white", linewidth=0.5)
        ax.set_xticks(range(len(METHODS)))
        ax.set_xticklabels([LABELS[m] for m in METHODS], fontsize=9.5, rotation=25, ha="right")
        arrow = "↓ better" if metric == "rmse" else "↑ better"
        ax.set_title(f"{ds}\n{metric.upper()} ({arrow})")
        lo = min(means) * 0.9 if metric == "rmse" else min(means) * 0.95
        ax.set_ylim(lo, max(means) * 1.06)
        ax.margins(x=0.02)
    fig.suptitle("Downstream performance by pretraining regime (frozen encoder + head)", y=1.02, fontsize=14)
    fig.tight_layout()
    fig.savefig(out / "fig1_core.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_lift(df, out):
    """% improvement over the random floor, direction-aware (positive = better)."""
    methods = ["unsup", "sup", "mixed"]
    lifts = {m: [] for m in methods}
    for ds in TASKS:
        metric = df[df.dataset == ds]["main_metric"].iloc[0]
        rnd, _ = _method_value(df, ds, "random")
        for m in methods:
            val, _ = _method_value(df, ds, m)
            lift = (rnd - val) / rnd if metric == "rmse" else (val - rnd) / rnd
            lifts[m].append(100 * lift)
    x = np.arange(len(TASKS)); w = 0.26
    fig, ax = plt.subplots(figsize=(11, 4.4))
    for i, m in enumerate(methods):
        ax.bar(x + (i - 1) * w, lifts[m], w, label=LABELS[m].replace("\n", " "), color=COLORS[m])
    ax.axhline(0, color="#333", lw=1)
    ax.set_xticks(x); ax.set_xticklabels(TASKS)
    ax.set_ylabel("% improvement over random floor")
    ax.set_title("Does pretraining help?  Unsupervised MLM lifts 4/5 tasks; supervised co-training barely moves off the floor")
    ax.legend(frameon=False, ncol=3)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out / "fig2_lift.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_scaling(df, out):
    fig, axes = plt.subplots(1, 5, figsize=(16, 3.6))
    sc = df[df.run_type == "scaling"].copy()
    sc["x"] = sc["subset_fraction"].apply(lambda f: CORPUS if pd.isna(f) else float(f) * CORPUS)
    for ax, ds in zip(axes, TASKS):
        metric = df[df.dataset == ds]["main_metric"].iloc[0]
        d = sc[sc.dataset == ds]
        for aug, color in (("canonical", "#2f6fb0"), ("enumerated", "#c44e52")):
            s = d[d.augmentation == aug].sort_values("x")
            if len(s):
                ax.plot(s["x"], s["mean"], marker="o", ms=4, label=aug, color=color)
        rnd, _ = _method_value(df, ds, "random")
        ecfp, _ = _method_value(df, ds, "ecfp4")
        ax.axhline(rnd, ls="--", color="#b0b0b0", lw=1, label="random")
        ax.axhline(ecfp, ls=":", color="#4a4a4a", lw=1, label="ECFP4")
        ax.set_xscale("log")
        arrow = "↓" if metric == "rmse" else "↑"
        ax.set_title(f"{ds} ({metric.upper()} {arrow})")
        ax.set_xlabel("# unique molecules")
        if ds == TASKS[0]:
            ax.legend(fontsize=7.5, frameon=False)
    fig.suptitle("Pretraining-data scaling at fixed compute — performance saturates by ~10⁵ molecules; enumeration ≈ canonical", y=1.02, fontsize=14)
    fig.tight_layout()
    fig.savefig(out / "fig3_scaling.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results_root", default="experiments/climb_v2")
    p.add_argument("--output_dir", default="experiments/climb_v2/report/deck")
    args = p.parse_args()
    root = Path(args.results_root)
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    df = _load(root)
    fig_core(df, out); fig_lift(df, out); fig_scaling(df, out)
    # dump the summary table the deck quotes
    rows = []
    for ds in TASKS:
        metric = df[df.dataset == ds]["main_metric"].iloc[0]
        r = {"task": ds, "metric": metric}
        for m in METHODS:
            r[m] = round(_method_value(df, ds, m)[0], 3)
        rows.append(r)
    pd.DataFrame(rows).to_csv(out / "deck_table.csv", index=False)
    print(f"wrote figures + deck_table.csv to {out}")
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
