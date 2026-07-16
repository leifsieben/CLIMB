"""Build the v2 paper report: ABSOLUTE per-task metrics + figures (no z-scores).

Reads every <results_root>/<run_id>/moleculenet/moleculenet_summary.csv (and the
sibling metadata.json when present), then emits:

  combined_results.csv  — tidy per-(run, dataset) MEAN/STD table (absolute metrics)
  fig_core_<dataset>.png — Exp B: bar of each featurizer/run at full train size
  fig_scaling_<dataset>.png — Exp C: metric vs unique-molecule count (canonical vs
                              enumerated), with random-floor and ECFP4 anchor lines
  fig_labeleff_<dataset>.png — Exp D: metric vs #labeled examples per featurizer
                               (only if subsampled-eval rows are present)
  report.md — per-task absolute tables

Run classification is by run_id prefix; scaling coordinates (augmentation,
unique-molecule fraction) come from metadata.json / the run_id.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CORPUS_SIZE = 124_000_000  # PubChem filtered; for the scaling x-axis
CORE_ORDER = ["random_baseline", "ecfp4_anchor", "unsup_only", "sup_only", "mixed"]


def _run_type_from_id(run_id: str) -> str:
    for t in ("random_baseline", "ecfp4_anchor", "unsup_only", "sup_only", "mixed", "scaling", "smoke"):
        if run_id.startswith(t):
            return t
    return "other"


def _frac_from(run_id: str, meta: Optional[dict]) -> Optional[float]:
    if meta and meta.get("unsupervised_subset_fraction") is not None:
        return float(meta["unsupervised_subset_fraction"])
    m = re.search(r"frac([0-9p]+|full)", run_id)
    if not m:
        return None
    tag = m.group(1)
    return None if tag == "full" else float(tag.replace("p", "."))


def _aug_from(run_id: str, meta: Optional[dict]) -> Optional[str]:
    if meta and meta.get("augmentation"):
        return meta["augmentation"]
    if "enumerated" in run_id:
        return "enumerated"
    if "canonical" in run_id:
        return "canonical"
    return None


def collect(results_root: Path) -> pd.DataFrame:
    rows = []
    for summ in sorted(results_root.glob("*/moleculenet/moleculenet_summary.csv")):
        run_dir = summ.parent.parent
        run_id = run_dir.name
        meta = None
        meta_path = run_dir / "metadata.json"
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text())
            except Exception:
                meta = None
        df = pd.read_csv(summ)
        for _, r in df.iterrows():
            if r["head_seed"] not in ("MEAN", "STD"):
                continue
            rows.append({
                "run_id": run_id,
                "run_type": _run_type_from_id(run_id),
                "dataset": r["dataset"],
                "task_type": r["task_type"],
                "main_metric": r["main_metric"],
                "featurizer": r.get("featurizer", "encoder"),
                "head": r.get("head", ""),
                "n_train": int(r.get("n_train", -1)),
                "stat": r["head_seed"],
                "value": r["main_value"],
                "augmentation": _aug_from(run_id, meta),
                "subset_fraction": _frac_from(run_id, meta),
            })
    long = pd.DataFrame(rows)
    if long.empty:
        return long
    idx_cols = ["run_id", "run_type", "dataset", "task_type", "main_metric",
                "featurizer", "head", "n_train", "augmentation", "subset_fraction"]
    # Merge (not pivot_table) so we only keep observed (run, dataset) rows — a
    # multi-index pivot would fill the full cartesian product with NaNs.
    mean_df = (long[long.stat == "MEAN"].drop(columns="stat")
               .rename(columns={"value": "mean"}))
    std_df = (long[long.stat == "STD"][idx_cols + ["value"]]
              .rename(columns={"value": "std"}))
    return mean_df.merge(std_df, on=idx_cols, how="left")


def _fig_core(df: pd.DataFrame, dataset: str, out: Path):
    d = df[(df.dataset == dataset) & (df.run_type.isin(CORE_ORDER))]
    if d.empty:
        return
    agg = (d.groupby("run_type")
             .agg(mean=("mean", "mean"), std=("mean", "std"))
             .reindex([t for t in CORE_ORDER if t in set(d.run_type)]))
    metric = d["main_metric"].iloc[0]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(range(len(agg)), agg["mean"], yerr=agg["std"].fillna(0), capsize=4,
           color=["#999", "#555", "#1f77b4", "#ff7f0e", "#2ca02c"][:len(agg)])
    ax.set_xticks(range(len(agg)))
    ax.set_xticklabels(agg.index, rotation=20, ha="right")
    ax.set_ylabel(metric.upper())
    ax.set_title(f"{dataset} — core comparison ({metric}, {'lower' if metric=='rmse' else 'higher'} better)")
    fig.tight_layout()
    fig.savefig(out / f"fig_core_{dataset}.png", dpi=120)
    plt.close(fig)


def _fig_scaling(df: pd.DataFrame, dataset: str, out: Path):
    d = df[(df.dataset == dataset) & (df.run_type == "scaling")].copy()
    if d.empty:
        return
    metric = d["main_metric"].iloc[0]
    d["x"] = d["subset_fraction"].apply(lambda f: CORPUS_SIZE if f is None or pd.isna(f) else f * CORPUS_SIZE)
    fig, ax = plt.subplots(figsize=(6, 4))
    for aug, sub in d.groupby("augmentation"):
        sub = sub.sort_values("x")
        ax.plot(sub["x"], sub["mean"], marker="o", label=aug)
    for rt, style, color in (("random_baseline", "--", "#999"), ("ecfp4_anchor", ":", "#000")):
        a = df[(df.dataset == dataset) & (df.run_type == rt)]
        if not a.empty:
            ax.axhline(a["mean"].mean(), ls=style, color=color, label=rt)
    ax.set_xscale("log")
    ax.set_xlabel("unique molecules seen by MLM (fixed compute)")
    ax.set_ylabel(metric.upper())
    ax.set_title(f"{dataset} — pretraining-data scaling ({metric})")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / f"fig_scaling_{dataset}.png", dpi=120)
    plt.close(fig)


def _fig_labeleff(df: pd.DataFrame, dataset: str, out: Path):
    d = df[(df.dataset == dataset) & (df.n_train > 0)]
    if d.empty or (d.groupby("run_type")["n_train"].nunique() > 1).sum() == 0:
        return  # no subsample sweep present yet
    metric = d["main_metric"].iloc[0]
    fig, ax = plt.subplots(figsize=(6, 4))
    for rt, sub in d.groupby("run_type"):
        sub = sub.groupby("n_train")["mean"].mean().reset_index().sort_values("n_train")
        if len(sub) > 1:
            ax.plot(sub["n_train"], sub["mean"], marker="o", label=rt)
    ax.set_xscale("log")
    ax.set_xlabel("# labeled training molecules")
    ax.set_ylabel(metric.upper())
    ax.set_title(f"{dataset} — label efficiency ({metric})")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / f"fig_labeleff_{dataset}.png", dpi=120)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results_root", required=True)
    p.add_argument("--output_dir", required=True)
    args = p.parse_args()

    results_root = Path(args.results_root)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    df = collect(results_root)
    if df.empty:
        print(f"[build_v2_report] no results found under {results_root}")
        return
    df.to_csv(out / "combined_results.csv", index=False)

    datasets = sorted(df["dataset"].unique())
    lines = ["# CLIMB v2 report — absolute per-task metrics\n",
             "Anchors: random-encoder floor + ECFP4/XGBoost ceiling. No z-scores.\n"]
    for ds in datasets:
        _fig_core(df, ds, out)
        _fig_scaling(df, ds, out)
        _fig_labeleff(df, ds, out)
        d = df[df.dataset == ds]
        metric = d["main_metric"].iloc[0]
        lines.append(f"\n## {ds} ({metric}, {'lower' if metric=='rmse' else 'higher'} is better)\n")
        lines.append("| run | featurizer | n_train | mean | std |")
        lines.append("|---|---|---:|---:|---:|")
        for _, r in d.sort_values(["run_type", "run_id"]).iterrows():
            std = "" if pd.isna(r.get("std")) else f"{r['std']:.4f}"
            lines.append(f"| {r['run_id']} | {r['featurizer']} | {r['n_train']} | {r['mean']:.4f} | {std} |")
    (out / "report.md").write_text("\n".join(lines))
    print(f"[build_v2_report] wrote {out}/report.md, combined_results.csv, and figures for {len(datasets)} datasets")


if __name__ == "__main__":
    main()
