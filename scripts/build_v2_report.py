"""Build the v2 paper report: aggregate, score, and figure-render.

Reads every <run_dir>/moleculenet/moleculenet_summary.csv plus the manifest, then:
  1. Builds raw_results.csv (one row per (run, dataset, head_seed)).
  2. Computes Score_v2 anchored on the random_baseline mean as zero, with sd as 1.
  3. Renders Figures 1-4, 6 from the README/plan analysis.
  4. Writes a markdown report.

Score_v2 (per (run, dataset)):
    if classification: z = (auc - random_mean) / random_sd
    if regression:     z = -(rmse - random_mean) / random_sd  (negate so higher = better)
Score_v2_run = mean over the 9 datasets.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter

LOWER_IS_BETTER = {"rmse"}


def _load_run_summaries(manifest: dict, project_root: Path) -> pd.DataFrame:
    rows = []
    for run in manifest["runs"]:
        run_id = run["run_id"]
        run_type = run["run_type"]
        sel = run.get("selection", {}) or run["pretrain_config"].get("selection", {})
        eval_csv = project_root / run["evaluation_output_dir"] / "moleculenet_summary.csv"
        if not eval_csv.exists():
            continue
        with eval_csv.open() as f:
            reader = csv.DictReader(f)
            for r in reader:
                rows.append({
                    "run_id": run_id,
                    "run_type": run_type,
                    "dataset": r["dataset"],
                    "task_type": r["task_type"],
                    "main_metric": r["main_metric"],
                    "head_seed": r["head_seed"],
                    "main_value": float(r["main_value"]) if r["main_value"] not in ("", "nan") else np.nan,
                    "mixing_ratio": sel.get("mixing_ratio"),
                    "n_families": sel.get("n_families"),
                    "pretraining_seed": sel.get("pretraining_seed"),
                    "total_forward_passes": sel.get("total_forward_passes"),
                })
    return pd.DataFrame(rows)


def _compute_score_v2(df: pd.DataFrame) -> pd.DataFrame:
    """Anchor on random_baseline. Compute Score_v2 per run."""
    # Use only per-seed rows, not the MEAN/STD aggregate rows from the eval CSV.
    valid = df[~df["head_seed"].isin(["MEAN", "STD"])].copy()

    # Random baseline stats per (dataset, main_metric)
    baseline = valid[valid["run_type"] == "random_baseline"]
    if baseline.empty:
        # No baseline yet — fall back to within-pool z-scoring.
        print("[build_v2_report] WARNING: no random_baseline rows; using within-pool z-scoring")
        stats = valid.groupby("dataset")["main_value"].agg(mu="mean", sigma="std").reset_index()
    else:
        stats = baseline.groupby("dataset")["main_value"].agg(mu="mean", sigma="std").reset_index()

    valid = valid.merge(stats, on="dataset", how="left")
    valid["sigma"] = valid["sigma"].replace(0, np.nan)
    valid["z"] = (valid["main_value"] - valid["mu"]) / valid["sigma"]
    valid.loc[valid["main_metric"].isin(LOWER_IS_BETTER), "z"] *= -1.0

    run_scores = (
        valid.groupby(["run_id", "run_type"], as_index=False)
        .agg(score_v2_mean=("z", "mean"),
             score_v2_sd=("z", "std"),
             n_datasets=("dataset", "nunique"))
    )

    sel_cols = valid.groupby("run_id", as_index=False).first()[
        ["run_id", "mixing_ratio", "n_families", "pretraining_seed", "total_forward_passes"]
    ]
    run_scores = run_scores.merge(sel_cols, on="run_id", how="left")
    return run_scores, valid


def _fig_unsup_ramp(df: pd.DataFrame, out: Path):
    sub = df[df["run_type"] == "unsup_only"].copy()
    if sub.empty:
        return
    summary = sub.groupby("total_forward_passes", as_index=False).agg(
        m=("score_v2_mean", "mean"),
        s=("score_v2_mean", "std"),
        n=("score_v2_mean", "size"),
    ).sort_values("total_forward_passes")
    summary["se"] = summary["s"].fillna(0) / np.sqrt(summary["n"])

    fig, ax = plt.subplots(figsize=(7, 4.6))
    ax.scatter(sub["total_forward_passes"], sub["score_v2_mean"], s=42, color="#5B84B1", alpha=0.4)
    ax.errorbar(summary["total_forward_passes"], summary["m"], yerr=1.96 * summary["se"],
                marker="o", color="#1D3557", capsize=4)
    ax.axhline(0, color="black", lw=1, alpha=0.5, label="random baseline")
    ax.set_xscale("log")
    ax.set_xlabel("Unsupervised forward passes")
    ax.set_ylabel("Score_v2 (z vs random baseline)")
    ax.set_title("Figure 1 — Unsupervised scaling")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "fig1_unsup_ramp.png", bbox_inches="tight")
    plt.close(fig)


def _fig_sup_ramp(df: pd.DataFrame, out: Path):
    sub = df[df["run_type"] == "sup_only"].copy()
    if sub.empty:
        return
    sub["n_fam_int"] = sub["n_families"].astype(int)
    summary = sub.groupby("n_fam_int", as_index=False).agg(
        m=("score_v2_mean", "mean"),
        s=("score_v2_mean", "std"),
        n=("score_v2_mean", "size"),
    ).sort_values("n_fam_int")
    summary["se"] = summary["s"].fillna(0) / np.sqrt(summary["n"])

    fig, ax = plt.subplots(figsize=(7, 4.6))
    ax.scatter(sub["n_fam_int"], sub["score_v2_mean"], s=42, color="#B23A48", alpha=0.5)
    ax.errorbar(summary["n_fam_int"], summary["m"], yerr=1.96 * summary["se"],
                marker="o", color="#7F1D1D", capsize=4)
    ax.axhline(0, color="black", lw=1, alpha=0.5, label="random baseline")
    ax.set_xticks(summary["n_fam_int"])
    ax.set_xlabel("# supervised families")
    ax.set_ylabel("Score_v2")
    ax.set_title("Figure 2 — Supervised scaling")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "fig2_sup_ramp.png", bbox_inches="tight")
    plt.close(fig)


def _fig_landscape(df: pd.DataFrame, out: Path):
    pts = []
    for _, r in df.iterrows():
        if r["run_type"] == "unsup_only":
            x, y = float(r["total_forward_passes"]), 0.0
        elif r["run_type"] == "sup_only":
            x, y = 0.0, float(r["total_forward_passes"]) * float(r["n_families"]) / 5.0
        elif r["run_type"] == "mixed":
            ratio = float(r["mixing_ratio"]) if r["mixing_ratio"] is not None else 0.5
            b = float(r["total_forward_passes"])
            x, y = b * ratio, b * (1 - ratio)
        else:
            continue
        pts.append((x, y, r["score_v2_mean"], r["run_type"]))
    if not pts:
        return
    pts = pd.DataFrame(pts, columns=["x", "y", "z", "type"])
    fig, ax = plt.subplots(figsize=(7.5, 5.4))
    pts["lx"] = np.log10(pts["x"] + 1)
    pts["ly"] = np.log10(pts["y"] + 1)
    sc = ax.scatter(pts["lx"], pts["ly"], c=pts["z"], cmap="viridis", s=120, edgecolor="white", linewidth=0.5)
    fig.colorbar(sc, ax=ax, label="Score_v2")
    ax.set_xlabel("log10(unsup FPs + 1)")
    ax.set_ylabel("log10(sup FPs + 1)")
    ax.set_title("Figure 3 — Pretraining landscape")
    fig.tight_layout()
    fig.savefig(out / "fig3_landscape.png", bbox_inches="tight")
    plt.close(fig)


def _fig_family_order(df: pd.DataFrame, out: Path):
    sub = df[df["run_type"] == "family_order"].copy()
    if sub.empty:
        return
    fig, ax = plt.subplots(figsize=(7, 4.6))
    grouped = sub.groupby("run_id")["score_v2_mean"].first().reset_index()
    grouped["ord"] = grouped["run_id"].str.extract(r"v(\d)_")[0].astype(int)
    by_ord = [grouped[grouped["ord"] == i]["score_v2_mean"].values for i in sorted(grouped["ord"].unique())]
    ax.boxplot(by_ord, tick_labels=[f"order_{i}" for i in sorted(grouped["ord"].unique())], showmeans=True)
    ax.axhline(0, color="black", lw=1, alpha=0.5)
    ax.set_ylabel("Score_v2")
    ax.set_title("Figure 6 — Family-order ablation (3-of-5)")
    fig.tight_layout()
    fig.savefig(out / "fig6_family_order.png", bbox_inches="tight")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--project_root", default=".")
    p.add_argument("--output", default="experiments/robust_matrix_v2/report")
    args = p.parse_args()

    project_root = Path(args.project_root).resolve()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    manifest = json.loads(Path(args.manifest).read_text())
    raw = _load_run_summaries(manifest, project_root)
    if raw.empty:
        print("[build_v2_report] no eval data found")
        return
    raw.to_csv(out / "raw_results.csv", index=False)

    run_scores, valid = _compute_score_v2(raw)
    run_scores.to_csv(out / "run_scores.csv", index=False)

    _fig_unsup_ramp(run_scores, out)
    _fig_sup_ramp(run_scores, out)
    _fig_landscape(run_scores, out)
    _fig_family_order(run_scores, out)

    md = ["# CLIMB v2 results report\n",
          f"_Generated from {len(run_scores)} evaluated runs._\n\n"]

    md.append("## Run-type counts\n\n")
    md.append(run_scores["run_type"].value_counts().to_frame("n").to_markdown() + "\n\n")

    md.append("## Top 10 runs by Score_v2\n\n")
    top = run_scores.sort_values("score_v2_mean", ascending=False).head(10)
    md.append(top[["run_id", "run_type", "score_v2_mean", "n_datasets"]].to_markdown(index=False) + "\n\n")

    md.append("## Bottom 5 runs by Score_v2\n\n")
    bot = run_scores.sort_values("score_v2_mean", ascending=True).head(5)
    md.append(bot[["run_id", "run_type", "score_v2_mean", "n_datasets"]].to_markdown(index=False) + "\n\n")

    md.append("## Figures\n\n")
    for f in ("fig1_unsup_ramp.png", "fig2_sup_ramp.png", "fig3_landscape.png", "fig6_family_order.png"):
        if (out / f).exists():
            md.append(f"![{f}]({f})\n\n")

    (out / "report.md").write_text("".join(md))
    print(f"[build_v2_report] wrote {out / 'report.md'}")


if __name__ == "__main__":
    main()
