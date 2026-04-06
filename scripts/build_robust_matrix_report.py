#!/usr/bin/env python3
"""Build robust matrix plots and a reproducible notebook."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter


ROOT = Path("/Users/lsieben/VSCode/CLIMB")
RAW = ROOT / "experiments/robust_matrix/aggregate/raw_results.csv"
OUTDIR = ROOT / "experiments/robust_matrix/aggregate"
NOTEBOOK = ROOT / "experiments/robust_matrix/robust_matrix_analysis.ipynb"

LOWER_IS_BETTER = {"rmse"}
LEAKED_DATASETS = {"PCBA"}
FULL_UNSUP_CORPUS_TOKENS = 551_133_440


def apply_style() -> None:
    plt.style.use("default")
    plt.rcParams.update(
        {
            "figure.dpi": 170,
            "savefig.dpi": 170,
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "grid.alpha": 0.25,
            "grid.linestyle": "-",
            "axes.grid": True,
            "lines.linewidth": 2.4,
        }
    )


def token_fmt(x: float, _pos: int | None = None) -> str:
    if x >= 1e9:
        return f"{x/1e9:.0f}B"
    if x >= 1e6:
        return f"{x/1e6:.0f}M"
    return f"{x:,.0f}"


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(RAW)
    df["main_value"] = pd.to_numeric(df["main_value"], errors="coerce")
    df["token_budget_total"] = pd.to_numeric(df["token_budget_total"], errors="coerce")

    valid = df.dropna(subset=["dataset", "main_metric", "main_value"]).copy()
    stats = valid.groupby("dataset")["main_value"].agg(mu="mean", sigma="std").reset_index()
    valid = valid.merge(stats, on="dataset", how="left")
    valid["sigma"] = valid["sigma"].replace(0, np.nan)
    valid["z"] = (valid["main_value"] - valid["mu"]) / valid["sigma"]
    valid.loc[valid["main_metric"].isin(LOWER_IS_BETTER), "z"] *= -1.0
    valid["z"] = valid["z"].fillna(0.0)

    run_scores = (
        valid.groupby("run_id", as_index=False)
        .agg(
            z_all=("z", "mean"),
            n_datasets=("dataset", "nunique"),
            token_budget_total=("token_budget_total", "first"),
        )
    )

    no_leak = (
        valid[~valid["dataset"].isin(LEAKED_DATASETS)]
        .groupby("run_id", as_index=False)
        .agg(z_no_leak=("z", "mean"))
    )
    run_scores = run_scores.merge(no_leak, on="run_id", how="left")
    return valid, run_scores


def build_run_tables(valid: pd.DataFrame, run_scores: pd.DataFrame) -> dict[str, pd.DataFrame]:
    unsup = run_scores[run_scores["run_id"].str.match(r"^unsup_baseline_\d+_\d+$", na=False)].copy()
    m = unsup["run_id"].str.extract(r"^unsup_baseline_(\d+)_(\d+)$")
    unsup["unsup_tokens"] = m[0].astype(float)
    unsup["replicate"] = m[1].astype(int)

    sup = run_scores[run_scores["run_id"].str.match(r"^sup_order_\dof5_\d+$", na=False)].copy()
    m = sup["run_id"].str.extract(r"^sup_order_(\d)of5_(\d+)$")
    sup["n_families"] = m[0].astype(int)
    sup["replicate"] = m[1].astype(int)
    sup["sup_tokens"] = sup["token_budget_total"] * sup["n_families"] / 5.0

    mixed = run_scores[run_scores["run_id"].str.match(r"^mixed_\d+_\d+_\d+$", na=False)].copy()
    m = mixed["run_id"].str.extract(r"^mixed_(\d+)_(\d+)_(\d+)$")
    mixed["unsup_pct"] = m[0].astype(int)
    mixed["sup_pct"] = m[1].astype(int)
    mixed["replicate"] = m[2].astype(int)
    mixed["unsup_tokens"] = mixed["token_budget_total"] * mixed["unsup_pct"] / 100.0
    mixed["sup_tokens"] = mixed["token_budget_total"] * mixed["sup_pct"] / 100.0

    dataset_unsup = valid[valid["run_id"].isin(unsup["run_id"])].copy()
    dataset_unsup = dataset_unsup.merge(
        unsup[["run_id", "unsup_tokens", "replicate"]], on="run_id", how="left"
    )

    dataset_sup = valid[valid["run_id"].isin(sup["run_id"])].copy()
    dataset_sup = dataset_sup.merge(
        sup[["run_id", "n_families", "sup_tokens", "replicate"]], on="run_id", how="left"
    )

    landscape = pd.concat(
        [
            unsup.assign(sup_tokens=0.0)[["run_id", "z_all", "z_no_leak", "unsup_tokens", "sup_tokens"]],
            sup.assign(unsup_tokens=0.0)[["run_id", "z_all", "z_no_leak", "unsup_tokens", "sup_tokens"]],
            mixed[["run_id", "z_all", "z_no_leak", "unsup_tokens", "sup_tokens"]],
        ],
        ignore_index=True,
    )
    return {
        "unsup": unsup,
        "sup": sup,
        "mixed": mixed,
        "dataset_unsup": dataset_unsup,
        "dataset_sup": dataset_sup,
        "landscape": landscape,
    }


def plot_unsup_ramp(unsup: pd.DataFrame) -> Path:
    summary = (
        unsup.groupby("unsup_tokens", as_index=False)
        .agg(mean_z=("z_all", "mean"), sd=("z_all", "std"), n=("z_all", "size"))
        .sort_values("unsup_tokens")
    )
    summary["ci95"] = 1.96 * summary["sd"].fillna(0.0) / np.sqrt(summary["n"])

    fig, ax = plt.subplots(figsize=(7.1, 4.6))
    ax.scatter(unsup["unsup_tokens"], unsup["z_all"], s=42, color="#5B84B1", alpha=0.35, zorder=2)
    ax.plot(summary["unsup_tokens"], summary["mean_z"], marker="o", color="#1D3557", zorder=3)
    ax.fill_between(
        summary["unsup_tokens"],
        summary["mean_z"] - summary["ci95"],
        summary["mean_z"] + summary["ci95"],
        color="#5B84B1",
        alpha=0.18,
        zorder=1,
    )
    ax.axhline(0, color="black", lw=1, alpha=0.5)
    ax.axvline(FULL_UNSUP_CORPUS_TOKENS, color="black", ls="--", lw=1.2, alpha=0.8)
    ax.text(
        FULL_UNSUP_CORPUS_TOKENS,
        ax.get_ylim()[1],
        " full unsup corpus",
        rotation=90,
        va="top",
        ha="left",
        fontsize=9,
    )
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(FuncFormatter(token_fmt))
    ax.set_xlabel("Unsupervised token budget")
    ax.set_ylabel("Aggregate MoleculeNet z-score")
    ax.set_title("Unsupervised Ramp-Up")
    fig.tight_layout()
    out = OUTDIR / "overall_unsup_ramp.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_sup_ramp(sup: pd.DataFrame) -> Path:
    summary = (
        sup.groupby("n_families", as_index=False)
        .agg(
            mean_all=("z_all", "mean"),
            sd_all=("z_all", "std"),
            mean_nl=("z_no_leak", "mean"),
            sd_nl=("z_no_leak", "std"),
            n=("run_id", "size"),
            tokens=("sup_tokens", "median"),
        )
        .sort_values("n_families")
    )
    summary["ci_all"] = 1.96 * summary["sd_all"].fillna(0.0) / np.sqrt(summary["n"])
    summary["ci_nl"] = 1.96 * summary["sd_nl"].fillna(0.0) / np.sqrt(summary["n"])

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.plot(summary["n_families"], summary["mean_all"], marker="o", color="#B23A48", label="z-score (all tasks)")
    ax.fill_between(
        summary["n_families"],
        summary["mean_all"] - summary["ci_all"],
        summary["mean_all"] + summary["ci_all"],
        color="#B23A48",
        alpha=0.16,
    )
    ax.plot(summary["n_families"], summary["mean_nl"], marker="o", color="#2A6F97", label="z-score (excluding PCBA)")
    ax.fill_between(
        summary["n_families"],
        summary["mean_nl"] - summary["ci_nl"],
        summary["mean_nl"] + summary["ci_nl"],
        color="#2A6F97",
        alpha=0.16,
    )
    ax.axhline(0, color="black", lw=1, alpha=0.5)
    ax.set_xlabel("Supervised ramp-up")
    ax.set_ylabel("Aggregate MoleculeNet z-score")
    ax.set_xticks(summary["n_families"])
    ax.set_xticklabels([f"{int(x)}/5" for x in summary["n_families"]])
    ax.set_title("Supervised Ramp-Up")
    ax.legend(loc="best")
    top = ax.secondary_xaxis("top")
    top.set_xticks(summary["n_families"])
    top.set_xticklabels([token_fmt(x) for x in summary["tokens"]])
    top.set_xlabel("Supervised tokens")
    fig.tight_layout()
    out = OUTDIR / "overall_sup_ramp.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_landscape(landscape: pd.DataFrame) -> Path:
    fig = plt.figure(figsize=(8.2, 6.2))
    ax = fig.add_subplot(111, projection="3d")

    plot_df = landscape.dropna(subset=["unsup_tokens", "sup_tokens", "z_all"]).copy()
    plot_df["xlog"] = np.log10(plot_df["unsup_tokens"] + 1e6)
    plot_df["ylog"] = np.log10(plot_df["sup_tokens"] + 1e6)

    sc = ax.scatter(
        plot_df["xlog"],
        plot_df["ylog"],
        plot_df["z_all"],
        c=plot_df["z_all"],
        cmap="viridis",
        s=44,
        alpha=0.95,
        depthshade=True,
    )
    if len(plot_df) >= 3:
        try:
            ax.plot_trisurf(
                plot_df["xlog"],
                plot_df["ylog"],
                plot_df["z_all"],
                cmap="viridis",
                alpha=0.28,
                linewidth=0.15,
                antialiased=True,
            )
        except Exception:
            pass

    xticks = np.linspace(plot_df["xlog"].min(), plot_df["xlog"].max(), 4)
    yticks = np.linspace(plot_df["ylog"].min(), plot_df["ylog"].max(), 4)
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    ax.set_xticklabels([token_fmt((10**x) - 1e6) for x in xticks])
    ax.set_yticklabels([token_fmt((10**y) - 1e6) for y in yticks])
    ax.set_xlabel("Unsupervised tokens")
    ax.set_ylabel("Supervised tokens")
    ax.set_zlabel("Aggregate z-score")
    ax.set_title("Unified Pretraining Landscape")
    ax.view_init(elev=26, azim=-130)
    fig.colorbar(sc, ax=ax, shrink=0.72, pad=0.08, label="Aggregate z-score")
    fig.tight_layout()
    out = OUTDIR / "overall_landscape_3d.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_dataset_panels_unsup(dataset_unsup: pd.DataFrame) -> Path:
    datasets = sorted(dataset_unsup["dataset"].dropna().unique())
    ncols = 4
    nrows = math.ceil(len(datasets) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(15.5, 3.1 * nrows), sharex=False)
    axes = np.array(axes).reshape(-1)
    for ax, dataset in zip(axes, datasets):
        sub = dataset_unsup[dataset_unsup["dataset"] == dataset].copy()
        sub = sub.sort_values("unsup_tokens")
        summary = sub.groupby("unsup_tokens", as_index=False).agg(mean_value=("main_value", "mean"))
        ax.scatter(sub["unsup_tokens"], sub["main_value"], s=24, alpha=0.35, color="#5B84B1")
        ax.plot(summary["unsup_tokens"], summary["mean_value"], marker="o", color="#1D3557")
        ax.axvline(FULL_UNSUP_CORPUS_TOKENS, color="black", ls="--", lw=0.9, alpha=0.7)
        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(FuncFormatter(token_fmt))
        ax.set_title(dataset)
        metric = sub["main_metric"].iloc[0]
        ax.set_ylabel(metric.upper())
        ax.tick_params(axis="x", labelrotation=35)
    for ax in axes[len(datasets):]:
        ax.axis("off")
    fig.suptitle("Per-Dataset Unsupervised Ramp-Up", y=0.995)
    fig.tight_layout()
    out = OUTDIR / "dataset_panels_unsup.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_dataset_panels_sup(dataset_sup: pd.DataFrame) -> Path:
    datasets = sorted(dataset_sup["dataset"].dropna().unique())
    ncols = 4
    nrows = math.ceil(len(datasets) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(15.5, 3.1 * nrows), sharex=False)
    axes = np.array(axes).reshape(-1)
    for ax, dataset in zip(axes, datasets):
        sub = dataset_sup[dataset_sup["dataset"] == dataset].copy()
        sub = sub.sort_values("n_families")
        summary = sub.groupby("n_families", as_index=False).agg(mean_value=("main_value", "mean"))
        ax.scatter(sub["n_families"], sub["main_value"], s=24, alpha=0.35, color="#B23A48")
        ax.plot(summary["n_families"], summary["mean_value"], marker="o", color="#7F1D1D")
        ax.set_xticks(sorted(sub["n_families"].dropna().unique()))
        ax.set_xticklabels([f"{int(x)}/5" for x in sorted(sub["n_families"].dropna().unique())])
        ax.set_title(dataset)
        metric = sub["main_metric"].iloc[0]
        ax.set_ylabel(metric.upper())
    for ax in axes[len(datasets):]:
        ax.axis("off")
    fig.suptitle("Per-Dataset Supervised Ramp-Up", y=0.995)
    fig.tight_layout()
    out = OUTDIR / "dataset_panels_sup.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def write_notebook() -> None:
    source = f"""from pathlib import Path
import math
import re
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter

ROOT = Path("{ROOT}")
RAW = ROOT / "experiments/robust_matrix/aggregate/raw_results.csv"
OUTDIR = ROOT / "experiments/robust_matrix/aggregate"

LOWER_IS_BETTER = {sorted(LOWER_IS_BETTER)}
LEAKED_DATASETS = {sorted(LEAKED_DATASETS)}
FULL_UNSUP_CORPUS_TOKENS = {FULL_UNSUP_CORPUS_TOKENS}

plt.style.use("default")
plt.rcParams.update({{
    "figure.dpi": 170,
    "savefig.dpi": 170,
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
    "grid.alpha": 0.25,
    "axes.grid": True,
    "lines.linewidth": 2.4,
}})

def token_fmt(x, _pos=None):
    if x >= 1e9:
        return f"{{x/1e9:.0f}}B"
    if x >= 1e6:
        return f"{{x/1e6:.0f}}M"
    return f"{{x:,.0f}}"

df = pd.read_csv(RAW)
df["main_value"] = pd.to_numeric(df["main_value"], errors="coerce")
df["token_budget_total"] = pd.to_numeric(df["token_budget_total"], errors="coerce")
valid = df.dropna(subset=["dataset", "main_metric", "main_value"]).copy()
stats = valid.groupby("dataset")["main_value"].agg(mu="mean", sigma="std").reset_index()
valid = valid.merge(stats, on="dataset", how="left")
valid["sigma"] = valid["sigma"].replace(0, np.nan)
valid["z"] = (valid["main_value"] - valid["mu"]) / valid["sigma"]
valid.loc[valid["main_metric"].isin(LOWER_IS_BETTER), "z"] *= -1.0
valid["z"] = valid["z"].fillna(0.0)

run_scores = valid.groupby("run_id", as_index=False).agg(
    z_all=("z", "mean"),
    n_datasets=("dataset", "nunique"),
    token_budget_total=("token_budget_total", "first"),
)
no_leak = valid[~valid["dataset"].isin(LEAKED_DATASETS)].groupby("run_id", as_index=False).agg(
    z_no_leak=("z", "mean")
)
run_scores = run_scores.merge(no_leak, on="run_id", how="left")

unsup = run_scores[run_scores["run_id"].str.match(r"^unsup_baseline_\\d+_\\d+$", na=False)].copy()
m = unsup["run_id"].str.extract(r"^unsup_baseline_(\\d+)_(\\d+)$")
unsup["unsup_tokens"] = m[0].astype(float)
unsup["replicate"] = m[1].astype(int)

sup = run_scores[run_scores["run_id"].str.match(r"^sup_order_\\dof5_\\d+$", na=False)].copy()
m = sup["run_id"].str.extract(r"^sup_order_(\\d)of5_(\\d+)$")
sup["n_families"] = m[0].astype(int)
sup["replicate"] = m[1].astype(int)
sup["sup_tokens"] = sup["token_budget_total"] * sup["n_families"] / 5.0

mixed = run_scores[run_scores["run_id"].str.match(r"^mixed_\\d+_\\d+_\\d+$", na=False)].copy()
m = mixed["run_id"].str.extract(r"^mixed_(\\d+)_(\\d+)_(\\d+)$")
mixed["unsup_pct"] = m[0].astype(int)
mixed["sup_pct"] = m[1].astype(int)
mixed["replicate"] = m[2].astype(int)
mixed["unsup_tokens"] = mixed["token_budget_total"] * mixed["unsup_pct"] / 100.0
mixed["sup_tokens"] = mixed["token_budget_total"] * mixed["sup_pct"] / 100.0

dataset_unsup = valid[valid["run_id"].isin(unsup["run_id"])].merge(
    unsup[["run_id", "unsup_tokens", "replicate"]], on="run_id", how="left"
)
dataset_sup = valid[valid["run_id"].isin(sup["run_id"])].merge(
    sup[["run_id", "n_families", "sup_tokens", "replicate"]], on="run_id", how="left"
)
"""

    notebook = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# Robust Matrix Analysis\\n",
                    "\\n",
                    "This notebook reproduces the aggregate and per-dataset plots for the robust matrix runs.\\n",
                    "The `z_no_leak` aggregate excludes `PCBA` as a provisional leakage-sensitive summary.\\n",
                ],
            },
            {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": source.splitlines(True)},
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Overall unsupervised ramp-up\\n",
                    "summary = unsup.groupby('unsup_tokens', as_index=False).agg(mean_z=('z_all', 'mean'), sd=('z_all', 'std'), n=('z_all', 'size')).sort_values('unsup_tokens')\\n",
                    "summary['ci95'] = 1.96 * summary['sd'].fillna(0.0) / np.sqrt(summary['n'])\\n",
                    "fig, ax = plt.subplots(figsize=(7.1, 4.6))\\n",
                    "ax.scatter(unsup['unsup_tokens'], unsup['z_all'], s=42, color='#5B84B1', alpha=0.35)\\n",
                    "ax.plot(summary['unsup_tokens'], summary['mean_z'], marker='o', color='#1D3557')\\n",
                    "ax.fill_between(summary['unsup_tokens'], summary['mean_z'] - summary['ci95'], summary['mean_z'] + summary['ci95'], color='#5B84B1', alpha=0.18)\\n",
                    "ax.axhline(0, color='black', lw=1, alpha=0.5)\\n",
                    "ax.axvline(FULL_UNSUP_CORPUS_TOKENS, color='black', ls='--', lw=1.2, alpha=0.8)\\n",
                    "ax.set_xscale('log')\\n",
                    "ax.xaxis.set_major_formatter(FuncFormatter(token_fmt))\\n",
                    "ax.set_xlabel('Unsupervised token budget')\\n",
                    "ax.set_ylabel('Aggregate MoleculeNet z-score')\\n",
                    "ax.set_title('Unsupervised Ramp-Up')\\n",
                    "plt.show()\\n",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Overall supervised ramp-up\\n",
                    "summary = sup.groupby('n_families', as_index=False).agg(mean_all=('z_all', 'mean'), sd_all=('z_all', 'std'), mean_nl=('z_no_leak', 'mean'), sd_nl=('z_no_leak', 'std'), n=('run_id', 'size'), tokens=('sup_tokens', 'median')).sort_values('n_families')\\n",
                    "summary['ci_all'] = 1.96 * summary['sd_all'].fillna(0.0) / np.sqrt(summary['n'])\\n",
                    "summary['ci_nl'] = 1.96 * summary['sd_nl'].fillna(0.0) / np.sqrt(summary['n'])\\n",
                    "fig, ax = plt.subplots(figsize=(7.2, 4.8))\\n",
                    "ax.plot(summary['n_families'], summary['mean_all'], marker='o', color='#B23A48', label='z-score (all tasks)')\\n",
                    "ax.fill_between(summary['n_families'], summary['mean_all'] - summary['ci_all'], summary['mean_all'] + summary['ci_all'], color='#B23A48', alpha=0.16)\\n",
                    "ax.plot(summary['n_families'], summary['mean_nl'], marker='o', color='#2A6F97', label='z-score (excluding PCBA)')\\n",
                    "ax.fill_between(summary['n_families'], summary['mean_nl'] - summary['ci_nl'], summary['mean_nl'] + summary['ci_nl'], color='#2A6F97', alpha=0.16)\\n",
                    "ax.set_xticks(summary['n_families'])\\n",
                    "ax.set_xticklabels([f'{int(x)}/5' for x in summary['n_families']])\\n",
                    "ax.axhline(0, color='black', lw=1, alpha=0.5)\\n",
                    "ax.set_xlabel('Supervised ramp-up')\\n",
                    "ax.set_ylabel('Aggregate MoleculeNet z-score')\\n",
                    "ax.legend(loc='best')\\n",
                    "plt.show()\\n",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# 3D landscape\\n",
                    "landscape = pd.concat([\\n",
                    "    unsup.assign(sup_tokens=0.0)[['run_id', 'z_all', 'unsup_tokens', 'sup_tokens']],\\n",
                    "    sup.assign(unsup_tokens=0.0)[['run_id', 'z_all', 'unsup_tokens', 'sup_tokens']],\\n",
                    "    mixed[['run_id', 'z_all', 'unsup_tokens', 'sup_tokens']],\\n",
                    "], ignore_index=True)\\n",
                    "plot_df = landscape.dropna(subset=['unsup_tokens', 'sup_tokens', 'z_all']).copy()\\n",
                    "plot_df['xlog'] = np.log10(plot_df['unsup_tokens'] + 1e6)\\n",
                    "plot_df['ylog'] = np.log10(plot_df['sup_tokens'] + 1e6)\\n",
                    "fig = plt.figure(figsize=(8.2, 6.2))\\n",
                    "ax = fig.add_subplot(111, projection='3d')\\n",
                    "sc = ax.scatter(plot_df['xlog'], plot_df['ylog'], plot_df['z_all'], c=plot_df['z_all'], cmap='viridis', s=44, alpha=0.95)\\n",
                    "if len(plot_df) >= 3:\\n",
                    "    try:\\n",
                    "        ax.plot_trisurf(plot_df['xlog'], plot_df['ylog'], plot_df['z_all'], cmap='viridis', alpha=0.28, linewidth=0.15)\\n",
                    "    except Exception:\\n",
                    "        pass\\n",
                    "ax.set_xlabel('Unsupervised tokens')\\n",
                    "ax.set_ylabel('Supervised tokens')\\n",
                    "ax.set_zlabel('Aggregate z-score')\\n",
                    "ax.set_title('Unified Pretraining Landscape')\\n",
                    "plt.show()\\n",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Per-dataset unsupervised panels\\n",
                    "datasets = sorted(dataset_unsup['dataset'].dropna().unique())\\n",
                    "ncols = 4\\n",
                    "nrows = math.ceil(len(datasets) / ncols)\\n",
                    "fig, axes = plt.subplots(nrows, ncols, figsize=(15.5, 3.1 * nrows))\\n",
                    "axes = np.array(axes).reshape(-1)\\n",
                    "for ax, dataset in zip(axes, datasets):\\n",
                    "    sub = dataset_unsup[dataset_unsup['dataset'] == dataset].sort_values('unsup_tokens')\\n",
                    "    summary = sub.groupby('unsup_tokens', as_index=False).agg(mean_value=('main_value', 'mean'))\\n",
                    "    ax.scatter(sub['unsup_tokens'], sub['main_value'], s=24, alpha=0.35, color='#5B84B1')\\n",
                    "    ax.plot(summary['unsup_tokens'], summary['mean_value'], marker='o', color='#1D3557')\\n",
                    "    ax.axvline(FULL_UNSUP_CORPUS_TOKENS, color='black', ls='--', lw=0.9, alpha=0.7)\\n",
                    "    ax.set_xscale('log')\\n",
                    "    ax.xaxis.set_major_formatter(FuncFormatter(token_fmt))\\n",
                    "    ax.set_title(dataset)\\n",
                    "for ax in axes[len(datasets):]:\\n",
                    "    ax.axis('off')\\n",
                    "plt.tight_layout()\\n",
                    "plt.show()\\n",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Per-dataset supervised panels\\n",
                    "datasets = sorted(dataset_sup['dataset'].dropna().unique())\\n",
                    "ncols = 4\\n",
                    "nrows = math.ceil(len(datasets) / ncols)\\n",
                    "fig, axes = plt.subplots(nrows, ncols, figsize=(15.5, 3.1 * nrows))\\n",
                    "axes = np.array(axes).reshape(-1)\\n",
                    "for ax, dataset in zip(axes, datasets):\\n",
                    "    sub = dataset_sup[dataset_sup['dataset'] == dataset].sort_values('n_families')\\n",
                    "    summary = sub.groupby('n_families', as_index=False).agg(mean_value=('main_value', 'mean'))\\n",
                    "    ax.scatter(sub['n_families'], sub['main_value'], s=24, alpha=0.35, color='#B23A48')\\n",
                    "    ax.plot(summary['n_families'], summary['mean_value'], marker='o', color='#7F1D1D')\\n",
                    "    ax.set_xticks(sorted(sub['n_families'].dropna().unique()))\\n",
                    "    ax.set_xticklabels([f'{int(x)}/5' for x in sorted(sub['n_families'].dropna().unique())])\\n",
                    "    ax.set_title(dataset)\\n",
                    "for ax in axes[len(datasets):]:\\n",
                    "    ax.axis('off')\\n",
                    "plt.tight_layout()\\n",
                    "plt.show()\\n",
                ],
            },
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.x"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    NOTEBOOK.write_text(json.dumps(notebook, indent=2))


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    apply_style()
    valid, run_scores = load_data()
    tables = build_run_tables(valid, run_scores)
    plot_unsup_ramp(tables["unsup"])
    plot_sup_ramp(tables["sup"])
    plot_landscape(tables["landscape"])
    plot_dataset_panels_unsup(tables["dataset_unsup"])
    plot_dataset_panels_sup(tables["dataset_sup"])
    write_notebook()
    print(f"Wrote plots to {OUTDIR}")
    print(f"Wrote notebook to {NOTEBOOK}")


if __name__ == "__main__":
    main()
