#!/usr/bin/env python3
"""Build paper-ready figures and a report from the robust_matrix aggregate.

Implements the analysis plan in README.md (Section "Paper Analysis Plan"):
- Score_full   = mean z-score across ALL MoleculeNet datasets
- Score_unbiased = mean z-score EXCLUDING PCBA (potential pretraining/eval overlap)

Z-scores are computed across all evaluated runs (per dataset), with RMSE
metrics negated so that "higher is better" holds for every score.

Note: a random-encoder baseline (the README's true normalization reference)
is not in the matrix yet, so we report the within-pool z-score and flag this
as a gap in the report.
"""
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
OUT = ROOT / "experiments/robust_matrix/report"
OUT.mkdir(parents=True, exist_ok=True)

LOWER_IS_BETTER = {"rmse"}
LEAKED = {"PCBA"}
FULL_UNSUP_CORPUS_TOKENS = 551_133_440  # for reference line on unsup ramp

plt.rcParams.update({
    "figure.dpi": 140,
    "savefig.dpi": 140,
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
    "grid.alpha": 0.25,
    "axes.grid": True,
    "lines.linewidth": 2.0,
})


def token_fmt(x: float, _pos=None) -> str:
    if x >= 1e9:
        return f"{x/1e9:.0f}B"
    if x >= 1e6:
        return f"{x/1e6:.0f}M"
    if x >= 1e3:
        return f"{x/1e3:.0f}K"
    return f"{x:,.0f}"


# ---------- 1. Load & build canonical tables ----------

def load() -> tuple[pd.DataFrame, pd.DataFrame]:
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

    run = (
        valid.groupby(["run_id", "run_type"], as_index=False)
        .agg(score_full=("z", "mean"),
             n_datasets=("dataset", "nunique"),
             token_budget_total=("token_budget_total", "first"),
             selection=("selection", "first"))
    )
    nopcba = (
        valid[~valid["dataset"].isin(LEAKED)]
        .groupby("run_id", as_index=False)
        .agg(score_unbiased=("z", "mean"))
    )
    run = run.merge(nopcba, on="run_id", how="left")
    return valid, run


def parse_runs(run: pd.DataFrame):
    unsup = run[run["run_id"].str.match(r"^unsup_baseline_\d+_\d+$", na=False)].copy()
    m = unsup["run_id"].str.extract(r"^unsup_baseline_(\d+)_(\d+)$")
    unsup["unsup_tokens"] = m[0].astype(float)
    unsup["replicate"] = m[1].astype(int)

    sup = run[run["run_id"].str.match(r"^sup_order_\dof5_\d+$", na=False)].copy()
    m = sup["run_id"].str.extract(r"^sup_order_(\d)of5_(\d+)$")
    sup["n_families"] = m[0].astype(int)
    sup["replicate"] = m[1].astype(int)
    # sup_tokens estimated as n/5 * total budget per run; budget = 1B (configured)
    sup["sup_tokens"] = sup["token_budget_total"].fillna(1e9) * sup["n_families"] / 5.0

    mixed = run[run["run_id"].str.match(r"^mixed_\d+_\d+_\d+$", na=False)].copy()
    m = mixed["run_id"].str.extract(r"^mixed_(\d+)_(\d+)_(\d+)$")
    mixed["unsup_pct"] = m[0].astype(int)
    mixed["sup_pct"] = m[1].astype(int)
    mixed["replicate"] = m[2].astype(int)
    # mixed runs use 1B token budget per worker2 patch
    mixed_budget = mixed["token_budget_total"].fillna(1e9)
    mixed["unsup_tokens"] = mixed_budget * mixed["unsup_pct"] / 100.0
    mixed["sup_tokens"] = mixed_budget * mixed["sup_pct"] / 100.0

    cov = run[run["run_id"].str.match(r"^unsup_cov_\d+pct_\d+b$", na=False)].copy()
    m = cov["run_id"].str.extract(r"^unsup_cov_(\d+)pct_(\d+b)$")
    cov["coverage_pct"] = m[0].astype(int)

    return unsup, sup, mixed, cov


# ---------- 2. Figures ----------

def fig_unsup_ramp(unsup: pd.DataFrame, valid: pd.DataFrame, score_col="score_unbiased") -> Path:
    summary = (
        unsup.groupby("unsup_tokens", as_index=False)
        .agg(mean=(score_col, "mean"), sd=(score_col, "std"), n=(score_col, "size"))
        .sort_values("unsup_tokens")
    )
    summary["se"] = summary["sd"].fillna(0) / np.sqrt(summary["n"])

    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    ax.scatter(unsup["unsup_tokens"], unsup[score_col], s=42, color="#5B84B1", alpha=0.4, zorder=2)
    ax.errorbar(summary["unsup_tokens"], summary["mean"], yerr=1.96 * summary["se"],
                marker="o", color="#1D3557", capsize=4, zorder=3)
    ax.axhline(0, color="black", lw=1, alpha=0.5)
    ax.axvline(FULL_UNSUP_CORPUS_TOKENS, color="grey", ls="--", lw=1, alpha=0.6)
    ax.text(FULL_UNSUP_CORPUS_TOKENS, ax.get_ylim()[1] * 0.95,
            " full corpus", rotation=90, va="top", ha="left", fontsize=9, color="grey")
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(FuncFormatter(token_fmt))
    ax.set_xlabel("Unsupervised token budget")
    ax.set_ylabel(f"{score_col} (z-score across pool)")
    ax.set_title(f"Figure 1 — Unsupervised scaling curve ({score_col})")
    fig.tight_layout()
    out = OUT / f"fig1_unsup_ramp_{score_col}.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def fig_sup_ramp(sup: pd.DataFrame, score_col="score_unbiased") -> Path:
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    summary = (
        sup.groupby("n_families", as_index=False)
        .agg(mean=(score_col, "mean"), sd=(score_col, "std"), n=(score_col, "size"),
             tokens=("sup_tokens", "median"))
        .sort_values("n_families")
    )
    summary["se"] = summary["sd"].fillna(0) / np.sqrt(summary["n"])

    # individual replicates as faint dots
    ax.scatter(sup["n_families"], sup[score_col], s=44, alpha=0.35, color="#B23A48", zorder=2)
    ax.errorbar(summary["n_families"], summary["mean"], yerr=1.96 * summary["se"],
                marker="o", color="#7F1D1D", capsize=4, zorder=3)

    ax.axhline(0, color="black", lw=1, alpha=0.5)
    ax.set_xticks(summary["n_families"])
    ax.set_xticklabels([f"{int(x)}/5" for x in summary["n_families"]])
    ax.set_xlabel("# supervised families used during pretraining")
    ax.set_ylabel(f"{score_col}")
    ax.set_title(f"Figure 2 — Supervised scaling curve ({score_col})")
    top = ax.secondary_xaxis("top")
    top.set_xticks(summary["n_families"])
    top.set_xticklabels([token_fmt(x) for x in summary["tokens"]])
    top.set_xlabel("Approx supervised tokens")
    fig.tight_layout()
    out = OUT / f"fig2_sup_ramp_{score_col}.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def fig_landscape_heatmap(unsup, sup, mixed, score_col="score_unbiased") -> Path:
    # Build a regularised heatmap. Unsup-only runs sit on the y=0 (sup_tokens=0) row;
    # sup-only on x=0; mixed in interior.
    pts = []
    for _, r in unsup.iterrows():
        pts.append((r["unsup_tokens"], 0.0, r[score_col]))
    for _, r in sup.iterrows():
        pts.append((0.0, r["sup_tokens"], r[score_col]))
    for _, r in mixed.iterrows():
        pts.append((r["unsup_tokens"], r["sup_tokens"], r[score_col]))
    pts = pd.DataFrame(pts, columns=["unsup_tokens", "sup_tokens", "z"])
    pts = pts.dropna()

    fig, ax = plt.subplots(figsize=(7.6, 5.4))
    pts["x"] = np.log10(pts["unsup_tokens"] + 1)
    pts["y"] = np.log10(pts["sup_tokens"] + 1)
    sc = ax.scatter(pts["x"], pts["y"], c=pts["z"], cmap="viridis", s=120, edgecolor="white", linewidth=0.5)
    fig.colorbar(sc, ax=ax, label=score_col)
    ax.set_xlabel("log10(unsup tokens + 1)")
    ax.set_ylabel("log10(sup tokens + 1)")
    ax.set_title(f"Figure 3 — Pretraining landscape ({score_col})")
    fig.tight_layout()
    out = OUT / f"fig3_landscape_{score_col}.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def fig_pcba_leak(run: pd.DataFrame) -> Path:
    df = run.copy()
    df["gap"] = df["score_full"] - df["score_unbiased"]
    df["family"] = df["run_id"].apply(_run_family)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    # Left: scatter of full vs unbiased coloured by family
    families = sorted(df["family"].dropna().unique())
    cmap = plt.get_cmap("tab10")
    color = {f: cmap(i % 10) for i, f in enumerate(families)}
    for f, sub in df.groupby("family"):
        axes[0].scatter(sub["score_unbiased"], sub["score_full"], s=46, label=f,
                        color=color[f], alpha=0.85, edgecolor="white", linewidth=0.4)
    lim = [df[["score_full", "score_unbiased"]].min().min() - 0.3,
           df[["score_full", "score_unbiased"]].max().max() + 0.3]
    axes[0].plot(lim, lim, "--", color="grey", alpha=0.6)
    axes[0].set_xlim(lim); axes[0].set_ylim(lim)
    axes[0].set_xlabel("Score_unbiased")
    axes[0].set_ylabel("Score_full (incl PCBA)")
    axes[0].set_title("Score_full vs Score_unbiased")
    axes[0].legend(fontsize=8, loc="lower right")

    # Right: gap distribution by family
    parts = [df[df["family"] == f]["gap"].dropna() for f in families]
    axes[1].boxplot(parts, labels=families, showmeans=True)
    axes[1].axhline(0, color="black", lw=1, alpha=0.5)
    axes[1].set_ylabel("Score_full − Score_unbiased")
    axes[1].set_title("PCBA leakage gap by run family")
    axes[1].tick_params(axis="x", rotation=20)

    fig.suptitle("Figure 4 — PCBA leakage diagnostic", weight="bold")
    fig.tight_layout()
    out = OUT / "fig4_pcba_leakage.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def fig_family_order_box(sup: pd.DataFrame) -> Path:
    # Each n_families slot has 5 replicates; show them
    fig, ax = plt.subplots(figsize=(8, 4.8))
    by = sup.groupby("n_families")["score_unbiased"].apply(list)
    ax.boxplot([by[k] for k in sorted(by.index)], labels=[f"{k}/5" for k in sorted(by.index)], showmeans=True)
    # also overlay each point coloured by replicate index (replicate is family-order)
    for k in sorted(by.index):
        ys = by[k]
        xs = [k + np.random.uniform(-0.12, 0.12) for _ in ys]
        # convert position
        pos = sorted(by.index).index(k) + 1
        xs = [pos + np.random.uniform(-0.12, 0.12) for _ in ys]
        ax.scatter(xs, ys, s=30, alpha=0.6, color="#B23A48")
    ax.set_xlabel("# supervised families")
    ax.set_ylabel("Score_unbiased")
    ax.set_title("Figure 6 — Replicate spread per supervised ramp step")
    ax.axhline(0, color="black", lw=1, alpha=0.5)
    fig.tight_layout()
    out = OUT / "fig6_family_order.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def fig_coverage(cov: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    cov_s = cov.sort_values("coverage_pct")
    ax.plot(cov_s["coverage_pct"], cov_s["score_unbiased"], marker="o", color="#1D3557")
    for _, r in cov_s.iterrows():
        ax.annotate(r["run_id"].split("_")[-2], (r["coverage_pct"], r["score_unbiased"]),
                    textcoords="offset points", xytext=(6, 4), fontsize=8, color="grey")
    ax.axhline(0, color="black", lw=1, alpha=0.5)
    ax.set_xlabel("% of unsupervised corpus")
    ax.set_ylabel("Score_unbiased")
    ax.set_title("Figure 7 — Coverage ablation (1B token budget, 1 rep each)")
    fig.tight_layout()
    out = OUT / "fig7_coverage.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def fig_per_dataset_unsup(valid: pd.DataFrame, unsup: pd.DataFrame) -> Path:
    sub = valid[valid["run_id"].isin(unsup["run_id"])].merge(
        unsup[["run_id", "unsup_tokens"]], on="run_id"
    )
    datasets = sorted(sub["dataset"].dropna().unique())
    ncols = 4
    nrows = math.ceil(len(datasets) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 3.0 * nrows))
    axes = np.array(axes).reshape(-1)
    for ax, ds in zip(axes, datasets):
        d = sub[sub["dataset"] == ds].copy().sort_values("unsup_tokens")
        ax.scatter(d["unsup_tokens"], d["main_value"], s=22, alpha=0.4, color="#5B84B1")
        s = d.groupby("unsup_tokens", as_index=False)["main_value"].mean()
        ax.plot(s["unsup_tokens"], s["main_value"], marker="o", color="#1D3557")
        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(FuncFormatter(token_fmt))
        m = d["main_metric"].iloc[0]
        ax.set_title(f"{ds} ({m})")
        ax.tick_params(axis="x", labelrotation=30)
        ax.axvline(FULL_UNSUP_CORPUS_TOKENS, color="grey", ls="--", lw=0.7, alpha=0.6)
    for ax in axes[len(datasets):]:
        ax.axis("off")
    fig.suptitle("Per-dataset unsupervised ramp (raw metrics)", y=1.0, weight="bold")
    fig.tight_layout()
    out = OUT / "fig_per_dataset_unsup.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


# ---------- 3. helpers ----------

def _run_family(run_id: str) -> str:
    if re.match(r"^unsup_baseline_\d+_\d+$", run_id):
        return "unsup_baseline"
    if re.match(r"^sup_order_\dof5_\d+$", run_id):
        return "sup_order"
    if re.match(r"^mixed_\d+_\d+_\d+$", run_id):
        return "mixed"
    if re.match(r"^unsup_cov_\d+pct_\d+b$", run_id):
        return "unsup_cov"
    if run_id.startswith("smoke"):
        return "smoke"
    return "other"


def write_report(valid, run, unsup, sup, mixed, cov, figs):
    md = []
    md.append("# CLIMB robust_matrix paper-prep report\n")
    md.append(f"_Generated 2026-05-01 from {len(run)} evaluated runs._\n")
    md.append("\n## Research question\n")
    md.append("**Does unsupervised pretraining benefit chemical language models, "
              "and how does it interact with supervised pretraining?**\n")
    md.append(
        "\nTwo aggregate scores per the README plan:\n"
        "- **Score_full** — mean within-pool z-score across all 15 MoleculeNet datasets.\n"
        "- **Score_unbiased** — same, **excluding PCBA** (PCBA family is also part of supervised pretraining; potential leak).\n"
    )
    md.append("\n_Caveat:_ no random-encoder baseline is available yet, so z-scores are computed within-pool. "
              "Magnitudes are relative to the matrix average, not to the no-pretraining floor.\n")

    md.append("\n## Matrix completion status\n\n")
    md.append("| Group | Planned runs | Evaluated | % done |\n|---|---:|---:|---:|\n")
    g = {
        "smoke": (3, run["run_id"].str.startswith("smoke").sum()),
        "unsup_baseline (8 token budgets × 3 reps)": (24, len(unsup)),
        "sup_order_ramp (5 ramp steps × 5 family-orders)": (25, len(sup)),
        "unsup_cov (5 coverage levels × 1 rep)": (5, len(cov)),
        "mixed_fixed_budget (5 ratios × 3 reps)": (15, len(mixed)),
        "random-encoder baseline (planned, F1)": (5, 0),
    }
    for k, (planned, done) in g.items():
        pct = f"{100*done/planned:.0f}%" if planned else ""
        md.append(f"| {k} | {planned} | {done} | {pct} |\n")
    md.append(f"\n**Total evaluated: {len(run)} of 72 manifest runs (+ 5 missing random baseline).**\n")

    md.append("\n## Headline numbers\n\n")
    md.append("Top-5 runs by Score_unbiased:\n\n")
    top5 = run.sort_values("score_unbiased", ascending=False).head(5)
    md.append("| run | family | Score_unbiased | Score_full | n_datasets |\n|---|---|---:|---:|---:|\n")
    for _, r in top5.iterrows():
        md.append(f"| {r['run_id']} | {_run_family(r['run_id'])} | {r['score_unbiased']:+.3f} | {r['score_full']:+.3f} | {int(r['n_datasets'])} |\n")
    md.append("\nBottom-5 runs by Score_unbiased:\n\n")
    bot5 = run.sort_values("score_unbiased", ascending=True).head(5)
    md.append("| run | family | Score_unbiased | Score_full | n_datasets |\n|---|---|---:|---:|---:|\n")
    for _, r in bot5.iterrows():
        md.append(f"| {r['run_id']} | {_run_family(r['run_id'])} | {r['score_unbiased']:+.3f} | {r['score_full']:+.3f} | {int(r['n_datasets'])} |\n")

    md.append("\n## Figures\n\n")
    for k, p in figs.items():
        rel = p.relative_to(ROOT)
        md.append(f"### {k}\n![{k}]({rel})\n\n")

    md.append("\n## Per-group score summaries\n\n")
    for label, frame, key in [
        ("Unsupervised ramp", unsup, "unsup_tokens"),
        ("Supervised ramp", sup, "n_families"),
        ("Mixed budget", mixed, "unsup_pct"),
        ("Coverage ablation", cov, "coverage_pct"),
    ]:
        md.append(f"### {label}\n\n")
        if frame.empty:
            md.append("_no runs evaluated yet_\n\n")
            continue
        s = (frame.groupby(key, as_index=False)
             .agg(score_unbiased_mean=("score_unbiased", "mean"),
                  score_unbiased_sd=("score_unbiased", "std"),
                  score_full_mean=("score_full", "mean"),
                  n=("score_full", "size")))
        md.append(s.to_markdown(index=False, floatfmt=".3f") + "\n\n")

    out = OUT / "report.md"
    out.write_text("".join(md))
    print(f"wrote {out}")
    return out


def main():
    valid, run = load()
    unsup, sup, mixed, cov = parse_runs(run)

    # save tidy tables
    run.to_csv(OUT / "run_scores.csv", index=False)

    figs = {}
    figs["Figure 1 — Unsupervised ramp (Score_unbiased)"] = fig_unsup_ramp(unsup, valid, "score_unbiased")
    figs["Figure 1b — Unsupervised ramp (Score_full)"] = fig_unsup_ramp(unsup, valid, "score_full")
    figs["Figure 2 — Supervised ramp (Score_unbiased)"] = fig_sup_ramp(sup, "score_unbiased")
    figs["Figure 2b — Supervised ramp (Score_full)"] = fig_sup_ramp(sup, "score_full")
    figs["Figure 3 — Pretraining landscape"] = fig_landscape_heatmap(unsup, sup, mixed, "score_unbiased")
    figs["Figure 4 — PCBA leakage diagnostic"] = fig_pcba_leak(run)
    figs["Figure 6 — Family-order spread"] = fig_family_order_box(sup)
    figs["Figure 7 — Coverage ablation"] = fig_coverage(cov)
    figs["Per-dataset unsup ramp"] = fig_per_dataset_unsup(valid, unsup)

    write_report(valid, run, unsup, sup, mixed, cov, figs)


if __name__ == "__main__":
    main()
