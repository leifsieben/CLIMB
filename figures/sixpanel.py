"""Loaders for the canonical 6-panel results + the ranking maths shared by Fig A1 / Table A2.

All numbers come from figure_data/six_panel/, produced by scripts/six_panel_aggregate.py.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

from figures.arms import ARMS, ARM_ORDER, PANELS, PANEL_ORDER

ROOT = Path(__file__).resolve().parent.parent
SP = ROOT / "figure_data" / "six_panel"


def load_mainline() -> pd.DataFrame:
    """arm x panel point estimates at the 8M budget."""
    df = pd.read_csv(SP / "mainline_8M.csv")
    df["arm"] = pd.Categorical(df["arm"], ARM_ORDER, ordered=True)
    df["panel"] = pd.Categorical(df["panel"], PANEL_ORDER, ordered=True)
    return df


def load_long() -> pd.DataFrame:
    """Replicate-level values: per target (MoleculeACE) / per seed x fold (MoleculeNet)."""
    return pd.read_csv(SP / "mainline_8M_long.csv")


def load_bootstrap() -> pd.DataFrame:
    return pd.read_csv(SP / "mainline_8M_bootstrap.csv")


def score_matrix(arms=None) -> pd.DataFrame:
    """Wide matrix of raw scores: rows = arms, columns = the 6 panels (NaN where not run)."""
    arms = arms or ARM_ORDER
    df = load_mainline()
    m = df.pivot_table(index="arm", columns="panel", values="value", observed=True)
    return m.reindex(index=arms, columns=PANEL_ORDER)


def rank_table(arms=None) -> pd.DataFrame:
    """Per-panel rank (1 = best) rescaled to a common 1..N scale, plus the mean rank.

    Rescaling matters because a panel where only k of the N arms were run would otherwise hand
    those k arms artificially good ranks (1..k instead of 1..N). Rank within the panel, then map
    [1..k] -> [1..N] linearly.

    Returns rows = arms, columns = the 6 panels + mean_rank, se_rank, n_panels, worst, best.
    """
    arms = arms or ARM_ORDER
    S = score_matrix(arms)
    N = len(arms)
    R = pd.DataFrame(index=S.index, columns=PANEL_ORDER, dtype=float)
    for p in PANEL_ORDER:
        col = S[p].dropna()
        if col.empty:
            continue
        r = col.rank(ascending=PANELS[p]["higher_better"] is False)   # 1 = best
        k = len(col)
        R.loc[r.index, p] = 1 + (N - 1) * (r - 1) / (k - 1) if k > 1 else 1.0
    R["n_panels"] = R[PANEL_ORDER].notna().sum(axis=1)
    R["mean_rank"] = R[PANEL_ORDER].mean(axis=1)
    R["se_rank"] = R[PANEL_ORDER].std(axis=1, ddof=1) / np.sqrt(R["n_panels"])
    R["best"] = R[PANEL_ORDER].min(axis=1)
    R["worst"] = R[PANEL_ORDER].max(axis=1)
    return R.sort_values("mean_rank")


def shortfall_table(arms=None) -> pd.DataFrame:
    """Per-panel % behind that panel's best model, plus the mean and its SE across panels.

    The effect-size counterpart to rank_table(): ranking treats a panel where the field spans
    1.8% of the metric (BBBP) exactly like one where it spans 22% (MoleculeACE), which flatters
    models that lose narrowly on compressed panels. This keeps the size of the gap.
    """
    arms = arms or ARM_ORDER
    S = score_matrix(arms)
    G = {}
    for p in PANEL_ORDER:
        col = S[p].dropna()
        if col.empty:
            continue
        best = col.max() if PANELS[p]["higher_better"] else col.min()
        G[p] = 100 * (best - col).abs() / abs(best)
    G = pd.DataFrame(G).reindex(index=arms, columns=PANEL_ORDER)
    G["n_panels"] = G[PANEL_ORDER].notna().sum(axis=1)
    G["mean_gap"] = G[PANEL_ORDER].mean(axis=1)
    G["se_gap"] = G[PANEL_ORDER].std(axis=1, ddof=1) / np.sqrt(G["n_panels"])
    G["best"] = G[PANEL_ORDER].min(axis=1)
    G["worst"] = G[PANEL_ORDER].max(axis=1)
    return G.sort_values("mean_gap")


def wins(arms=None) -> pd.DataFrame:
    """Per-arm win counts: how often it is the best / top-3 arm on a panel (raw, un-rescaled)."""
    arms = arms or ARM_ORDER
    S = score_matrix(arms)
    out = {}
    for p in PANEL_ORDER:
        col = S[p].dropna()
        if col.empty:
            continue
        out[p] = col.rank(ascending=PANELS[p]["higher_better"] is False)
    R = pd.DataFrame(out)
    return pd.DataFrame({
        "n_panels": R.notna().sum(axis=1),
        "n_best": (R == 1).sum(axis=1),
        "n_top3": (R <= 3).sum(axis=1),
    })


def fmt_value(panel: str, v: float) -> str:
    """Format a raw score the way its panel wants it."""
    if not np.isfinite(v):
        return "—"
    return f"{v:.1f}" if panel == "QM7" else f"{v:.3f}"


def arm_labels(index) -> list:
    return [ARMS[a]["label"] if a in ARMS else a for a in index]


def arm_colors(index) -> list:
    return [ARMS[a]["color"] if a in ARMS else "#999999" for a in index]
