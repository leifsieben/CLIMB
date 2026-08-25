"""Fig B — pretraining scaling ladders on the canonical 6 panels (x = tokens).

ONE script, ONE figure: figures_v2/figB.png / .pdf

What it shows
-------------
Each panel is one benchmark of the canonical six; each coloured line is one pretraining ladder
(seed-0 rungs only): supervised, dense · unsupervised (MLM) · unsup->sup, dense. X = tokens
actually processed (trainer's non-padding `tokens_seen`, log scale) — NOT forward passes x a
constant, and unsup->sup counts its true total (MLM base + 2M-FP SFT stage). The two top
unsupervised rungs (50M/100M, open markers) are trained on the LARGER RDKit-canonical corpus,
not the 12M corpus of the lower rungs — same ladder, different corpus at the top. The
unsupervised MoleculeACE line jumps 24M -> 50M (unsup_48M was never scored there). Reference
lines: both XGBoost anchors (ECFP dash-dot, ECFP+desc dashed) and the random encoder (dotted).

NO error bars (user decision 2026-08-17: they made every panel unreadable — single clean
variant, no banded variant). The underlying
spread is sd_total in figure_data/six_panel/scaling_ladders.csv — 5-fold SD at every rung
(MoleculeACE: SD across the 3 eval-seed macro-means; hERG: SD across 3 eval seeds) — the same
estimand at every rung of every line, available if a referee asks. Pretraining-seed replicates
(8M rung only) are deliberately ignored so every point means the same thing. CheMeleon is
excluded (curiosity comparator only — never in ablation/scaling figures).

Run:  python3 -m figures.fig_B
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D

from figures.style import STYLE, FS, save, check_font, row_ncol, LEGEND_BOX
from figures.arms import ARMS, PANELS, PANEL_ORDER
from figures.sixpanel import ROOT

check_font()
INK = "#000000"

DF = pd.read_csv(ROOT / "figure_data" / "six_panel" / "scaling_ladders.csv")

# ladder display order + style: colour from arms.py (single source of truth), markers distinct
LADDERS = ["sup_dense", "unsup", "u2s_dense"]
MARKER = {"sup_dense": "o", "unsup": "D", "u2s_dense": "P"}

# reference lines (user request 2026-08-17): the stronger XGBoost anchor only, plus the
# untrained random encoder (plain ECFP was added then dropped on request).
REF_LINES = [("ecfp_desc", "--"), ("random_encoder", ":")]

# anchor / control reference levels (compute-independent), from the mainline table
MAIN = pd.read_csv(ROOT / "figure_data" / "six_panel" / "mainline_8M.csv")
REF = {a: dict(zip(MAIN[MAIN.arm == a].panel, MAIN[MAIN.arm == a].value))
       for a, _ in REF_LINES}

YMARGIN = 0.18
# fixed log-spaced ticks, kept sparse (user request 2026-08-17: the dense set was crowded)
XTICKS = [1e8, 5e8, 1e9, 5e9]


def ladder_df(ladder, panel):
    return DF[(DF.ladder == ladder) & (DF.panel == panel)].sort_values("tokens")


def _fmt_tokens(v, _):
    return f"{v/1e9:g}B" if v >= 1e9 else f"{v/1e6:g}M"


def _big_marker(ax, g, color):
    """Open markers on the big-corpus rungs (unsup 50M/100M)."""
    b = g[g.big_corpus == 1]
    if len(b):
        ax.plot(b.tokens, b.value, marker="o", mfc="none", mec=color, mew=1.1, ms=7.5,
                ls="none", zorder=4)


def _panels(banded):
    # 2x3 at FULL page width. One row of six was tried and reverted (user 2026-08-19: "too
    # extreme... they become super distorted") -- six panels across 6.69in leaves ~1.05in
    # each, taller than they are wide, which squashes the curves. 2x3 gives ~2.0in panels.
    # The height saving comes from tighter spacing and ONE shared x-axis label instead of
    # six, not from collapsing the grid. Width is ~3.5% over col2 because savefig("tight")
    # trims back to about the text block.
    fig, axes = plt.subplots(2, 3, figsize=(STYLE["col2"] * 1.035, 3.75))
    for ax, p in zip(axes.ravel(), PANEL_ORDER):
        d = PANELS[p]
        lo, hi = np.inf, -np.inf
        for ladder in LADDERS:
            g = ladder_df(ladder, p)
            if g.empty:
                continue
            c = ARMS[ladder]["color"]
            if banded:
                ax.fill_between(g.tokens, g.value - g.sd_total, g.value + g.sd_total,
                                color=c, alpha=0.15, lw=0, zorder=2)
            ax.plot(g.tokens, g.value, color=c, ls="-", lw=STYLE["lw"],
                    marker=MARKER[ladder], ms=4.6, mec="white", mew=0.6, zorder=3)
            _big_marker(ax, g, c)
            lo = min(lo, (g.value - g.sd_total).min()); hi = max(hi, (g.value + g.sd_total).max())
        for a, ls in REF_LINES:
            if p in REF[a]:
                ax.axhline(REF[a][p], color=ARMS[a]["color"], ls=ls, lw=1.1, zorder=1)
                lo = min(lo, REF[a][p]); hi = max(hi, REF[a][p])
        ax.set_xscale("log")
        ax.xaxis.set_major_locator(ticker.FixedLocator(XTICKS))
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(_fmt_tokens))
        ax.xaxis.set_minor_locator(ticker.NullLocator())
        ax.tick_params(axis="x", which="minor", bottom=False)
        ax.set_xlim(6.5e7, 6.5e9)
        pad = YMARGIN * max(hi - lo, 1e-9)
        y0, y1 = lo - pad, hi + pad
        if d["metric"] == "roc_auc":
            y1 = min(y1, 1.0)
        ax.set_ylim(y0, y1)
        arrow = "↑" if d["higher_better"] else "↓"
        ax.set_title(f"{d['label']} {arrow}", fontsize=FS["title"], fontweight="bold", color=INK, pad=4)
        ax.set_ylabel(d["metric_short"], fontsize=FS["annot"], color=INK)
        # x label drawn ONCE under the row (below), not six times.

        ax.grid(ls=":", lw=0.6, color=STYLE["grid"]); ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    handles = [Line2D([], [], color=ARMS[l]["color"], marker=MARKER[l], ms=4.5, lw=1.2,
                      label=ARMS[l]["label"]) for l in LADDERS]
    handles.append(Line2D([], [], color=INK, marker="o", mfc="none", mew=1.0, ls="none",
                          label="larger corpus (unsup 50M/100M)"))
    for a, ls in REF_LINES:
        handles.append(Line2D([], [], color=ARMS[a]["color"], ls=ls, lw=1.2, label=ARMS[a]["label"]))
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.050),
               ncol=row_ncol(handles), fontsize=FS["legend"], handletextpad=0.5, labelspacing=0.3,
               columnspacing=1.2, borderpad=0.30, **LEGEND_BOX, labelcolor=INK)
    # Axes -> shared x-label -> legend, each about one text-height apart (user 2026-08-19:
    # "too much white space"). The legend is anchored just BELOW the x-label rather than
    # near the canvas floor: with loc="upper center" a low anchor hangs the legend body off
    # the canvas, and savefig("tight") then GROWS the image downward to contain it -- which
    # adds exactly the white band it looks like it should remove.
    fig.tight_layout(rect=(0, 0.112, 1, 1), w_pad=0.35)
    fig.text(0.5, 0.068, "pretraining tokens", ha="center", va="bottom",
             fontsize=FS["annot"], color=INK)
    return fig


def main():
    # single clean variant (user decision 2026-08-17: no error display; sd_total stays
    # available in scaling_ladders.csv if a referee asks)
    fig = _panels(banded=False)
    save(fig, "fig_B")
    plt.close(fig)


if __name__ == "__main__":
    main()
