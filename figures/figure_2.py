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


EXCLUDED_RUNGS = {"skip_dense_48M"}

DF = pd.read_csv(ROOT / "figure_data" / "six_panel" / "scaling_ladders.csv")
_before = set(DF.rung)
DF = DF[~DF.rung.isin(EXCLUDED_RUNGS)].reset_index(drop=True)


_dropped = sorted(_before & EXCLUDED_RUNGS)
_stale = sorted(EXCLUDED_RUNGS - _before)
if _dropped:
    print(f"  [figure_2] excluded {len(_dropped)} rung(s) still present in scaling_ladders.csv: "
          f"{', '.join(_dropped)} -- rebuild with scripts/six_panel_scaling.py to drop at source")
assert not set(DF.rung) & EXCLUDED_RUNGS


CORPUS_12M = 12_000_000
import re as _re


def _fwd_passes(rung):
    m = _re.search(r"(\d+)M", rung)
    return int(m.group(1)) * 1_000_000 if m else np.nan


def epochs(rung, big_corpus):
    """How many times the rung re-reads its corpus. 1.0 = every molecule seen once."""
    fp = _fwd_passes(rung)
    return 1.0 if big_corpus else fp / CORPUS_12M


DF["epochs"] = [epochs(r.rung, r.big_corpus) for r in DF.itertuples()]
DF["repeated"] = DF["epochs"] > 1.0


LADDERS = ["sup_dense", "unsup", "u2s_dense"]
MARKER = {"sup_dense": "o", "unsup": "D", "u2s_dense": "P"}


REF_LINES = [("ecfp_desc", "--"), ("random_encoder", ":")]


MAIN = pd.read_csv(ROOT / "figure_data" / "six_panel" / "mainline_8M.csv")
REF = {a: dict(zip(MAIN[MAIN.arm == a].panel, MAIN[MAIN.arm == a].value))
       for a, _ in REF_LINES}

YMARGIN = 0.18

XTICKS = [1e8, 5e8, 1e9, 5e9]


def ladder_df(ladder, panel):
    return DF[(DF.ladder == ladder) & (DF.panel == panel)].sort_values("tokens")


def _fmt_tokens(v, _):
    return f"{v/1e9:g}B" if v >= 1e9 else f"{v/1e6:g}M"


def _panels():


    fig, axes = plt.subplots(2, 3, figsize=(STYLE["col2"] * 1.035, 3.75))
    for ax, p in zip(axes.ravel(), PANEL_ORDER):
        d = PANELS[p]
        lo, hi = np.inf, -np.inf
        for ladder in LADDERS:
            g = ladder_df(ladder, p)
            g = g[~g.repeated]
            if g.empty:
                continue
            g = g.sort_values("tokens")
            c = ARMS[ladder]["color"]
            ax.plot(g.tokens, g.value, color=c, ls="-", lw=STYLE["lw"],
                    marker=MARKER[ladder], ms=4.6, mec="white", mew=0.6, zorder=3)
            lo = min(lo, (g.value - g.sd_total).min()); hi = max(hi, (g.value + g.sd_total).max())
        for a, ls in REF_LINES:
            if p in REF[a]:
                ax.axhline(REF[a][p], color=ARMS[a]["color"], ls=ls, lw=1.1, zorder=1)
                lo = min(lo, REF[a][p]); hi = max(hi, REF[a][p])
        ax.set_xscale("log")
        ax.xaxis.set_major_locator(ticker.FixedLocator(XTICKS))
        ax.set_xlim(6.5e7, 6.5e9)
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(_fmt_tokens))
        ax.xaxis.set_minor_locator(ticker.NullLocator())
        ax.tick_params(axis="x", which="minor", bottom=False)
        pad = YMARGIN * max(hi - lo, 1e-9)
        y0, y1 = lo - pad, hi + pad
        if d["metric"] == "roc_auc":
            y1 = min(y1, 1.0)
        ax.set_ylim(y0, y1)
        arrow = "↑" if d["higher_better"] else "↓"
        ax.set_title(f"{d['label']} {arrow}", fontsize=FS["title"], fontweight="bold", color=INK, pad=4)
        ax.set_ylabel(d["metric_short"], fontsize=FS["annot"], color=INK)


        ax.grid(ls=":", lw=0.6, color=STYLE["grid"]); ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    handles = [Line2D([], [], color=ARMS[l]["color"], marker=MARKER[l], ms=4.5, lw=1.2,
                      label=ARMS[l]["label"]) for l in LADDERS]
    for a, ls in REF_LINES:
        handles.append(Line2D([], [], color=ARMS[a]["color"], ls=ls, lw=1.2, label=ARMS[a]["label"]))


    _rows = 1 if len(handles) <= 5 else 2
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.050),
               ncol=row_ncol(handles, rows=_rows), fontsize=FS["legend"], handletextpad=0.5,
               labelspacing=0.3,
               columnspacing=1.2, borderpad=0.30, **LEGEND_BOX, labelcolor=INK)


    fig.tight_layout(rect=(0, 0.098, 1, 1), w_pad=0.35)


    fig.text(0.5, 0.082, "pretraining tokens", ha="center", va="bottom", fontsize=FS["annot"], color=INK)
    return fig


def main():


    fig = _panels()
    save(fig, "figure_2")
    plt.close(fig)


if __name__ == "__main__":
    main()
