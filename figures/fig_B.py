"""Fig B — pretraining scaling ladders on the canonical 6 panels (x = tokens).

ONE script, ONE figure: figures_v2/figB.png / .pdf

What it shows
-------------
Each panel is one benchmark of the canonical six; each coloured line is one pretraining ladder
(seed-0 rungs only): supervised, dense · unsupervised (MLM) · unsup->sup, dense. X = tokens
actually processed (trainer's non-padding `tokens_seen`, log scale) — NOT forward passes x a
constant, and unsup->sup counts its true total (MLM base + 2M-FP SFT stage). Reference lines are
the ECFP4+desc XGBoost anchor (dashed) and the untrained random encoder (dotted) — ONE anchor,
not two; plain ECFP was drawn for a while and dropped (see REF_LINES).

THE PLATE IS THE "nodup" CUT, and three things follow from that rather than from this paragraph:
every rung that re-reads the 12M corpus is dropped entirely, so no point needs a corpus caveat;
`_big_marker` is never called, so there are NO open markers and no corpus key in the legend; and
the rungs actually drawn are whatever survives `repeated`, which is a property of the data, not a
list anyone maintains here. Read the inventory `report()` prints, not a rung list in prose — an
earlier version of this docstring described a "24M -> 50M jump" on a line whose 24M rung the cut
had already removed.

Each line therefore carries ONE point per budget. skip_dense_8M and skip_dense_8M_c124 are the
same 8M forward passes on the 12M and the 124M corpus and land 0.343B vs 0.330B apart — the same
x for plotting purposes — so joining them drew a real corpus effect as a vertical zigzag. Only
the _c124 rung is on the line (Leif 2026-08-28); the pair is a caption number, and at that fixed
budget the 124M corpus wins on all six panels: MoleculeACE -0.0391, Ames +0.0299, Tox21 +0.0247,
HIV +0.0169, QM7 -0.64, BACE +0.0012.

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

# RUNGS DROPPED FROM EVERY CUT (Leif 2026-08-26: "just drop it everywhere").
#
# skip_dense_48M began 2026-07-16T09:34:03Z, 2h43m BEFORE the canonical descriptor-stats object
# was first written. It therefore took pretrain_v2's refit branch and fit its own normalizer on a
# 20k sample -- and it did so on a box whose venv carried a shadowed rdkit-pypi 2022.9.5 exposing
# 208 of the 217 descriptors. Its config is also the only one in the dense ladder with no
# descriptor_precompute_dir, so it computed those 208 live. Three differences in one rung:
#
#     descriptor set   208, not 217
#     normalizer       self-fit on 20k, not the canonical July stats
#     pathway          live, where 2M/8M/24M/96M all read the precompute
#
# IT WAS INVISIBLE IN THE VALUES. MoleculeACE read 0.7674, between 24M's 0.7687 and 96M's 0.7748;
# BACE, Tox21 and the rest were equally unremarkable. A run trained against a different objective
# with a self-fit normalizer produced numbers indistinguishable from its neighbours, which is why
# nothing caught it for six weeks -- and why a footnote on a normal-looking point was the weaker
# remedy. It is not a noisy point; it answers a different question.
#
# Dropped in six_panel_scaling.py's LADDERS as well, so a rebuild agrees with this file. The
# filter here is the belt to that braces: a stale scaling_ladders.csv on someone's disk cannot
# put the rung back on the plate without failing the assert below.
EXCLUDED_RUNGS = {"skip_dense_48M"}

DF = pd.read_csv(ROOT / "figure_data" / "six_panel" / "scaling_ladders.csv")
_before = set(DF.rung)
DF = DF[~DF.rung.isin(EXCLUDED_RUNGS)].reset_index(drop=True)
# Report what was dropped rather than dropping it quietly: a filter that silently matches nothing
# is indistinguishable from a filter that silently matches everything, and both look like success.
_dropped = sorted(_before & EXCLUDED_RUNGS)
_stale = sorted(EXCLUDED_RUNGS - _before)
if _dropped:
    print(f"  [fig_B] excluded {len(_dropped)} rung(s) still present in scaling_ladders.csv: "
          f"{', '.join(_dropped)} -- rebuild with scripts/six_panel_scaling.py to drop at source")
assert not set(DF.rung) & EXCLUDED_RUNGS

# ---------------------------------------------------------------- unique molecules -------------
# THE BASE CORPUS IS 12M MOLECULES. A rung above 12M forward passes is RE-READING it, so forward
# passes (and therefore tokens) stop being a data axis there and become a repetition axis. Only the
# two `unsup` big-corpus rungs draw from the 124M full corpus and keep seeing new molecules.
# Measured across the whole table: sup_dense 0 of 5 rungs escape the cap, sup_dense_sparse 0 of 4,
# sup_sparse 0 of 4, u2s_dense 0 of 4, unsup 2 of 6.
# See notes/scaling-ladder-unique-molecule-confound.md -- plotting tokens alone puts
# skip_dense_96M at 4.12B tokens far to the right while it holds the same 12M molecules as the
# rung three positions to its left.
CORPUS_12M = 12_000_000
import re as _re


def _fwd_passes(rung):
    m = _re.search(r"(\d+)M", rung)
    return int(m.group(1)) * 1_000_000 if m else np.nan


# The unsup->sup rungs run TWO stages: an MLM base of <rung>M forward passes plus a fixed 2M-FP
# SFT. Their token count already includes both -- u2s_dense_from2M is 0.17B against unsup_2M's
# 0.09B, i.e. 85.8 tokens per molecule where every pure ladder is a flat 40-43 -- so reading the
# unique-molecule count off the RUN ID puts this family at half its true position. Derived from the
# stages instead.
#
# CAVEAT, stated because it cannot be resolved from the artifacts: this counts the SFT molecules as
# NEW. If the SFT table overlaps the MLM sample, the true figure is lower, bounded below by the MLM
# count alone. The u2s points therefore carry an upper bound on their x, which is the conservative
# direction for the claim being made (it cannot make u2s look better than it is on a data axis).
SFT_STAGE_FP = 2_000_000


def unique_molecules(rung, big_corpus):
    fp = _fwd_passes(rung)
    base = fp if big_corpus else min(fp, CORPUS_12M)
    return base + SFT_STAGE_FP if rung.startswith("u2s_") else base


def epochs(rung, big_corpus):
    """How many times the rung re-reads its corpus. 1.0 = every molecule seen once."""
    fp = _fwd_passes(rung)
    return 1.0 if big_corpus else fp / CORPUS_12M


DF["uniq"] = [unique_molecules(r.rung, r.big_corpus) for r in DF.itertuples()]
DF["epochs"] = [epochs(r.rung, r.big_corpus) for r in DF.itertuples()]
DF["repeated"] = DF["epochs"] > 1.0

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


def _panels(banded, variant="marked"):
    """variant:
       "marked"  x = tokens, every rung drawn; rungs that RE-READ the corpus are hollow and
                 carry their epoch count, so a reader can see which points add data and which
                 only add passes.
       "nodup"   x = tokens, but ONLY rungs that actually add unique molecules are plotted.
                 The supervised ladders lose their top three rungs and keep two points, which
                 is the honest extent of their data-scaling evidence.
       "unique"  x = unique molecules. Repeated rungs collapse onto x = 12M, so the vertical
                 stack there IS the confound, drawn.
    """
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
            if variant == "nodup":
                g = g[~g.repeated]
            if g.empty:
                continue
            xcol = "uniq" if variant == "unique" else "tokens"
            g = g.sort_values(xcol)
            c = ARMS[ladder]["color"]
            if banded:
                ax.fill_between(g.tokens, g.value - g.sd_total, g.value + g.sd_total,
                                color=c, alpha=0.15, lw=0, zorder=2)
            if variant == "unique":
                # Points that share an x are the SAME data read more times; join them with a
                # thin vertical spine rather than a line that pretends to travel along x.
                solid = g[~g.repeated]
                ax.plot(solid[xcol], solid.value, color=c, ls="-", lw=STYLE["lw"],
                        marker=MARKER[ladder], ms=4.6, mec="white", mew=0.6, zorder=3)
                rep = g[g.repeated]
                if len(rep):
                    ax.plot(rep[xcol], rep.value, color=c, ls="none", marker=MARKER[ladder],
                            ms=4.0, mfc="none", mew=1.0, zorder=3)
                    ax.plot([CORPUS_12M] * 2, [rep.value.min(), rep.value.max()],
                            color=c, ls=":", lw=0.9, zorder=2)
            else:
                ax.plot(g[xcol], g.value, color=c, ls="-", lw=STYLE["lw"],
                        marker=MARKER[ladder], ms=4.6, mec="white", mew=0.6, zorder=3)
            if variant == "marked":
                rep = g[g.repeated]
                for r in rep.itertuples():
                    ax.plot(r.tokens, r.value, marker=MARKER[ladder], ms=4.6, mfc="white",
                            mec=c, mew=1.1, ls="none", zorder=4)
            if variant != "nodup":
                # The open ring marks the big-corpus rungs. In the paper cut every rung samples
                # new molecules, so the distinction it encodes no longer exists -- and it has no
                # legend key there either, which would leave an unexplained symbol on the plate.
                _big_marker(ax, g, c)
            lo = min(lo, (g.value - g.sd_total).min()); hi = max(hi, (g.value + g.sd_total).max())
        for a, ls in REF_LINES:
            if p in REF[a]:
                ax.axhline(REF[a][p], color=ARMS[a]["color"], ls=ls, lw=1.1, zorder=1)
                lo = min(lo, REF[a][p]); hi = max(hi, REF[a][p])
        ax.set_xscale("log")
        if variant == "unique":
            ax.xaxis.set_major_locator(ticker.FixedLocator([2e6, 12e6, 100e6]))
            ax.set_xlim(1.4e6, 1.7e8)
            ax.axvline(CORPUS_12M, color=INK, ls=(0, (2, 2)), lw=0.7, zorder=1)
        else:
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
        # x label drawn ONCE under the row (below), not six times.

        ax.grid(ls=":", lw=0.6, color=STYLE["grid"]); ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    handles = [Line2D([], [], color=ARMS[l]["color"], marker=MARKER[l], ms=4.5, lw=1.2,
                      label=ARMS[l]["label"]) for l in LADDERS]
    if variant != "nodup":
        handles.append(Line2D([], [], color=INK, marker="o", mfc="none", mew=1.0, ls="none",
                              label="larger corpus (unsup 50M/100M)"))
    if variant == "marked":
        handles.append(Line2D([], [], color=INK, marker="s", mfc="white", mew=1.1, ls="none",
                              label="re-reads the 12M corpus (no new molecules)"))
    if variant == "unique":
        handles.append(Line2D([], [], color=INK, ls=(0, (2, 2)), lw=0.9,
                              label="12M corpus ceiling"))
    for a, ls in REF_LINES:
        handles.append(Line2D([], [], color=ARMS[a]["color"], ls=ls, lw=1.2, label=ARMS[a]["label"]))
    # ROWS BY HANDLE COUNT, measured not guessed: the "re-reads the corpus" key takes the
    # marked variant to 6 handles, and one row rendered 8.91in against a 6.69in text block.
    # ONE ROW for the paper cut: three ladders plus two reference lines is five keys, and with
    # every repeated rung dropped there is no corpus key to explain any more.
    _rows = 1 if len(handles) <= 5 else 2
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.050),
               ncol=row_ncol(handles, rows=_rows), fontsize=FS["legend"], handletextpad=0.5,
               labelspacing=0.3,
               columnspacing=1.2, borderpad=0.30, **LEGEND_BOX, labelcolor=INK)
    # Axes -> shared x-label -> legend, each about one text-height apart (user 2026-08-19:
    # "too much white space"). The legend is anchored just BELOW the x-label rather than
    # near the canvas floor: with loc="upper center" a low anchor hangs the legend body off
    # the canvas, and savefig("tight") then GROWS the image downward to contain it -- which
    # adds exactly the white band it looks like it should remove.
    fig.tight_layout(rect=(0, 0.112, 1, 1), w_pad=0.35)
    xlab = {"marked": "pretraining tokens  (hollow = corpus re-read, no new molecules)",
            "nodup":  "pretraining tokens   (every rung samples new molecules; none re-reads)",
            "unique": "unique molecules seen"}[variant]
    fig.text(0.5, 0.068, xlab, ha="center", va="bottom", fontsize=FS["annot"], color=INK)
    return fig


def main():
    # single clean variant (user decision 2026-08-17: no error display; sd_total stays
    # available in scaling_ladders.csv if a referee asks)
    #
    # THREE VARIANTS for review (Leif 2026-08-26). Tokens is the standard unit and v1 keeps it;
    # v2 and v3 exist because tokens alone cannot show that the supervised ladders stop adding
    # molecules after the 8M rung.
    # ONLY RUNGS THAT SAMPLE NEW CHEMISTRY (Leif 2026-08-26). Every repeated rung is dropped, so
    # no point on the plate needs a corpus caveat and there is nothing to explain about resampling
    # vs new molecules -- all points are the same kind of point.
    #
    # THE TOKENS AXIS SURVIVES THAT, which is why it stays. On non-repeating rungs tokens per
    # molecule is a flat 40-43 for every pure ladder, so tokens IS the unique-molecule axis
    # rescaled; keeping the standard unit costs nothing once the repeats are gone. (u2s runs 53-86
    # because its tokens include the SFT stage -- see unique_molecules().)
    fig = _panels(banded=False, variant="nodup")
    save(fig, "fig_B")
    plt.close(fig)
    # NO SI CUTS (Leif 2026-08-26: "don't even produce the fig_B SI cuts, these are not needed").
    # The "marked" and "unique" variants remain reachable as _panels(variant=...) for a referee
    # question about re-reads, but nothing renders them, so figures_v2/ holds one fig_B artefact
    # and there is no second plate to keep in sync with the first.
    report()


def report():
    d = DF.drop_duplicates("rung")[["ladder", "rung", "tokens", "uniq", "epochs", "repeated"]]
    print("\nRung inventory -- what each point on fig_B actually represents:\n")
    print(f"   {'ladder':<20}{'rung':<28}{'tokens':>9}{'unique mols':>13}{'epochs':>9}")
    for lad, g in d.groupby("ladder"):
        for r in g.sort_values("tokens").itertuples():
            flag = "  re-read" if r.repeated else ""
            print(f"   {lad:<20}{r.rung:<28}{r.tokens/1e9:>8.2f}B{r.uniq/1e6:>12.0f}M"
                  f"{r.epochs:>9.1f}{flag}")


if __name__ == "__main__":
    main()
