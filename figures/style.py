"""Shared matplotlib style + save helper for the CLIMB paper figures.

Every figure script starts with `from figures.style import *` (or imports what it needs) so the
typography, sizes and output paths are identical across the paper. Figures are written to
figures_v2/ as both PNG (screen/review) and PDF (vector, for LaTeX).
"""
from __future__ import annotations
from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
OUTDIR = ROOT / "figures_v2"

# ---------------------------------------------------------------------------------------------
# ONE font, ONE size scale. These figures get combined into multi-panel layouts later, so every
# script must use these exact point sizes -- never a local tweak like `fs_annot - 0.5`, which would
# make one panel's text a different size from its neighbour's.
# ---------------------------------------------------------------------------------------------
FONT = "Arial"                                  # pinned, no silent fallback to a different face
FS = dict(
    title=9,        # panel title / suptitle
    label=8,        # axis labels
    tick=8,         # tick labels, model names
    annot=7,        # in-plot numbers and keys
    legend=7,       # legend entries
    caption=6.5,    # figure caption (screen review only; the paper caption is LaTeX)
    panel_tag=10,   # the "a", "b", "c" panel letters
)

# WIDTHS. The paper is set on A4 (210 mm) with 20 mm margins, so the text block is 170 mm =
# 6.69 in. `col2` IS that text block: every full-width figure uses it, so figures arrive at 1:1
# scale in LaTeX (\includegraphics[width=\textwidth]) with no downscaling, and font sizes on the
# page are exactly the point sizes set in FS below. Do not hard-code a width in a figure script.
A4_TEXT = 6.69                                  # 170 mm text block on A4
STYLE = dict(
    col1=3.25, col15=4.75, col2=A4_TEXT,       # single, 1.5, and full text-block widths (inches)
    lw=1.2, lw_thin=0.7, marker_size=5.0, cap_size=2.0,
    dpi_screen=120, dpi_save=300,
    grid="#A6A6A6", ink="#000000", mute="#000000", faint="#E6E6E6",
    **{f"fs_{k}": v for k, v in FS.items()},   # STYLE["fs_title"] etc. stay available
)


def install():
    mpl.rcParams.update({
        "figure.dpi": STYLE["dpi_screen"], "savefig.dpi": STYLE["dpi_save"],
        "figure.facecolor": "white", "savefig.facecolor": "white",
        "savefig.bbox": "tight", "savefig.pad_inches": 0.04,
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "font.family": "sans-serif",
        "font.sans-serif": [FONT],
        "font.size": FS["tick"],
        "axes.titlesize": FS["title"], "axes.labelsize": FS["label"],
        "xtick.labelsize": FS["tick"], "ytick.labelsize": FS["tick"],
        "legend.fontsize": FS["legend"],
        "mathtext.default": "regular", "mathtext.fontset": "custom",
        "mathtext.rm": FONT, "mathtext.it": FONT, "mathtext.bf": FONT,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": 0.8, "axes.edgecolor": "#000000", "axes.labelcolor": "#000000",
        "axes.titlepad": 5.0,
        "xtick.direction": "out", "ytick.direction": "out",
        "xtick.major.size": 3.0, "ytick.major.size": 3.0,
        "xtick.major.width": 0.8, "ytick.major.width": 0.8,
        "xtick.color": "#000000", "ytick.color": "#000000",
        "text.color": "#000000",
        "xtick.minor.visible": False, "ytick.minor.visible": False,
        "lines.linewidth": STYLE["lw"], "lines.markersize": STYLE["marker_size"],
        "hatch.linewidth": 0.35,          # fine dots, not fat blobs
        # grid.alpha was 0.30 on top of a #C8C8C8 grid, i.e. an effective ~#EFEFEF -- the gridlines
        # were essentially invisible in print. Alpha is now 1.0 and the colour carries the
        # lightness, so what you set is what you get.
        "axes.grid": False, "grid.color": STYLE["grid"], "grid.linewidth": 0.6, "grid.alpha": 1.0,
        "axes.axisbelow": True,
        "legend.frameon": False, "legend.handlelength": 1.3,
        "legend.columnspacing": 1.0, "legend.labelspacing": 0.35,
    })
    OUTDIR.mkdir(exist_ok=True)


def _pdf_width_in(path):
    """Width of a saved PDF's media box, in inches (None if it cannot be parsed)."""
    import re
    m = re.search(rb"/MediaBox\s*\[\s*([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)",
                  path.read_bytes()[:4000])
    return (float(m.group(3)) - float(m.group(1))) / 72 if m else None


def save(fig, name, formats=("png", "pdf"), subdir=None, wide=False):
    """Save to figures_v2/<name>.<ext>. Returns the PNG path.

    savefig uses bbox_inches="tight", so the width actually written is NOT the figsize width: it
    shrinks when a figure has slack margins and GROWS when anything (a legend anchored outside the
    axes, a suptitle above the canvas) sits beyond the canvas. Two figures authored at the same
    width can therefore land 1.5in apart, and LaTeX then scales them differently at
    \includegraphics[width=\textwidth] -- so their fonts print at different sizes even though every
    script sets the same points. This check makes that loud instead of silent."""
    # `subdir` is for COMPONENT panels that are assembled into another figure (fig_C1/C2/D ->
    # fig_C_D). They are still rendered standalone for review, but they are not paper figures, so
    # they are kept out of figures_v2/ proper -- that folder should hold only what goes in the
    # paper.
    out = OUTDIR / subdir if subdir else OUTDIR
    out.mkdir(parents=True, exist_ok=True)
    for ext in formats:
        fig.savefig(out / f"{name}.{ext}")
    rel = f"figures_v2/{subdir}/{name}" if subdir else f"figures_v2/{name}"
    print(f"  saved  {rel}." + "/".join(formats))
    if "pdf" in formats:
        w = _pdf_width_in(out / f"{name}.pdf")
        if wide:
            print(f"  (wide figure: {w:.2f}in — set landscape/full-bleed on purpose, "
                  f"not scaled to the {A4_TEXT:.2f}in text block)")
        elif w is not None and abs(w - A4_TEXT) / A4_TEXT > 0.05:
            print(f"  WARNING  {name}: rendered {w:.2f}in vs page width {A4_TEXT:.2f}in "
                  f"({(w / A4_TEXT - 1) * 100:+.0f}%) -- fonts will not match the rest of the set")
    return out / f"{name}.png"


def title(target, text, pad=6, **kw):
    """Title a figure. Pass an Axes for a single-panel figure -- an axes title sits directly above
    the plot, whereas a suptitle floats at the top of the canvas and leaves a band of white space.
    Pass a Figure only for multi-panel layouts."""
    if hasattr(target, "set_title"):
        target.set_title(text, fontsize=FS["title"], fontweight="bold", color=STYLE["ink"],
                         pad=pad, **kw)
    else:
        target.suptitle(text, fontsize=FS["title"], fontweight="bold", color=STYLE["ink"], **kw)


# NOTE: there is deliberately no caption() helper. Captions are NEVER drawn into the figure --
# they belong in the LaTeX \caption{}. Document a figure in the script's docstring instead.


def check_font():
    """Fail loudly if the pinned font is missing -- a silent fallback would change every figure."""
    from matplotlib import font_manager as fm
    if FONT not in {f.name for f in fm.fontManager.ttflist}:
        raise RuntimeError(f"font {FONT!r} not installed; figures would silently use a different face")


install()
