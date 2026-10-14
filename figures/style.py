from __future__ import annotations
from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.text as mtext

ROOT = Path(__file__).resolve().parent.parent
OUTDIR = ROOT / "plots"


FONT = "Arial"
FS = dict(
    title=9,
    label=8,
    tick=8,
    annot=7,
    legend=7,
    caption=6.5,
    panel_tag=10,
)


A4_TEXT = 6.69
STYLE = dict(
    col1=3.25, col15=4.75, col2=A4_TEXT,
    lw=1.2, lw_thin=0.7, marker_size=5.0, cap_size=2.0,
    dpi_screen=120, dpi_save=300,
    grid="#A6A6A6", ink="#000000", mute="#000000", faint="#E6E6E6",
    **{f"fs_{k}": v for k, v in FS.items()},
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
        "hatch.linewidth": 0.35,


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


LEGEND_BOX = dict(frameon=True, framealpha=1.0, edgecolor=STYLE["ink"], facecolor="white")


def row_ncol(handles, rows=1):

    n = len(list(handles))
    if n <= 0:
        return 1
    if rows <= 1:
        return n
    return -(-n // int(rows))


def mark_empty(ax, why="no data for this panel"):

    ax._climb_intentionally_empty = why


def _panel_has_data(ax):

    from matplotlib.transforms import BlendedGenericTransform
    for ln in ax.lines:
        if isinstance(ln.get_transform(), BlendedGenericTransform):
            continue
        if len(ln.get_xdata()):
            return True
    return bool(ax.patches or ax.collections or ax.images or ax.containers)


def check_no_empty_panels(fig, name):

    empty = [ax for ax in fig.axes
             if not getattr(ax, "_climb_intentionally_empty", None)
             and not _panel_has_data(ax)
             and ax.get_label() != "<colorbar>"]
    if empty:
        where = ", ".join(ax.get_title() or f"axes at {ax.get_position().bounds}" for ax in empty)
        raise AssertionError(
            f"{name}: {len(empty)} panel(s) drew no data -- {where}. If that is intended, call "
            f"figures.style.mark_empty(ax, why) at the draw site; otherwise the reader is "
            f"resolving the wrong key (see the 2026-08-19 HIV panel bugs).")


def check_text_is_ink(fig, name):

    import matplotlib.colors as mcolors
    bad = []
    for t in fig.findobj(mtext.Text):
        if not t.get_visible() or not (t.get_text() or "").strip():
            continue
        r, g, b, a = mcolors.to_rgba(t.get_color())
        if a == 0:
            continue
        if (r, g, b) in ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)):
            continue
        bad.append((t.get_text()[:40].replace("\n", " "), mcolors.to_hex((r, g, b))))
    assert not bad, (
        f"{name}: text drawn in a non-ink colour -- grey text is forbidden, use INK and "
        f"carry de-emphasis with italics or size instead. Offenders: "
        + "; ".join(f"{txt!r} at {hexc}" for txt, hexc in bad[:8])
        + (f" (+{len(bad) - 8} more)" if len(bad) > 8 else ""))


def save(fig, name, formats=("png", "pdf"), subdir=None, wide=False):


    check_no_empty_panels(fig, name)
    check_text_is_ink(fig, name)
    out = OUTDIR / subdir if subdir else OUTDIR
    out.mkdir(parents=True, exist_ok=True)
    for ext in formats:
        fig.savefig(out / f"{name}.{ext}")
    rel = f"plots/{subdir}/{name}" if subdir else f"plots/{name}"
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

    if hasattr(target, "set_title"):
        target.set_title(text, fontsize=FS["title"], fontweight="bold", color=STYLE["ink"],
                         pad=pad, **kw)
    else:
        target.suptitle(text, fontsize=FS["title"], fontweight="bold", color=STYLE["ink"], **kw)


def check_font():
    """Fail loudly if the pinned font is missing -- a silent fallback would change every figure."""
    from matplotlib import font_manager as fm
    if FONT not in {f.name for f in fm.fontManager.ttflist}:
        raise RuntimeError(f"font {FONT!r} not installed; figures would silently use a different face")


install()
