from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D

from figures.style import STYLE, FS, save, check_font, mark_empty, row_ncol, LEGEND_BOX
from figures.arms import ARMS, PANELS, PANEL_ORDER, E2E_PAIRS, series_label
from figures.sixpanel import ROOT

check_font()
INK = "#000000"

DF = pd.read_csv(ROOT / "figure_data" / "SI_fig_a" / "SI_fig_a_e2e_need.csv")


SERIES = [(series_label(fr), fr, series_label(fr)) for fr, _ in E2E_PAIRS]
PROBES = ["frozen", "end2end"]


_missing = [k for k, _, _ in SERIES if k not in set(DF.encoder)] if len(DF) else []
assert not _missing, (f"SI fig a: series key(s) {_missing} are not in the table's encoder column "
                      f"{sorted(set(DF.encoder))} -- rebuild with scripts/build_SI_fig_a_table.py")


ANCHOR_ARM = "ecfp_desc"


def _anchor_values(protocols):
    """{panel: (value, source)} for the classical anchor, from the same wave as every point."""
    import csv as _csv
    del protocols
    out = {}
    main = ROOT / "figure_data" / "six_panel" / "mainline_8M.csv"
    if main.exists():
        for r in _csv.DictReader(main.open()):
            if (r["arm"] == ANCHOR_ARM and r["panel"] in PANELS
                    and r["metric"] == PANELS[r["panel"]]["metric"]
                    and r["value"] not in ("", "nan")):
                out[r["panel"]] = (float(r["value"]), "mainline")
    return out


XTICKS = ["frozen", "end2end"]


def main():


    _waves = sorted(set(DF.protocol.astype(str))) if len(DF) else []
    assert len(_waves) <= 1, (f"SI fig a: the table mixes protocols {_waves}. Every point in this "
                             f"figure must come from one wave -- see the builder's docstring.")
    PROTO = {p: (_waves[0] if _waves else "") for p in PANEL_ORDER}
    ANCHOR = _anchor_values(PROTO)


    _ls_kinds = []
    fig, axes = plt.subplots(2, 3, figsize=(STYLE["col2"] * 1.035, 3.3))
    for ax, p in zip(axes.ravel(), PANEL_ORDER):
        d = PANELS[p]
        g_all = DF[DF.panel == p]
        arrow = "↑" if d["higher_better"] else "↓"
        ax.set_title(f"{d['label']} {arrow}", fontsize=FS["title"], fontweight="bold",
                     color=INK, pad=4)
        ax.set_ylabel(d["metric_short"], fontsize=FS["annot"], color=INK)
        ax.grid(axis="y", ls=":", lw=0.6, color=STYLE["grid"])
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)


        av = ANCHOR.get(p, (None, None))[0]
        if av is not None:
            ax.axhline(av, color=ARMS[ANCHOR_ARM]["color"], ls=":", lw=1.3, zorder=2)

        if g_all.empty:


            ax.set_ylabel("")
            ax.text(0.5, 0.5, "end2end\nnot run", transform=ax.transAxes,
                    ha="center", va="center", fontsize=FS["annot"] - 0.5, color=INK)
            ax.set_xticks([])
            ax.set_yticks([])


            mark_empty(ax, f"{p}: no end2end run of a pretrained encoder on this panel")
            continue


        def _ls_for(enc_label):
            del enc_label
            return "-"


        vals, lohi = [], []
        for enc_label, arm_key, _ in SERIES:
            ys, dn, up = [], [], []
            for probe in PROBES:
                r = g_all[(g_all.encoder == enc_label) & (g_all.probe == probe)]
                if not len(r):
                    ys.append(np.nan); dn.append(0.0); up.append(0.0); continue
                v, lo_, hi_ = (float(r.value.iloc[0]), float(r.lo.iloc[0]), float(r.hi.iloc[0]))
                ys.append(v)
                dn.append(max(0.0, v - lo_))
                up.append(max(0.0, hi_ - v))
                lohi += [lo_, hi_]
            colour = ARMS[arm_key]["color"]
            _ls = _ls_for(enc_label)
            if _ls != "-":
                _ls_kinds.append((p, enc_label))
            ax.errorbar([0, 1], ys, yerr=[dn, up], color=colour, lw=STYLE["lw"], marker="o",
                        ls=_ls,
                        ms=4.4, mec="white", mew=0.8, elinewidth=1.0, capsize=3.0,
                        capthick=1.1, ecolor=colour, zorder=3)
            vals += [v for v in ys if np.isfinite(v)]

        ax.set_xticks([0, 1])
        ax.set_xticklabels(XTICKS, fontsize=FS["annot"])
        ax.set_xlim(-0.32, 1.32)
        ax.xaxis.set_minor_locator(ticker.NullLocator())
        ax.tick_params(axis="x", which="minor", bottom=False)
        if av is not None:
            lohi.append(av)
        lo, hi = min(lohi or vals), max(lohi or vals)
        pad = 0.22 * max(hi - lo, 1e-9)
        y0, y1 = lo - pad, hi + pad
        if d["metric"] == "roc_auc":
            y1 = min(y1, 1.0)
        ax.set_ylim(y0, y1)

    handles = [Line2D([], [], color=ARMS[k]["color"], marker="o", ms=5.0, lw=1.4, label=lab)
               for _, k, lab in SERIES]
    handles.append(Line2D([], [], color=ARMS[ANCHOR_ARM]["color"], ls=":", lw=1.3,
                          label="XGBoost, ECFP4+desc"))


    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.068), ncol=row_ncol(handles),
               fontsize=FS["legend"], handletextpad=0.5, labelspacing=0.3, columnspacing=1.4,
               borderpad=0.30, **LEGEND_BOX, labelcolor=INK)

    fig.tight_layout(rect=(0, 0.088, 1, 1), w_pad=0.35)
    save(fig, "figure_E1")
    plt.close(fig)

    print("\nSI Fig a — end2end minus frozen at full data (+ = end2end better):")
    for p in PANEL_ORDER:
        g = DF[DF.panel == p]
        if g.empty:
            print(f"   {p:<12} — no end2end run of a pretrained encoder")
            continue
        sign = 1 if g.higher_better.iloc[0] else -1
        for label, _, _ in SERIES:
            fr = g[(g.encoder == label) & (g.probe == "frozen")]
            ee = g[(g.encoder == label) & (g.probe == "end2end")]
            if not len(fr) or not len(ee):
                continue
            fr, ee = fr.iloc[0], ee.iloc[0]
            delta = sign * (float(ee.value) - float(fr.value))


            flag = "*" if (fr.hi < ee.lo or ee.hi < fr.lo) else " "
            print(f"   {p:<12}{label:<20}{delta:>+10.4f}{flag}   ({fr.protocol})")
    print("   * = the frozen and end2end intervals are disjoint")


if __name__ == "__main__":
    main()
