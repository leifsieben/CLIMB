from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

from figures.style import STYLE, FS, save, check_font, row_ncol, LEGEND_BOX
from figures.arms import ARMS, SHADES

check_font()
ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "figure_data" / "embedding_resolution" / "separability_auc.csv"
INK = "#000000"
TINT = "#F0EDE6"


SERIES = [("ECFP",      ARMS["ecfp"]["color"],      "ECFP4"),
          ("ECFP+d",    ARMS["ecfp_desc"]["color"], "ECFP4+desc"),
          ("uns-ENUM",  SHADES["unsup"][0],         "CLIMB unsup., augmented"),
          ("uns-CANON", SHADES["unsup"][2],         "CLIMB unsup., canonical"),
          ("sup",       ARMS["sup_dense"]["color"], "CLIMB supervised"),


          ("rand",      ARMS["random_encoder"]["color"], "no pretrain, random")]


MODES = [("A", "stereo_flip",         "Inverted\nstereocentre"),
         ("A", "ez_flip",             "Flipped E/Z\ndouble bond"),
         ("A", "c_to_n",              "Aromatic C→N\n(benzene→pyridine)"),
         ("A", "add_methyl",          "One methyl\nadded"),
         ("A", "ch2_homologue",       "One CH$_2$ inserted\n(homologue)"),
         ("A", "halogen_swap",        "Cl→F, skeleton\nunchanged"),
         ("A", "add_fluorine",        "One fluorine\nadded"),
         ("A", "isotope_13c",         "$^{12}$C→$^{13}$C,\ngraph unchanged"),
         ("A", "ring_size",           "Cyclopentyl ↔\ncyclohexyl"),
         ("A", "regioisomer",         "para vs meta\nsubstitution"),
         ("B", "smiles_enumeration",  "Re-written\nSMILES"),
         ("B", "kekule",              "Kekulé\nform"),
         ("B", "symmetry_equivalent", "Equivalent\npositions")]


NOTATION = "notation"


INPUT_OF = {"A": "canonical", "B": "as_written"}


def compute():

    raw = pd.read_csv(SRC)
    arms = [a for a, _, _ in SERIES]
    keep = []
    for kl, want in INPUT_OF.items():
        sel = raw[(raw["klass"] == kl) & (raw["input"] == want)
                  & (raw["short"].isin(arms + [NOTATION]))]
        want_n = len(set(raw.loc[raw["klass"] == kl, "mode"])) * (len(arms) + 1)
        assert len(sel) == want_n, (
            f"class {kl} on {want!r} input: {len(sel)} rows, expected {want_n}. Class A is a "
            f"chemistry question (canonical); class B IS the notation question (as written). "
            f"Drawing either on the other's input inverts the figure.")
        keep.append(sel)
    d = pd.concat(keep, ignore_index=True)
    missing = set(arms + [NOTATION]) - set(d["short"])
    assert not missing, f"fig_G: arms missing from {SRC.name}: {sorted(missing)}"
    n = sorted(set(d["n_pairs"]))
    assert n and min(n) >= 900, f"fig_G: expected ~1000 pairs per cell, found n in {n[:5]}"
    return {c: d.pivot_table(index=["klass", "mode"], columns="short", values=c)
            for c in ("auc_mean", "auc_sd", "n_degenerate", "n_pairs")}


def _row(frame, kl, mode):
    """One panel's bar values, in SERIES order, NaN where the cell is absent."""
    return np.array([frame.loc[(kl, mode), lab] if (kl, mode) in frame.index else np.nan
                     for lab, _, _ in SERIES], dtype=float)


CHANCE = 0.5


YMAX = 1.035


SUBTICK = 0.006


def _panel(ax, vals, sds, floor, title, klass):

    if klass == "B":
        ax.set_facecolor(TINT)
    x = np.arange(len(SERIES))
    v = np.asarray(vals, dtype=float)
    drawn = np.clip(v, CHANCE, None)
    ax.bar(x, drawn - CHANCE, bottom=CHANCE, width=0.80,
           color=[c for _, c, _ in SERIES], edgecolor=INK, linewidth=0.45, zorder=3)

    for xi, val in zip(x, v):
        if np.isfinite(val) and val <= CHANCE:
            ax.bar([xi], [SUBTICK], bottom=CHANCE, width=0.80, color=INK, lw=0, zorder=4)


    err = np.asarray(sds, dtype=float)


    ok = np.isfinite(v) & np.isfinite(err) & (v > CHANCE)
    if ok.any():
        ax.errorbar(x[ok], v[ok], yerr=err[ok], fmt="none", ecolor=INK, elinewidth=0.55,
                    capsize=1.3, capthick=0.55, zorder=5)
    if np.isfinite(floor):
        ax.axhline(floor, color=INK, ls=(0, (2, 1.6)), lw=0.8, zorder=6)
    ax.set_ylim(CHANCE, YMAX)
    ax.set_yticks([0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.5", "", "1"])
    ax.set_xticks([])
    ax.set_xlim(-0.70, len(SERIES) - 0.30)
    ax.grid(axis="y", ls=":", lw=0.6, color=STYLE["grid"])
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(axis="y", labelsize=FS["annot"] - 1)
    ax.set_title(title, fontsize=FS["annot"], fontweight="bold", color=INK, pad=3, loc="center")


def _legend_handles():
    """Shared with fig_G so the two figures cannot drift in arm order, colour, or label."""


    h = [Patch(facecolor=c, edgecolor=INK, lw=0.6, label=lab) for _, c, lab in SERIES]


    h.append(Line2D([], [], color=INK, ls=(0, (2, 1.6)), lw=0.9,
                    label="notation floor (char n-grams)"))
    return h


def report(R, modes, heading):
    """The printed table. Same resolution path as the bars, so the console cannot disagree."""
    print(f"\n{heading}\n")
    print(f"   {'mode':<22}{'floor':>8}" + "".join(f"{lab:>15s}" for lab, _, _ in SERIES))
    for kl, mode, _ in modes:
        row = f"   {mode:<22}"
        auc, sd, dg, npr = (_row(R[c], kl, mode) for c in
                            ("auc_mean", "auc_sd", "n_degenerate", "n_pairs"))
        fl = R["auc_mean"].loc[(kl, mode), NOTATION] if (kl, mode) in R["auc_mean"].index else np.nan
        row += f"{fl:>8.3f}" if np.isfinite(fl) else f"{'—':>8}"
        for k in range(len(SERIES)):
            if not np.isfinite(auc[k]):
                row += f"{'—':>15}"
                continue
            mark = "0" if npr[k] and dg[k] / npr[k] >= 0.5 else " "
            row += f"{auc[k]:>8.3f}±{sd[k]:.3f} {mark}"
        print(row)
    print("\n   held-out ROC-AUC, mean ± 1 SD over 5 split seeds. 0.500 = the edit leaves no")
    print("   signature a tree can learn; 1.000 = every held-out pair separated.")
    print("   'floor' = character n-grams of the SMILES, no chemistry: available to any string")
    print("   model for free, so a bar below it carries less than the raw text does.")
    print("   0 marks a cell where the arm maps BOTH members of most pairs to the same vector —")
    print("   it cannot possibly differ, which is 'not resolved' in its strongest form.")


NCOL_A = 5
NCOL = NCOL_A + 1


CONTROLS = ["smiles_enumeration", "kekule"]


def main():
    R = compute()
    A = [m for m in MODES if m[0] == "A"]
    assert len(A) == 10, f"fig_G expects the ten class-A modes, found {[m[1] for m in A]}"
    B = [m for m in MODES if m[0] == "B" and m[1] in CONTROLS]
    B.sort(key=lambda m: CONTROLS.index(m[1]))
    assert len(B) == 2, f"fig_G expects the two class-B controls, found {[m[1] for m in B]}"

    fig = plt.figure(figsize=(STYLE["col2"], 3.85))
    gs = fig.add_gridspec(2, NCOL, left=0.068, right=0.995, top=0.875, bottom=0.250,
                          wspace=0.38, hspace=0.72)
    tags = "abcdefghij"
    for i, (kl, mode, title) in enumerate(A):
        ax = fig.add_subplot(gs[i // NCOL_A, i % NCOL_A])
        auc, sd = (_row(R[c], kl, mode) for c in ("auc_mean", "auc_sd"))
        floor = R["auc_mean"].loc[(kl, mode), NOTATION]
        assert np.isfinite(auc).any(), f"fig_G: no data for {mode}"
        _panel(ax, auc, sd, floor, title, kl)
        ax.text(0.0, 1.30, tags[i], transform=ax.transAxes, fontsize=FS["panel_tag"],
                fontweight="bold", va="bottom", ha="left", color=INK)
        if i % NCOL_A == 0:
            ax.set_ylabel("can a tree tell the pair\napart?  (held-out AUC)",
                          fontsize=FS["annot"])


    for j, (kl, mode, title) in enumerate(B):
        ax = fig.add_subplot(gs[j, NCOL_A])
        auc, sd = (_row(R[c], kl, mode) for c in ("auc_mean", "auc_sd"))
        floor = R["auc_mean"].loc[(kl, mode), NOTATION]
        assert np.isfinite(auc).any(), f"fig_G: no data for control {mode}"
        _panel(ax, auc, sd, floor, title, kl)
        ax.text(0.0, 1.30, "kl"[j], transform=ax.transAxes, fontsize=FS["panel_tag"],
                fontweight="bold", va="bottom", ha="left", color=INK)


    _h = _legend_handles()
    fig.legend(handles=_h, loc="upper center", bbox_to_anchor=(0.500, 0.212),
               ncol=row_ncol(_h, rows=2), fontsize=FS["legend"], handletextpad=0.4,
               columnspacing=1.0, labelspacing=0.35, borderpad=0.30, **LEGEND_BOX)
    save(fig, "figure_6")
    plt.close(fig)

    report(R, A + B, "Fig G — class A (canonical) plus the two class-B controls (as written)")


if __name__ == "__main__":
    main()
