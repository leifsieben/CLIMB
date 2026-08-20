"""SI Fig a — do you need to train the models end-to-end on the downstream data?

ONE script, ONE figure: figures_v2/SI_fig_a.png / .pdf

The same pretrained encoder used two ways at FULL downstream data: frozen (encoder fixed, probe
trained on the labels) versus end-to-end (whole network fine-tuned). Two encoders, `unsupervised`
and `supervised, dense`, on each canonical panel.

THE ANSWER IS MOSTLY YES, but it is not universal. All six panels are populated as of 2026-08-20.
End-to-end wins 10 of the 12 encoder x panel cells and clears the combined SD in 8. The largest
gains are QM7 unsupervised (-12.0 RMSE), HIV supervised (+0.050 NEF1) and Tox21 supervised
(+0.039). Both exceptions involve the SUPERVISED encoder -- BACE (-0.013) and QM7 (-3.3 RMSE) --
where freezing is as good or better. Read together: end-to-end training mostly buys back what a
weak pretraining objective failed to learn, and buys least where the frozen features were already
good.

HIV WAS EMPTY UNTIL 2026-08-20 AND THE HOLE WAS IN THIS BUILDER, NOT IN THE DATA. Its end-to-end
runs are 5-fold CV in climb_v2_phase2 (the mainline wave), and build_SI_fig_a_table.py had only a
label-efficiency branch for MolNet panels plus a mainline branch listing MoleculeACE and Ames by
name. HIV belonged to neither list, so the panel printed "end2end not run" while the runs existed.
That is worth recording because the figure said something false about the experiment, in a way
that reads as a candid admission rather than a bug.

CBS is computed by the builder and NOT drawn -- it is not in the canonical panel set. Its rows are
left in the table on purpose (its +0.125 NEF1 is the largest single gain anywhere here, and CBS may
return as an SI panel). Counts above are over the DRAWN cells; this paragraph once said "9 of 12"
by counting CBS as a panel the reader could see.

So end-to-end fine-tuning is the better default, but the frozen probe is not far behind on several
panels, and it is the cheaper option by far (SI Fig c). SI Fig e shows how this trade depends on
how many labels you have: the frozen probe's advantage lives in the small-data regime.

Error bars are +-1 SD of that panel's replicate unit, and each panel's frozen and end2end numbers
come from the SAME wave, split and seed grid, so the within-panel comparison is like-for-like.

PROTOCOL WARNING — the protocol DIFFERS BETWEEN PANELS (MoleculeACE/Ames use the mainline wave;
BACE/Tox21/QM7 the label-efficiency wave at its 100% fraction). Compare frozen vs end2end WITHIN a
panel; never compare a value in one panel against a value in another.

HIV IS NOW POPULATED. Both encoders' end-to-end HIV runs (5-fold CV, 3 pretraining seeds each)
landed 2026-08-19 and this figure reads them from climb_v2_phase2. Its error bars are the
pretraining-seed spread on BOTH ends of the slope: mainline_8M.csv also offers sd_total for HIV,
but that is over 15 cells (3 seeds x 5 folds) and reads 0.104 against the end2end side's 0.012,
so pairing them would put a fold-spread bar opposite a seed-spread bar on one slope.

Data: figure_data/SI_fig_a/SI_fig_a_e2e_need.csv, built by scripts/build_SI_fig_a_table.py.

Run:  python3 scripts/build_SI_fig_a_table.py && python3 -m figures.SI_fig_a
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D

from figures.style import STYLE, FS, save, check_font, mark_empty
from figures.arms import ARMS, PANELS, PANEL_ORDER
from figures.sixpanel import ROOT

check_font()
INK = "#000000"

DF = pd.read_csv(ROOT / "figure_data" / "SI_fig_a" / "SI_fig_a_e2e_need.csv")

# A SLOPE plot, not bars (user request 2026-08-17): each encoder is ONE line joining its frozen
# value to its end2end value, so the only thing the reader has to judge is the DIRECTION and
# steepness of the change — which is the whole question — rather than comparing four bar heights.
# Colour = encoder family: blue = unsupervised, red = supervised, dense.
# (join key, arms.py key, legend label) -- all three from arms.py, never a literal. The join key
# must be byte-identical to what build_SI_fig_a_table.py wrote into the `encoder` column, and the
# two were separate literals until 2026-08-20, when arms.py's rename of sup_dense to "supervised,
# desc" left this file asking for "supervised, dense". The join matched nothing and the supervised
# line disappeared from every panel without any check firing.
SERIES = [(ARMS["unsup"]["label"],     "unsup",     ARMS["unsup"]["label"]),
          (ARMS["sup_dense"]["label"], "sup_dense", ARMS["sup_dense"]["label"]),
          # The external comparator, added 2026-08-20. Both of its ends come from the MAINLINE
          # wave because CheMeleon is not in the label-efficiency wave at all, so on the three
          # label-efficiency panels its LEVEL is not comparable to the CLIMB lines beside it even
          # though its own slope is valid. Those panels draw it DASHED -- see _ls_for below.
          ("CheMeleon", "chemeleon_frozen", "CheMeleon")]
PROBES = ["frozen", "end2end"]

# FAIL LOUDLY ON A DEAD JOIN KEY. A series whose key is absent from the table draws nothing and
# says nothing -- the panels stay populated by the other series, so no empty-panel check fires.
# Both sides now come from arms.py, so this should be unreachable; it is here because it was
# reachable once and cost the supervised encoder's line across all six panels.
_missing = [k for k, _, _ in SERIES if k not in set(DF.encoder)] if len(DF) else []
assert not _missing, (f"SI fig a: series key(s) {_missing} are not in the table's encoder column "
                      f"{sorted(set(DF.encoder))} -- rebuild with scripts/build_SI_fig_a_table.py")

# The classical anchor as a reference line, ON EVERY PANEL, and PROTOCOL-MATCHED PER PANEL.
#
# This figure is built from two waves and the anchor is now measured in both, so each panel's line
# comes from the same wave as its points:
#   MoleculeACE, Ames      mainline wave           -> six_panel/mainline_8M.csv
#   BACE, Tox21, QM7, HIV  label-efficiency 100%   -> analysis/rigor/label_efficiency_fp_desc_anchor_summary.csv
#
# It is worth recording why this mattered, because the mismatched version was drawn first and
# looked entirely plausible. The two waves disagree by far more than the models do: the SAME
# ECFP4+desc features through the SAME XGBoost read 0.8712 on BACE in the mainline wave and 0.7836
# under label-efficiency -- 8.8 points, larger than the entire spread between arms on that panel.
# A mainline line on a label-efficiency panel therefore drew the anchor roughly a protocol above
# where it belongs, and every CLIMB-to-anchor gap read off those four panels was measuring the
# wave.
#
# THE LABEL-EFFICIENCY ANCHOR IS CORRECT, AND THE 8.8-POINT GAP TO MAINLINE IS THE SPLIT.
# Verified by re-running it through eval_v2's own single-hold-out path, which reproduces BACE
# 0.7836 to four decimals from the same features and the same head. There is no defect in the
# label-efficiency classical path.
#
# What the two waves differ in is the SPLIT CONSTRUCTION, and only that: mainline is scaffold
# 5-fold CV, label-efficiency is a single scaffold hold-out. Training-set size is identical (1,210).
# The single hold-out is markedly harder, and it is harder FOR THE FINGERPRINT SPECIFICALLY:
#
#   anchor  per CV fold 0.8419-0.8891   single hold-out 0.7836   BELOW its worst fold
#   unsup   per CV fold 0.8186-0.8810   single hold-out 0.8251   INSIDE its fold range
#
# So on that split the frozen encoders do beat the classical anchor on BACE (0.8251 vs 0.7836),
# while under 5-fold CV the anchor wins 4 folds of 5 -- losing fold0 by 0.002. Both are true of
# their own protocol. The mechanism is plausible rather than established: a held-out scaffold
# group punishes substructure-matching features more than a learned embedding, so an extrapolative
# split costs the fingerprint more.
#
# QUOTE IT WITH THE PROTOCOL ATTACHED, AND NOTE n=1. The hold-out result rests on ONE split of
# ~303 test molecules; at fraction=1.0 label_eff_fractions uses a single subsample seed, so there
# is no split-to-split spread behind it. It is enough to say the verdict is protocol-dependent; it
# is NOT enough to say the encoders beat the anchor on BACE full stop.
#
# An earlier note here called this anchor a 5.2-fold-SD outlier and retracted the encoder result on
# that basis. That reasoning was wrong: it compared a single-hold-out value against the spread of
# 5-fold CV cells, which are different estimands, so the SD it was measured against did not apply.
#
# METRIC IS MATCHED EXPLICITLY, not positionally: the anchor summary carries BOTH roc_auc and nef1
# for BACE/Tox21/HIV, so a positional read would silently take whichever sorted first. HIV's line
# is its nef1 (0.6190), which is quantised -- zero spread across three seeds -- and BACE's nef1 is
# pinned at 1.0 and is not plotted anywhere. That is small-active-count quantisation rather than a
# bug, and it is only safe here because a reference LINE needs a level and not an interval.
ANCHOR_ARM = "ecfp_desc"
LABELEFF_ANCHOR = ROOT / "analysis" / "rigor" / "label_efficiency_fp_desc_anchor_summary.csv"


def _anchor_values(protocols):
    """{panel: (value, source)} for the classical anchor, matched to each panel's own wave."""
    import csv as _csv
    out = {}
    main = ROOT / "figure_data" / "six_panel" / "mainline_8M.csv"
    if main.exists():
        for r in _csv.DictReader(main.open()):
            if r["arm"] == ANCHOR_ARM and r["value"] not in ("", "nan"):
                out[r["panel"]] = (float(r["value"]), "mainline")
    if LABELEFF_ANCHOR.exists():
        for r in _csv.DictReader(LABELEFF_ANCHOR.open()):
            p = r["task"]
            if r["split"] != "test" or p not in PANELS:
                continue
            # match the metric the PANEL plots; both roc_auc and nef1 are present for some tasks
            if r["metric"] != PANELS[p]["metric"]:
                continue
            # ONLY panels whose points are known to be label-efficiency. A panel with no points
            # has no protocol to match and MUST NOT be guessed: HIV was briefly pinned to the
            # label-efficiency anchor on the assumption its end2end run belonged to that wave, and
            # it does not. scripts/hiv_e2e_molnet_run.sh calls evaluate_finetuned(cv_folds=5),
            # i.e. scaffold 5-fold -- the MAINLINE protocol -- while label_eff_fractions.py says in
            # its own docstring that it subsamples "a single-hold-out train split". HIV's matched
            # anchor is therefore mainline 0.7373, not 0.6190.
            #
            # The trap that made the wrong guess look right: LE HIV n_train is 32,896 and scaffold
            # 5-fold train is 4/5 x 41,127 = 32,902. The training sizes agree to 0.02%, so "same
            # n_train" reads as confirmation that the protocols match. Same size, different split
            # construction. Match on how the split was BUILT, never on how big it is.
            if str(protocols.get(p, "")).startswith("label-efficiency"):
                out[p] = (float(r["mean"]), "label-efficiency")
    return out
# "end2end" spelled out (user 2026-08-19: "e2e that is not commonly understood"). It does not fit
# horizontally under a ~1.1in panel, so the x tick labels are rotated instead of abbreviated --
# shortening to jargon to win space is the wrong trade.
XTICKS = ["frozen", "end2end"]


def main():
    # PROTO IS THE PANEL'S OWN PROTOCOL, AND IT MUST COME FROM THE CLIMB ROWS ONLY.
    #
    # This was `{r.panel: str(r.protocol) for r in DF.itertuples()}` -- last row per panel wins --
    # which was fine while every row in a panel shared a protocol. Adding the CheMeleon series
    # broke it silently in the worst possible way: CheMeleon's rows are all mainline and are
    # appended last, so every panel's protocol flipped to "mainline" and the anchor resolver below
    # started drawing the MAINLINE anchor on the three label-efficiency panels. That is precisely
    # the defect this file's own docstring spends twenty lines on (BACE reads 0.8712 mainline
    # against 0.7836 label-efficiency -- 8.8 points, larger than the spread between arms).
    #
    # The external comparator is excluded by FAMILY from arms.py, not by name.
    _external = {lab for lab, key, _ in SERIES if ARMS[key]["family"] == "chemeleon"}
    _own = DF[~DF.encoder.isin(_external)] if len(DF) else DF
    PROTO = {}
    for panel, g in (_own.groupby("panel") if len(_own) else []):
        seen = sorted(set(g.protocol.astype(str)))
        assert len(seen) == 1, f"SI fig a: panel {panel} mixes protocols {seen} in its CLIMB rows"
        PROTO[panel] = seen[0]
    ANCHOR = _anchor_values(PROTO)
    # 2x3 at FULL page width. One row of six was tried and reverted (user 2026-08-19: "too
    # extreme... they become super distorted") -- six panels across 6.69in leaves ~1.05in
    # each, taller than they are wide, which squashes the curves. 2x3 gives ~2.0in panels.
    # The height saving comes from tighter spacing and ONE shared x-axis label instead of
    # six, not from collapsing the grid. Width is ~3.5% over col2 because savefig("tight")
    # trims back to about the text block.
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

        # Anchor first, so it is drawn even on panels with no encoder data (HIV) -- there the line
        # is the only content and it is what makes the empty panel worth printing.
        av = ANCHOR.get(p, (None, None))[0]
        if av is not None:
            ax.axhline(av, color=ARMS[ANCHOR_ARM]["color"], ls=":", lw=1.3, zorder=2)

        if g_all.empty:
            # Short enough to sit INSIDE a ~1.1in panel. The long form overran into both
            # neighbours' y-axis labels once the grid went to one row.
            ax.set_ylabel("")
            ax.text(0.5, 0.5, "end2end\nnot run", transform=ax.transAxes,
                    ha="center", va="center", fontsize=FS["annot"] - 0.5, color=INK)
            ax.set_xticks([])
            ax.set_yticks([])
            # DECLARED empty, so style.check_no_empty_panels passes it and fails on any panel that
            # is empty by accident instead. As of 2026-08-19 this fires for HIV only, and that is a
            # real hole, not a resolver bug: unsup_8M_e2e and skip_dense_8M_e2e are suite-track-only
            # runs with no MolNet summary at all, so the cell needs an end-to-end fine-tune on HIV
            # (peer session, needs a GPU box). The placeholder text says so on the figure.
            mark_empty(ax, f"{p}: no end2end run of a pretrained encoder on this panel")
            continue

        # A CROSS-PROTOCOL SERIES IS DASHED, and the rule is computed rather than listed: a series
        # whose own rows are all "mainline" while the panel's CLIMB points are label-efficiency is
        # measured on a different split, so its level cannot be read against theirs. Solid means
        # "same protocol as this panel"; dashed means "read the direction, not the height".
        def _ls_for(enc_label):
            """Solid when the series shares this panel's protocol, dashed when it does not."""
            own = set(g_all[g_all.encoder == enc_label].protocol.astype(str))
            panel_proto = str(PROTO.get(p, ""))
            return "-" if (not own or own == {panel_proto}) else (0, (4, 2))

        vals, errs = [], []
        for enc_label, arm_key, _ in SERIES:
            ys, es = [], []
            for probe in PROBES:
                r = g_all[(g_all.encoder == enc_label) & (g_all.probe == probe)]
                ys.append(float(r.value.iloc[0]) if len(r) else np.nan)
                e = float(pd.to_numeric(r.sd, errors="coerce").iloc[0]) if len(r) else 0.0
                es.append(0.0 if not np.isfinite(e) else e)
            colour = ARMS[arm_key]["color"]
            _ls = _ls_for(enc_label)
            if _ls != "-":
                _ls_kinds.append((p, enc_label))
            ax.errorbar([0, 1], ys, yerr=es, color=colour, lw=STYLE["lw"], marker="o",
                        ls=_ls,
                        ms=4.4, mec="white", mew=0.8, elinewidth=1.0, capsize=3.0,
                        capthick=1.1, ecolor=colour, zorder=3)
            vals += [v for v in ys if np.isfinite(v)]
            errs += es

        ax.set_xticks([0, 1])
        ax.set_xticklabels(XTICKS, fontsize=FS["annot"])
        ax.set_xlim(-0.32, 1.32)
        ax.xaxis.set_minor_locator(ticker.NullLocator())
        ax.tick_params(axis="x", which="minor", bottom=False)
        if av is not None:
            vals.append(av)
        lo, hi = min(vals) - max(errs), max(vals) + max(errs)
        pad = 0.22 * max(hi - lo, 1e-9)
        y0, y1 = lo - pad, hi + pad
        if d["metric"] == "roc_auc":
            y1 = min(y1, 1.0)
        ax.set_ylim(y0, y1)

    handles = [Line2D([], [], color=ARMS[k]["color"], marker="o", ms=5.0, lw=1.4, label=lab)
               for _, k, lab in SERIES]
    handles.append(Line2D([], [], color=ARMS[ANCHOR_ARM]["color"], ls=":", lw=1.3,
                          label="XGBoost, ECFP4+desc"))
    # The dashed style has to be decodable or it is just an inconsistency. Only added when some
    # panel actually draws it, so the key never describes something absent from the canvas.
    if any(_ls_kinds):
        handles.append(Line2D([], [], color=INK, ls=(0, (4, 2)), lw=1.3,
                              label="dashed = mainline wave on a label-efficiency panel;\n"
                                    "read its slope, not its height"))
    # WIDTH FIRST: spend the page's width on the legend before its height (user 2026-08-19).
    # A legend row costs every figure below it on the page; a legend column costs nothing
    # until it runs past the text block, and these entries do not.
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.068), ncol=3,
               fontsize=FS["legend"], handletextpad=0.5, labelspacing=0.3, columnspacing=1.4,
               borderpad=0.0, frameon=False, labelcolor=INK)
    # Legend sits one text-height under the tick labels; see the SI b/d/e note.
    fig.tight_layout(rect=(0, 0.088, 1, 1), w_pad=0.35)
    save(fig, "SI_fig_a")
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
            delta = sign * (float(ee.value.iloc[0]) - float(fr.value.iloc[0]))
            sd = np.hypot(pd.to_numeric(fr.sd, errors="coerce").iloc[0],
                          pd.to_numeric(ee.sd, errors="coerce").iloc[0])
            flag = "*" if np.isfinite(sd) and abs(delta) > sd else " "
            # the SERIES' own protocol, not the panel's -- they differ for the external
            # comparator on the label-efficiency panels, which is the whole reason it is dashed
            print(f"   {p:<12}{label:<20}{delta:>+10.4f}{flag}   ({fr.protocol.iloc[0]})")
    print("   * = |delta| exceeds the combined SD")


if __name__ == "__main__":
    main()
