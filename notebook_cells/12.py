# ---------- A2 · scaling ladders, on the 5-fold CV only, on BOTH scaling axes ----------
# A2.a x = forward passes (compute). A2.b x = UNIQUE MOLECULES actually seen. They are the same
# runs and the same y-values; only the x-coordinate changes -- and the two tell different stories,
# because past one epoch of a stream extra forward passes buy no new molecules. The sparse SFT
# recipes exhaust their capped pool at ~0.52M molecules, so in A2.b their ladders go VERTICAL:
# every rung above 2M FP is pure repetition. That is invisible on the compute axis.
#
# EVERY error bar here is the SAME quantity: the spread across the 5 scaffold folds. The single
# hold-out is not used and pretraining-seed replicates are deliberately ignored, because those
# replicates exist only at the 8M rung -- mixing them in would make the 8M interval mean something
# different from every other point on the same line.
A2_KEYS = ["unsup_only", "unsup2sup:dense",
           "sup_only:dense", "sup_only:sparse_all", "sup_only:dense_plus_sparse"]
# The classical anchors are compute-independent, so they are horizontal references on BOTH axes,
# not ladders. Greys (not their A1 bar colours) so they read as "the bar to clear" rather than as
# two more models competing with five coloured ladders.
# Only the two classical anchors. Both no_pretrain references (frozen floor and end-to-end) were
# drawn here too, which put FOUR grey/near-grey horizontal lines behind five coloured ladders --
# the panel became unreadable and the references stopped functioning as references. The
# no_pretrain arms are still in A1.a/A1.b, where they are bars and can be read properly.
A2_ANCHORS = [("ecfp4",   "#333333", (0, (6, 2)), "Morgan+XGBoost (ECFP4)"),
              ("fp_desc", "#B3B3B3", (0, (2, 2)), "Morgan+desc+XGBoost")]

def ladder_cv(task, key):
    """CV mean + across-fold sd per budget, seed-0 runs only. Empty frame if nothing is scored."""
    d = DF_CV[(DF_CV.wave == "climb_v2_phase2") & (DF_CV.task == task) & (DF_CV.seed == 0)]
    if key == "unsup_only":            s = d[d.regime == "unsup_only"]
    elif key.startswith("unsup2sup:"): s = d[(d.regime == "unsup2sup") & (d.recipe == key.split(":")[1])]
    else:                              s = d[(d.regime == "sup_only") & (d.recipe == key.split(":")[1])]
    s = s[~s.truncated]
    if not len(s): return s
    g = (s.groupby("budget_fp").agg(value=("value", "mean"), err=("std", "mean"))
          .reset_index().sort_values("budget_fp"))
    g["mols"] = [unique_molecules(key, b) for b in g.budget_fp]
    return g

def anchor_cv(task, key):
    """(mean, across-fold sd) for a compute-independent reference, or (nan, nan)."""
    s = DF_CV[(DF_CV.wave == "climb_v2_phase2") & (DF_CV.task == task) & (DF_CV.regime == key)]
    return (float(s.value.mean()), float(s["std"].mean())) if len(s) else (np.nan, np.nan)

# Rungs that exist in the CV eval. 96M was scored on the hold-out only, and unsup->sup has no 48M
# CV run, so both are absent here rather than silently interpolated. Stated in the caption.
BUDG = [2e6, 8e6, 24e6, 48e6]
N_RUNGS = len(BUDG)

def draw_A2(xcol, tag, xlabel, extra_note, fname):
    ncol = 2; nrow = int(np.ceil(len(CORE_TASKS) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(STYLE["col2"], 2.05 * nrow)); axes = axes.ravel()
    for i, (ax, task) in enumerate(zip(axes, CORE_TASKS)):
        drew = False; gaps = []
        for key in A2_KEYS:
            g = ladder_cv(task, key)
            if not len(g):
                gaps.append(f"{rc_label(key)}: no CV data"); continue
            drew = True
            if len(g) < N_RUNGS: gaps.append(f"{rc_label(key)}: {len(g)}/{N_RUNGS} rungs")
            ax.errorbar(g[xcol], g.value, yerr=g.err, color=rc_color(key), ls=rc_ls(key),
                        lw=STYLE["lw"], marker=rc_marker(key), ms=STYLE["marker_size"], mec="white",
                        capsize=2, elinewidth=0.8, zorder=3)
        for key, col, dsh, _lab in A2_ANCHORS:   # horizontal reference + its own fold-spread band
            v, e = anchor_cv(task, key)
            if not np.isfinite(v):
                gaps.append(f"{_lab}: no CV data"); continue
            ax.axhline(v, color=col, ls=dsh, lw=1.1, zorder=2)
            if np.isfinite(e): ax.axhspan(v - e, v + e, color=col, alpha=0.16, lw=0, zorder=0)
        if xcol == "budget_fp":
            set_fp_axis(ax, BUDG); ax.set_xlim(1.6e6, 6.0e7)
        else:
            ax.set_xscale("log"); ax.set_xlim(3.5e5, 1.6e7)
            ax.xaxis.set_major_locator(ticker.FixedLocator([5e5, 1e6, 2e6, 4e6, 8e6, 12e6]))
            ax.xaxis.set_major_formatter(ticker.FixedFormatter(
                ["0.5M", "1M", "2M", "4M", "8M", "12M"]))
            ax.xaxis.set_minor_locator(ticker.NullLocator()); ax.tick_params(axis="x", which="minor", bottom=False)
            # the corpus is only ~12M molecules, so this is where the unsupervised axis simply stops
            ax.axvline(UNSUP_CORPUS, color="#999", ls=(0, (1, 2)), lw=0.7, zorder=1)
        ax.set_title(ttitle(task, oneline=True), pad=6); ax.set_xlabel(xlabel)
        ax.set_ylabel(re.sub(r"\s*[↑↓]\s*$", "", mlabel(task)))   # arrow lives in the title
        label_all_yticks(ax)
        # A line that simply stops short is indistinguishable from a line that plateaued. Name the gaps.
        if gaps:
            ax.set_ylim(top=ax.get_ylim()[1] * (1.06 + 0.07 * int(np.ceil(len(gaps) / 2))))
            ax.text(0.02, 0.98, "incomplete:\n" + "\n".join(gaps), transform=ax.transAxes,
                    ha="left", va="top", fontsize=STYLE["fs_annot"] - 2, color="#B00020",
                    linespacing=1.2, bbox=dict(fc="white", ec="none", alpha=0.85, pad=1.0))
        if not drew: no_data_watermark(ax, "no CV-scored runs")
    for ax in axes[len(CORE_TASKS):]: ax.axis("off")

    handles = [plt.Line2D([], [], color=rc_color(k), marker=rc_marker(k), ls=rc_ls(k), label=rc_label(k))
               for k in A2_KEYS]
    handles += [plt.Line2D([], [], color=c, ls=d, lw=1.1, label=lab) for _, c, d, lab in A2_ANCHORS]
    if xcol == "mols":
        handles += [plt.Line2D([], [], color="#999", ls=(0, (1, 2)), lw=0.7,
                               label="corpus exhausted (12M molecules)")]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=3,
               fontsize=STYLE["fs_legend"], columnspacing=1.2)
    fig.suptitle(f"Fig {tag} - scaling of the primary regimes in {xlabel}",
                 fontsize=STYLE["fs_title"], y=1.075)
    note = ("pooled 5-fold scaffold CV ONLY  ·  error bars on the ladders and the shaded band "
            "around each classical anchor are both ±1 sd across the 5 folds "
            "(no head-seed, no pretraining-seed spread), so every interval means the same thing  ·  "
            "the no_pretrain references are omitted here to keep the panels readable; see A1.a/A1.b  ·  "
            "the 96M rung and unsup→sup's 48M rung are hold-out-only, so they are absent rather "
            "than interpolated  ·  " + extra_note)
    fig.text(0.5, 0.995, "\n".join(_tw.wrap(note, 120)), ha="center", va="top",
             fontsize=STYLE["fs_annot"] - 0.5, color="#666")
    fig.subplots_adjust(top=0.88, bottom=0.10, hspace=0.62, wspace=0.30)
    save_fig(fig, fname); plt.show()

draw_A2("budget_fp", "A2.a", "forward passes",
        "unsup→sup is at its TRUE total (MLM base + 2M-FP SFT)",
        "figA2a_scaling_forward_passes")
draw_A2("mols", "A2.b", "unique molecules seen",
        "x = unique molecules, so a ladder that runs VERTICAL is spending compute on repetition, "
        "not on new chemistry; the sparse SFT pool caps at ~0.52M and the unsupervised corpus at 12M",
        "figA2b_scaling_unique_molecules")

print(f"\nA2 change from the smallest to the largest CV-scored rung (per arm, mean over tasks):")
for key in A2_KEYS:
    spans = [100 * lift(g.value.iloc[-1], t, g.value.iloc[0])
             for t in CORE_TASKS for g in [ladder_cv(t, key)] if len(g) >= 2]
    if spans: print(f"   {rc_label(key):<30} {np.mean(spans):+6.1f}%   (n={len(spans)} tasks)")

print("\nunique molecules behind each rung (why A2.b differs from A2.a):")
for key in A2_KEYS:
    g = ladder_cv("ESOL", key)
    if len(g):
        print(f"   {rc_label(key):<30} " +
              "  ".join(f"{b/1e6:.0f}M FP→{m/1e6:.2f}M mol" for b, m in zip(g.budget_fp, g.mols)))