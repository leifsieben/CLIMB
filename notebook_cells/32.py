# ---------- SI · vocabulary-size scaling (wave climb_v2_vocab), on the 5-fold CV only ----------
# One MLM-only encoder per tokenizer at a matched 2M forward passes; the ONLY thing that moves is the
# tokenizer vocabulary (and, by construction, the embedding-param count that tracks it). Both panels
# read the tidy per-run CV summary the compute session shipped -- `mean`/`fold_std` are the mean and
# across-fold sd of the 3-head-seed-averaged prediction, i.e. the identical CV scheme used everywhere
# else in this notebook. x-axis is the ACTUAL (measured) vocab, not the nominal target.
_VOCAB = pd.read_csv(DATA_ROOT / "climb_v2_vocab" / "vocab_cv_summary.csv")
VOCAB_FAMILY = {                                   # BPE = main-paper family; Unigram = warm contrast
    "bpe":     dict(color=PALETTE["blue"],   marker="o", label="Byte-level BPE"),
    "unigram": dict(color=PALETTE["orange"], marker="s", label="Unigram-LM"),
}
# (task, metric-in-CSV, lower_is_better) -- HIV uses NEF1% (its virtual-screening readout), not ROC-AUC.
VOCAB_PANELS = [("ESOL", "rmse", True),  ("QM7", "rmse", True),  ("BBBP",  "roc_auc", False),
                ("BACE", "roc_auc", False), ("Tox21", "roc_auc", False), ("HIV", "nef1", False)]

def _vseries(family, task, metric):
    """(actual_vocab, mean, fold_std) sorted by vocab for one family/task/metric."""
    s = _VOCAB[(_VOCAB.family == family) & (_VOCAB.dataset == task) & (_VOCAB.metric == metric)]
    s = s.sort_values("actual_vocab")
    return s.actual_vocab.values.astype(float), s["mean"].values.astype(float), s.fold_std.values.astype(float)

def _vtitle(task, metric):
    arrow = "↓" if task in ("ESOL", "QM7") else "↑"
    pretty = {"rmse": "RMSE", "roc_auc": "ROC-AUC", "nef1": "NEF1%"}[metric]
    return f'{TASKS[task]["pretty"]} — {pretty} {arrow}'

# ----- one combined figure: (a) per-task scaling grid, (b) effect-size summary -----
# Stacked like figC1J1_I1_combined: rows 0-1 hold the six task panels (2x3), row 2 is the
# full-width effect-size summary. Both read the SAME data, so panel b can never disagree with a.
figV = plt.figure(figsize=(STYLE["col2"], 8.7))
_outer = figV.add_gridspec(3, 1, height_ratios=[1.0, 1.0, 0.86], hspace=0.55)
_g0 = _outer[0].subgridspec(1, 3, wspace=0.34); _g1 = _outer[1].subgridspec(1, 3, wspace=0.34)
_taxes = [figV.add_subplot(_g0[j]) for j in range(3)] + [figV.add_subplot(_g1[j]) for j in range(3)]
axE = figV.add_subplot(_outer[2])

# ----- (a) per-task metric vs actual vocab, BPE vs Unigram -----
for ax, (task, metric, lower) in zip(_taxes, VOCAB_PANELS):
    for fam, sty in VOCAB_FAMILY.items():
        x, y, e = _vseries(fam, task, metric)
        if not len(x): continue
        ax.errorbar(x, y, yerr=e, color=sty["color"], marker=sty["marker"], ms=STYLE["marker_size"],
                    lw=STYLE["lw"], capsize=STYLE["cap_size"], elinewidth=STYLE["lw_thin"], mec="white",
                    label=sty["label"], zorder=3)
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(ticker.FixedLocator([261, 1000, 3000, 12000]))
    ax.xaxis.set_major_formatter(ticker.FixedFormatter(["261", "1k", "3k", "12k"]))
    ax.xaxis.set_minor_locator(ticker.NullLocator()); ax.tick_params(axis="x", which="minor", bottom=False)
    ax.set_title(_vtitle(task, metric), pad=6); ax.set_xlabel("tokenizer vocab (actual, log)")
    ax.margins(x=0.10); label_all_yticks(ax)
_taxes[0].legend(loc="best", fontsize=STYLE["fs_legend"])

# ----- (b) largest-vocab change vs the vocab-261 baseline, in fold-std units -----
# One number per family/task: how far the LARGEST reachable vocab moves the metric away from the
# character-level (261) baseline, signed so + = better, divided by that task's fold std at the large
# vocab. |effect| < 1 (inside the shaded band) means "within evaluation noise".
_dodge = {"bpe": -0.16, "unigram": +0.16}
for fam, sty in VOCAB_FAMILY.items():
    for k, (task, metric, lower) in enumerate(VOCAB_PANELS):
        x, y, e = _vseries(fam, task, metric)
        if len(x) < 2: continue
        delta = (y[0] - y[-1]) if lower else (y[-1] - y[0])       # + = large vocab better than vocab-261
        denom = e[-1] if e[-1] > 0 else np.nan
        eff = delta / denom
        if not np.isfinite(eff): continue
        xp = k + _dodge[fam]
        axE.plot([xp, xp], [0, eff], color=sty["color"], lw=2.4, zorder=2, solid_capstyle="round")
        axE.plot([xp], [eff], color=sty["color"], marker=sty["marker"], ms=7, mec="white", zorder=3)
axE.axhspan(-1, 1, color="#888888", alpha=0.13, zorder=0)
axE.axhline(0, color=PALETTE["black"], lw=0.9)
axE.set_xticks(range(len(VOCAB_PANELS))); axE.set_xticklabels([p[0] for p in VOCAB_PANELS])
axE.set_ylabel("largest vocab vs vocab-261\n(fold-std units, + = better)")
axE.set_title("effect size: largest reachable vocab vs the character-level (261) baseline "
              "— shaded band = ±1 fold std (within noise)", pad=6)
axE.set_ylim(-2.6, 2.6); axE.margins(x=0.05); label_all_yticks(axE)

_suptitle(figV, "Fig SV - vocabulary size barely affects unsupervised (MLM) pretraining "
                "(frozen-probe 5-fold scaffold CV, matched 2M forward passes)",
          fontsize=STYLE["fs_title"], y=0.995)
figV.subplots_adjust(left=0.10, right=0.965, top=0.945, bottom=0.055)
# a/b block tags, flush-left above the scaling grid and the effect panel (figC1J1_I1 pattern)
for _ax, _t in [(_taxes[0], "a"), (axE, "b")]:
    figV.text(0.012, _ax.get_position().y1 + 0.008, _t, fontsize=STYLE["fs_panel_tag"],
              fontweight="bold", va="bottom", ha="left")
save_fig(figV, "figSV_vocab"); plt.show()

# ----- printed readout: the near-null result, stated in numbers -----
print("\nVocabulary-size effect (largest reachable vocab vs vocab-261, in fold-std units; "
      "|.|<1 = within fold noise):")
for fam in VOCAB_FAMILY:
    parts = []
    for task, metric, lower in VOCAB_PANELS:
        x, y, e = _vseries(fam, task, metric)
        if len(x) < 2 or not (e[-1] > 0): continue
        delta = (y[0] - y[-1]) if lower else (y[-1] - y[0])
        parts.append(f"{task} {delta/e[-1]:+.2f}σ")
    print(f"   {VOCAB_FAMILY[fam]['label']:14} " + "   ".join(parts))
print("   (the only >1σ move is ESOL/BPE, which degrades as vocab grows while Unigram does not; "
      "everything else is within fold noise, and vocab-261 is already competitive.)")
