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

# ----- figSV_vocab_scaling : per-task metric vs actual vocab, BPE vs Unigram -----
ncol = 3; nrow = int(np.ceil(len(VOCAB_PANELS) / ncol))
figV, axesV = plt.subplots(nrow, ncol, figsize=(STYLE["col2"], 2.15 * nrow)); axesV = axesV.ravel()
for ax, (task, metric, lower) in zip(axesV, VOCAB_PANELS):
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
for ax in axesV[len(VOCAB_PANELS):]: ax.axis("off")
axesV[0].legend(loc="best", fontsize=STYLE["fs_legend"])
_suptitle(figV, "Fig SV.a - vocabulary size barely affects unsupervised (MLM) pretraining "
                "(frozen-probe 5-fold scaffold CV, matched 2M forward passes)",
          fontsize=STYLE["fs_title"], y=1.02)
_vnote = ("frozen-probe pooled 5-fold scaffold CV  ·  error bars = ±1 std across the 5 held-out "
          "scaffold folds  ·  x = actual tokenizer vocab (not the nominal target)  ·  single "
          "pretraining seed  ·  vocab and embedding-param count are confounded by construction "
          "(larger vocab ⇒ more params); disclosed, not removed")
_caption(figV, 0.5, 0.995, "\n".join(_tw.wrap(_vnote, 120)), ha="center", va="top",
         fontsize=STYLE["fs_annot"] - 0.5, color="#666")
figV.subplots_adjust(top=0.90, bottom=0.11, hspace=0.55, wspace=0.34)
save_fig(figV, "figSV_vocab_scaling"); plt.show()

# ----- figSV_vocab_effect : largest-vocab change vs the vocab-261 baseline, in fold-std units -----
# One number per family/task: how far the LARGEST reachable vocab moves the metric away from the
# character-level (261) baseline, signed so + = better, divided by that task's fold std at the large
# vocab. |effect| < 1 (inside the shaded band) means "within evaluation noise".
figE, axE = plt.subplots(figsize=(STYLE["col15"], 3.6))
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
        axE.plot([xp], [eff], color=sty["color"], marker=sty["marker"], ms=7, mec="white",
                 label=(sty["label"] if k == 0 else None), zorder=3)
axE.axhspan(-1, 1, color="#888888", alpha=0.13, zorder=0)
axE.axhline(0, color=PALETTE["black"], lw=0.9)
axE.set_xticks(range(len(VOCAB_PANELS))); axE.set_xticklabels([p[0] for p in VOCAB_PANELS])
axE.set_ylabel("largest vocab vs vocab-261\n(fold-std units, + = better)")
axE.set_ylim(-2.6, 2.6); axE.margins(x=0.05); label_all_yticks(axE)
axE.legend(loc="upper right", fontsize=STYLE["fs_legend"])
_suptitle(figE, "Fig SV.b - does enlarging the vocabulary help? Largest reachable vocab vs the "
                "character-level (261) baseline; shaded band = ±1 fold std (within-noise)",
          fontsize=STYLE["fs_title"], y=1.02)
_caption(figE, 0.5, 0.995, "\n".join(_tw.wrap(
    "effect = (metric at the largest reachable vocab − metric at vocab 261), signed so positive is "
    "better, in units of that task's across-fold std at the large vocab. Markers inside the shaded "
    "±1σ band are within evaluation noise.", 118)), ha="center", va="top",
    fontsize=STYLE["fs_annot"] - 0.5, color="#666")
figE.subplots_adjust(top=0.90, bottom=0.12); save_fig(figE, "figSV_vocab_effect"); plt.show()

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
