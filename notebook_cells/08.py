# ---------- A1 · headline bars at the matched budget, drawn on BOTH evaluation schemes ----------
# A1.a = single DeepChem scaffold hold-out (rarest scaffolds in test, 113-204 molecules on the
#        small tasks). A1.b = pooled 5-fold scaffold CV (README §8.1's protocol of record;
#        10-35x the test molecules, so it is the one that can actually separate arms).
# Same arms, same order, same colours in both panels so they are read side by side.
import textwrap as _tw          # module scope: later figures (E1) reuse this alias
A1_ORDER = ["ecfp4", "fp_desc", "no_pretrain", "no_pretrain_e2e", "unsup_only",
            "sup_only:dense", "sup_only:sparse_all", "sup_only:dense_plus_sparse",
            "unsup2sup:sparse_all", "unsup2sup:dense", "unsup2sup:dense_plus_sparse"]
# `minimol_full` and `mixed` are deliberately absent from the BARS (11 is already crowded); both
# recipes are still reported for both regimes in Table A1 below, which covers all five.

def a1_title(task, sub):
    """`ESOL (1,128 mols)` over `(RMSE ↓ · n_test = 113)` -- the second number is the whole point
    of splitting A1 into a/b, so it belongs in the panel, not only in the caption."""
    tot, n = n_molecules(task, "moleculenet_cv"), n_molecules(task, sub)
    head = f'{TASKS[task]["pretty"]} ({tot:,} mols)' if np.isfinite(tot) else TASKS[task]["pretty"]
    if not np.isfinite(n):        tail = mlabel(task)
    elif n == tot:                tail = f'{mlabel(task)} · all scored, OOF'   # CV: no held-out remainder
    else:                         tail = f'{mlabel(task)} · n_test = {n:,}'
    return f"{head}\n({tail})"

def draw_A1(df, sub, tag, scheme_note, err_note, fname):
    ncol = 3; nrow = int(np.ceil(len(CORE_TASKS) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(STYLE["col2"], 2.35 * nrow)); axes = axes.ravel()
    for ax, task in zip(axes, CORE_TASKS):
        vals = [arm_value(df, task, k) for k in A1_ORDER]
        errs = [arm_err(df, task, k)   for k in A1_ORDER]
        x = np.arange(len(A1_ORDER))
        for xi, k, v, e in zip(x, A1_ORDER, vals, errs):
            if not np.isfinite(v): continue
            ax.bar(xi, v, color=rc_color(k), edgecolor="white", lw=0.4, width=0.82,
                   yerr=(e if np.isfinite(e) else None),
                   error_kw=dict(ecolor="#333333", elinewidth=0.9, capsize=2, capthick=0.9))
        # A missing bar simply vanishes, which a reader cannot tell from a zero-height bar. Name them.
        pending = [rc_label(k) for k, v in zip(A1_ORDER, vals) if not np.isfinite(v)]
        if pending:
            ax.set_ylim(top=ax.get_ylim()[1] * (1.10 + 0.09 * int(np.ceil(len(pending) / 2))))
            ax.text(0.02, 0.985, "pending:\n" + "\n".join(pending), transform=ax.transAxes,
                    ha="left", va="top", fontsize=STYLE["fs_annot"] - 1.5, color="#B00020",
                    linespacing=1.25, bbox=dict(fc="white", ec="none", alpha=0.85, pad=1.0))
        add_chance_line(ax, task)
        ax.set_title(a1_title(task, sub), pad=6); ax.set_xticks(x); ax.set_xticklabels([])
        ax.tick_params(axis="x", length=0); ax.margins(x=0.04); label_all_yticks(ax)
    for ax in axes[len(CORE_TASKS):]: ax.axis("off")
    for i in range(0, len(CORE_TASKS), ncol): axes[i].set_ylabel("metric value")

    handles = [mpl.patches.Patch(facecolor=rc_color(k), label=rc_label(k)) for k in A1_ORDER]
    handles += [plt.Line2D([], [], color="#999999", ls=(0, (1, 1.5)), lw=1.0, label="random (chance)")]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=4,
               fontsize=STYLE["fs_legend"], handlelength=1.5, columnspacing=1.3)

    # replication note; the count of replicated arms is read from the data so it cannot go stale
    _multi = sum(1 for k in A1_ORDER if n_seeds(df, "ESOL", k) > 1)
    _note = (f"{scheme_note}  ·  {err_note}"
             + (f" ({_multi} arms have pretraining-seed replicates)" if _multi else "")
             + "  ·  classical anchors: 3 XGBoost seeds  ·  no_pretrain (frozen) = random-init "
               "encoder, features frozen; no_pretrain (end-to-end) = same encoder fine-tuned on the task")
    _suptitle(fig, f"Fig {tag} - which model performs best? (matched at {MATCHED_BUDGET} forward passes)",
                 fontsize=STYLE["fs_title"], y=1.10)
    _caption(fig, 0.5, 0.995, "\n".join(_tw.wrap(_note, 125)), ha="center", va="top",
             fontsize=STYLE["fs_annot"] - 0.5, color="#666")
    fig.subplots_adjust(top=0.82, bottom=0.13, hspace=0.42, wspace=0.42)
    save_fig(fig, fname); plt.show()

    print(f"\n{tag} mean lift over no_pretrain at {MATCHED_BUDGET} (tasks scored / {len(CORE_TASKS)}):")
    for k in A1_ORDER:
        ls = [lift(arm_value(df, t, k), t, npt_floor(df, t)) for t in CORE_TASKS]
        ls = [v for v in ls if np.isfinite(v)]
        if ls: print(f"   {rc_label(k):<30} {100*np.mean(ls):+6.1f}%   (n={len(ls)})")

draw_A1(DF,    "moleculenet",    "A1.a", "single DeepChem scaffold hold-out (rarest scaffolds in test; 113-204 "
                        "test molecules on ESOL/BACE/BBBP)",
        "error bars = ±1 sd over pretraining seeds where replicates exist, else over head seeds",
        "figA1a_best_model_headline_holdout")
draw_A1(DF_CV, "moleculenet_cv", "A1.b", "pooled 5-fold scaffold cross-validation (README §8.1 protocol of record)",
        "error bars = ±1 sd across the 5 scaffold folds",
        "figA1b_best_model_headline_cv")