# ---------- B2 · corrupted-pretraining control ----------
# Colour encodes the PAIR, lightness encodes real-vs-corrupted: dark blue / light blue for the MLM
# pair, dark red / light orange for the MTR pair. All four come from the shared PALETTE, so the
# figure still reads as part of the same set even though the pairing is B2-specific.
B2_SERIES=[("unsup_only",     "unsup_only (MLM) - real",          PALETTE["blue"]),
           ("corrupt_mlm",    "corrupted MLM (shuffled tokens)",  PALETTE["cyan"]),
           ("sup_only:dense", "sup_only: dense (MTR) - real",     PALETTE["red"]),
           ("corrupt_mtr",    "corrupted MTR (shuffled targets)", PALETTE["orange"])]

# PREFER the 5-fold CV: on the single hold-out this panel is scored on 113-204 molecules per task
# and HIV's NEF1% comes off the top 1% of ~4.1k molecules (~41 molecules), which is why the HIV
# bars swing wildly there. The corrupted encoders have not been CV-scored yet, so the figure falls
# back to the hold-out and says so on its face rather than quietly using the weak split.
_b2_have_cv=all(len(arm_rows(DF_CV,"ESOL",k)) for k,_,_ in B2_SERIES)
B2_DF   = DF_CV if _b2_have_cv else DF
B2_SCHEME=("pooled 5-fold scaffold CV" if _b2_have_cv else
           "single scaffold hold-out (113-204 test molecules) - CV PENDING for the corrupted arms")
_have_b2=any(len(arm_rows(B2_DF,"ESOL",k)) for k,_,_ in B2_SERIES if k.startswith("corrupt"))

fig,ax=plt.subplots(figsize=(STYLE["col2"],2.9))
x=np.arange(len(CORE_TASKS)); w=0.19
for i,(key,lab,c) in enumerate(B2_SERIES):
    ys=[100*lift(arm_value(B2_DF,t,key),t,npt_floor(B2_DF,t)) for t in CORE_TASKS]
    ax.bar(x+(i-1.5)*w,ys,width=w,color=c,edgecolor="white",lw=0.4,label=lab)
ax.axhline(0,color=PALETTE["black"],lw=0.8)
# "Skip pretraining and just fine-tune" is the reference that makes the corrupted/real contrast
# actionable: a corrupted arm that clears no_pretrain but not this one has bought nothing real.
_e2e=[100*lift(arm_value(B2_DF,t,"no_pretrain_e2e"),t,npt_floor(B2_DF,t)) for t in CORE_TASKS]
if any(np.isfinite(v) for v in _e2e):
    for xi,v in zip(x,_e2e):
        if np.isfinite(v):
            ax.plot([xi-0.5,xi+0.5],[v,v],color=rc_color("no_pretrain_e2e"),ls=(0,(1,1.5)),lw=1.3,
                    zorder=5,solid_capstyle="butt")
    _e2e_h=[plt.Line2D([],[],color=rc_color("no_pretrain_e2e"),ls=(0,(1,1.5)),lw=1.3,
                       label="no_pretrain (end-to-end)")]
else:
    _e2e_h=[]
    ax.text(0.995,0.02,"no_pretrain (end-to-end): E1 runs pending",transform=ax.transAxes,
            ha="right",va="bottom",fontsize=STYLE["fs_annot"]-1.5,color="#B00020")
ax.set_xticks(x); ax.set_xticklabels([TASKS[t]["pretty"] for t in CORE_TASKS])
ax.set_ylabel("lift over no_pretrain (%)"); label_all_yticks(ax)
if not _have_b2:
    ax.text(0.5,1.02,"CORRUPTED ARMS NOT YET AVAILABLE — corrupt_mlm_8M / corrupt_mtr_8M still training",
            transform=ax.transAxes,ha="center",va="bottom",fontsize=STYLE["fs_annot"],
            color="#B00020",fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3",fc="#FCE8E8",ec="#B00020",lw=0.8))
elif not _b2_have_cv:
    ax.text(0.5,1.02,"SCHEME FALLBACK — corrupted arms have no 5-fold CV eval yet, so this panel is "
            "on the small single hold-out; HIV especially is unreliable here",
            transform=ax.transAxes,ha="center",va="bottom",fontsize=STYLE["fs_annot"]-1,
            color="#B00020",bbox=dict(boxstyle="round,pad=0.3",fc="#FCE8E8",ec="#B00020",lw=0.8))
# Column-major fill: this order puts MLM-real top-left, MLM-corrupted top-right, MTR-real
# bottom-left, MTR-corrupted bottom-right, so each pair is read down a column... no: matplotlib
# fills column-first, so the handles are reordered here to land as [TL, TR / BL, BR].
_h,_l=ax.get_legend_handles_labels()
_ord=[0,2,1,3]                       # -> col0 = (MLM real, MTR real); col1 = (MLM corr, MTR corr)
fig.legend(handles=[_h[i] for i in _ord]+_e2e_h,labels=[_l[i] for i in _ord]+[h.get_label() for h in _e2e_h],
           loc="upper center",bbox_to_anchor=(0.5,0.02),ncol=2,fontsize=STYLE["fs_legend"])
fig.suptitle("Fig B2 - does content-free pretraining help just as much?",
             fontsize=STYLE["fs_title"],y=1.18)
fig.text(0.5,1.06,f"zero = no_pretrain  ·  {B2_SCHEME}. Each corrupted arm is matched to its real "
         "counterpart in objective, data volume, compute and schedule; only chemical content is destroyed.",
         ha="center",va="top",fontsize=STYLE["fs_annot"]-0.5,color="#666")
fig.subplots_adjust(bottom=0.28)
save_fig(fig,"figB2_corrupted_control"+("" if _have_b2 else "_PLACEHOLDER")); plt.show()

if _have_b2:
    print(f"B2 mean lift over no_pretrain across the core tasks ({B2_SCHEME}):")
    for key,lab,_ in B2_SERIES:
        ls=[lift(arm_value(B2_DF,t,key),t,npt_floor(B2_DF,t)) for t in CORE_TASKS]
        ls=[v for v in ls if np.isfinite(v)]
        if ls: print(f"   {lab:<40} {100*np.mean(ls):+6.1f}%  (n={len(ls)})")
    print("\nper-task lift (%), to expose any single task driving the mean:")
    print(pd.DataFrame({lab:[100*lift(arm_value(B2_DF,t,key),t,npt_floor(B2_DF,t)) for t in CORE_TASKS]
                        for key,lab,_ in B2_SERIES},index=CORE_TASKS).round(1).to_string())
else:
    print("B2: no corrupted-control data yet. Expected runs: corrupt_mlm_8M, corrupt_mtr_8M "
          "(8M FP each, matched to unsup_8M / skip_dense_8M).")