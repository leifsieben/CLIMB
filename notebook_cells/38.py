# ---------- SI figSA · synthetic-statistics ladder (Experiment A) ----------
_EXPA_CSV="analysis/rigor/expA_ladder_summary.csv"
EXPA=pd.read_csv(_EXPA_CSV) if os.path.exists(_EXPA_CSV) else pd.DataFrame(columns=["arm","dataset","metric","mean","std"])
EXPA_FLOOR="no_pretrain (frozen)"                 # random-init frozen probe = the 0-lift reference
# arms ordered by how much corpus structure they preserve (most -> least); colours are a sequential
# blue ramp so "less structure preserved" reads as lighter. bigram_resample (local adjacency) lands
# next and slots between shuffle and unigram when its rows appear in the CSV.
EXPA_ARMS=[("real (unsup_only)", "real",                                  "#08519c"),
           ("shuffle_tokens",    "shuffle tokens\n(order destroyed)",     "#4292c6"),
           ("bigram_resample",   "bigram resample\n(local adjacency)",    "#88bedc"),
           ("unigram_resample",  "unigram resample\n(corpus marginal)",   "#c6dbef")]
EXPA_ARMS=[a for a in EXPA_ARMS if a[0] in set(EXPA.get("arm",[]))]        # drop rungs not yet run
EXPA_TASKS=[t for t in ["ESOL","BBBP","BACE","Tox21","QM7","HIV","Lipophilicity"] if t in set(EXPA.get("dataset",[]))]

def _expa(arm,task):
    s=EXPA[(EXPA.arm==arm)&(EXPA.dataset==task)]
    return (float(s["mean"].iloc[0]),float(s["std"].iloc[0])) if len(s) else (np.nan,np.nan)
def _expa_lift(arm,task):
    """lift over the frozen no_pretrain floor (%), sign-corrected for metric direction, with the
    seed-sd of both the arm and the floor propagated in quadrature."""
    fm,fs=_expa(EXPA_FLOOR,task); am,as_=_expa(arm,task)
    if not (np.isfinite(fm) and np.isfinite(am) and fm!=0): return (np.nan,np.nan)
    hb=TASKS[task]["higher_better"]
    g=(am/fm-1.0) if hb else (1.0-am/fm)          # + = better than the random-init floor
    err=np.sqrt((as_/fm)**2+(am*fs/fm**2)**2)
    return (100*g,100*err)

if EXPA_TASKS and EXPA_ARMS:
    fig,ax=plt.subplots(figsize=(STYLE["col2"],3.3))
    x=np.arange(len(EXPA_TASKS)); n=len(EXPA_ARMS); w=0.8/n
    for i,(arm,lab,c) in enumerate(EXPA_ARMS):
        ys=[_expa_lift(arm,t)[0] for t in EXPA_TASKS]; es=[_expa_lift(arm,t)[1] for t in EXPA_TASKS]
        ax.bar(x+(i-(n-1)/2)*w,ys,width=w,color=c,edgecolor="white",lw=0.4,
               yerr=es,capsize=1.8,error_kw=dict(lw=0.7),label=lab)
    ax.axhline(0,color=PALETTE["black"],lw=0.8)   # 0 = no_pretrain (frozen)
    ax.set_xticks(x); ax.set_xticklabels([TASKS[t]["pretty"] for t in EXPA_TASKS])
    ax.set_ylabel("lift over no_pretrain (frozen) (%)")
    ax.set_xlabel("evaluation task")
    label_all_yticks(ax)
    ax.legend(loc="upper center",bbox_to_anchor=(0.5,-0.16),ncol=n,fontsize=STYLE["fs_legend"],
              frameon=False,columnspacing=1.2,handletextpad=0.4)
    _suptitle(fig,"Fig SA - synthetic-statistics ladder: the MLM benefit is per-molecule token composition",
              fontsize=STYLE["fs_title"],y=1.02)
    _note=("frozen probe (5-fold scaffold CV, 3 seeds), native units, each corpus bit-cloned from "
           "unsup_8M. Lift over the random-init frozen floor (0). shuffle keeps the per-molecule token "
           "multiset but destroys order; bigram keeps only local token adjacency; unigram keeps only the "
           "corpus token marginal. shuffle≈real and unigram≈floor => the benefit is per-molecule "
           "composition; bigram is intermediate on tasks with real spread (ESOL/BACE/Tox21/Lipo), so "
           "local adjacency carries a partial, task-dependent share. BBBP/HIV are saturated (all arms "
           "within ~0.01), so the ladder is clearest on ESOL/Lipo/BACE/Tox21/QM7. HIV = ROC-AUC.")
    _caption(fig,0.5,-0.30,"\n".join(_tw.wrap(_note,120)),ha="center",va="top",
             fontsize=STYLE["fs_annot"],color="#555")
    fig.subplots_adjust(bottom=0.20)
    save_fig(fig,"figSA_synthetic_statistics_ladder"); plt.show()
else:
    print("ExpA ladder: analysis/rigor/expA_ladder_summary.csv not found or empty.")

# quantify the ladder: mean lift over floor, and how much of 'real' each rung recovers
if EXPA_TASKS and EXPA_ARMS:
    print("\nExpA ladder - mean lift over no_pretrain(frozen), across tasks:")
    for arm,lab,_ in EXPA_ARMS:
        v=[_expa_lift(arm,t)[0] for t in EXPA_TASKS]
        print(f"   {lab.splitlines()[0]:22s} {np.nanmean(v):+5.1f}%")
    rl=np.array([_expa_lift("real (unsup_only)",t)[0] for t in EXPA_TASKS])
    for arm,lab,_ in EXPA_ARMS:
        if arm=="real (unsup_only)": continue
        al=np.array([_expa_lift(arm,t)[0] for t in EXPA_TASKS])
        with np.errstate(invalid="ignore",divide="ignore"):
            frac=np.nanmean(np.where(np.abs(rl)>1e-9,al/rl,np.nan))*100
        print(f"   -> {lab.splitlines()[0]} recovers {frac:.0f}% of real's benefit (mean over tasks)")
