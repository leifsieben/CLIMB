# ---------- SI figSA · synthetic-statistics ladder (Exp A) + Wikipedia transfer (Exp B) ----------
_EXPA_CSV="analysis/rigor/expA_ladder_summary.csv"
_EXPB_CSV="analysis/rigor/expB_wiki_summary.csv"
EXPA=pd.read_csv(_EXPA_CSV) if os.path.exists(_EXPA_CSV) else pd.DataFrame(columns=["arm","dataset","metric","mean","std"])
# Fold in Exp B's Wikipedia arm. Exp B reuses the SAME frozen floor and `real`/unsup_8M comparators
# (native _baselines) as Exp A, so lift-over-floor is directly comparable; take only the wiki_real rows.
if os.path.exists(_EXPB_CSV):
    _eb=pd.read_csv(_EXPB_CSV)
    EXPA=pd.concat([EXPA,_eb[_eb.arm=="wiki_real"]],ignore_index=True)
EXPA_FLOOR="no_pretrain (frozen)"                 # random-init frozen probe = the 0-lift reference
# Exp A rungs are a sequential BLUE ramp (most -> least corpus structure preserved). Wikipedia (Exp B --
# English text, zero chemistry, same SMILES tokenizer) is RED and sits on the far right. bigram slots in
# when present; any arm absent from the CSV is dropped.
EXPA_ARMS=[("real (unsup_only)", "real",                                  "#08519c"),
           ("shuffle_tokens",    "shuffle tokens\n(order destroyed)",     "#4292c6"),
           ("bigram_resample",   "bigram resample\n(local adjacency)",    "#88bedc"),
           ("unigram_resample",  "unigram resample\n(corpus marginal)",   "#c6dbef"),
           ("wiki_real",         "wiki (English,\nzero chemistry)",       "#d62728")]
EXPA_ARMS=[a for a in EXPA_ARMS if a[0] in set(EXPA.get("arm",[]))]        # drop arms not yet run
EXPA_TASKS=[t for t in CORE_TASKS if t in set(EXPA.get("dataset",[]))]     # Lipophilicity excluded (not in the paper's task set)

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
        xb=x+(i-(n-1)/2)*w
        ys=[_expa_lift(arm,t)[0] for t in EXPA_TASKS]; es=[_expa_lift(arm,t)[1] for t in EXPA_TASKS]
        ax.bar(xb,ys,width=w,color=c,edgecolor="white",lw=0.4,
               yerr=es,capsize=1.8,error_kw=dict(lw=0.7),label=lab)
        for xi,v,e in zip(xb,ys,es):     # write each bar's lift value on the bar (vertical, above the cap)
            if not np.isfinite(v): continue
            yt=(v+e+0.4) if v>=0 else (v-e-0.4)
            ax.text(xi,yt,f"{v:.1f}",rotation=90,ha="center",va="bottom" if v>=0 else "top",
                    fontsize=STYLE["fs_annot"]-2.5,color="#333")
    ax.axhline(0,color=PALETTE["black"],lw=0.8)   # 0 = no_pretrain (frozen)
    ax.set_xticks(x); ax.set_xticklabels([TASKS[t]["pretty"] for t in EXPA_TASKS])
    ax.set_ylabel("lift over no_pretrain (frozen) (%)")
    ax.set_xlabel("evaluation task")
    label_all_yticks(ax)
    ax.legend(loc="upper center",bbox_to_anchor=(0.5,-0.16),ncol=n,fontsize=STYLE["fs_legend"],
              frameon=False,columnspacing=1.2,handletextpad=0.4)
    _suptitle(fig,"Fig SA - where the MLM benefit lives: per-molecule composition (blue) + domain-general transfer (Wikipedia, red)",
              fontsize=STYLE["fs_title"],y=1.02)
    _note=("frozen probe (5-fold scaffold CV, 3 seeds), native units, lift over the random-init frozen "
           "floor (0). BLUE = Exp A structure ladder, each corpus bit-cloned from unsup_8M: shuffle keeps "
           "the per-molecule token multiset but drops order; bigram keeps local adjacency; unigram keeps "
           "only the corpus marginal. shuffle≈real and unigram≈floor => the benefit is per-molecule "
           "composition; bigram is intermediate on tasks with real spread (ESOL/BACE/Tox21). RED = Exp B: "
           "a model pretrained on English Wikipedia (zero chemistry) through the SAME SMILES tokenizer, "
           "length-matched; it beats the floor on most tasks and matches real on QM7 => much of the "
           "benefit is domain-general. Guards: Wikipedia covers 96.9% of eval-token mass >=1x (QM7 lowest "
           "yet transfers fully) and its token marginal is near-maximally divergent (JS=0.93 bits), so "
           "transfer is neither an undertrained-embedding artifact nor the marginal. BBBP/HIV saturate. "
           "HIV = ROC-AUC.")
    _caption(fig,0.5,-0.30,"\n".join(_tw.wrap(_note,120)),ha="center",va="top",
             fontsize=STYLE["fs_annot"],color="#555")
    fig.subplots_adjust(bottom=0.20)
    save_fig(fig,"figSA_synthetic_statistics_ladder"); plt.show()
else:
    print("ExpA ladder: analysis/rigor/expA_ladder_summary.csv not found or empty.")

# quantify the ladder as two tables: lift over the floor, and each arm's recovery of real's benefit
if EXPA_TASKS and EXPA_ARMS:
    _lab={"real (unsup_only)":"real","shuffle_tokens":"shuffle","bigram_resample":"bigram",
          "unigram_resample":"unigram","wiki_real":"wiki"}
    _rl={t:_expa_lift("real (unsup_only)",t)[0] for t in EXPA_TASKS}
    _SAT={"BBBP","HIV"}                        # real lift <1.5% here => the recovery ratio is unstable
    _inf=[t for t in EXPA_TASKS if t not in _SAT]
    _hdr="   "+"arm".ljust(9)+"".join(f"{TASKS[t]['pretty']:>7}" for t in EXPA_TASKS)
    print("\nLift over no_pretrain (frozen) (%):"); print(_hdr)
    for arm,lab,_ in EXPA_ARMS:
        print("   "+_lab[arm].ljust(9)+"".join(f"{_expa_lift(arm,t)[0]:7.1f}" for t in EXPA_TASKS))
    print("\nRecovery = arm lift / real lift  (% of real's pretraining benefit):")
    print(_hdr+"   mean*")
    for arm,lab,_ in EXPA_ARMS:
        if arm=="real (unsup_only)": continue
        rec={t:(_expa_lift(arm,t)[0]/_rl[t]*100 if abs(_rl[t])>1e-9 else np.nan) for t in EXPA_TASKS}
        print("   "+_lab[arm].ljust(9)+"".join(f"{rec[t]:7.0f}" for t in EXPA_TASKS)
              +f"   {np.nanmean([rec[t] for t in _inf]):5.0f}")
    print("   * mean over non-saturated tasks (excl. BBBP/HIV, where real lift <1.5% makes the ratio unstable)")
