# ---------- SI figSB · Wikipedia-through-SMILES-tokenizer transfer (Experiment B) ----------
_EXPB_CSV="analysis/rigor/expB_wiki_summary.csv"
EXPB=pd.read_csv(_EXPB_CSV) if os.path.exists(_EXPB_CSV) else pd.DataFrame(columns=["arm","dataset","metric","mean","std"])
EXPB_FLOOR="no_pretrain (frozen)"                 # random-init frozen probe = the 0-lift reference
EXPB_ARMS=[("real (unsup_only)","real (SMILES corpus)",        "#08519c"),
           ("wiki_real",        "wiki (English, zero chemistry)","#d95f02")]
EXPB_ARMS=[a for a in EXPB_ARMS if a[0] in set(EXPB.get("arm",[]))]
EXPB_TASKS=[t for t in ["ESOL","BBBP","BACE","Tox21","QM7","HIV","Lipophilicity"] if t in set(EXPB.get("dataset",[]))]

def _expb(arm,task):
    s=EXPB[(EXPB.arm==arm)&(EXPB.dataset==task)]
    return (float(s["mean"].iloc[0]),float(s["std"].iloc[0])) if len(s) else (np.nan,np.nan)
def _expb_lift(arm,task):
    """lift over the frozen no_pretrain floor (%), sign-corrected, with arm+floor seed sd propagated."""
    fm,fs=_expb(EXPB_FLOOR,task); am,as_=_expb(arm,task)
    if not (np.isfinite(fm) and np.isfinite(am) and fm!=0): return (np.nan,np.nan)
    hb=TASKS[task]["higher_better"]
    g=(am/fm-1.0) if hb else (1.0-am/fm)
    err=np.sqrt((as_/fm)**2+(am*fs/fm**2)**2)
    return (100*g,100*err)

if EXPB_TASKS and EXPB_ARMS:
    fig,ax=plt.subplots(figsize=(STYLE["col2"],3.3))
    x=np.arange(len(EXPB_TASKS)); n=len(EXPB_ARMS); w=0.8/n
    for i,(arm,lab,c) in enumerate(EXPB_ARMS):
        ys=[_expb_lift(arm,t)[0] for t in EXPB_TASKS]; es=[_expb_lift(arm,t)[1] for t in EXPB_TASKS]
        ax.bar(x+(i-(n-1)/2)*w,ys,width=w,color=c,edgecolor="white",lw=0.4,
               yerr=es,capsize=1.8,error_kw=dict(lw=0.7),label=lab)
    ax.axhline(0,color=PALETTE["black"],lw=0.8)   # 0 = no_pretrain (frozen)
    ax.set_xticks(x); ax.set_xticklabels([TASKS[t]["pretty"] for t in EXPB_TASKS])
    ax.set_ylabel("lift over no_pretrain (frozen) (%)"); ax.set_xlabel("evaluation task")
    label_all_yticks(ax)
    ax.legend(loc="upper center",bbox_to_anchor=(0.5,-0.16),ncol=n,fontsize=STYLE["fs_legend"],
              frameon=False,columnspacing=1.4,handletextpad=0.4)
    _suptitle(fig,"Fig SB - English Wikipedia through the SMILES tokenizer: the benefit is domain-general",
              fontsize=STYLE["fs_title"],y=1.02)
    _note=("frozen probe (5-fold scaffold CV, 3 seeds), native units. `wiki` is pretrained on English "
           "Wikipedia (zero chemistry) with the SAME SMILES tokenizer, chunk lengths sampled from the "
           "SMILES length distribution. Lift over the random-init frozen floor (0). Guards: Wikipedia "
           "trained 96.9% of eval-token mass >=1x (88.6% >=1000x), so this is not an undertrained-"
           "embedding artifact (QM7 has the lowest coverage yet transfers fully); and the token marginals "
           "are near-maximally divergent (JS=0.93 bits, 435 chemistry-only tokens unfilled), so transfer "
           "is not the corpus marginal. HIV = ROC-AUC.")
    _caption(fig,0.5,-0.30,"\n".join(_tw.wrap(_note,120)),ha="center",va="top",
             fontsize=STYLE["fs_annot"],color="#555")
    fig.subplots_adjust(bottom=0.20)
    save_fig(fig,"figSB_wikipedia_transfer"); plt.show()
else:
    print("ExpB wiki: analysis/rigor/expB_wiki_summary.csv not found or empty.")

if EXPB_TASKS and "wiki_real" in set(EXPB.get("arm",[])):
    print("\nExpB - wiki lift over no_pretrain(frozen), and % of real's benefit recovered:")
    for t in EXPB_TASKS:
        wl=_expb_lift("wiki_real",t)[0]; rl=_expb_lift("real (unsup_only)",t)[0]
        frac=(wl/rl*100) if (np.isfinite(rl) and abs(rl)>1e-9) else np.nan
        print(f"   {t:14s} wiki {wl:+5.1f}%   ({frac:+4.0f}% of real)")
