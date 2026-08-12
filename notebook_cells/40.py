# ---------- SI figSA2 · same ladder, lift over the end-to-end baseline ----------
# Reuses cell 38's combined EXPA frame (Exp A rungs + Exp B wiki) and its EXPA_ARMS/EXPA_TASKS, but
# swaps the floor from no_pretrain(frozen) to no_pretrain(END-TO-END) = phase2 e2e_random_0* (native,
# mean over 3 seeds) -- the same honest baseline Fig I1/B2 use. e2e is a fine-tuned baseline, not a
# frozen probe, so it lives in phase2, not the expA _baselines.
def _e2e_floor(task):
    vals=[]
    for r in ("e2e_random_00","e2e_random_01","e2e_random_02"):
        p=DATA_ROOT/"climb_v2_phase2"/r/"moleculenet_cv"/"suite_summary.json"
        if p.exists():
            v=json.loads(p.read_text()).get(task+"_MEAN")
            if v is not None and np.isfinite(v): vals.append(v)
    return float(np.mean(vals)) if vals else np.nan
def _lift_e2e(arm,task):
    fm=_e2e_floor(task); am,as_=_expa(arm,task)
    if not (np.isfinite(fm) and np.isfinite(am) and fm!=0): return (np.nan,np.nan)
    hb=TASKS[task]["higher_better"]
    return (100*((am/fm-1) if hb else (1-am/fm)), 100*abs(as_/fm))

_have_e2e=bool(EXPA_TASKS) and bool(EXPA_ARMS) and all(np.isfinite(_e2e_floor(t)) for t in EXPA_TASKS)
if _have_e2e:
    fig,ax=plt.subplots(figsize=(STYLE["col2"],3.3))
    x=np.arange(len(EXPA_TASKS)); n=len(EXPA_ARMS); w=0.8/n
    for i,(arm,lab,c) in enumerate(EXPA_ARMS):
        xb=x+(i-(n-1)/2)*w
        ys=[_lift_e2e(arm,t)[0] for t in EXPA_TASKS]; es=[_lift_e2e(arm,t)[1] for t in EXPA_TASKS]
        ax.bar(xb,ys,width=w,color=c,edgecolor="white",lw=0.4,yerr=es,capsize=1.8,error_kw=dict(lw=0.7),label=lab)
        for xi,v,e in zip(xb,ys,es):
            if not np.isfinite(v): continue
            yt=(v+e+0.4) if v>=0 else (v-e-0.4)
            ax.text(xi,yt,f"{v:.1f}",rotation=90,ha="center",va="bottom" if v>=0 else "top",
                    fontsize=STYLE["fs_annot"]-2.5,color="#333")
    ax.axhline(0,color=PALETTE["black"],lw=0.8)   # 0 = no_pretrain (end-to-end)
    ax.set_xticks(x); ax.set_xticklabels([TASKS[t]["pretty"] for t in EXPA_TASKS])
    ax.set_ylabel("lift over no_pretrain (end-to-end) (%)"); ax.set_xlabel("evaluation task")
    label_all_yticks(ax)
    ax.legend(loc="upper center",bbox_to_anchor=(0.5,-0.16),ncol=n,fontsize=STYLE["fs_legend"],
              frameon=False,columnspacing=1.2,handletextpad=0.4)
    _suptitle(fig,"Fig SA (companion) - same ladder, lift over the end-to-end baseline",
              fontsize=STYLE["fs_title"],y=1.02)
    _note=("identical to Fig SA but the floor is no_pretrain (END-TO-END) -- a random encoder fine-tuned "
           "per task (phase2 e2e_random_0*, native, mean over 3 seeds; the Fig I1/B2 baseline), not the "
           "frozen random probe. Composition story is unchanged (shuffle≈real, unigram far below), but "
           "QM7 turns slightly negative for every arm -- end-to-end fine-tuning already beats the frozen "
           "MLM probe there. HIV = ROC-AUC.")
    _caption(fig,0.5,-0.28,"\n".join(_tw.wrap(_note,120)),ha="center",va="top",
             fontsize=STYLE["fs_annot"],color="#555")
    fig.subplots_adjust(bottom=0.20)
    save_fig(fig,"figSA2_synthetic_ladder_vs_e2e"); plt.show()
    print("\nfigSA2 e2e floor (phase2 e2e_random native) per task:",
          {t:round(_e2e_floor(t),3) for t in EXPA_TASKS})
    print("lift over e2e (%):   "+"".join(f"{TASKS[t]['pretty']:>7}" for t in EXPA_TASKS))
    for arm,lab,_ in EXPA_ARMS:
        print(f"   {lab.splitlines()[0][:9]:9s} "+"".join(f"{_lift_e2e(arm,t)[0]:7.1f}" for t in EXPA_TASKS))
else:
    print("figSA2: e2e floor unavailable (need figure_data/climb_v2_phase2/e2e_random_0*/moleculenet_cv).")
