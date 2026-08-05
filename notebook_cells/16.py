# ---------- E1 · H5: regime-split eval ceiling ----------
# The first version of this figure plotted ONE frozen and ONE fine-tuned line, both from the
# unsup_only ladder. That answers a ladder-SHAPE question; it cannot answer H5, which is about the
# ORDERING of two regimes. sup_only is fine-tuned by scripts/run_e1_sup_gpu.sh into a separate
# output dir, merged here, so the 48 unsup fine-tunes are not redone.
CEIL     = DATA_ROOT/"_eval_ceiling"/"eval_ceiling.csv"
CEIL_SUP = DATA_ROOT/"_eval_ceiling_sup"/"eval_ceiling.csv"
E1_SUP_RECIPE = "dense"          # best sup_only recipe by mean lift in A1

def _regime_of(run_id):
    if run_id.startswith("random_baseline"): return "no_pretrain"
    if run_id.startswith("unsup_"):          return "unsup_only"
    m=re.match(r"skip_(.+)_\d+M$",run_id)
    return f"sup_only:{m[1]}" if m else None

_parts=[pd.read_csv(q) for q in (CEIL,CEIL_SUP) if q.exists()]
C = pd.concat(_parts,ignore_index=True) if _parts else pd.DataFrame()
if len(C):
    if "seed" not in C.columns: C["seed"]=0
    C["regime"]=C.run_id.map(_regime_of)

E1_REGIMES=[k for k in ("unsup_only",f"sup_only:{E1_SUP_RECIPE}")
            if len(C) and k in set(C.regime)]
_E1_PENDING=[k for k in ("unsup_only",f"sup_only:{E1_SUP_RECIPE}") if k not in E1_REGIMES]

if not len(C):
    fig,ax=plt.subplots(figsize=(STYLE["col15"],2.6))
    no_data_watermark(ax,"run scripts/run_e1_gpu.sh")
    _suptitle(fig, "Fig E1 (H5) [NO DATA]",fontsize=STYLE["fs_title"])
    save_fig(fig,"figE1_H5_eval_ceiling"); plt.show()
else:
    nseed=C.seed.nunique()
    e1_tasks=[t for t in ("BACE","BBBP","ESOL","HIV") if t in set(C.task)]

    # Frozen values come from the run summaries PER HEAD SEED, not from the cached CSV: the CSV
    # collapsed them to a single MEAN and copied it onto every seed row, so its spread was zero.
    # Reading the summaries also keeps E1 on the same scorer as A1/A2.
    def _frozen_seeds(run_id,task):
        f=DATA_ROOT/"climb_v2_phase2"/run_id/"moleculenet"/"moleculenet_summary.csv"
        if not f.exists(): return np.array([])
        d=pd.read_csv(f)
        d=d[(d.dataset==task)&(d.main_metric.isin(["roc_auc","rmse"]))     # not _train, not nef1
            &(~d.head_seed.astype(str).isin(["MEAN","STD"]))]
        return d.main_value.values.astype(float)

    def e1_points(regime,task):
        """-> tidy frame: one row per budget with frozen/fine-tuned mean and sd."""
        s=C[(C.regime==regime)&(C.task==task)&(C.budget>0)]
        rows=[]
        for (rid,b),g in s.groupby(["run_id","budget"]):
            fz=_frozen_seeds(rid,task)
            rows.append(dict(run_id=rid,budget=b,
                             fz=(np.mean(fz) if len(fz) else np.nan),
                             fz_sd=(np.std(fz,ddof=1) if len(fz)>1 else np.nan),
                             ft=g.finetune.mean(),
                             ft_sd=(g.finetune.std() if len(g)>1 else np.nan)))
        return pd.DataFrame(rows).sort_values("budget") if rows else pd.DataFrame()

    def e1_random(task,col):
        s=C[(C.regime=="no_pretrain")&(C.task==task)]
        v=s[col].mean() if len(s) else np.nan
        if col=="frozen" and not np.isfinite(v):
            # The eval-ceiling CSV carries no frozen value for the random-init runs on HIV, so the
            # no_pretrain FROZEN reference vanished there. Recover it from the SAME frozen-probe
            # summaries the pretrained ladder is read from (identical scorer), so HIV gets its line.
            fz=[x for rid in ("random_baseline_00","random_baseline_01","random_baseline_02")
                  for x in _frozen_seeds(rid,task)]
            v=float(np.mean(fz)) if fz else np.nan
        return v

    E1_STYLE={k:dict(c=rc_color(k),m=rc_marker(k) or "o") for k in E1_REGIMES}
    ncol=len(e1_tasks)
    fig,axes=plt.subplots(2,ncol,figsize=(STYLE["col2"],6.2),
                          gridspec_kw=dict(height_ratios=[1,1],hspace=0.55,wspace=0.42))
    axes=np.atleast_2d(axes)

    # ---------- (a) the ladders: 2 regimes x {frozen, fine-tuned} ----------
    for j,task in enumerate(e1_tasks):
        ax=axes[0,j]
        for k in E1_REGIMES:
            d=e1_points(k,task)
            if d.empty: continue
            st=E1_STYLE[k]
            ax.errorbar(d.budget,d.fz,yerr=d.fz_sd.fillna(0),color=st["c"],marker=st["m"],
                        ls=(0,(3,2)),lw=STYLE["lw_thin"],mfc="white",ms=4.5,
                        capsize=2,elinewidth=0.7,zorder=3)
            ax.errorbar(d.budget,d.ft,yerr=d.ft_sd.fillna(0),color=st["c"],marker=st["m"],
                        ls="-",lw=STYLE["lw"],ms=4.5,capsize=2,elinewidth=0.7,zorder=4)
        # Both random-init references. The end-to-end one sat unused in the same CSV -- without it
        # the panel shows what pretraining buys over a FROZEN random encoder, which is the easy
        # comparison; the fine-tuned random encoder is the one a practitioner would actually use.
        _rf=e1_random(task,"frozen")
        if np.isfinite(_rf):
            ax.axhline(_rf,ls=(0,(1,1)),color=PALETTE["grey2"],lw=STYLE["lw_thin"],zorder=1)
        _re=e1_random(task,"finetune")
        if np.isfinite(_re):
            ax.axhline(_re,ls=(0,(1,1.5)),color=rc_color("no_pretrain_e2e"),lw=1.0,zorder=1)
        _bud=sorted(C[(C.task==task)&(C.budget>0)].budget.unique())
        if _bud: set_fp_axis(ax,_bud)
        ax.set_xlabel("forward passes")
        ax.set_title(("HIV (ROC-AUC ↑)" if task=="HIV" else ttitle(task,oneline=True)),pad=6)
        label_all_yticks(ax)
    axes[0,0].set_ylabel("metric value")
    panel_tag(axes[0,0],"a",dx=-0.42)

    # ---------- (b) the H5 ordering test at the matched budget ----------
    _flips=[]
    for j,task in enumerate(e1_tasks):
        ax=axes[1,j]
        hb=TASKS[task]["higher_better"] if task!="HIV" else True   # ROC-AUC here, so higher=better
        vals={}
        for k in E1_REGIMES:
            d=e1_points(k,task)
            if d.empty: continue
            r=d.iloc[(d.budget-BUDGET_FP[MATCHED_BUDGET]).abs().argsort().iloc[0]]
            st=E1_STYLE[k]
            ax.errorbar([0,1],[r.fz,r.ft],
                        yerr=[0 if not np.isfinite(r.fz_sd) else r.fz_sd,
                              0 if not np.isfinite(r.ft_sd) else r.ft_sd],
                        color=st["c"],marker=st["m"],ms=6,lw=STYLE["lw"]+0.4,
                        capsize=3,elinewidth=0.8,zorder=3)
            ax.plot([0],[r.fz],marker=st["m"],ms=6,color=st["c"],mfc="white",zorder=4)  # open=frozen
            vals[k]=(r.fz,r.ft)
        for _c,_v in ((PALETTE["grey2"],e1_random(task,"frozen")),
                      (rc_color("no_pretrain_e2e"),e1_random(task,"finetune"))):
            if np.isfinite(_v):
                ax.axhline(_v,ls=(0,(1,1.5)),color=_c,lw=1.0,zorder=1)
        ax.set_xlim(-0.35,1.35); ax.set_xticks([0,1])
        ax.set_xticklabels(["frozen","fine-\ntuned"])
        ax.set_title(("HIV (ROC-AUC ↑)" if task=="HIV" else ttitle(task,oneline=True)),pad=6)
        label_all_yticks(ax)
        # Deliberately NOT sharing y with the ladder above: on the ladder's range the two segments
        # sit almost on top of each other and a crossing is invisible, which is the one thing this
        # panel exists to show. It is zoomed to its own data; the caption says so.
        ax.margins(y=0.18)
        # Does the ordering survive? Compare the sign of (unsup - sup) in each condition.
        if len(vals)==2:
            u,s_=vals["unsup_only"],vals[f"sup_only:{E1_SUP_RECIPE}"]
            better=lambda a,b:(a>b) if hb else (a<b)
            fz_u_wins,ft_u_wins=better(u[0],s_[0]),better(u[1],s_[1])
            flip = fz_u_wins!=ft_u_wins
            # The flip verdict is still computed and PRINTED below the figure, but no longer
            # stamped on the panel: a bold red "ORDERING FLIPS" reads as a conclusion when the
            # error bars here routinely overlap, and it is one word doing work the reader should
            # do by looking at the two segments.
            _flips.append((task,fz_u_wins,ft_u_wins,flip))
    axes[1,0].set_ylabel("metric value")
    panel_tag(axes[1,0],"b",dx=-0.42)

    handles=[plt.Line2D([],[],color=E1_STYLE[k]["c"],marker=E1_STYLE[k]["m"],ls="-",
                        label=rc_label(k)) for k in E1_REGIMES]
    handles+=[plt.Line2D([],[],color="#555",marker="o",mfc="white",ls=(0,(3,2)),
                         lw=STYLE["lw_thin"],label="frozen probe (open marker)"),
              plt.Line2D([],[],color="#555",marker="o",ls="-",label="fine-tuned end-to-end (filled)"),
              plt.Line2D([],[],color=PALETTE["grey2"],ls=(0,(1,1)),lw=STYLE["lw_thin"],
                         label="no_pretrain (frozen)"),
              plt.Line2D([],[],color=rc_color("no_pretrain_e2e"),ls=(0,(1,1.5)),lw=1.0,
                         label="no_pretrain (end-to-end)")]
    fig.legend(handles=handles,loc="upper center",bbox_to_anchor=(0.5,0.155),ncol=3,
               fontsize=STYLE["fs_legend"])
    _suptitle(fig, "Fig E1 (H5) - is the sup_only / unsup_only ordering a frozen-probe artifact?",
                 fontsize=STYLE["fs_title"],y=0.99)
    # One figure-level pending note: per-panel boxes landed on the data in all four panels, and
    # the missing arm is missing everywhere, so saying it four times bought nothing.
    if _E1_PENDING:
        fig.text(0.5,0.955,"PENDING - "+", ".join(rc_label(k) for k in _E1_PENDING)
                 +" not fine-tuned yet, so (b) cannot test the ordering",
                 ha="center",va="top",fontsize=STYLE["fs_annot"],color="#B00020",fontweight="bold")
    _fzn=max([len(_frozen_seeds(r,e1_tasks[0])) for r in C.run_id.unique()]+[0])
    _note=(f"(b) is the matched {MATCHED_BUDGET} rung of (a), zoomed to its own range so a "
           f"crossing is visible. Error bars: dashed/open = ±1 sd over "
           f"{_fzn} head seeds - a frozen probe has no other randomness, the encoder is fixed; "
           f"solid/filled = ±1 sd over {nseed} fine-tuning seeds, which re-randomise the head AND "
           f"the encoder optimisation, so the two are not the same quantity. Single scaffold "
           f"hold-out (not the 5-fold CV of A1/A2). HIV is reported as ROC-AUC here, deliberately: "
           f"this figure tests whether end-to-end fine-tuning changes the frozen-probe ORDERING, for "
           f"which ROC-AUC is the right endpoint; the virtual-screening endpoint (NEF1%) is the "
           f"headline in A1/A2. (NEF1% is also unavailable here -- the fine-tuning harness saved "
           f"no predictions.) "
           f"The no_pretrain FROZEN reference on HIV is recovered from the random-init runs' "
           f"frozen-probe summaries (the eval-ceiling CSV lacks a frozen HIV value); elsewhere it "
           f"comes from that CSV. Both no_pretrain references are present on all four tasks.")
    fig.subplots_adjust(top=0.88,bottom=0.24)
    _caption(fig, 0.5,0.085,"\n".join(_tw.wrap(_note,132)),ha="center",va="top",
             fontsize=STYLE["fs_annot"]-0.5,color="#666")
    save_fig(fig,"figE1_H5_eval_ceiling"); plt.show()

    # ---------- printed readouts ----------
    if _E1_PENDING:
        print("E1: STILL PENDING -> "+", ".join(rc_label(k) for k in _E1_PENDING)
              +"  (run scripts/run_e1_sup_gpu.sh)")
    print(f"\nE1 (a) ladder shape - spread across the pretrained ladder vs the head-seed noise "
          f"it must clear:")
    for k in E1_REGIMES:
        for task in e1_tasks:
            d=e1_points(k,task)
            if len(d)<2: continue
            noise=np.nanmean(d.fz_sd.values)
            dz,dt=d.fz.max()-d.fz.min(),d.ft.max()-d.ft.min()
            verdict=("fine-tuning COMPRESSES it: the probe was NOT the ceiling" if dt<dz
                     else "fine-tuning SEPARATES them: the probe may be the ceiling")
            print(f"   {rc_label(k):<22} {task:<5} frozen={dz:.4f} ({dz/noise:.1f}x head-seed sd) "
                  f"finetuned={dt:.4f}  -> {verdict}")
    if _flips:
        print(f"\nE1 (b) H5 ordering at {MATCHED_BUDGET} (does unsup_only beat sup_only?):")
        for task,fz_u,ft_u,flip in _flips:
            print(f"   {task:<5} frozen: {'unsup' if fz_u else 'sup'} ahead   "
                  f"fine-tuned: {'unsup' if ft_u else 'sup'} ahead   "
                  f"-> {'FLIPS' if flip else 'holds'}")
        nf=sum(f for _,_,_,f in _flips)
        print(f"   {nf}/{len(_flips)} tasks flip. "
              +("The frozen ordering is NOT preserved under fine-tuning." if nf>len(_flips)/2
                else "The frozen ordering largely survives fine-tuning, so it is not a probe artifact."))