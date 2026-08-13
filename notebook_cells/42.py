# ---------- MAINLINE · external CBS-inhibitor VS benchmark (Truong 2026) ----------
_CBS="experiment_cbs/cbs_nef1_summary.csv"
CBS   =pd.read_csv(_CBS) if os.path.exists(_CBS) else pd.DataFrame(columns=["arm","metric","mean","std_over_seeds","n_seeds"])
CBS_RUN=pd.read_csv("experiment_cbs/cbs_per_run.csv") if os.path.exists("experiment_cbs/cbs_per_run.csv") else pd.DataFrame()
CBS_REF=pd.read_csv("experiment_cbs/cbs_reference_lines.csv") if os.path.exists("experiment_cbs/cbs_reference_lines.csv") else pd.DataFrame()

def _cbs_val(arm,metric="nef1"):
    s=CBS[(CBS.arm==arm)&(CBS.metric==metric)]
    return float(s["mean"].iloc[0]) if len(s) else np.nan
def _cbs_err(arm,metric="nef1"):
    """seed-spread where >1 pretraining seed exists; else the fold-spread (single-run XGBoost anchors)."""
    s=CBS[(CBS.arm==arm)&(CBS.metric==metric)]
    if not len(s): return 0.0
    sd=float(s["std_over_seeds"].iloc[0]); n=int(s["n_seeds"].iloc[0])
    if n>1 and np.isfinite(sd) and sd>0: return sd
    r=CBS_RUN[CBS_RUN.arm==arm]; col=("nef1_std_fold" if metric=="nef1" else "roc_auc_std_fold")
    return float(r[col].iloc[0]) if (len(r) and col in r.columns) else 0.0
def _cbs_ref(substr):
    r=CBS_REF[CBS_REF.label.str.contains(substr,case=False,na=False,regex=False)] if len(CBS_REF) else CBS_REF
    if not len(r): return (np.nan,np.nan)
    e=r["err"].iloc[0]
    return (float(r["value"].iloc[0]), float(e) if pd.notna(e) else np.nan)

_cbs_arms=[a for a in A1_ORDER if len(CBS[(CBS.arm==a)&(CBS.metric=="nef1")])]
if _cbs_arms:
    vals=[_cbs_val(a) for a in _cbs_arms]; errs=[_cbs_err(a) for a in _cbs_arms]
    y=np.arange(len(_cbs_arms))[::-1]                 # first arm of A1_ORDER at the TOP
    fig,ax=plt.subplots(figsize=(STYLE["col15"],0.40*len(_cbs_arms)+1.3))
    ax.barh(y,vals,color=[rc_color(a) for a in _cbs_arms],edgecolor="white",lw=0.4,height=0.72,
            xerr=errs,error_kw=dict(lw=0.8,capsize=2.5,ecolor="#333"),zorder=3)
    for yi,v,e in zip(y,vals,errs):
        ax.text(v+e+0.015,yi,f"{v:.2f}",va="center",ha="left",fontsize=STYLE["fs_annot"],zorder=4)
    ax.set_yticks(y); ax.set_yticklabels([rc_label(a) for a in _cbs_arms],fontsize=STYLE["fs_annot"])
    ax.set_xlim(0,1.08); ax.set_xlabel("NEF1%  (normalized enrichment @ top-1%, provided 5-fold split)")
    # --- Truong reference overlays (from cbs_reference_lines.csv) ---
    tt,tte=_cbs_ref("target-trained")                # CBS-specific, structure-based
    if np.isfinite(tt):
        if np.isfinite(tte): ax.axvspan(tt-tte,tt+tte,color="#607d8b",alpha=0.13,zorder=0)
        ax.axvline(tt,color="#455a64",ls=(0,(5,2)),lw=1.15,zorder=2)
    ax.axvspan(0,0.125,color="#B00020",alpha=0.06,zorder=0)   # ligand-only + generic both ~0-0.13
    ax.axvline(0.0625,color="#B00020",ls=(0,(1,1.5)),lw=0.9,zorder=2)
    _tt_lab=(f"Truong 2026 target-trained (structure-based)  {tt:.3f} ± {tte:.3f}" if np.isfinite(tt) else "")
    _h=[plt.Line2D([],[],color="#455a64",ls=(0,(5,2)),lw=1.15,label=_tt_lab),
        plt.Line2D([],[],color="#B00020",ls=(0,(1,1.5)),lw=0.9,
                   label="Truong ligand-only (descriptors) & SOTA generic docking/co-folding  ≈ 0–0.13")]
    # legend BELOW the x-axis label (outside the plot) so it can't overlap the bars
    ax.legend(handles=_h,loc="upper center",bbox_to_anchor=(0.5,-0.16),fontsize=STYLE["fs_legend"]-0.5,
              framealpha=0.0,handlelength=2.4,ncol=1,borderaxespad=0.0)
    ax.margins(y=0.02)
    _suptitle(fig,"External validation — CBS inhibitor virtual screening (Truong 2026 benchmark)",
              fontsize=STYLE["fs_title"],y=1.0)
    _cap=("NEF1% on the benchmark's provided leakage-controlled 5-fold split (max inter-fold Tanimoto "
          "<0.70). Error bars = ±1 sd over 3 pretraining seeds (all but the two XGBoost anchors) or over "
          "the 5 folds (anchors). Morgan+desc+XGBoost (fp_desc) is best and clears Truong's target-trained "
          "structure-based models (0.764) with no docking/3D; MLM (unsup_only) does not beat the frozen "
          "random-init encoder. Faithful to the published benchmark (folds match Table 1, no leakage, their "
          "descriptor baseline reproduced at 0.07) so 'matches/beats structure-based ON THIS benchmark' is "
          "solid; the retrospective-VS decoy/analogue-bias limitation (Sieg 2019; Chen 2019) means a "
          "prospective-screener claim is not established. ROC-AUC is near-ceiling (0.92–0.995) and less "
          "informative.")
    _caption(fig,0.5,-0.16,"\n".join(_tw.wrap(_cap,118)),ha="center",va="top",
             fontsize=STYLE["fs_annot"]-0.5,color="#555")
    fig.subplots_adjust(bottom=0.14)
    save_fig(fig,"figCBS_external_validation"); plt.show()

    print("CBS external benchmark — NEF1% (provided folds), sorted:")
    for a in sorted(_cbs_arms,key=lambda k:-_cbs_val(k)):
        print(f"   {rc_label(a):<34} {_cbs_val(a):.3f} ± {_cbs_err(a):.3f}   (ROC-AUC {_cbs_val(a,'roc_auc'):.3f})")
    print(f"\n   reference: Truong target-trained {tt:.3f}±{tte:.3f}  ·  their ligand-only ~0.06  ·  SOTA generic ~0")
else:
    print("figCBS: experiment_cbs/cbs_nef1_summary.csv not found or empty.")
