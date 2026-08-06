# ---------- B1p1 · label-efficiency, train vs test (per-task FRACTION sweep) ----------
LE_SUM="analysis/rigor/label_efficiency_fractions_summary.csv"
# Frozen-probe label-efficiency at per-task FRACTIONS of each task's own hold-out train split
# {5,10,25,50,100%}. This REPLACES the earlier shared-absolute-budget sweep {100,300,1k,3k,full},
# which CAPPED on small tasks -- ESOL n=1000/3000/full were the identical 902 molecules, BACE/BBBP
# n=3000==full -- so the old figure plotted the same result 2-3x (reviewer-flagged). The x-axis is
# now n_train, which is per-task, so points sit at different x per task: that is the fix, not a bug.
# Regression is NATIVE units, matching the A1/A2 native-units fix (old label-eff RMSEs were in
# normalized units, ~2x smaller); classification (roc_auc/nef1) is unchanged.
# Arm keys mirror the A1/A2 encoders (scripts/label_eff_fractions.py, bit-identical to the
# eval_v2.evaluate single-hold-out path -- same _subsample_train rng, zscore standardizer fit on the
# subsample, target scaler, MLP head, native-unit unscale):
#   random    -> climb_v2_phase2/random_baseline_00   (random init, frozen)
#   unsup     -> climb_v2_phase2/unsup_8M             (MLM only -- no supervised labels at all)
#   sup       -> climb_v2_phase2/skip_dense_8M        (random init -> MTR on descriptors)
#   unsup2sup -> climb_v2_phase2/u2s_dense_from8M     (MLM 8M -> MTR on descriptors)
LE_REGIMES=[("random","no_pretrain (frozen)",rc_color("no_pretrain")),
            ("unsup","unsup_only (MLM, no labels)",rc_color("unsup_only")),
            ("sup","sup_only: dense (MTR)",rc_color("sup_only:dense")),
            ("unsup2sup","unsup→sup: dense (MTR)",rc_color("unsup2sup:dense")),
            # Declared so its absence is visible. It cannot come from this frozen-probe wave: every
            # series above is a frozen probe re-fit at each budget, whereas the end-to-end arm needs
            # the ENCODER re-fine-tuned at each budget -- a separate sweep re-run at the SAME 5
            # fractions (peer compute session, ~2.5h). Its rows will be appended to LE_SUM.
            ("e2e","no_pretrain (end-to-end)",rc_color("no_pretrain_e2e"))]

_led=pd.read_csv(LE_SUM) if os.path.exists(LE_SUM) else pd.DataFrame(
    columns=["arm","task","task_type","metric","split","n_train","mean","std"])
def _lemetric(task):        # this panel uses ONE shared train+test metric per task
    tt=_led[_led.task==task]
    return "rmse" if (len(tt) and tt.task_type.iloc[0]=="regression") else "roc_auc"
def le(reg,task,kind="test"):
    m=_lemetric(task)
    s=_led[(_led.arm==reg)&(_led.task==task)&(_led.split==kind)&(_led.metric==m)].sort_values("n_train")
    return s.n_train.to_numpy(), s["mean"].to_numpy(), s["std"].to_numpy()
def _kfmt(v):
    return (f"{v/1000:.1f}k".replace(".0k","k")) if v>=1000 else f"{int(round(v))}"

ncol=3; nrow=int(np.ceil(len(CORE_TASKS)/ncol))
fig,axes=plt.subplots(nrow,ncol,figsize=(STYLE["col2"],2.4*nrow)); axes=axes.ravel()
_le_missing=set()
for ax,task in zip(axes,CORE_TASKS):
    drew=False; xticks=None
    for reg,_lab,c in LE_REGIMES:
        xs,te,ee=le(reg,task,"test")
        if not len(xs): _le_missing.add(_lab); continue
        _,tr,_=le(reg,task,"train")
        drew=True; xticks=xs
        ax.errorbar(xs,te,yerr=ee,color=c,marker="o",ms=3.4,lw=STYLE["lw"],mec="white",
                    capsize=2,elinewidth=0.8,zorder=3)
        if len(tr)==len(xs):
            ax.plot(xs,tr,color=c,marker="^",ms=3.0,lw=STYLE["lw_thin"],ls=(0,(4,2)),alpha=0.85,zorder=2)
    # Every classification panel (incl. HIV) uses ROC-AUC here: this is the train-vs-test-gap
    # (fit/generalize) panel, so a shared train/test metric is required and NEF1% has no train-set
    # counterpart. NEF1% is the HIV headline in the screening figures (A1/A2).
    _t=("HIV (ROC-AUC ↑)" if task=="HIV" else ttitle(task,oneline=True))
    ax.set_title(_t,pad=6); ax.set_xlabel("# labeled training molecules")
    if xticks is not None:
        ax.set_xscale("log")
        ax.xaxis.set_major_locator(ticker.FixedLocator(list(xticks)))
        ax.xaxis.set_major_formatter(ticker.FixedFormatter([_kfmt(v) for v in xticks]))
        ax.xaxis.set_minor_locator(ticker.NullLocator()); ax.tick_params(axis="x",which="minor",bottom=False)
        ax.tick_params(axis="x",labelsize=STYLE["fs_annot"]-1.0)
    label_all_yticks(ax)
    if not drew: no_data_watermark(ax,"label-efficiency sweep missing")
for ax in axes[len(CORE_TASKS):]: ax.axis("off")
if _le_missing:                     # a legend entry with no line is a claim; say it is missing
    axes[0].text(0.02,0.02,"not run:\n"+"\n".join(sorted(_le_missing)),transform=axes[0].transAxes,
                 ha="left",va="bottom",fontsize=STYLE["fs_annot"]-1.5,color="#B00020",
                 linespacing=1.25,bbox=dict(fc="white",ec="none",alpha=0.85,pad=1.0))
for i in range(0,len(CORE_TASKS),ncol): axes[i].set_ylabel("metric value")

reg_h=[plt.Line2D([],[],color=c,marker="o",
                  label=lab+(" - NOT RUN" if lab in _le_missing else "")) for _,lab,c in LE_REGIMES]
sty_h=[plt.Line2D([],[],color="#444",marker="o",ls="-",label="test"),
       plt.Line2D([],[],color="#444",marker="^",ls=(0,(4,2)),label="train")]
fig.legend(handles=reg_h+sty_h,loc="upper center",bbox_to_anchor=(0.5,0.0),ncol=6,
           fontsize=STYLE["fs_legend"])
_suptitle(fig, "Fig B1p1 - label-efficiency and mechanism: does the frozen probe fit, or generalize?",
             fontsize=STYLE["fs_title"],y=1.075)
_b1note=("frozen probe re-fit at each label budget on the 8M encoders, single scaffold hold-out  ·  "
         "budgets are 5/10/25/50/100% of EACH task's own train split (x = # labeled training "
         "molecules, per-task -- this replaces the old shared absolute budget that capped small "
         "tasks)  ·  both SFT arms are the SAME `dense` (RDKit-MTR) recipe, so sup_only vs unsup→sup "
         "differs only in the MLM stage -- neither is an average over recipes  ·  3 subsample × 3 head "
         "seeds per point (100% = 3 head seeds; subsampling is a no-op at full)  ·  regression RMSE is "
         "in NATIVE units  ·  Tox21 'labels' = molecules (each carries up to 12 assay columns; AUC "
         "averaged over columns with both classes present)  ·  HIV keeps ROC-AUC here because this is "
         "the train-vs-test-gap panel and NEF1% has no train-set counterpart; NEF1% is the HIV "
         "headline in A1/A2")
_caption(fig, 0.5,0.995,"\n".join(_tw.wrap(_b1note,120)),
         ha="center",va="top",fontsize=STYLE["fs_annot"]-0.5,color="#666")
fig.subplots_adjust(top=0.86,bottom=0.12,hspace=0.52,wspace=0.34)
save_fig(fig,"figB1p1_label_efficiency_train_test"); plt.show()

# mean |train - test| gap, split by task type so native-unit RMSE (QM7 ~10^2) is not averaged
# together with unitless ROC-AUC. Both should shrink as labels grow (over-fit closes).
_CLS=[t for t in CORE_TASKS if _lemetric(t)=="roc_auc"]
_REG=[t for t in CORE_TASKS if _lemetric(t)=="rmse"]
def _gap(reg,tasks,which):
    g=[]
    for t in tasks:
        xs,te,_=le(reg,t,"test"); _,tr,_=le(reg,t,"train")
        if not len(xs) or len(tr)!=len(xs): continue
        j=(0 if which=="min" else len(xs)-1)
        if np.isfinite(te[j]) and np.isfinite(tr[j]): g.append(abs(tr[j]-te[j]))
    return np.mean(g) if g else np.nan
print("\nB1p1 mean |train - test| gap, smallest vs full budget:")
print(f"   {'':30} classification ROC-AUC        regression RMSE (native)")
for reg,lab,_ in LE_REGIMES:
    c0,c1=_gap(reg,_CLS,'min'),_gap(reg,_CLS,'max')
    r0,r1=_gap(reg,_REG,'min'),_gap(reg,_REG,'max')
    print(f"   {lab:<30} {c0:6.3f} -> {c1:6.3f}            {r0:7.3f} -> {r1:7.3f}")
