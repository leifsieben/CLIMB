# ---------- B1p1 · label-efficiency, train vs test ----------
LE="figure_data/climb_v2_labeleff_v2"
# Directory prefix -> (label, colour). The labels name the EXACT encoder behind each series
# (scripts/b1_replicates_v2.sh), because "sup_only" and "unsup→sup" are five-recipe families
# elsewhere in this notebook and an unqualified label here would read as an average over them.
# It is not an average: both SFT arms are the `dense` (RDKit-MTR) recipe at the 8M rung, chosen so
# they differ ONLY in whether an MLM stage came first. Colours match A1/A2 for the same models.
#   random    -> climb_v2_phase2/random_baseline_00   (random init, frozen)
#   unsup     -> climb_v2_phase2/unsup_8M             (MLM only -- no supervised labels at all)
#   sup       -> climb_v2_phase2/skip_dense_8M        (random init -> MTR on descriptors)
#   unsup2sup -> climb_v2_phase2/u2s_dense_from8M     (MLM 8M -> MTR on descriptors)
LE_REGIMES=[("random","no_pretrain (frozen)",rc_color("no_pretrain")),
            ("unsup","unsup_only (MLM, no labels)",rc_color("unsup_only")),
            ("sup","sup_only: dense (MTR)",rc_color("sup_only:dense")),
            ("unsup2sup","unsup→sup: dense (MTR)",rc_color("unsup2sup:dense")),
            # Declared so its absence is visible. It cannot come from this wave: every series
            # above is a frozen probe re-fit at each label budget, whereas the end-to-end arm
            # needs the ENCODER re-fine-tuned at each budget -- a separate sweep that has not
            # been run (scripts/b1_replicates_v2.sh says so explicitly).
            ("e2e","no_pretrain (end-to-end)",rc_color("no_pretrain_e2e"))]
LE_SIZES=[("100",100),("300",300),("1000",1000),("3000",3000),("0",10000)]

_LE={}
for f in glob.glob(f"{LE}/*/moleculenet/moleculenet_summary.csv"):
    t=f.split("/")[-3].split("_"); size=t[-2].replace("n",""); reg="_".join(t[:-2])
    for _,r in pd.read_csv(f).query("head_seed=='MEAN'").iterrows():
        mm=str(r.main_metric)
        if   mm in ("rmse","roc_auc"): kind="test"
        elif mm.endswith("_train"):    kind="train"
        else: continue                            # nef1 etc: no train counterpart, skip
        _LE.setdefault((reg,size,r.dataset,kind),[]).append(r.main_value)
def le(reg,size,task,kind="test"):
    v=_LE.get((reg,size,task,kind),[])
    return (float(np.mean(v)), float(np.std(v)) if len(v)>1 else 0.0) if v else (np.nan,0.0)

ncol=3; nrow=int(np.ceil(len(CORE_TASKS)/ncol))
fig,axes=plt.subplots(nrow,ncol,figsize=(STYLE["col2"],2.4*nrow)); axes=axes.ravel()
xs=[x for _,x in LE_SIZES]
_le_missing=set()
for ax,task in zip(axes,CORE_TASKS):
    drew=False
    for reg,_lab,c in LE_REGIMES:
        te=[le(reg,s,task,"test")[0]  for s,_ in LE_SIZES]
        tr=[le(reg,s,task,"train")[0] for s,_ in LE_SIZES]
        ee=[le(reg,s,task,"test")[1]  for s,_ in LE_SIZES]
        if all(not np.isfinite(v) for v in te): _le_missing.add(_lab); continue
        drew=True
        ax.errorbar(xs,te,yerr=ee,color=c,marker="o",ms=3.4,lw=STYLE["lw"],mec="white",
                    capsize=2,elinewidth=0.8,zorder=3)
        ax.plot(xs,tr,color=c,marker="^",ms=3.0,lw=STYLE["lw_thin"],ls=(0,(4,2)),alpha=0.85,zorder=2)
    # HIV is scored by ROC-AUC in this figure (train has no NEF1% counterpart) -- the title has
    # to say so, or it silently contradicts the axis.
    _t=("HIV (ROC-AUC ↑)" if task=="HIV" else ttitle(task,oneline=True))
    ax.set_xscale("log"); ax.set_title(_t,pad=6)
    ax.set_xlabel("# training labels")
    ax.xaxis.set_major_locator(ticker.FixedLocator(xs))
    ax.xaxis.set_major_formatter(ticker.FixedFormatter(["100","300","1k","3k","full"]))
    ax.xaxis.set_minor_locator(ticker.NullLocator()); ax.tick_params(axis="x",which="minor",bottom=False)
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
_b1note=("frozen probe re-fit at each label budget on the 8M encoders  ·  both SFT arms are the "
         "SAME `dense` (RDKit-MTR) recipe, so sup_only vs unsup→sup differs only in the MLM stage "
         "-- neither is an average over recipes  ·  3 subsample draws × 3 head seeds per point  ·  "
         "HIV scored by ROC-AUC here so train and test share a metric")
_caption(fig, 0.5,0.995,"\n".join(_tw.wrap(_b1note,120)),
         ha="center",va="top",fontsize=STYLE["fs_annot"]-0.5,color="#666")
fig.subplots_adjust(top=0.86,bottom=0.12,hspace=0.48,wspace=0.34)
save_fig(fig,"figB1p1_label_efficiency_train_test"); plt.show()

print("\nB1p1 mean |train - test| gap, smallest vs largest label budget:")
for reg,lab,_ in LE_REGIMES:
    out=[]
    for size in ("100","0"):
        g=[abs(le(reg,size,t,"train")[0]-le(reg,size,t,"test")[0]) for t in CORE_TASKS]
        g=[v for v in g if np.isfinite(v)]
        out.append(np.mean(g) if g else np.nan)
    print(f"   {lab:<30} n=100: {out[0]:.3f}   full: {out[1]:.3f}")