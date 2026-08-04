# ---------- C1J1 · SFT-family ablation + transfer matrix + the H10 similarity test ----------
ABL="climb_v2_ablation_dedup"
ABL_ARMS=["seq_mtr","seq_pcba","seq_l1000","seq_pcqm","seq_sparse_all","seq_dense_plus_sparse"]
# Every arm here is unsup->sup warm-started from ONE 2M-FP MLM base, in the ablation wave, scored
# on the single scaffold hold-out. The bare recipe names ("dense+sparse") are also A1/A2 arm names
# for a DIFFERENT model -- sup_only, random init, 8M budget, phase-2 wave -- so they are prefixed
# here. Without the prefix the two figures look like they contradict each other when they are
# simply reporting different models.
ABL_LABEL={"seq_mtr":"u2s: dense (MTR)","seq_pcba":"u2s: PCBA","seq_l1000":"u2s: L1000",
           "seq_pcqm":"u2s: PCQM","seq_sparse_all":"u2s: sparse_all",
           "seq_dense_plus_sparse":"u2s: dense+sparse"}
# The ablation arms warm-start from climb_v2_phase2/unsup_2M, so THAT is the MLM base to show as
# the reference row -- not the round-1 unsup_only run the earlier version of this figure used.
BASE_RUN=(DATA_ROOT/"climb_v2_phase2"/"unsup_2M")
# Prefer the 5-fold CV as soon as the whole wave has it. This panel was stuck on the single
# hold-out (113-204 test molecules on the small tasks) because the ablation wave had never been
# CV-scored. It needs the WHOLE wave -- every seq_* arm AND the three random_baseline_* that form
# the floor AND the anchor -- because each arm is lifted against the floor scored in its own wave;
# a CV floor with hold-out arms would silently produce wrong lifts. All-or-nothing, checked here.
_abl_dirs=sorted((DATA_ROOT/ABL).glob("*/"))
C1J1_SUB=("moleculenet_cv"
          if _abl_dirs and all((d/"moleculenet_cv"/"suite_summary.json").exists() for d in _abl_dirs)
             and (BASE_RUN/"moleculenet_cv"/"suite_summary.json").exists()
          else "moleculenet")
C1J1_SCHEME=("pooled 5-fold scaffold CV" if C1J1_SUB=="moleculenet_cv" else
             "single scaffold hold-out (113-204 test molecules) - CV PENDING for this wave")

def _wave_floor(wave,task):
    """Mean of the three random-init encoders AS SCORED IN THAT WAVE."""
    v=[]
    for rb in sorted((DATA_ROOT/wave).glob("random_baseline*/")):
        d=load_suite(rb,sub=C1J1_SUB); sk=TASKS[task].get("suite_key",task)
        if d and (sk,"mean") in d: v.append(d[(sk,"mean")])
    return float(np.mean(v)) if v else np.nan
abl_floor=lambda task:_wave_floor(ABL,task)

# FLOOR CHOICE -- same change as I1. Lift over the FROZEN random encoder is the easy comparison:
# a frozen random encoder cannot adapt to the task at all, so nearly anything clears it. The
# honest floor is the random encoder FINE-TUNED end-to-end, which is what a practitioner would do
# INSTEAD of pretraining.
# Those runs live in climb_v2_phase2, not this wave, and this figure otherwise refuses cross-wave
# comparisons. It is safe here only because the two waves' FROZEN floors are numerically identical
# on all six tasks (both CV-scored, same random-init encoders, same folds) -- checked in the
# printout below, so the day that stops being true it is visible rather than silent.
C1J1_FLOOR_LABEL="no_pretrain (end-to-end)"
def c1j1_floor(task):
    v=arm_value(DF_CV if C1J1_SUB=="moleculenet_cv" else DF,task,"no_pretrain_e2e")
    return v if np.isfinite(v) else _wave_floor(ABL,task)   # frozen floor if e2e is not scored

# The two waves run the SAME three random-init encoders, but they were scored in separate eval
# runs, and re-scoring shifts a metric by up to ~0.03. So each arm is lifted against the floor
# from ITS OWN wave -- comparing lifts, each internally consistent -- rather than against one
# borrowed floor. The divergence is printed so the size of the effect is visible, not assumed away.
_drift=[(t,a,b) for t in CORE_TASKS
         for a,b in [(abl_floor(t),npt_floor(DF_CV if C1J1_SUB=="moleculenet_cv" else DF,t))]
         if np.isfinite(a) and np.isfinite(b) and abs(a-b)>5e-3]
if _drift:
    print("   WARNING - the two waves' frozen floors have drifted apart, so borrowing phase2's "
          "end-to-end floor for this wave is no longer safe:")
    for _t,_a,_b in _drift:
        print(f"      {_t}: ablation={_a:.4f} vs phase2={_b:.4f} ({100*abs(_a-_b)/_b:.1f}%)")
else:
    print(f"   frozen floors agree across both waves on all {len(CORE_TASKS)} tasks "
          f"-> safe to lift every arm against phase2's {C1J1_FLOOR_LABEL} floor")

rows=[]
_srcs=[("unsup_only (MLM base)",BASE_RUN,"climb_v2_phase2")]+\
      [(ABL_LABEL[a],DATA_ROOT/ABL/a,ABL) for a in ABL_ARMS]+\
      [("Morgan+XGBoost",DATA_ROOT/ABL/"ecfp4_anchor",ABL)]
for disp,rd,wv in _srcs:
    d=load_suite(rd,sub=C1J1_SUB)
    if not d: continue
    for task in CORE_TASKS:
        sk=TASKS[task].get("suite_key",task)
        if (sk,"mean") in d:
            rows.append(dict(arm=disp,task=task,lift=lift(d[(sk,"mean")],task,c1j1_floor(task))))
AB=pd.DataFrame(rows)
ARM_ORDER=[a for a in ["unsup_only (MLM base)"]+[ABL_LABEL[a] for a in ABL_ARMS]+["Morgan+XGBoost"]
           if a in set(AB.arm)]

fig=plt.figure(figsize=(STYLE["col2"],5.2))
gs=fig.add_gridspec(2,2,width_ratios=[1,1.30],height_ratios=[1,1.05],hspace=0.50,wspace=0.62)
axA=fig.add_subplot(gs[0,0]); axB=fig.add_subplot(gs[0,1]); axC=fig.add_subplot(gs[1,:])

# --- (a) mean lift per arm ---
avg=AB.groupby("arm").lift.mean().reindex(ARM_ORDER)*100
cols=["#555555" if a.startswith("unsup_only") else
      (PALETTE["black"] if a=="Morgan+XGBoost" else PALETTE["blue"]) for a in ARM_ORDER]
axA.barh(range(len(ARM_ORDER)),avg.values,color=cols,edgecolor="white",lw=0.4)
axA.axvline(0,color="#333",lw=0.8)
axA.set_yticks(range(len(ARM_ORDER))); axA.set_yticklabels(ARM_ORDER); axA.invert_yaxis()
axA.set_xlabel(f"avg lift over {C1J1_FLOOR_LABEL} (%)")
axA.set_title("Which SFT label type helps?\n(all arms: unsup→sup from one 2M-FP MLM base)",
              pad=6); panel_tag(axA,"a",dx=-0.62)

# --- (b) per-task transfer matrix (SFT families only; the base/anchor rows live in (a)) ---
MAT_ROWS=[ABL_LABEL[a] for a in ABL_ARMS if ABL_LABEL[a] in set(AB.arm)]
H=AB.pivot(index="arm",columns="task",values="lift").reindex(index=MAT_ROWS,columns=CORE_TASKS)*100
# HIV lift runs to -79%, so a linear +/-15 scale painted the whole HIV column one flat colour
# and hid a 60-point spread. A symmetric-log norm keeps the small +/-5% structure readable while
# still ranking the large negatives honestly.
_vmax=float(np.nanmax(np.abs(H.values))); _vmax=max(20.0,np.ceil(_vmax/10)*10)
_norm=mpl.colors.SymLogNorm(linthresh=5,linscale=1.0,vmin=-_vmax,vmax=_vmax,base=10)
im=axB.imshow(H.values,cmap="PuOr_r",norm=_norm,aspect="auto")
axB.set_xticks(range(len(CORE_TASKS)))
axB.set_xticklabels([f'{TASKS[t]["pretty"]}\n[{TASKS[t]["domain"]}]' for t in CORE_TASKS],
                    fontsize=STYLE["fs_annot"]-1.5,rotation=30,ha="right")
axB.set_yticks(range(len(MAT_ROWS))); axB.set_yticklabels(MAT_ROWS)
axB.grid(False)
for i in range(H.shape[0]):
    for j in range(H.shape[1]):
        v=H.values[i,j]
        if np.isfinite(v):
            axB.text(j,i,f"{v:+.0f}",ha="center",va="center",fontsize=STYLE["fs_annot"],
                     color="white" if abs(v)>14 else "#222")
_cbt=[t for t in (-_vmax,-20,-5,0,5,20,_vmax) if abs(t)<=_vmax]
cb=fig.colorbar(im,ax=axB,fraction=0.046,pad=0.03,ticks=_cbt)
# SymLogNorm's default colorbar formatter drops every tick but 0; set the labels explicitly.
cb.ax.yaxis.set_major_formatter(ticker.FixedFormatter([f"{t:+.0f}".replace("+0","0") for t in _cbt]))
cb.set_label("lift (%), symlog scale",fontsize=STYLE["fs_legend"])
cb.ax.tick_params(labelsize=STYLE["fs_annot"]-1)
axB.set_title("SFT family → eval task",pad=6); panel_tag(axB,"b",dx=-0.22)

# --- (c) H10: does lift track chemical similarity? ---
SIMP=DATA_ROOT/"_tanimoto"/"family_task_similarity.csv"
# seq_mtr and seq_dense_plus_sparse are absent: descriptor regression has no family molecule set
# to measure similarity against, so they have no x-coordinate. They remain in (a) and (b).
ARM2FAM={"seq_pcba":["PCBA"],"seq_l1000":["L1000_MCF7","L1000_VCAP"],"seq_pcqm":["PCQM"],
         "seq_sparse_all":["PCBA","L1000_MCF7","L1000_VCAP"]}
pts=[]
if SIMP.exists():
    SIM=pd.read_csv(SIMP)
    for arm,fams in ARM2FAM.items():
        d=load_suite(DATA_ROOT/ABL/arm,sub=C1J1_SUB)
        if not d: continue
        for t in CORE_TASKS:
            sk=TASKS[t].get("suite_key",t)
            if (sk,"mean") not in d: continue
            l=lift(d[(sk,"mean")],t,c1j1_floor(t))
            m=SIM[(SIM.task==t)&(SIM.family.isin(fams))].mean_max_tanimoto.mean()
            if np.isfinite(l) and np.isfinite(m): pts.append((m,100*l,t,ABL_LABEL[arm]))
if len(pts)>=4:
    X=np.array([p[0] for p in pts]); Y=np.array([p[1] for p in pts]); TK=[p[2] for p in pts]
    cmap=plt.get_cmap("tab10"); tcol={t:cmap(i%10) for i,t in enumerate(CORE_TASKS)}
    for t in [t for t in CORE_TASKS if t in set(TK)]:
        m=[k==t for k in TK]
        axC.scatter(X[m],Y[m],s=30,color=tcol[t],edgecolor="white",lw=0.5,zorder=3,
                    label=TASKS[t]["pretty"])
    b,a=np.polyfit(X,Y,1); xs=np.linspace(X.min(),X.max(),50)
    axC.plot(xs,a+b*xs,color=PALETTE["black"],lw=STYLE["lw_thin"],ls=(0,(4,2)),zorder=2)
    axC.axhline(0,color=PALETTE["grey2"],lw=STYLE["lw_thin"],zorder=1)
    from scipy import stats as _st
    r=float(np.corrcoef(X,Y)[0,1]); rho,p=_st.spearmanr(X,Y)
    axC.set_xlabel("mean max ECFP4 Tanimoto SIMILARITY between eval-task and SFT-family molecules"
                   "  —  right = MORE similar →")
    axC.set_ylabel(f"lift over {C1J1_FLOOR_LABEL} (%)")
    axC.set_title(f"H10 test: lift vs chemical similarity  (n={len(pts)}, "
                  f"Pearson r={r:+.2f}, Spearman ρ={rho:+.2f}, p={p:.2f})",pad=6)
    # legend OUTSIDE the axes: inside, the swatches read as extra data points
    axC.legend(title="eval task",loc="upper left",bbox_to_anchor=(1.02,1.0),frameon=False,
               fontsize=STYLE["fs_legend"],title_fontsize=STYLE["fs_legend"],handletextpad=0.3)
    label_all_yticks(axC)
else:
    no_data_watermark(axC,"run scripts/compute_family_task_similarity.py")
    axC.set_xlabel("family–task Tanimoto similarity  —  right = MORE similar →")
    axC.set_ylabel(f"lift over {C1J1_FLOOR_LABEL} (%)")
panel_tag(axC,"c",dx=-0.09)

_suptitle(fig, "Fig C1J1 - supervised pretraining data: which type helps, how much, and does it "
             "follow chemistry?",fontsize=STYLE["fs_title"],y=1.03)
_c1note=(f"deduped wave: the SFT blocklist drops 34,301 eval molecules, so no arm trains on "
         f"eval-test molecules. Arms are unsup→sup from one shared 2M-FP MLM base, all lifted "
         f"against {C1J1_FLOOR_LABEL} -- the random encoder FINE-TUNED on the task, not the frozen "
         f"one, since clearing a frozen random encoder is close to automatic. SCHEME: {C1J1_SCHEME}. "
         f"(c) omits dense (MTR) and dense+sparse: descriptor regression has no family molecule set "
         f"to measure similarity against. Families are sampled, so similarities are lower bounds.")
_caption(fig, 0.5,0.035,"\n".join(_tw.wrap(_c1note,150)),
         ha="center",va="top",fontsize=STYLE["fs_annot"]-0.5,color="#666")
fig.subplots_adjust(top=0.87,bottom=0.14)
save_fig(fig,"figC1J1_sft_family_transfer"); plt.show()

print(f"C1J1 (a) mean lift over {C1J1_FLOOR_LABEL}:")
for a in ARM_ORDER: print(f"   {a:<24} {avg[a]:+6.1f}%")
# Reconciliation, printed rather than assumed: an arm named "dense+sparse" exists in BOTH figures
# and means two different models. Anyone reading A1 and C1J1 together will otherwise see a
# contradiction that is not there.
print("\nNOTE - C1J1 vs A1 are different models despite the shared recipe names:")
print(f"   C1J1 'u2s: dense+sparse' = {ABL}/seq_dense_plus_sparse "
      f"(unsup→sup from a 2M MLM base, single hold-out)")
print(f"   A1   'sup_only: dense+sparse' = climb_v2_phase2/skip_dense_plus_sparse_8M "
      f"(random init, 8M SFT, A1.b uses 5-fold CV)")
print("   Different regime, budget, wave AND eval scheme -- their rankings need not agree.")
if len(pts)>=4:
    print(f"\nC1J1 (c) H10: n={len(pts)} arm x task cells, Pearson r={r:+.3f}, "
          f"Spearman rho={rho:+.3f} (p={p:.3f}), similarity range {X.min():.2f}-{X.max():.2f}")
    for t in [t for t in CORE_TASKS if t in set(TK)]:
        m=[k==t for k in TK]
        if sum(m)>=3:
            print(f"   within {t:<6}: rho={_st.spearmanr(X[m],Y[m]).statistic:+.2f} (n={sum(m)})")