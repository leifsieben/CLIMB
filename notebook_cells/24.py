# ---------- H1 · canonical vs enumerated SMILES ----------
# SCHEME: this round-1 wave was only ever scored on the single DeepChem scaffold hold-out, so the
# y-values are 113-204-molecule estimates on the small tasks -- not the 5-fold CV every other
# panel now uses. It CANNOT simply be re-scored: `climb_v2`'s encoder weights were never uploaded
# (zero `encoder/` objects under s3://climb-s3-bucket/experiments/climb_v2/), so moving H1 to CV
# means RETRAINING all ten runs -- which is now under way. This cell upgrades itself the moment a
# CV eval appears, so no edit is needed when that lands.
H1_SUB="moleculenet_cv" if list(Path(f"{DATA_ROOT}/climb_v2").glob("scaling_*/moleculenet_cv")) else "moleculenet"
FRACS=[("frac0p001",0.001),("frac0p01",0.01),("frac0p1",0.1),("frac0p3",0.3),("fracfull",1.0)]
# Six panels in a single row overlapped titles, tick labels and axis labels. 3x2 matches A1.
ncol=3; nrow=int(np.ceil(len(CORE_TASKS)/ncol))
fig,axes=plt.subplots(nrow,ncol,figsize=(STYLE["col2"],2.35*nrow)); axes=axes.ravel()
for ax,task in zip(axes,CORE_TASKS):
    drew=False
    for mode,c,mk in [("canonical",PALETTE["blue"],"o"),("enumerated",PALETTE["red"],"s")]:
        xs,ys=[],[]
        for fk,fv in FRACS:
            d=load_suite(DATA_ROOT/"climb_v2"/f"scaling_{mode}_{fk}",sub=H1_SUB)
            sk=TASKS[task].get("suite_key",task)
            if d and (sk,"mean") in d: xs.append(fv); ys.append(d[(sk,"mean")])
        if xs: drew=True; ax.plot(xs,ys,color=c,marker=mk,ms=3.5,lw=STYLE["lw"],mec="white")
    ax.set_xscale("log"); ax.set_title(ttitle(task,oneline=True),pad=6)
    ax.set_xlabel("unique-molecule\nfraction of corpus")
    ax.xaxis.set_minor_locator(ticker.NullLocator()); ax.tick_params(axis="x",which="minor",bottom=False)
    label_all_yticks(ax)
    if not drew: no_data_watermark(ax,"HIV post-dates this\nround-1 sweep")
for ax in axes[len(CORE_TASKS):]: ax.axis("off")
for i in range(0,len(CORE_TASKS),ncol): axes[i].set_ylabel("metric value")
handles=[plt.Line2D([],[],color=PALETTE["blue"],marker="o",label="canonical (identical string re-shown)"),
         plt.Line2D([],[],color=PALETTE["red"], marker="s",label="enumerated (different valid SMILES)")]
fig.legend(handles=handles,loc="upper center",bbox_to_anchor=(0.5,0.02),ncol=2,fontsize=STYLE["fs_legend"])
_h1scheme=("pooled 5-fold scaffold CV" if H1_SUB=="moleculenet_cv" else
           "single scaffold hold-out (113-204 test molecules) - this wave has NO CV eval and its "
           "encoders were never saved; the ten runs are being RETRAINED so this can move to CV")
fig.suptitle("Fig H1 - does SMILES enumeration beat canonical repetition? (unsup_only)",
             fontsize=STYLE["fs_title"],y=1.06)
fig.text(0.5,0.995,"\n".join(_tw.wrap(f"SCHEME: {_h1scheme}",110)),ha="center",va="top",
         fontsize=STYLE["fs_annot"]-0.5,color=("#666" if H1_SUB=="moleculenet_cv" else "#B00020"))
fig.subplots_adjust(top=0.90,bottom=0.14,hspace=0.55,wspace=0.36)
save_fig(fig,"figH1_canonical_vs_enumerated"); plt.show()

print(f"H1 scheme: {_h1scheme}")
print("H1 enumerated minus canonical (positive = enumeration better), by unique-molecule fraction:")
for fk,fv in FRACS:
    ds={m:load_suite(DATA_ROOT/"climb_v2"/f"scaling_{m}_{fk}",sub=H1_SUB) for m in ("canonical","enumerated")}
    if not all(ds.values()): continue
    d=[]
    for t in CORE_TASKS:
        sk=TASKS[t].get("suite_key",t)
        if all((sk,"mean") in ds[m] for m in ds):
            f=ds["canonical"][(sk,"mean")]
            d.append(lift(ds["enumerated"][(sk,"mean")],t,f))
    if d: print(f"   frac={fv:<6} mean relative change {100*np.nanmean(d):+.2f}%  (n={len(d)})")