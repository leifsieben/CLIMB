# ---------- H1 · canonical vs enumerated SMILES (5-fold CV, 3 pretraining-seed replicates) ----------
# The round-1 wave (climb_v2) was hold-out only, had no HIV, and its encoders were never saved, so
# it could not be re-scored. The RETRAINED wave `climb_v2_h1` fixes all three: pooled 5-fold
# scaffold CV, all six tasks incl. HIV, and THREE pretraining-seed replicates per (mode, fraction)
# -- so H1 finally has an error bar (spread across seeds) instead of a single noisy hold-out draw.
H1_WAVE="climb_v2_h1"; H1_SUB="moleculenet_cv"; H1_SEEDS=[0,1,2]
H1_YPAD=0.02          # small fixed pad; the min-span below is what stops tiny gaps being oversold
H1_MIN_SPAN=0.20      # no panel may make a <0.2 difference fill its whole height
FRACS=[("frac0p001",0.001),("frac0p01",0.01),("frac0p1",0.1),("frac0p3",0.3),("fracfull",1.0)]

def _h1_seed_vals(mode,fk,task):
    """(mean, sd) of the task metric across the pretraining-seed replicates, or (nan, nan)."""
    sk=TASKS[task].get("suite_key",task); vs=[]
    for s in H1_SEEDS:
        d=load_suite(DATA_ROOT/H1_WAVE/f"scaling_{mode}_{fk}_s{s}",sub=H1_SUB)
        if d and (sk,"mean") in d: vs.append(d[(sk,"mean")])
    if not vs: return (np.nan,np.nan)
    return (float(np.mean(vs)), float(np.std(vs,ddof=1)) if len(vs)>1 else 0.0)

# Six panels in a single row overlapped titles, tick labels and axis labels. 3x2 matches A1.
ncol=3; nrow=int(np.ceil(len(CORE_TASKS)/ncol))
fig,axes=plt.subplots(nrow,ncol,figsize=(STYLE["col2"],2.35*nrow)); axes=axes.ravel()
for ax,task in zip(axes,CORE_TASKS):
    drew=False; _band=[]
    for mode,c,mk in [("canonical",PALETTE["blue"],"o"),("enumerated",PALETTE["red"],"s")]:
        xs,ys,es=[],[],[]
        for fk,fv in FRACS:
            m,sd=_h1_seed_vals(mode,fk,task)
            if np.isfinite(m): xs.append(fv); ys.append(m); es.append(sd)
        if xs:
            drew=True
            ax.errorbar(xs,ys,yerr=es,color=c,marker=mk,ms=3.5,lw=STYLE["lw"],mec="white",
                        capsize=2,elinewidth=0.8)
            _band+=[y-e for y,e in zip(ys,es)]+[y+e for y,e in zip(ys,es)]
    # Enforce a floor on the visible range: auto-scaling made Tox21/BBBP span ~0.02 of ROC-AUC --
    # differences well inside the seed spread, stretched to fill the panel. A fixed pad plus a
    # minimum span means a difference this small is DRAWN as small.
    if drew:
        _lo,_hi=min(_band)-H1_YPAD,max(_band)+H1_YPAD
        if (_hi-_lo)<H1_MIN_SPAN:
            _mid=(_lo+_hi)/2; _lo,_hi=_mid-H1_MIN_SPAN/2,_mid+H1_MIN_SPAN/2
        if "AUC" in TASKS[task]["metric"] and _hi>1.0:   # AUC ceiling: keep the span, drop top to 1.0
            _lo-=_hi-1.0; _hi=1.0
        ax.set_ylim(_lo,_hi)
    ax.set_xscale("log"); ax.set_title(ttitle(task,oneline=True),pad=6)
    ax.set_xlabel("unique-molecule\nfraction of corpus")
    ax.xaxis.set_minor_locator(ticker.NullLocator()); ax.tick_params(axis="x",which="minor",bottom=False)
    label_all_yticks(ax)
    if not drew: no_data_watermark(ax,f"no {H1_WAVE} CV data")
for ax in axes[len(CORE_TASKS):]: ax.axis("off")
for i in range(0,len(CORE_TASKS),ncol): axes[i].set_ylabel("metric value")
handles=[plt.Line2D([],[],color=PALETTE["blue"],marker="o",label="canonical (identical string re-shown)"),
         plt.Line2D([],[],color=PALETTE["red"], marker="s",label="enumerated (different valid SMILES)")]
fig.legend(handles=handles,loc="upper center",bbox_to_anchor=(0.5,0.02),ncol=2,fontsize=STYLE["fs_legend"])
_h1scheme=(f"pooled 5-fold scaffold CV · {len(H1_SEEDS)} pretraining-seed replicates "
           f"(error bars = ±1 sd across seeds) · all six tasks incl. HIV")
_suptitle(fig, "Fig H1 - does SMILES enumeration beat canonical repetition? (unsup_only)",
             fontsize=STYLE["fs_title"],y=1.06)
_caption(fig, 0.5,0.995,"\n".join(_tw.wrap(f"SCHEME: {_h1scheme}",110)),ha="center",va="top",
         fontsize=STYLE["fs_annot"]-0.5,color="#666")
fig.subplots_adjust(top=0.90,bottom=0.14,hspace=0.55,wspace=0.36)
save_fig(fig,"figH1_canonical_vs_enumerated"); plt.show()

print(f"H1 scheme: {_h1scheme}")
print("H1 enumerated minus canonical (positive = enumeration better), by unique-molecule fraction:")
for fk,fv in FRACS:
    d=[]
    for t in CORE_TASKS:
        cm,_=_h1_seed_vals("canonical",fk,t); em,_=_h1_seed_vals("enumerated",fk,t)
        if np.isfinite(cm) and np.isfinite(em): d.append(lift(em,t,cm))
    if d: print(f"   frac={fv:<6} mean relative change {100*np.nanmean(d):+.2f}%  (n={len(d)})")
