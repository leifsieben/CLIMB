# ---------- I1 · memorization vs representation ----------
TANI=DATA_ROOT/"_tanimoto"/"corpus_similarity.csv"
I1_MODEL=f"unsup_{MATCHED_BUDGET}"
# BASELINE CHOICE IS THE WHOLE POINT OF THIS PANEL.
# Lift over the FROZEN random encoder is close to trivial: that baseline cannot adapt to the task
# at all, so almost any pretraining clears it and the resulting "+34% on ESOL" says more about the
# baseline than about pretraining. The honest comparator is the random-init encoder FINE-TUNED
# end-to-end -- what a practitioner would actually do INSTEAD of pretraining. Lift over that is
# what decides whether the MLM stage buys anything, and it may well be ~0.
I1_BASELINES={"no_pretrain_e2e":["e2e_random_00","e2e_random_01","e2e_random_02"],
              "no_pretrain"    :["random_baseline_00","random_baseline_01","random_baseline_02"]}
I1_BASE_KEY="no_pretrain_e2e"          # primary; the frozen one is printed only as contrast
I1_TASKS=[t for t in ("ESOL","QM7") if t in TASKS]

def _preds(run):
    # per-molecule predictions exist only for the CV evaluation, so BOTH the model and its
    # baselines are read from CV here -- same scheme on both sides of the comparison.
    p=DATA_ROOT/"climb_v2_phase2"/run/"moleculenet_cv"/"test_predictions.csv"
    if not p.exists(): return None
    d=pd.read_csv(p)
    return (d.groupby(["dataset","raw_smiles"],as_index=False)
              .agg(y_true=("y_true","first"),y_pred=("y_pred","mean")))

def _base_preds(base_key):
    """Mean prediction across the baseline's replicates, or None if none are scored yet."""
    bs=[b for b in (_preds(r) for r in I1_BASELINES[base_key]) if b is not None]
    if not bs: return None
    return (pd.concat(bs).groupby(["dataset","raw_smiles"],as_index=False)
              .agg(y_true=("y_true","first"),y_pred=("y_pred","mean")))

def _lift_rmse(se_m,se_b):
    rm,rb=np.sqrt(np.mean(se_m)),np.sqrt(np.mean(se_b))
    return np.nan if (not np.isfinite(rb) or rb==0) else 100*(rb-rm)/rb

def binned_lift(task,base_key=None,nbins=5,nboot=400,seed=0):
    """-> (centres, lift%, lo, hi, n_per_bin) with a percentile bootstrap over molecules."""
    base_key=base_key or I1_BASE_KEY
    if not TANI.exists(): return (None,)*5
    mod=_preds(I1_MODEL); base=_base_preds(base_key)
    if mod is None or base is None: return (None,)*5
    m=mod[mod.dataset==task].merge(base[base.dataset==task],on=["dataset","raw_smiles"],
                                   suffixes=("_m","_b"))
    m=m.merge(pd.read_csv(TANI)[["raw_smiles","max_tanimoto_to_corpus"]],on="raw_smiles",how="inner")
    if len(m)<50: return (None,)*5
    m["se_m"]=(m.y_pred_m-m.y_true_m)**2; m["se_b"]=(m.y_pred_b-m.y_true_b)**2
    edges=m.max_tanimoto_to_corpus.quantile(np.linspace(0,1,nbins+1)).values; edges[0]-=1e-9
    m["bin"]=pd.cut(m.max_tanimoto_to_corpus,bins=edges,labels=False,include_lowest=True)
    rng=np.random.default_rng(seed); xs,ys,los,his,ns=[],[],[],[],[]
    for b in range(nbins):
        s=m[m.bin==b]
        if len(s)<15: continue
        sm,sb=s.se_m.values,s.se_b.values
        pt=_lift_rmse(sm,sb)
        if not np.isfinite(pt): continue
        idx=rng.integers(0,len(sm),size=(nboot,len(sm)))
        boot=np.array([_lift_rmse(sm[i],sb[i]) for i in idx]); boot=boot[np.isfinite(boot)]
        lo,hi=(np.percentile(boot,[2.5,97.5]) if len(boot) else (np.nan,np.nan))
        xs.append(float(s.max_tanimoto_to_corpus.mean())); ys.append(pt)
        los.append(lo); his.append(hi); ns.append(int(len(s)))
    return xs,ys,los,his,ns

def quartile_lift(task,base_key=None):
    """(most-similar, most-novel) lift% with bootstrap CIs, from the outer quartiles."""
    xs,ys,lo,hi,_=binned_lift(task,base_key,nbins=4)
    if not xs or len(ys)<4: return None
    return (ys[-1],lo[-1],hi[-1]),(ys[0],lo[0],hi[0])

_I1_BASE_LABEL={"no_pretrain_e2e":"no_pretrain (end-to-end)","no_pretrain":"no_pretrain (frozen)"}
_I1_HAVE=_base_preds(I1_BASE_KEY) is not None
_I1_NEED=("needs e2e_random_0*/moleculenet_cv\n(E1 end-to-end wave still running)"
          if not _I1_HAVE else "needs corpus_similarity.csv")

# Height + bottom margin reserve room for the caption INSIDE the canvas; at negative figure
# coords it only clears in the exported PNG (bbox="tight" grows it), not in the inline render.
fig,(ax0,ax1)=plt.subplots(1,2,figsize=(STYLE["col2"],3.3))
pairs=[(t,v) for t,v in ((t,quartile_lift(t)) for t in I1_TASKS) if v]
if pairs:
    sim=float(np.mean([v[0][0] for _,v in pairs])); nov=float(np.mean([v[1][0] for _,v in pairs]))
    _se=lambda k: float(np.sqrt(np.sum([((v[k][2]-v[k][1])/2/len(pairs))**2 for _,v in pairs])))
    se_s,se_n=_se(0),_se(1)
    ax0.bar([0,1],[sim,nov],color=[PALETTE["purple"],PALETTE["teal"]],width=0.6,
            yerr=[se_s,se_n],capsize=STYLE["cap_size"],error_kw=dict(lw=STYLE["lw_thin"]))
    for xi,v,e in zip([0,1],[sim,nov],[se_s,se_n]):
        ax0.text(xi,v+e+0.6,f"{v:+.1f}%",ha="center",fontsize=STYLE["fs_annot"])
    ax0.axhline(0,color=PALETTE["black"],lw=0.6)
    lo_,hi_=min(0,sim-se_s,nov-se_n),max(0,sim+se_s,nov+se_n)
    ax0.set_ylim(lo_-abs(lo_)*0.35-2,hi_+abs(hi_)*0.35+3)
else:
    ax0.set_ylim(-5,15); no_data_watermark(ax0,_I1_NEED)
ax0.set_xticks([0,1]); ax0.set_xticklabels(["most corpus-similar\n(top quartile)","most novel\n(bottom quartile)"])
_ylab=f"lift over {_I1_BASE_LABEL[I1_BASE_KEY]} (%)"
ax0.set_ylabel(_ylab); label_all_yticks(ax0); panel_tag(ax0,"a",dx=-0.20)

col={"ESOL":PALETTE["blue"],"QM7":PALETTE["orange"]}; drew=False
for t in I1_TASKS:
    xs,ys,lo,hi,nn=binned_lift(t)
    if not xs: continue
    drew=True
    err=np.vstack([np.array(ys)-np.array(lo),np.array(hi)-np.array(ys)])
    ax1.errorbar(xs,ys,yerr=err,color=col.get(t,PALETTE["grey"]),marker="o",lw=STYLE["lw"],
                 capsize=STYLE["cap_size"],label=f"{t} (n/bin≈{int(np.median(nn))})")
if drew:
    ax1.axhline(0,color=PALETTE["black"],lw=0.6)
    ax1.legend(loc="best",fontsize=STYLE["fs_legend"])
else:
    ax1.set_ylim(-5,15); no_data_watermark(ax1,_I1_NEED)
ax1.set_xlabel("max ECFP4 Tanimoto to corpus (bin mean)\nright = MORE similar →")
ax1.set_ylabel(_ylab); label_all_yticks(ax1); panel_tag(ax1,"b",dx=-0.18)

_suptitle(fig, "Fig I1 - memorization or representation? Who benefits from MLM pretraining",
             fontsize=STYLE["fs_title"],y=1.04)
fig.subplots_adjust(top=0.88,bottom=0.30,wspace=0.35)
_caption(fig, 0.5,0.05,f"baseline = {_I1_BASE_LABEL[I1_BASE_KEY]}, on the pooled 5-fold CV. Regression "
         "tasks only (needs a per-molecule error). Corpus similarity is max Tanimoto to a SAMPLE "
         "of the corpus, so it is a lower bound.",
         ha="center",va="top",fontsize=STYLE["fs_annot"],color="#555")
save_fig(fig,"figI1_memorization_vs_representation"); plt.show()

for t,v in pairs:
    print(f"I1 {t} (vs {_I1_BASE_LABEL[I1_BASE_KEY]}): most-similar {v[0][0]:+.1f}% "
          f"[{v[0][1]:+.1f},{v[0][2]:+.1f}]   most-novel {v[1][0]:+.1f}% "
          f"[{v[1][1]:+.1f},{v[1][2]:+.1f}]  (95% bootstrap CI)")
if pairs:
    # The verdict is DERIVED, not asserted. The previous version printed "overlapping CIs => no
    # evidence" unconditionally, so it would have claimed that even when the intervals separated.
    _sep=[t for t,v in pairs if v[0][1]>v[1][2] or v[1][1]>v[0][2]]
    if _sep:
        print(f"Non-overlapping CIs on {', '.join(_sep)} => the gain DOES depend on corpus "
              f"similarity there; memorization cannot be ruled out.")
    else:
        print("CIs overlap on every task => no evidence the gain concentrates on corpus-similar "
              "molecules, i.e. consistent with representation rather than memorization.")
    # Same analysis against the weak frozen baseline, printed for contrast so the effect of the
    # baseline choice on the headline number is visible rather than argued about.
    if _base_preds("no_pretrain") is not None:
        print("\n   contrast - the SAME comparison against the frozen baseline (why we switched):")
        for t in I1_TASKS:
            v=quartile_lift(t,"no_pretrain")
            if v: print(f"   {t}: most-similar {v[0][0]:+.1f}%   most-novel {v[1][0]:+.1f}%  "
                        f"(vs no_pretrain frozen)")
else:
    print(f"I1: no data yet. Baseline is {_I1_BASE_LABEL[I1_BASE_KEY]}, whose CV predictions "
          f"({', '.join(I1_BASELINES[I1_BASE_KEY])}) land with the E1 end-to-end wave.")