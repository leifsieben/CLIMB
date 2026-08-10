# ---------- I1 · memorization vs representation ----------
# Reviewer point (Note C): the top-Tanimoto bin could mix interpolation with outright memorization,
# because molecules IDENTICAL to a corpus molecule sit in that bin by construction. This is real and
# material for ESOL -- 41.9% of ESOL eval molecules are ECFP4-identical (Tanimoto=1.0) to a corpus
# molecule (QM7 only 15.7%; its median max-Tanimoto is 0.63, so QM7 is the clean control). The number
# that matters is fingerprint-identity, NOT the literal exact-match rate (1.3% ESOL / 1.9% QM7): Fig I1
# BINS on ECFP4 Tanimoto, so what saturates the top bin is fingerprint identity, not SMILES identity.
# We defuse the circularity two ways:
#   (1) similarity is the TRUE max ECFP4 Tanimoto to the FULL 12M corpus (full_corpus_similarity_i1.csv),
#       not the old 500k-subsample lower bound; and
#   (2) corpus-IDENTICAL molecules (ECFP4 Tanimoto = 1.0, or a literal isomeric-canonical match) are
#       REMOVED from the similarity bins and reported as their own bar. Near-duplicates (0.95<=T<1.0 --
#       salt/tautomer/stereo variants) are DISTINCT inputs to the model, so they stay in the trend as
#       the genuine interpolation regime; excluding them too does not change the conclusion.
# The salt-stripped "73%" overlap is an artifact (toluene/pyridine as the largest fragment of unrelated
# PubChem mixtures) and is deliberately NOT used.
DEDUP=Path("analysis/dedup_i1")
TANI=DEDUP/"full_corpus_similarity_i1.csv"        # true max Tanimoto to the full 12M corpus
EXACT=DEDUP/"exact_match_per_molecule.csv"        # literal isomeric-canonical corpus match
IDENTICAL_THR=0.99999                             # ECFP4 Tanimoto = 1.0 => corpus-identical (the circularity)
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

def _simframe():
    """Full-corpus similarity + a `memorized` flag (exact match OR fingerprint near-dup >= 0.95)."""
    if not TANI.exists(): return None
    s=pd.read_csv(TANI).rename(columns={"max_tanimoto_to_corpus_full":"max_tanimoto_to_corpus"})
    if EXACT.exists():
        e=pd.read_csv(EXACT)[["raw_smiles","dataset","exact_nosalt"]]
        s=s.merge(e,on=["raw_smiles","dataset"],how="left")
    s["exact_nosalt"]=s.get("exact_nosalt",0)
    s["exact_nosalt"]=s.exact_nosalt.fillna(0).astype(int)
    # "memorized" here == corpus-IDENTICAL (ECFP4 Tanimoto = 1.0, or a literal match). Near-dups
    # (0.95<=T<1.0) are deliberately NOT flagged -- they are distinct inputs and stay in the trend.
    s["memorized"]=(s.exact_nosalt==1)|(s.max_tanimoto_to_corpus>=IDENTICAL_THR)
    return s
_SIM=_simframe()

def _lift_rmse(se_m,se_b):
    rm,rb=np.sqrt(np.mean(se_m)),np.sqrt(np.mean(se_b))
    return np.nan if (not np.isfinite(rb) or rb==0) else 100*(rb-rm)/rb

def _merged(task,base_key):
    """model + baseline + full-corpus similarity, with squared errors and the memorized flag."""
    if _SIM is None: return None
    mod=_preds(I1_MODEL); base=_base_preds(base_key)
    if mod is None or base is None: return None
    m=mod[mod.dataset==task].merge(base[base.dataset==task],on=["dataset","raw_smiles"],
                                   suffixes=("_m","_b"))
    m=m.merge(_SIM[_SIM.dataset==task][["raw_smiles","max_tanimoto_to_corpus","memorized"]],
              on="raw_smiles",how="inner")
    if len(m)<50: return None
    m["se_m"]=(m.y_pred_m-m.y_true_m)**2; m["se_b"]=(m.y_pred_b-m.y_true_b)**2
    return m

def _boot_ci(se_m,se_b,nboot,seed):
    pt=_lift_rmse(se_m,se_b)
    if not np.isfinite(pt): return None
    rng=np.random.default_rng(seed)
    idx=rng.integers(0,len(se_m),size=(nboot,len(se_m)))
    boot=np.array([_lift_rmse(se_m[i],se_b[i]) for i in idx]); boot=boot[np.isfinite(boot)]
    lo,hi=(np.percentile(boot,[2.5,97.5]) if len(boot) else (np.nan,np.nan))
    return pt,lo,hi

def binned_lift(task,base_key=None,nbins=5,nboot=400,seed=0):
    """-> (centres, lift%, lo, hi, n_per_bin) over the NON-memorized (not-in-corpus) molecules."""
    base_key=base_key or I1_BASE_KEY
    m=_merged(task,base_key)
    if m is None: return (None,)*5
    m=m[~m.memorized]                      # corpus matches are removed from the trend, reported apart
    if len(m)<50: return (None,)*5
    edges=np.unique(m.max_tanimoto_to_corpus.quantile(np.linspace(0,1,nbins+1)).values)
    if len(edges)<3: return (None,)*5
    edges[0]-=1e-9
    m=m.assign(bin=pd.cut(m.max_tanimoto_to_corpus,bins=edges,labels=False,include_lowest=True))
    xs,ys,los,his,ns=[],[],[],[],[]
    for b in range(len(edges)-1):
        s=m[m.bin==b]
        if len(s)<15: continue
        ci=_boot_ci(s.se_m.values,s.se_b.values,nboot,seed)
        if ci is None: continue
        pt,lo,hi=ci
        xs.append(float(s.max_tanimoto_to_corpus.mean())); ys.append(pt)
        los.append(lo); his.append(hi); ns.append(int(len(s)))
    return xs,ys,los,his,ns

def quartile_lift(task,base_key=None):
    """(most-similar, most-novel) lift% with bootstrap CIs, from the outer quartiles (non-memorized)."""
    xs,ys,lo,hi,_=binned_lift(task,base_key,nbins=4)
    if not xs or len(ys)<4: return None
    return (ys[-1],lo[-1],hi[-1]),(ys[0],lo[0],hi[0])

def memorized_lift(task,base_key=None,nboot=400,seed=1):
    """Lift% on the EXCLUDED corpus-match group (exact or Tanimoto>=0.95), for a side-by-side bar."""
    base_key=base_key or I1_BASE_KEY
    m=_merged(task,base_key)
    if m is None: return None
    s=m[m.memorized]
    if len(s)<15: return None
    ci=_boot_ci(s.se_m.values,s.se_b.values,nboot,seed)
    return None if ci is None else (ci[0],ci[1],ci[2],int(len(s)))

_I1_BASE_LABEL={"no_pretrain_e2e":"no_pretrain (end-to-end)","no_pretrain":"no_pretrain (frozen)"}
_I1_HAVE=_base_preds(I1_BASE_KEY) is not None
_I1_NEED=("needs e2e_random_0*/moleculenet_cv\n(E1 end-to-end wave still running)"
          if not _I1_HAVE else "needs full_corpus_similarity_i1.csv")

# Height + bottom margin reserve room for the caption INSIDE the canvas; at negative figure
# coords it only clears in the exported PNG (bbox="tight" grows it), not in the inline render.
fig,(ax0,ax1)=plt.subplots(1,2,figsize=(STYLE["col2"],3.3))
pairs=[(t,v) for t,v in ((t,quartile_lift(t)) for t in I1_TASKS) if v]
mpairs=[(t,v) for t,v in ((t,memorized_lift(t)) for t in I1_TASKS) if v]
def _agg(getter,items):                                       # mean over tasks + combined SE from CIs
    trips=[getter(v) for _,v in items]                        # each getter(v) is a (pt,lo,hi) triple
    val=float(np.mean([t[0] for t in trips]))
    se=float(np.sqrt(np.sum([((t[2]-t[1])/2/len(trips))**2 for t in trips])))
    return val,se
if pairs:
    sim,se_s=_agg(lambda v:v[0],pairs); nov,se_n=_agg(lambda v:v[1],pairs)
    bars=[("most corpus-similar\n(top quartile,\nnot identical)",sim,se_s,"#1b5e20"),
          ("most novel\n(bottom quartile)",nov,se_n,"#66bb6a")]
    if mpairs:                                                 # excluded corpus-identical group, shown apart
        mem,se_m=_agg(lambda v:v,mpairs)
        bars.append(("corpus-identical\n(Tanimoto=1.0,\nexcluded)",mem,se_m,"#9e9e9e"))
    xpos=list(range(len(bars)))
    ax0.bar(xpos,[b[1] for b in bars],color=[b[3] for b in bars],width=0.6,
            yerr=[b[2] for b in bars],capsize=STYLE["cap_size"],error_kw=dict(lw=STYLE["lw_thin"]))
    for xi,b in zip(xpos,bars):
        ax0.text(xi,b[1]+b[2]+0.6,f"{b[1]:+.1f}%",ha="center",fontsize=STYLE["fs_annot"])
    ax0.axhline(0,color=PALETTE["black"],lw=0.6)
    _vals=[b[1] for b in bars]; _errs=[b[2] for b in bars]
    lo_,hi_=min(0,*(v-e for v,e in zip(_vals,_errs))),max(0,*(v+e for v,e in zip(_vals,_errs)))
    ax0.set_ylim(lo_-abs(lo_)*0.35-2,hi_+abs(hi_)*0.35+3)
    ax0.set_xticks(xpos); ax0.set_xticklabels([b[0] for b in bars],fontsize=STYLE["fs_annot"]-0.5)
else:
    ax0.set_ylim(-5,15); no_data_watermark(ax0,_I1_NEED); ax0.set_xticks([])
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
ax1.set_xlabel("max ECFP4 Tanimoto to corpus (bin mean)")
ax1.set_ylabel(_ylab); label_all_yticks(ax1); panel_tag(ax1,"b",dx=-0.18)

_suptitle(fig, "Fig I1 - memorization or representation? Who benefits from MLM pretraining",
             fontsize=STYLE["fs_title"],y=1.04)
fig.subplots_adjust(top=0.88,bottom=0.34,wspace=0.35)
_caption(fig, 0.5,0.02,f"baseline = {_I1_BASE_LABEL[I1_BASE_KEY]}, pooled 5-fold CV, regression tasks "
         "only. Similarity = TRUE max ECFP4 Tanimoto to the full 12M corpus. Corpus-IDENTICAL molecules "
         "(ECFP4 Tanimoto = 1.0; 41.9% of ESOL, 15.7% of QM7) are excluded from the bins and shown "
         "separately in (a); near-duplicates (0.95≤T<1.0) are kept as the interpolation regime. The trend "
         "is over molecules that are NOT fingerprint-identical to any corpus molecule; the conclusion is "
         "unchanged if the near-dup band is also excluded.",
         ha="center",va="top",fontsize=STYLE["fs_annot"],color="#555")
save_fig(fig,"figI1_memorization_vs_representation"); plt.show()

for t,v in pairs:
    print(f"I1 {t} (vs {_I1_BASE_LABEL[I1_BASE_KEY]}, excl. corpus-identical): most-similar {v[0][0]:+.1f}% "
          f"[{v[0][1]:+.1f},{v[0][2]:+.1f}]   most-novel {v[1][0]:+.1f}% "
          f"[{v[1][1]:+.1f},{v[1][2]:+.1f}]  (95% bootstrap CI)")
for t,v in mpairs:
    print(f"   {t} corpus-identical group (Tani=1.0, excluded): {v[0]:+.1f}% [{v[1]:+.1f},{v[2]:+.1f}]  (n={v[3]})")
if pairs:
    # The verdict is DERIVED, not asserted. The previous version printed "overlapping CIs => no
    # evidence" unconditionally, so it would have claimed that even when the intervals separated.
    _sep=[t for t,v in pairs if v[0][1]>v[1][2] or v[1][1]>v[0][2]]
    if _sep:
        print(f"Non-overlapping CIs on {', '.join(_sep)} => even among non-identical molecules the gain "
              f"DOES depend on corpus similarity there.")
    else:
        print("Once corpus-identical molecules are removed, no task shows the lift concentrating on "
              "corpus-similar molecules (CIs overlap; if anything the novel quartile benefits more). The "
              "apparent top-bin advantage -- most visible for ESOL -- is carried by the corpus-identical "
              "group in (a): memorization of in-corpus structures, not genuine interpolation.")
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
