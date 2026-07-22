# ---- helpers every figure shares: the no_pretrain floor and arm selection -------------------
def npt_floor(df,task,wave="climb_v2_phase2"):
    """Mean of the three random-init encoders — the 'neither pretraining nor labels' reference."""
    s=df[(df.wave==wave)&(df.regime=="no_pretrain")&(df.task==task)]
    return s.value.mean() if len(s) else np.nan

def arm_rows(df,task,key,budget=None,wave="climb_v2_phase2"):
    """All rows (across pretraining seeds) for one arm at one budget. Truncated runs excluded."""
    budget=budget or MATCHED_BUDGET
    b=(df.wave==wave)&(df.task==task)
    if key=="no_pretrain":        return df[b&(df.regime=="no_pretrain")]
    if key in ("ecfp4","fp_desc"):return df[b&(df.regime==key)]
    if key=="unsup_only":         return df[b&(df.regime=="unsup_only")&(df.budget_label==budget)&(~df.truncated)]
    if key=="no_pretrain_e2e":    return df[b&(df.regime=="no_pretrain_e2e")]
    if key=="unsup2sup":          return df[b&(df.regime=="unsup2sup")&(df.budget_label==budget)&(~df.truncated)]
    if key.startswith("unsup2sup:"):   # one recipe, not the mean of all five
        return df[b&(df.regime=="unsup2sup")&(df.recipe==key.split(":")[1])&(df.budget_label==budget)&(~df.truncated)]
    if key.startswith("sup_only:"):
        return df[b&(df.regime=="sup_only")&(df.recipe==key.split(":")[1])&(df.budget_label==budget)&(~df.truncated)]
    if key.startswith("corrupt_"):return df[b&(df.regime==key)&(df.budget_label==budget)&(~df.truncated)]
    return df.iloc[0:0]

def arm_value(df,task,key,budget=None):
    s=arm_rows(df,task,key,budget); return s.value.mean() if len(s) else np.nan

def arm_err(df,task,key,budget=None):
    """Prefer PRETRAINING-SEED spread where replicates exist: it is the dominant noise source and
    what a reader assumes an error bar means. Head-seed std alone understates it. Arms with one
    pretraining seed fall back to head-seed std -- so the bars are NOT the same quantity across
    arms, which the caption states rather than hides."""
    s=arm_rows(df,task,key,budget)
    if not len(s): return np.nan
    if s.seed.nunique()>1:
        return float(np.nanstd(s.groupby("seed").value.mean().values,ddof=1))
    return float(np.nanmean(s["std"].values))

def n_seeds(df,task,key,budget=None):
    s=arm_rows(df,task,key,budget); return int(s.seed.nunique()) if len(s) else 0

# ---- how many UNIQUE molecules an arm actually sees -------------------------------------------
# Forward passes are not molecules. Within one epoch they coincide; past it the stream repeats.
# The unsupervised corpus is ~12M (README 6.1) and each SFT recipe is capped per family
# (config_v2.SUPERVISED_FAMILY_CAPS x README 6.3 sizes), so the sparse recipes exhaust their pool
# after ~0.5M molecules and every further forward pass is pure repetition. A2.b makes that visible;
# Table A1 quotes the same numbers, from here, so the two can never disagree.
UNSUP_CORPUS=UNSUP_CORPUS_12M          # default; 50M/100M override via unsup_corpus_size()
_FAM_AVAIL=dict(PCBA=1_517_000,L1000_MCF7=11_718,L1000_VCAP=7_800,PCQM=3_810_323,WONG=34_652)
_FAM_CAP  =dict(PCBA=500_000,L1000_MCF7=100_000,L1000_VCAP=100_000,PCQM=200_000,WONG=100_000)
_GROUP    =dict(sparse_all=["PCBA","L1000_MCF7","L1000_VCAP"],
                minimol_full=["PCBA","L1000_MCF7","L1000_VCAP","PCQM","WONG"])
ASSAY_POOL={g:sum(min(_FAM_AVAIL[f],_FAM_CAP[f]) for f in fams) for g,fams in _GROUP.items()}
# recipe -> (mtr weight, supervised weight, assay group). `dense` is MTR on the PubChem corpus with
# COMPUTED descriptor labels, so its labelled molecules come from the unsupervised corpus.
SFT_STREAMS={"dense":(1.0,0.0,None),"sparse_all":(0.0,1.0,"sparse_all"),
             "dense_plus_sparse":(0.5,0.5,"sparse_all"),"minimol_full":(0.0,1.0,"minimol_full"),
             "mixed":(0.5,0.5,"minimol_full")}

def sft_molecules(recipe,sft_fp):
    """(descriptor-labelled, assay-labelled) unique molecules an SFT stage of `sft_fp` FP touches."""
    w_mtr,w_sup,group=SFT_STREAMS[recipe]
    return (min(w_mtr*sft_fp,UNSUP_CORPUS) if w_mtr else 0,
            min(w_sup*sft_fp,ASSAY_POOL[group]) if w_sup else 0)

def unique_molecules(key,budget_fp,budget_label=None):
    """Total unique molecules behind an arm at one budget. `budget_fp` is the DF value, i.e. for
    unsup->sup it already includes the 2M-FP SFT stage."""
    if not np.isfinite(budget_fp): return np.nan
    _corpus=unsup_corpus_size(budget_label)   # 124M for the 50M/100M rungs, 12M otherwise
    if key=="unsup_only":        return min(budget_fp,_corpus)
    if key.startswith("sup_only:"):
        return sum(sft_molecules(key.split(":")[1],budget_fp))
    if key.startswith("unsup2sup:"):
        base=budget_fp-U2S_SFT_FP
        return min(base,_corpus)+sum(sft_molecules(key.split(":")[1],U2S_SFT_FP))
    return np.nan

_MOLCOUNT={}
def n_molecules(task,sub="moleculenet_cv",wave="climb_v2_phase2"):
    """How many molecules this task actually scores under one evaluation scheme.

    Derived from the per-molecule prediction dumps, never hard-coded: the suite has gained tasks
    (Lipophilicity, HIV) and been re-split during the project, and a stale constant in a panel
    title is a claim the reader cannot check. `moleculenet_cv` predicts EVERY molecule
    out-of-fold, so its count is the dataset size; `moleculenet` counts the single hold-out's
    test split only. Returns NaN if no run has scored the task yet."""
    key=(task,sub,wave)
    if key not in _MOLCOUNT:
        n=np.nan
        for rd in sorted((DATA_ROOT/wave).glob("*/")):
            p=rd/sub/"test_predictions.csv"
            if not p.exists(): continue
            d=pd.read_csv(p,usecols=["dataset","mol_index"])
            d=d[d.dataset==task]
            if len(d): n=int(d.mol_index.nunique()); break
        _MOLCOUNT[key]=n
    return _MOLCOUNT[key]

def best_sup_recipe(df,budget=None,tasks=None):
    """The sup_only recipe with the highest MEAN lift over no_pretrain. Resolved from the data so
    it cannot go stale as runs land."""
    tasks=tasks or CORE_TASKS; best,best_v=None,-np.inf
    for r in sup_recipes:
        vals=[lift(arm_value(df,t,f"sup_only:{r}",budget),t,npt_floor(df,t)) for t in tasks]
        vals=[v for v in vals if np.isfinite(v)]
        if vals and np.mean(vals)>best_v: best,best_v=r,float(np.mean(vals))
    return best or "dense", best_v

BEST_SUP,_bv = best_sup_recipe(DF)
print(f"best sup_only recipe at {MATCHED_BUDGET} = {BEST_SUP!r} (mean lift {100*_bv:+.1f}% over no_pretrain)")
print("PRIMARY set for A2/B1p1: no_pretrain, unsup_only, sup_only:"+BEST_SUP+", unsup→sup")