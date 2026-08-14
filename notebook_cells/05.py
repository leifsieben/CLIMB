import json, re, glob, os
import numpy as np, pandas as pd
DATA_ROOT=Path("figure_data")

def _achieved_fp(run_dir):
    """Last forward_passes_seen in metrics.jsonl; None if unavailable."""
    m=run_dir/"metrics.jsonl"
    if not m.exists(): return None
    last=""
    try:
        with m.open() as fh:
            for line in fh:
                if line.strip(): last=line
    except OSError: return None
    try: return float(json.loads(last).get("forward_passes_seen",0) or 0)
    except Exception: return None

def _achieved_tokens(run_dir):
    """Last tokens_seen in metrics.jsonl (trainer's own non-padding token count); nan if absent.
    The honest scaling x-axis: real tokens processed per run, not FP x a guessed constant -- and
    that ratio genuinely varies (12M corpus ~42.9 tok/seq, RDKit-canonical 124M ~40.4, sparse
    SFT ~40.0)."""
    m=run_dir/"metrics.jsonl"
    if not m.exists(): return np.nan
    last=""
    try:
        with m.open() as fh:
            for line in fh:
                if line.strip(): last=line
    except OSError: return np.nan
    try: return float(json.loads(last).get("tokens_seen", np.nan))
    except Exception: return np.nan

def run_completion(run_dir,budget_fp,tol=0.98):
    """-> (status, achieved_fp); status in {complete, truncated, unknown, n/a}."""
    if (run_dir/"verified.json").exists(): return "complete",budget_fp
    # Anchors and random baselines have no forward-pass budget, so nothing could be truncated.
    # Flagging them would train the reader to ignore the warning, defeating its purpose.
    if not np.isfinite(budget_fp) or not budget_fp: return "n/a",np.nan
    fp=_achieved_fp(run_dir)
    if fp is None: return "unknown",np.nan
    return ("complete" if fp>=tol*budget_fp else "truncated"),fp

_UNKNOWN_RUNS=set()

def load_suite(run_dir,sub="moleculenet"):
    p=run_dir/sub/"suite_summary.json"
    if not p.exists(): return None
    d=json.loads(p.read_text()); out={}
    for k,v in d.items():
        if k.endswith("_MEAN"): out[(k[:-5],"mean")]=v
        elif k.endswith("_STD"): out[(k[:-4],"std")]=v
    return out

def _eval_mtime(run_dir,sub):
    """When this run's summary was written. Re-scoring the SAME encoder with the current eval_v2
    moves metrics by up to ~0.03 (head-seed noise plus code drift), so a panel that mixes an old
    summary with a fresh one is comparing two protocols. Carrying the timestamp makes that
    detectable instead of invisible."""
    p=run_dir/sub/"suite_summary.json"
    return pd.Timestamp(p.stat().st_mtime,unit="s") if p.exists() else pd.NaT

def parse_run(name):
    """Directory name -> regime/recipe/budget/seed, or None if it is not a plottable run."""
    # Pretraining-seed replicates are "<base>_s1"/"_s2". Parse the base and carry the seed, so
    # replicates aggregate instead of being silently dropped.
    m=re.match(r"(.+)_s(\d+)$",name)
    if m:
        base=parse_run(m.group(1))
        return {**base,"seed":int(m.group(2))} if base else None
    if name=="ecfp4_anchor":   return dict(seed=0,regime="ecfp4",  recipe=None,budget_label=None,budget_fp=np.nan)
    if name=="fp_desc_anchor": return dict(seed=0,regime="fp_desc",recipe=None,budget_label=None,budget_fp=np.nan)
    if name.startswith("random_baseline"):
        return dict(seed=0,regime="no_pretrain",recipe=None,budget_label=None,budget_fp=np.nan)
    # External published comparator: CheMeleon (Burns 2025) fine-tuned end-to-end (chemeleon_e2e{,_s1,_s2}).
    if name=="chemeleon_e2e":
        return dict(seed=0,regime="chemeleon_e2e",recipe=None,budget_label=None,budget_fp=np.nan)
    # Random-init encoder fine-tuned END-TO-END on the eval task (E1), scored by the A1 protocol.
    # No pretraining budget, so nothing here can be truncated.
    if name.startswith("e2e_random"):
        return dict(seed=0,regime="no_pretrain_e2e",recipe=None,budget_label=None,budget_fp=np.nan)
    if name.startswith("corrupt_"):
        m=re.match(r"corrupt_(mlm|mtr)_(\d+M)$",name)
        if m: return dict(seed=0,regime=f"corrupt_{m[1]}",recipe=None,budget_label=m[2],budget_fp=BUDGET_FP[m[2]])
        return None
    m=re.match(r"unsup_(\d+M)$",name)
    if m: return dict(seed=0,regime="unsup_only",recipe=None,budget_label=m[1],budget_fp=BUDGET_FP[m[1]])
    m=re.match(r"skip_(.+)_(\d+M)$",name)      # dir names still 'skip_*'; the regime is sup_only
    if m: return dict(seed=0,regime="sup_only",recipe=m[1],budget_label=m[2],budget_fp=BUDGET_FP[m[2]])
    # unsup->sup: MLM base of <budget> FP, then a 2M-FP SFT stage. budget_label keys off the BASE
    # so it lines up with the same-budget arms; budget_fp is the TRUE total, so the compute axis
    # in A2 is not flattering to this arm.
    m=re.match(r"u2s_(.+)_from(\d+M)$",name)
    if m: return dict(seed=0,regime="unsup2sup",recipe=m[1],budget_label=m[2],
                      budget_fp=BUDGET_FP[m[2]]+U2S_SFT_FP)
    return None

def build_table(sub):
    rows=[]
    for wave in ["climb_v2_phase2","climb_v2_headline"]:
        for rd in sorted((DATA_ROOT/wave).glob("*/")):
            meta=parse_run(rd.name)
            if meta is None: continue
            suite=load_suite(rd,sub=sub)
            if not suite: continue
            st,fp=run_completion(rd,meta["budget_fp"])
            if st=="unknown": _UNKNOWN_RUNS.add(rd.name)
            for task in TASKS:
                sk=TASKS[task].get("suite_key",task)
                if (sk,"mean") in suite:
                    rows.append(dict(wave=wave,run=rd.name,task=task,value=suite[(sk,"mean")],
                                     std=suite.get((sk,"std"),np.nan),truncated=(st=="truncated"),
                                     evaluated_at=_eval_mtime(rd,sub),
                                     achieved_fp=(fp if np.isfinite(fp) else meta["budget_fp"]),
                                     tokens_seen=_achieved_tokens(rd),**meta))
    return pd.DataFrame(rows)

# DEFAULT for every bar/ladder figure: the single DeepChem scaffold hold-out. It puts the RAREST
# scaffolds in test and therefore actually discriminates; the balanced 5-fold CV saturates (a
# random encoder reaches 0.937 on BBBP vs ~0.70 published) and compresses regime gaps 2-3x.
DF    = build_table("moleculenet")
DF_CV = build_table("moleculenet_cv")     # used only where per-fold predictions are needed (I1)

print(f"single-split: {len(DF):5d} rows / {DF.run.nunique()} runs")
print(f"5-fold CV   : {len(DF_CV):5d} rows / {DF_CV.run.nunique()} runs")
trunc=sorted(DF[DF.truncated].run.unique())
if trunc:
    print(f"\nTRUNCATED, excluded from every figure ({len(trunc)}) - derived from achieved FP:")
    for r in trunc:
        row=DF[DF.run==r].iloc[0]
        print(f"   {r:32s} {row.achieved_fp/1e6:5.1f}M / {row.budget_fp/1e6:.0f}M "
              f"({100*row.achieved_fp/row.budget_fp:.0f}%)")
if _UNKNOWN_RUNS:
    print(f"\n⚠ UNKNOWN COMPLETENESS ({len(_UNKNOWN_RUNS)} runs): no verified.json and no metrics.jsonl.")
    print("  Plotted on trust. 'No evidence' is not 'fine'. Sync evidence with:")
    print("    aws s3 sync s3://climb-s3-bucket/experiments/<wave> figure_data/<wave> \\")
    print("      --exclude '*' --include '*/verified.json' --include '*/metrics.jsonl'")
    for r in sorted(_UNKNOWN_RUNS)[:10]: print(f"    - {r}")

# ---- evaluation-protocol audit ------------------------------------------------------------
# Every arm inside one panel must have been scored by the same eval code. Runs scored more than a
# few days apart may not have been.
for _name,_d in (("single-split",DF),("5-fold CV",DF_CV)):
    _t=_d.dropna(subset=["evaluated_at"]).evaluated_at
    if len(_t):
        _span=(_t.max()-_t.min()).days
        _flag="  <-- MIXED: re-scoring in progress, figures are PROVISIONAL" if _span>2 else ""
        print(f"{_name:<12} evaluated {_t.min():%Y-%m-%d} .. {_t.max():%Y-%m-%d} "
              f"(span {_span}d){_flag}")

# ---- HIV coverage audit -------------------------------------------------------------------
# HIV was added to the eval suite after most runs were scored, so it is the one task that can be
# missing for an arm that is otherwise complete. Print it rather than letting bars quietly vanish.
_hiv=set(DF[DF.task=="HIV"].run); _all=set(DF.run)
if _all-_hiv:
    print(f"\nHIV (NEF1%) not yet scored for {len(_all-_hiv)}/{len(_all)} single-split runs; their HIV "
          f"bars are drawn as 'pending', never as zero:")
    print("   "+", ".join(sorted(_all-_hiv)[:12])+(" ..." if len(_all-_hiv)>12 else ""))