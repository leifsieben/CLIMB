# ---------- Table A1 · win counts + data cost, ready to paste into LaTeX ----------
# "Best on a task" = leader by point estimate PLUS everyone not significantly different from the
# leader (README §8.1: Wilcoxon on per-molecule squared error for RMSE, DeLong paired-AUC for
# AUC/HIV). Ties count as a win for every arm in the tie set -- that is the point of the
# definition, and on the 113-molecule A1.a hold-out it is what the data actually supports.
# Unlike the bars, this table keeps ALL FIVE SFT recipes for both sup_only and unsup->sup.
import sys
if "scripts" not in sys.path: sys.path.insert(0, "scripts")
from compare_models import delong_test
from scipy import stats

ALPHA = 0.05
# Corpus sizes, family caps and the unique-molecule arithmetic live in the shared helpers cell so
# this table and Fig A2.b quote the same numbers by construction.

# ---- the arms: every model, in the A1 bar order, then the two recipes the bars omit -----------
_B = BUDGET_FP[MATCHED_BUDGET]
ARMS = {"ecfp4":           dict(label="Morgan+XGBoost (ECFP4)", runs=["ecfp4_anchor"],
                                unsup=0, dense=0, assay=0),
        "fp_desc":         dict(label="Morgan+desc+XGBoost", runs=["fp_desc_anchor"],
                                unsup=0, dense=0, assay=0),
        "no_pretrain":     dict(label="no_pretrain (frozen)", unsup=0, dense=0, assay=0,
                                runs=["random_baseline_00", "random_baseline_01", "random_baseline_02"]),
        # E1 re-run under the A1 protocol; the dirs appear once those jobs land, and every number
        # in this row switches from n.d. to real on the next notebook run. No hand-editing.
        "no_pretrain_e2e": dict(label="no_pretrain (end-to-end)", unsup=0, dense=0, assay=0,
                                runs=["e2e_random_00", "e2e_random_01", "e2e_random_02"]),
        "unsup_only":      dict(label="unsup_only (MLM)", unsup=min(_B, UNSUP_CORPUS), dense=0, assay=0,
                                runs=["unsup_8M", "unsup_8M_s1", "unsup_8M_s2"])}
for _r in sup_recipes:
    _d, _a = sft_molecules(_r, _B)
    ARMS[f"sup_only:{_r}"] = dict(label=f"sup_only: {_recipe_pretty[_r]}", unsup=0, dense=_d, assay=_a,
                                  runs=[f"skip_{_r}_8M", f"skip_{_r}_8M_s1", f"skip_{_r}_8M_s2"])
for _r in sup_recipes:
    _d, _a = sft_molecules(_r, U2S_SFT_FP)
    ARMS[f"unsup2sup:{_r}"] = dict(label=f"unsup→sup: {_recipe_pretty[_r]}",
                                   unsup=min(_B, UNSUP_CORPUS), dense=_d, assay=_a,
                                   runs=[f"u2s_{_r}_from{MATCHED_BUDGET}",
                                         f"u2s_{_r}_from{MATCHED_BUDGET}_s1",
                                         f"u2s_{_r}_from{MATCHED_BUDGET}_s2"])
for _k, _v in ARMS.items():
    _v["runs"] = [r for r in _v["runs"] if (DATA_ROOT / "climb_v2_phase2" / r).exists()]

# ---- data access + the paired tests ---------------------------------------------------------
_TBL_CACHE = {}
def _tbl_suite(run, task, sub):
    p = DATA_ROOT / "climb_v2_phase2" / run / sub / "suite_summary.json"
    if not p.exists(): return np.nan
    return json.loads(p.read_text()).get(TASKS[task].get("suite_key", task) + "_MEAN", np.nan)

def _tbl_point(arm, task, sub):
    """Ranking value = mean over pretraining seeds, i.e. the Fig A1 bar height."""
    v = [x for x in (_tbl_suite(r, task, sub) for r in ARMS[arm]["runs"]) if np.isfinite(x)]
    return float(np.mean(v)) if v else np.nan

def _tbl_preds(arm, task, sub):
    """Per-molecule predictions of the PRIMARY (seed-0) run -- the paired test needs one vector."""
    if not ARMS[arm]["runs"]: return None
    key = (ARMS[arm]["runs"][0], task, sub)
    if key not in _TBL_CACHE:
        p = DATA_ROOT / "climb_v2_phase2" / key[0] / sub / "test_predictions.csv"
        if not p.exists():
            _TBL_CACHE[key] = None
        else:
            d = pd.read_csv(p)
            d = d[d.dataset == task].drop_duplicates(["mol_index", "output_index"])
            _TBL_CACHE[key] = d if len(d) else None
    return _TBL_CACHE[key]

def _tbl_p(arm_a, arm_b, task, sub):
    """p for 'these two arms differ on this task', molecule-paired. NaN if untestable."""
    a, b = _tbl_preds(arm_a, task, sub), _tbl_preds(arm_b, task, sub)
    if a is None or b is None: return np.nan
    m = a.merge(b, on=["mol_index", "output_index"], suffixes=("_a", "_b"))
    if not len(m): return np.nan
    if TASKS[task]["higher_better"]:                 # DeLong per label column, median p (Tox21 = 12)
        ps = []
        for _, g in m.groupby("output_index"):
            g = g[np.isfinite(g.y_true_a)]
            if g.y_true_a.nunique() < 2: continue
            _, _, p = delong_test(g.y_true_a, g.y_pred_a, g.y_pred_b)
            if np.isfinite(p): ps.append(p)
        return float(np.median(ps)) if ps else np.nan
    ea = (m.y_pred_a - m.y_true_a) ** 2              # regression -> Wilcoxon on squared error
    eb = (m.y_pred_b - m.y_true_b) ** 2
    return 1.0 if np.allclose(ea, eb) else float(stats.wilcoxon(ea, eb).pvalue)

def _scored(arm, task, sub):
    return np.isfinite(_tbl_point(arm, task, sub)) and _tbl_preds(arm, task, sub) is not None

def _cobest(task, arms, sub):
    """Leader by point estimate + everyone statistically indistinguishable from it."""
    have = [a for a in arms if _scored(a, task, sub)]
    if not have: return set(), None
    hb = TASKS[task]["higher_better"]
    lead = max(have, key=lambda a: _tbl_point(a, task, sub) * (1 if hb else -1))
    keep = {lead}
    for a in have:
        if a != lead:
            p = _tbl_p(lead, a, task, sub)
            if not np.isfinite(p) or p >= ALPHA: keep.add(a)   # not separable -> co-best
    return keep, lead

# ---- assemble ------------------------------------------------------------------------------
def _fmt_mol(n):
    if n is None or n == 0: return "0"
    return f"{n/1e6:.2f} M" if n >= 1e6 else f"{n/1e3:.1f} k"

def a1_summary(sub):
    # An arm with no runs on disk yet (E1 end-to-end) must not drag every task out of the count,
    # so coverage is judged over ACTIVE arms only; inactive arms report n.d. across the row.
    active = [a for a in ARMS if any(_scored(a, t, sub) for t in CORE_TASKS)]
    climb  = [a for a in active if a not in ("ecfp4", "fp_desc")]   # drop only the XGBoost anchors
    # Win counts run ONLY over tasks where every active arm is scored -- otherwise an arm collects
    # a free win where its competitors were simply not evaluated, and x/n stops meaning the same
    # thing from row to row.
    scored_tasks = [t for t in CORE_TASKS if all(_scored(a, t, sub) for a in active)]
    # Two baselines, deliberately both reported. "beats no_pretrain (frozen)" is the easy bar --
    # a frozen random encoder cannot adapt at all -- and "beats no_pretrain (end-to-end)" is the
    # one a practitioner actually faces, matching B2/C1J1/I1. Showing them side by side is the
    # point: the gap between the two columns IS how much of an arm's apparent value came from
    # being compared against a baseline that could not learn.
    BEATS = {"beats no_pretrain (frozen)": "no_pretrain",
             "beats no_pretrain (e2e)":    "no_pretrain_e2e"}
    win = {a: 0 for a in active}; cwin = {a: 0 for a in climb}
    bnp = {lab: {a: 0 for a in active} for lab in BEATS}
    per_task, missing = [], []
    for t in CORE_TASKS:
        wa, la = _cobest(t, active, sub)
        wc, lc = _cobest(t, climb, sub)
        per_task.append(dict(task=t, leader=ARMS[la]["label"] if la else "n.d.", n_cobest=len(wa),
                             climb_leader=ARMS[lc]["label"] if lc else "n.d.",
                             arms_scored=f"{len([a for a in active if _scored(a, t, sub)])}/{len(active)}",
                             counted="yes" if t in scored_tasks else "NO - incomplete"))
        missing += [(ARMS[a]["label"], t) for a in active if not _scored(a, t, sub)]
        if t not in scored_tasks: continue
        for a in active:
            win[a] += int(a in wa)
            if a in cwin: cwin[a] += int(a in wc)
            for _lab, _base in BEATS.items():
                if a == _base or _base not in active: continue
                p = _tbl_p(a, _base, t, sub)
                if not np.isfinite(p): continue
                hb = TASKS[t]["higher_better"]
                better = (_tbl_point(a, t, sub) > _tbl_point(_base, t, sub)) if hb else \
                         (_tbl_point(a, t, sub) < _tbl_point(_base, t, sub))
                bnp[_lab][a] += int(better and p < ALPHA)
    N = len(scored_tasks)
    # Counts only, no percentages: x/n already carries the rate, and a second column restating it
    # as a percent doubled the table's width without adding information.
    def cell(x, a, pool):
        return f"{x[a]}/{N}" if a in pool else "n.d."
    rows = []
    for a, v in ARMS.items():
        row = {"model": v["label"], "unsup mols": _fmt_mol(v["unsup"]),
               "sup mols (desc)": _fmt_mol(v["dense"]), "sup mols (assay)": _fmt_mol(v["assay"]),
               "best x/n": cell(win, a, active),
               "CLIMB-only x/n": (cell(cwin, a, climb) if a in climb else
                                  ("n/a" if a in active else "n.d."))}
        for _lab, _base in BEATS.items():
            row[_lab] = ("--" if a == _base else
                         (cell(bnp[_lab], a, active) if _base in active else "n.d."))
        rows.append(row)
    return pd.DataFrame(rows), pd.DataFrame(per_task), missing, scored_tasks

for _sub, _tag, _title in [
        ("moleculenet",    "A1.a", "single scaffold hold-out -- the split Fig A1.a plots"),
        ("moleculenet_cv", "A1.b", "pooled 5-fold scaffold-CV OOF -- §8.1 test of record  <<< USE THIS ONE")]:
    _summary, _pt, _missing, _st = a1_summary(_sub)
    print("=" * 112); print(f"Table {_tag} -- {_title}"); print("=" * 112)
    print("molecules scored / dataset size per task:",
          ", ".join(f"{t}={n_molecules(t,_sub):,}/{n_molecules(t):,}" for t in CORE_TASKS))
    print(_pt.to_string(index=False))
    print(f"\nwin counts over the {len(_st)} task(s) with complete arm coverage: {', '.join(_st)}"
          + (f"   |  EXCLUDED: {', '.join(t for t in CORE_TASKS if t not in _st)}"
             if len(_st) < len(CORE_TASKS) else ""))
    print(); print(_summary.to_string(index=False)); print()
    if _missing:
        print(f"unscored arm x task cells ({len(_missing)}): "
              + ", ".join(f"{a}/{t}" for a, t in _missing[:12]) + (" ..." if len(_missing) > 12 else ""))
    print()

print("-" * 112)
print(f"alpha={ALPHA}. 'Best' = leader OR not separable from the leader, so co-best sets are large "
      f"where the split is small.\nRanking uses the pretraining-seed mean (the Fig A1 bar height); "
      f"the paired test uses the seed-0 run's predictions.\nHIV is ranked on NEF1% but tested with "
      f"DeLong AUC (README §8.1). Tox21 p = median over its 12 label columns.\n'sup mols (desc)' are "
      f"PubChem molecules carrying computed RDKit-descriptor labels (MTR), NOT assay data.\n"
      f"'sup mols (assay)' is capped by config_v2.SUPERVISED_FAMILY_CAPS, so the 8M-FP sparse arms "
      f"are ~15 epochs over ~0.5M molecules, not 8M unique ones.")
print("-" * 112)

# LaTeX for BOTH tables, in the paper's house style: booktabs rules, \makecell two-line headers,
# tight \tabcolsep, [t] float. Emitted from here rather than hand-formatted so the numbers can
# never drift from the notebook -- re-running after new data lands regenerates paste-ready LaTeX.
# REQUIRES: \usepackage{booktabs, makecell}
_TEX_ESC=[("\\","\\textbackslash{}"),("_",r"\_"),("%",r"\%"),("→",r"$\to$")]
def _esc(v):
    v=str(v)
    for a,b in _TEX_ESC[1:]: v=v.replace(a,b)   # no literal backslashes in our cells
    return v
# Explicit two-line header splits: guessing where to break reads badly ("CLIMB-\nonly").
_HDR={"unsup mols":("unsup","mols"), "sup mols (desc)":("sup mols","(desc)"),
      "sup mols (assay)":("sup mols","(assay)"), "best x/n":("best","x/n"),
      "CLIMB-only x/n":("CLIMB-only","x/n"),
      "beats no_pretrain (frozen)":("beats no_pretrain","(frozen)"),
      "beats no_pretrain (e2e)":("beats no_pretrain","(e2e)")}

def latex_table(df, caption, label):
    cols=list(df.columns); body=[[_esc(v) for v in r] for r in df.itertuples(index=False)]
    w=[max(len(x) for x in [cols[j]]+[r[j] for r in body]) for j in range(len(cols))]
    head=["model"]+[f" & \\makecell[r]{{{_esc(_HDR[c][0])}\\\\{_esc(_HDR[c][1])}}}" for c in cols[1:]]
    rows=[" & ".join([body_r[0].ljust(w[0])]+[body_r[j].ljust(w[j]) for j in range(1,len(cols))])
          + r" \\" for body_r in body]
    return "\n".join([r"\begin{table}[t]", r"\centering", r"\footnotesize",
                      r"\setlength{\tabcolsep}{4pt}", f"\\caption{{{caption}}}", f"\\label{{{label}}}",
                      r"\begin{tabular}{@{}l" + " r"*(len(cols)-1) + r"@{}}", r"\toprule",
                      "\n".join(head)+r" \\", r"\midrule", *rows,
                      r"\bottomrule", r"\end{tabular}", r"\end{table}"])

for _sub,_tag,_cap in [
        ("moleculenet",   "A1.a","single DeepChem scaffold hold-out"),
        ("moleculenet_cv","A1.b","pooled 5-fold scaffold cross-validation")]:
    _d,_,_,_st=a1_summary(_sub)
    if len(_st) < len(CORE_TASKS)-1:
        print(f"\n{'!'*100}\nTable {_tag}: NOT PAPER-READY -- only {len(_st)}/{len(CORE_TASKS)} tasks "
              f"have complete arm coverage ({', '.join(_st) or 'none'}). Usually a re-scoring pass "
              f"in flight; re-run when it finishes.\n"+"!"*100)
    print(f"\n{'='*100}\nLaTeX -- Table {_tag}   (needs \\usepackage{{booktabs, makecell}})\n{'='*100}\n")
    print(latex_table(_d,
          f"Fig.~{_tag} summary at the 8M forward-pass matched budget ({_cap}). "
          f"``Best'' = leader on a task or not statistically separable from it (paired Wilcoxon on "
          f"squared error for RMSE, DeLong paired-AUC for classification/HIV, $\\alpha=0.05$); "
          f"counted over the {len(_st)} task(s) with complete arm coverage. "
          f"``beats no\\_pretrain'' is reported against both baselines: the frozen random encoder "
          f"and the same encoder fine-tuned end-to-end.",
          f"tab:{_tag.replace('.','').lower()}_summary"))