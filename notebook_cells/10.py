# ---------- Table A1 · win counts + data cost (plain text; copy the numbers by hand) ----------
# ONE significance test throughout (README §8.1): a scaffold cluster-bootstrap of the paired metric
# difference, BH-FDR-corrected across the comparison family, PRECOMPUTED by
# scripts/best_model_bootstrap.py -> analysis/rigor/best_model_bootstrap.csv (too slow to run inline:
# HIV=41k molecules x ~15 arms x 2 schemes x 1,000 resamples). Definitions:
#   * A1.b (CV, test of record): an arm is "not distinguishable from the leader" when the
#     leader-minus-arm difference is NOT significant at BH-FDR q>=0.05 (leader trivially included);
#     "beats no_pretrain" = q<0.05 in the arm's favour.
#   * A1.a (hold-out): the rarest-scaffold split (113-4,112 test molecules, few scaffolds) is
#     underpowered and is NOT used to rank models -- each arm is reported as its point estimate with
#     a cluster-bootstrap 95% CI. The old 1-sigma hold-out rule is dropped (permissiveness hack).
#   * matrix (6 headline models): same test, FDR within that pre-specified set, on BOTH schemes.
# Unlike the bars, this table keeps ALL FIVE SFT recipes for both sup_only and unsup->sup.
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
        "unsup_only":      dict(label="unsup_only (MLM)", unsup=unique_molecules("unsup_only", _B, MATCHED_BUDGET), dense=0, assay=0,
                                runs=["unsup_8M", "unsup_8M_s1", "unsup_8M_s2"])}
for _r in sup_recipes:
    _d, _a = sft_molecules(_r, _B)
    ARMS[f"sup_only:{_r}"] = dict(label=f"sup_only: {_recipe_pretty[_r]}", unsup=0, dense=_d, assay=_a,
                                  runs=[f"skip_{_r}_8M", f"skip_{_r}_8M_s1", f"skip_{_r}_8M_s2"])
for _r in sup_recipes:
    _d, _a = sft_molecules(_r, U2S_SFT_FP)
    ARMS[f"unsup2sup:{_r}"] = dict(label=f"unsup→sup: {_recipe_pretty[_r]}",
                                   unsup=unique_molecules("unsup_only", _B, MATCHED_BUDGET), dense=_d, assay=_a,
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

def _scored(arm, task, sub):
    """An arm 'ran' on (task, sub) if it has a suite mean -- cheap coverage check for the data-cost rows."""
    return np.isfinite(_tbl_point(arm, task, sub))

# ---- 'best' = scaffold cluster-bootstrap + BH-FDR, read from the precomputed analysis --------
# (scripts/best_model_bootstrap.py). One test everywhere; co-best = leader-minus-arm NOT significant
# at BH-FDR q>=0.05. The hold-out is NOT ranked (underpowered) -> point estimate + CI instead.
_BMB = pd.read_csv("analysis/rigor/best_model_bootstrap.csv")
def _bmb(family, scheme, task):
    return _BMB[(_BMB.family == family) & (_BMB.scheme == scheme) & (_BMB.task == task)]
def _cobest(task, family="full", scheme="cv"):
    """(set of arms not distinguishable from the leader at FDR, leader arm key) for one task."""
    d = _bmb(family, scheme, task)
    lead = d.loc[d.is_leader, "arm"].iloc[0] if d.is_leader.any() else None
    return set(d.loc[d.cobest.fillna(False), "arm"]), lead
def _beats(task, arm, col):     # col in {"beats_frozen", "beats_e2e"}
    r = _bmb("full", "cv", task); r = r[r.arm == arm]
    return bool(r[col].iloc[0]) if len(r) and pd.notna(r[col].iloc[0]) else False
def _hold_est(task, arm):       # (point, ci_lo, ci_hi): hold-out estimate + cluster-bootstrap 95% CI
    r = _bmb("full", "holdout", task); r = r[r.arm == arm]
    if not len(r): return (np.nan, np.nan, np.nan)
    return (float(r.point.iloc[0]), float(r.ci_lo.iloc[0]), float(r.ci_hi.iloc[0]))

# ---- assemble ------------------------------------------------------------------------------
def _fmt_mol(n):
    if n is None or n == 0: return "0"
    return f"{n/1e6:.2f} M" if n >= 1e6 else f"{n/1e3:.1f} k"

def a1_summary_cv():
    """A1.b (CV) ranking table. 'co-best' = not distinguishable from the per-task leader at BH-FDR
    q>=0.05 (cluster bootstrap); 'beats no_pretrain' = q<0.05 in the arm's favour. All from the CSV."""
    active = [a for a in ARMS if any(_scored(a, t, "moleculenet_cv") for t in CORE_TASKS)]
    climb  = [a for a in active if a not in ("ecfp4", "fp_desc")]   # drop only the XGBoost anchors
    scored_tasks = [t for t in CORE_TASKS if all(_scored(a, t, "moleculenet_cv") for a in active)]
    BEATS = {"beats no_pretrain (frozen)": ("no_pretrain", "beats_frozen"),
             "beats no_pretrain (e2e)":    ("no_pretrain_e2e", "beats_e2e")}
    win = {a: 0 for a in active}; cwin = {a: 0 for a in climb}
    bnp = {lab: {a: 0 for a in active} for lab in BEATS}
    per_task, missing = [], []
    for t in CORE_TASKS:
        wa, la = _cobest(t, "full", "cv")
        wc, lc = _cobest(t, "climb", "cv")
        per_task.append(dict(task=t, leader=ARMS[la]["label"] if la in ARMS else "n.d.", n_cobest=len(wa),
                             climb_leader=ARMS[lc]["label"] if lc in ARMS else "n.d.",
                             arms_scored=f"{len([a for a in active if _scored(a, t, 'moleculenet_cv')])}/{len(active)}",
                             counted="yes" if t in scored_tasks else "NO - incomplete"))
        missing += [(ARMS[a]["label"], t) for a in active if not _scored(a, t, "moleculenet_cv")]
        if t not in scored_tasks: continue
        for a in active:
            win[a] += int(a in wa)
            if a in cwin: cwin[a] += int(a in wc)
            for _lab, (_base, _col) in BEATS.items():
                if a == _base or _base not in active: continue
                bnp[_lab][a] += int(_beats(t, a, _col))
    N = len(scored_tasks)
    def cell(x, a, pool):
        return f"{x[a]}/{N}" if a in pool else "n.d."
    rows = []
    for a, v in ARMS.items():
        row = {"model": v["label"], "unsup mols": _fmt_mol(v["unsup"]),
               "sup mols (desc)": _fmt_mol(v["dense"]), "sup mols (assay)": _fmt_mol(v["assay"]),
               "co-best x/n": cell(win, a, active),
               "CLIMB-only x/n": (cell(cwin, a, climb) if a in climb else
                                  ("n/a" if a in active else "n.d."))}
        for _lab, (_base, _col) in BEATS.items():
            row[_lab] = ("--" if a == _base else
                         (cell(bnp[_lab], a, active) if _base in active else "n.d."))
        rows.append(row)
    return pd.DataFrame(rows), pd.DataFrame(per_task), missing, scored_tasks

def a1_a_estimates():
    """A1.a (hold-out): NOT ranked -- point estimate with a 95% scaffold cluster-bootstrap CI per
    arm x task (the adversarial rarest-scaffold split is underpowered; the CV carries the rankings)."""
    rows = []
    for a, v in ARMS.items():
        row = {"model": v["label"]}
        for t in CORE_TASKS:
            pt, lo, hi = _hold_est(t, a)
            row[t] = f"{pt:.3g} [{lo:.3g}, {hi:.3g}]" if np.isfinite(pt) else "n.d."
        rows.append(row)
    return pd.DataFrame(rows)

# ---- A1.b (CV) = the ranking table of record; A1.a (hold-out) = estimates + CI, NOT ranked --------
_cv_summary, _cv_pt, _cv_missing, _cv_st = a1_summary_cv()
print("=" * 112)
print("Table A1.b -- pooled 5-fold scaffold-CV OOF -- test of record  <<< USE THIS ONE for rankings")
print("=" * 112)
print("molecules scored / dataset size per task:",
      ", ".join(f"{t}={n_molecules(t,'moleculenet_cv'):,}/{n_molecules(t):,}" for t in CORE_TASKS))
print(_cv_pt.to_string(index=False))
print(f"\nco-best counts over the {len(_cv_st)} task(s) with complete arm coverage: {', '.join(_cv_st)}"
      + (f"   |  EXCLUDED: {', '.join(t for t in CORE_TASKS if t not in _cv_st)}"
         if len(_cv_st) < len(CORE_TASKS) else ""))
print(); print(_cv_summary.to_string(index=False)); print()
if _cv_missing:
    print(f"unscored arm x task cells ({len(_cv_missing)}): "
          + ", ".join(f"{a}/{t}" for a, t in _cv_missing[:12]) + (" ..." if len(_cv_missing) > 12 else ""))

print("\n" + "=" * 112)
print("Table A1.a -- single scaffold hold-out -- point estimate [95% cluster-bootstrap CI], NOT ranked")
print("=" * 112)
print(a1_a_estimates().to_string(index=False)); print()

print("-" * 112)
print("ONE test throughout (scripts/best_model_bootstrap.py, scaffold cluster-bootstrap, n_boot=1000,\n"
      "BH-FDR across the arm x task family). 'co-best x/n' = tasks where the arm is NOT distinguishable\n"
      "from the per-task leader at q>=0.05 (leader included). 'beats no_pretrain' = q<0.05 in the arm's\n"
      "favour vs the frozen / end-to-end random baseline. 'CLIMB-only x/n' = the same, judged among the\n"
      "CLIMB arms only (anchors dropped). A1.a (hold-out) is NOT ranked: the rarest-scaffold split\n"
      "(113-4,112 test molecules, few scaffolds) is underpowered, so each arm is a point estimate with a\n"
      "95% cluster-bootstrap CI. HIV metric = NEF1%. Tox21 = mean over its 12 label columns.\n"
      "'sup mols (desc)' = PubChem molecules with computed RDKit-descriptor (MTR) labels; 'sup mols\n"
      "(assay)' capped by config_v2.SUPERVISED_FAMILY_CAPS (8M-FP sparse arms ~15 epochs over ~0.5M mols).")
print("-" * 112)

# ---- best-model matrix (6 headline models): same test, FDR within this pre-specified set, both schemes ----
MATRIX_MODELS = [("fp_desc", "Morgan+desc+XGBoost"), ("ecfp4", "Morgan+XGBoost (ECFP4)"),
                 ("unsup_only", "unsup_only (MLM)"), ("sup_only:dense_plus_sparse", "sup_only: dense+sparse"),
                 ("sup_only:mixed", "sup_only: mixed"), ("no_pretrain_e2e", "no_pretrain (end-to-end)")]
_cvb = {t: _cobest(t, "matrix", "cv")[0] for t in CORE_TASKS}
_hob = {t: _cobest(t, "matrix", "holdout")[0] for t in CORE_TASKS}
print("\nBest-model matrix (cluster-bootstrap + FDR within the 6-model set). cell = [CV][hold-out], "
      "# = co-best, . = distinguishable:")
print("model".ljust(26) + "".join(f"{t:>9}" for t in CORE_TASKS))
for _k, _lab in MATRIX_MODELS:
    _row = "".join(f"{('#' if _k in _cvb[t] else '.')+('#' if _k in _hob[t] else '.'):>9}" for t in CORE_TASKS)
    print(_lab.ljust(26) + _row)

if len(_cv_st) == len(CORE_TASKS):
    print(f"\nTable A1.b: COMPLETE -- all {len(CORE_TASKS)} tasks have every arm scored.")
else:
    print(f"\nTable A1.b: INCOMPLETE -- only {len(_cv_st)}/{len(CORE_TASKS)} tasks fully scored "
          f"({', '.join(_cv_st) or 'none'}); re-run once the scoring pass lands.")