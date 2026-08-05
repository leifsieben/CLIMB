# ---------- Table A1 · win counts + data cost (plain text; copy the numbers by hand) ----------
# "Best on a task" = leader by point estimate PLUS everyone not separable from the leader. The two
# evaluations use DIFFERENT separability rules: A1.b (CV, test of record) uses the paired Wilcoxon /
# DeLong test at alpha (README §8.1); A1.a (hold-out) uses a test-set bootstrap 1-sigma rule, because
# a few-hundred-molecule hold-out cannot resolve a 2-sigma significance test (see below).
# Unlike the bars, this table keeps ALL FIVE SFT recipes for both sup_only and unsup->sup.
import sys
for _p in (".", "scripts"):          # compare_models lives in scripts/, heads_v2 at the repo root
    if _p not in sys.path: sys.path.insert(0, _p)
from compare_models import delong_test
from scipy import stats
from heads_v2 import compute_metric, compute_nef   # the exact eval-pipeline metrics, reused below

ALPHA = 0.05
# TWO "best" definitions, one per evaluation. The CV (test of record) keeps the paired Wilcoxon /
# DeLong test at alpha=0.05. The single scaffold HOLD-OUT has only 113-784 test molecules, so that
# 2-sigma rule cannot separate even large point-estimate gaps (a 0.1-AUC spread reads as "tied").
# For the hold-out we instead resample the test molecules with replacement and call a model beaten
# only when the leader's advantage exceeds HOLDOUT_BOOT_K bootstrap standard errors -- a 1-sigma
# rule, more permissive but still accounting for small-sample noise. Same metric per task (RMSE /
# ROC-AUC / NEF1%); the CV table and every figure are unaffected.
HOLDOUT_BOOT_B, HOLDOUT_BOOT_K, HOLDOUT_BOOT_SEED = 2000, 1.0, 0
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

# ---- hold-out "best": paired test-set bootstrap (1-sigma rule) -------------------------------
def _hold_metric(task):
    """The task's reported metric as a function of (preds, labels) 2-D arrays -- numerically identical
    to the eval pipeline (RMSE / ROC-AUC over label columns / NEF1%), but the AUC uses a rank-sum
    formula rather than sklearn so the thousands of bootstrap re-evaluations stay fast."""
    m = TASKS[task]["metric"]
    if m == "NEF1%": return compute_nef
    if m == "RMSE":
        def _rmse(P, Y):
            out = [np.sqrt(np.mean((P[~np.isnan(Y[:, j]), j] - Y[~np.isnan(Y[:, j]), j]) ** 2))
                   for j in range(Y.shape[1]) if (~np.isnan(Y[:, j])).sum()]
            return float(np.mean(out)) if out else np.nan
        return _rmse
    def _auc(P, Y):                       # ROC-AUC via Mann-Whitney rank-sum == roc_auc_score
        out = []
        for j in range(Y.shape[1]):
            mask = ~np.isnan(Y[:, j]); y = Y[mask, j]
            if y.size == 0 or np.unique(y).size < 2: continue
            n1 = int((y > 0.5).sum()); n0 = y.size - n1
            if n1 == 0 or n0 == 0: continue
            r = stats.rankdata(P[mask, j])
            out.append((r[y > 0.5].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
        return float(np.mean(out)) if out else np.nan
    return _auc

_BOOT_CACHE = {}
def _boot_prep(task):
    """Bootstrap every arm ONCE per task: align all arms on the shared (molecule, output) grid of the
    hold-out, then resample MOLECULES with replacement (same resample for every arm -> paired) and
    recompute each arm's metric. Returns per-arm point value, per-arm bootstrap array, and direction."""
    if task in _BOOT_CACHE: return _BOOT_CACHE[task]
    have = [a for a in ARMS if _scored(a, task, "moleculenet")]
    dfs = {a: _tbl_preds(a, task, "moleculenet") for a in have}
    mols = sorted(set.intersection(*[set(d.mol_index) for d in dfs.values()]))
    outs = sorted(set(dfs[have[0]].output_index))
    mi = {m: i for i, m in enumerate(mols)}; oi = {o: j for j, o in enumerate(outs)}
    N, K = len(mols), len(outs)
    Yt = np.full((N, K), np.nan); P = {a: np.full((N, K), np.nan) for a in have}
    for a in have:
        d = dfs[a]; ri = d.mol_index.map(mi); ci = d.output_index.map(oi)
        ok = ri.notna() & ci.notna()
        P[a][ri[ok].astype(int).values, ci[ok].astype(int).values] = d.y_pred[ok].values
        Yt[ri[ok].astype(int).values, ci[ok].astype(int).values] = d.y_true[ok].values
    fn = _hold_metric(task); hb = TASKS[task]["higher_better"]
    point = {a: fn(P[a], Yt) for a in have}
    rng = np.random.default_rng(HOLDOUT_BOOT_SEED)
    idx = rng.integers(0, N, size=(HOLDOUT_BOOT_B, N))
    boot = {a: np.array([fn(P[a][ii], Yt[ii]) for ii in idx]) for a in have}
    _BOOT_CACHE[task] = (point, boot, hb)
    return _BOOT_CACHE[task]

def _boot_cobest(task, arms):
    """Hold-out best set: leader by point estimate + any arm whose deficit to the leader is under
    HOLDOUT_BOOT_K bootstrap SEs (1-sigma)."""
    have = [a for a in arms if _scored(a, task, "moleculenet")]
    if not have: return set(), None
    point, boot, hb = _boot_prep(task)
    have = [a for a in have if a in point]
    lead = max(have, key=lambda a: point[a] * (1 if hb else -1))
    keep = {lead}
    for a in have:
        if a == lead: continue
        dp = (point[lead] - point[a]) if hb else (point[a] - point[lead])   # + = leader ahead
        db = (boot[lead] - boot[a]) if hb else (boot[a] - boot[lead])
        se = float(np.nanstd(db))
        if not (dp > HOLDOUT_BOOT_K * se): keep.add(a)                      # within k-sigma -> co-best
    return keep, lead

def _best_set(task, arms, sub):
    """Dispatch: CV -> paired test (alpha), hold-out -> test-set bootstrap (k-sigma)."""
    return _boot_cobest(task, arms) if sub == "moleculenet" else _cobest(task, arms, sub)

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
        wa, la = _best_set(t, active, sub)
        wc, lc = _best_set(t, climb, sub)
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
print(f"'Best' = leader OR not separable from it. A1.b (CV): paired Wilcoxon (RMSE) / DeLong AUC "
      f"(classification/HIV) at alpha={ALPHA}. A1.a (hold-out): test-set bootstrap "
      f"({HOLDOUT_BOOT_B} resamples), a model is beaten only if the leader leads by > "
      f"{HOLDOUT_BOOT_K:g} bootstrap SE (a {HOLDOUT_BOOT_K:g}-sigma rule) -- more permissive, because "
      f"the small hold-out test sets (113 on ESOL, up to 4,112 on HIV) cannot resolve a 2-sigma test "
      f"on the smaller tasks.\nRanking uses the pretraining-seed "
      f"mean; the paired test uses the seed-0 run's predictions.\nHIV: ranked/bootstrapped on NEF1%, "
      f"CV-tested with DeLong AUC. Tox21 = mean/median over its 12 label columns. 'beats no_pretrain' "
      f"stays on the paired test in BOTH tables.\n'sup mols (desc)' are PubChem molecules carrying "
      f"computed RDKit-descriptor labels (MTR), NOT assay data.\n'sup mols (assay)' is capped by "
      f"config_v2.SUPERVISED_FAMILY_CAPS, so the 8M-FP sparse arms are ~15 epochs over ~0.5M "
      f"molecules, not 8M unique ones.")
print("-" * 112)

# ---- best-in-CV vs best-in-hold-out matrix for the headline models (source for the split-cell table) ----
MATRIX_MODELS = [("fp_desc", "Morgan+desc+XGBoost"), ("ecfp4", "Morgan+XGBoost (ECFP4)"),
                 ("unsup_only", "unsup_only (MLM)"), ("sup_only:dense_plus_sparse", "sup_only: dense+sparse"),
                 ("sup_only:mixed", "sup_only: mixed"), ("no_pretrain_e2e", "no_pretrain (end-to-end)")]
_cvb = {t: _best_set(t, [a for a in ARMS if _scored(a, t, "moleculenet_cv")], "moleculenet_cv")[0] for t in CORE_TASKS}
_hob = {t: _best_set(t, [a for a in ARMS if _scored(a, t, "moleculenet")], "moleculenet")[0] for t in CORE_TASKS}
print(f"\nBest-in-CV (paired, alpha={ALPHA}) vs best-in-hold-out (bootstrap {HOLDOUT_BOOT_K:g}-sigma), "
      f"headline models.  cell = [CV][hold-out],  # = best, . = not:")
print("model".ljust(26) + "".join(f"{t:>9}" for t in CORE_TASKS))
for _k, _lab in MATRIX_MODELS:
    _row = "".join(f"{('#' if _k in _cvb[t] else '.')+('#' if _k in _hob[t] else '.'):>9}" for t in CORE_TASKS)
    print(_lab.ljust(26) + _row)

# Per-task completeness guard, printed loudly: the win counts are only meaningful over tasks where
# EVERY active arm is scored. If a re-scoring pass is mid-flight and some arm x task cells are
# missing, say so here rather than letting the reader trust an x/n whose n silently shrank.
for _sub, _tag in [("moleculenet", "A1.a"), ("moleculenet_cv", "A1.b")]:
    _d, _, _missing, _st = a1_summary(_sub)
    if len(_st) == len(CORE_TASKS):
        print(f"Table {_tag}: COMPLETE -- all {len(CORE_TASKS)} tasks have every arm scored.")
    else:
        print(f"Table {_tag}: INCOMPLETE -- only {len(_st)}/{len(CORE_TASKS)} tasks fully scored "
              f"({', '.join(_st) or 'none'}); {len(_missing)} arm x task cells still missing. "
              f"x/n counts exclude the incomplete tasks; re-run once the scoring pass lands.")