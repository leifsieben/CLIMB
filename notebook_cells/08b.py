# ---------- A1.c · overall standing = mean rank across ALL FOUR benchmark suites ----------
# A1.a/A1.b rank the models WITHIN each MoleculeNet dataset; this pools EVERY benchmark we ran into one
# ranking: MoleculeNet, Polaris, MoleculeACE, CBS. Metrics are heterogeneous, so we rank (not average
# scores). Each suite is weighted EQUALLY (rank within suite -> rescale to a common 1..12 scale ->
# average the 4), so MoleculeACE(30)+Polaris(28) do not swamp MoleculeNet(6)+CBS(1). Same 12 models as
# A1.b. Polaris uses each task's primary_metric (matching scripts/chemeleon_suite_plots.py); MoleculeACE
# uses overall RMSE; CBS uses NEF1%. CheMeleon (e2e) has no native Polaris/MoleculeACE run -- those two
# come from Burns et al.'s PUBLISHED table (reference_long.csv), same footing as any published baseline;
# its MoleculeNet + CBS ranks are our native runs.
import csv, statistics as _st
from collections import defaultdict as _dd
_A1C = A1_ORDER                                       # the same 12 models as Fig A1.b
_SMAP = {"ecfp4": "ecfp4", "fp_desc": "fp_desc", "random_baseline_00": "no_pretrain",
         "no_pretrain_e2e_e2e": "no_pretrain_e2e", "unsup_8M": "unsup_only",
         "skip_dense_8M": "sup_only:dense", "skip_sparse_all_8M": "sup_only:sparse_all",
         "skip_dense_plus_sparse_8M": "sup_only:dense_plus_sparse",
         "u2s_dense_from8M": "unsup2sup:dense", "u2s_sparse_all_from8M": "unsup2sup:sparse_all",
         "u2s_dense_plus_sparse_from8M": "unsup2sup:dense_plus_sparse"}   # frozen dirs -> A1.b identity
_DIR = {v: k for k, v in _SMAP.items()}
_man = json.load(open("chemeleon_suite/data/polaris/polaris_manifest.json"))
_HIGHER = {"pearsonr", "spearmanr", "r2", "roc_auc", "pr_auc", "accuracy", "explained_var"}
_RAW = _dd(lambda: _dd(list))
for _r in csv.DictReader(open("chemeleon_suite/reference/reference_long.csv")):
    _RAW[(_r["track"], _r["model"])][(_r["task"], _r["metric"])].append(float(_r["value"]))

def _ours_pol(d):
    by = _dd(lambda: _dd(list))
    try:
        for r in csv.DictReader(open(f"figure_data/chemeleon_suite/polaris/{d}/polaris_scores.csv")):
            by[r["task"]][r["metric"]].append(float(r["value"]))
    except FileNotFoundError:
        return {}
    return {t: _st.mean(m[_man[t]["primary_metric"]]) for t, m in by.items()
            if _man.get(t, {}).get("primary_metric") in m}
def _ref_pol(name):
    return {t: _st.mean(v) for (t, m), v in _RAW[("polaris", name)].items()
            if m == _man.get(t, {}).get("primary_metric")}
def _ours_mace(d):
    by = _dd(list)
    try:
        for r in csv.DictReader(open(f"figure_data/chemeleon_suite/moleculeace/{d}/results.csv")):
            if r["subset"] == "overall" and r["metric"] == "rmse": by[r["task"]].append(float(r["value"]))
    except FileNotFoundError:
        return {}
    return {t: _st.mean(v) for t, v in by.items()}
def _ref_mace(name):
    return {t: _st.mean(v) for (t, m), v in _RAW[("moleculeace", name)].items() if m == "overall test rmse"}

_POL = {a: (_ref_pol("CheMeleon") if a == "chemeleon_e2e" else _ours_pol(_DIR.get(a, ""))) for a in _A1C}
_ACE = {a: (_ref_mace("CheMeleon") if a == "chemeleon_e2e" else _ours_mace(_DIR.get(a, ""))) for a in _A1C}
_MN  = {a: {t: arm_value(DF_CV, t, a) for t in CORE_TASKS if np.isfinite(arm_value(DF_CV, t, a))} for a in _A1C}
_cbsA = pd.read_csv("experiment_cbs/cbs_nef1_summary.csv"); _cbsA = _cbsA[_cbsA.metric == "nef1"]
_CBS = {a: ({"CBS": float(_cbsA[_cbsA.arm == a]["mean"].iloc[0])} if (_cbsA.arm == a).any() else {}) for a in _A1C}

def _suite_ranklists(vd, higher_of):
    """vd: arm -> {task: value}. Rank per task over the arms sharing that suite (1 = best). Returns
    arm -> list of per-dataset ranks, and the number of arms ranked."""
    arms = [a for a in _A1C if vd.get(a)]
    tasks = sorted(set.intersection(*[set(vd[a]) for a in arms])) if arms else []
    R = {a: [] for a in arms}
    for t in tasks:
        rk = pd.Series({a: vd[a][t] for a in arms}).rank(ascending=not higher_of(t))
        for a in arms: R[a].append(float(rk[a]))
    return R, len(arms)

_SPEC = [("MoleculeNet", _MN,  lambda t: TASKS[t]["higher_better"]),
         ("Polaris",     _POL, lambda t: _man[t]["primary_metric"] in _HIGHER),
         ("MoleculeACE", _ACE, lambda t: False),
         ("CBS",         _CBS, lambda t: True)]
_SUITES = {name: _suite_ranklists(vd, hof) for name, vd, hof in _SPEC}   # name -> (ranklists, n_arms)

def _mse(lst):                                         # (mean, standard error) of a rank list
    x = np.asarray(lst, float)
    return float(x.mean()), (float(x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else np.nan)

_rows = []
for _a in _A1C:
    _per, _eff = {}, {}          # suite -> (mean rank, SE) over the suite's datasets ; suite -> 1..12 rescale
    for _sn, (_R, _ns) in _SUITES.items():
        if _a in _R and _R[_a]:
            _m, _e = _mse(_R[_a]); _per[_sn] = (_m, _e); _eff[_sn] = 1 + 11 * (_m - 1) / (_ns - 1)
    if not _eff: continue
    _ev = np.array(list(_eff.values()))
    _rows.append(dict(arm=_a, overall=_ev.mean(), n=len(_ev), per=_per,
                      se=(_ev.std(ddof=1) / np.sqrt(len(_ev)) if len(_ev) > 1 else np.nan),
                      **{s: _eff.get(s, np.nan) for s in _SUITES}))
_S = pd.DataFrame(_rows).sort_values("overall").reset_index(drop=True)

# ---- forest: numbered colour circle per suite (see key), mean halo + ±1 SE across suites ----------
fig, ax = plt.subplots(figsize=(STYLE["col2"], 0.40 * len(_A1C) + 1.4))
_yy = np.arange(len(_S))[::-1]
_SNUM = {"MoleculeACE": 1, "MoleculeNet": 2, "Polaris": 3, "CBS": 4}      # legend numbering
_DYS = {1: 0.21, 2: 0.07, 3: -0.07, 4: -0.21}          # vertical stagger so equal ranks don't collide
for _yi, (_, _r) in zip(_yy, _S.iterrows()):
    _col = rc_color(_r.arm)
    if np.isfinite(_r.se):
        ax.errorbar(_r.overall, _yi, xerr=_r.se, fmt="none", ecolor=_col, elinewidth=1.3, capsize=2.5,
                    alpha=0.55, zorder=1)
    ax.scatter(_r.overall, _yi, s=180, color=_col, alpha=0.18, edgecolor="none", zorder=1)   # mean halo
    for _s, _num in _SNUM.items():
        if np.isfinite(_r[_s]):
            ax.scatter(_r[_s], _yi + _DYS[_num], s=92, color=_col, edgecolor="white", lw=0.6, zorder=3)
            ax.text(_r[_s], _yi + _DYS[_num], str(_num), ha="center", va="center",
                    fontsize=STYLE["fs_annot"] - 1.5, color="white", fontweight="bold", zorder=4)
    ax.text(_r.overall, _yi + 0.40, f"{_r.overall:.1f}", ha="center", va="bottom",
            fontsize=STYLE["fs_annot"] - 1, color="#333")
ax.set_yticks(_yy); ax.set_yticklabels([rc_label(_r.arm) for _, _r in _S.iterrows()], fontsize=STYLE["fs_annot"])
ax.set_xlim(0.4, len(_A1C) + 0.6); ax.set_xticks(range(1, len(_A1C) + 1))
ax.set_xlabel("mean rank across the 4 benchmark suites  (equal weight; 1 = best)", fontsize=STYLE["fs_axis_label"])
ax.grid(axis="x", ls=":", lw=0.6, color="#ccc", zorder=0); ax.set_axisbelow(True)
for _sp in ("top", "right", "left"): ax.spines[_sp].set_visible(False)
ax.tick_params(axis="y", length=0)
_key = "       ".join(f"({_n}) {_s if _s != 'CBS' else 'CBS Virtual Screen'}"
                     for _s, _n in sorted(_SNUM.items(), key=lambda kv: kv[1]))
ax.annotate(_key, xy=(0.5, -0.13), xycoords="axes fraction", ha="center", va="top",
            fontsize=STYLE["fs_annot"], color="#333")                 # legend OUTSIDE the axes
_suptitle(fig, "Fig A1.c - overall standing across four benchmark suites (equal weight per suite)",
          fontsize=STYLE["fs_title"], y=1.02)
_capc = ("Every benchmark we ran, pooled by RANK (metrics differ across suites, so scores can't be "
         "averaged). Each suite is weighted equally: rank the models within a suite, rescale to a common "
         "1..12 scale, average the four suites -- so MoleculeACE (30 tasks) and Polaris (28) do not swamp "
         "MoleculeNet (6) and CBS (1). Numbered circles = per-suite rank (see key), filled halo = mean, "
         "bar = ±1 SE across the four suites. Metrics: MoleculeNet native (RMSE/ROC-AUC/NEF1%), Polaris "
         "each task's primary metric, MoleculeACE overall RMSE, CBS NEF1%. Same 12 models as Fig A1.b. "
         "CheMeleon (e2e)'s Polaris & MoleculeACE ranks use Burns et al.'s PUBLISHED values (point "
         "estimates); all other cells are our native runs. Morgan+desc+XGBoost ranks best overall; "
         "CheMeleon (e2e) is 2nd -- strong on the bioactivity suites (MoleculeACE, Polaris), weaker on "
         "CBS; no CLIMB pretraining regime outranks the classical descriptor baseline.")
_caption(fig, 0.5, -0.19, "\n".join(_tw.wrap(_capc, 120)), ha="center", va="top",
         fontsize=STYLE["fs_annot"] - 0.5, color="#666")
save_fig(fig, "figA1c_overall_ranking_allsuites"); plt.show()

# ---- companion table: mean rank ± SE per suite (over its datasets) and overall (over the 4 suites) ----
_TCOLS = ["MoleculeNet", "Polaris", "MoleculeACE", "CBS"]
def _tcell(mse):
    if mse is None: return "—"
    _m, _e = mse
    return f"{_m:.2f} ± {_e:.2f}" if np.isfinite(_e) else f"{_m:.2f}"
print("\nFig A1.c - mean rank per suite ± SE (over the suite's datasets) and overall ± SE (over the 4 suites):")
print("| Model | " + " | ".join(_TCOLS) + " | Overall |")
print("|" + "---|" * (len(_TCOLS) + 2))
for _, _r in _S.iterrows():
    _ov = f"{_r.overall:.2f} ± {_r.se:.2f}" if np.isfinite(_r.se) else f"{_r.overall:.2f}"
    print(f"| {rc_label(_r.arm)} | " + " | ".join(_tcell(_r.per.get(c)) for c in _TCOLS) + f" | {_ov} |")
print("n datasets/suite: MoleculeNet=6, Polaris=28, MoleculeACE=30, CBS=1 (single benchmark -> no per-dataset SE)")
