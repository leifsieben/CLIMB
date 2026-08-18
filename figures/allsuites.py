"""Every dataset we ran, in one wide table — for the all-suites ranking figure (Fig A1-wide).

The 6-panel figures show per-dataset detail, which caps them at six columns. A rank average does
not need to show detail, so it can pool EVERY benchmark: 7 MoleculeNet datasets + 30 MoleculeACE
targets + 28 Polaris tasks + CBS = 66 datasets. More datasets, less sensitivity to any one
benchmark's quirks (see notes/bbbp-anchor-verification-2026-08-16.md for why that matters).

Sources, all under figure_data/:
  MoleculeNet  climb_v2_phase2/<seed dir>/moleculenet_cv/moleculenet_summary.csv   (pooled over
               pretraining-seed dirs, then over head-seed x fold cells)
  MoleculeACE  chemeleon_suite/moleculeace/<dir>/results.csv                       (mean over seeds)
  Polaris      chemeleon_suite/polaris/<dir>/polaris_scores.csv, each task scored on its own
               primary metric from chemeleon_suite/data/polaris/polaris_manifest.json
  CBS          experiment_cbs/cbs_nef1_summary.csv                                 (NEF1%)
"""
from __future__ import annotations
import csv
import json, statistics as st, collections
from pathlib import Path
import numpy as np
import pandas as pd

from figures.sixpanel import NATIVE_SUBDIRS
from figures.arms import ARMS, ARM_ORDER

ROOT = Path(__file__).resolve().parent.parent
FD = ROOT / "figure_data"

SUITES = ["MoleculeNet", "MoleculeACE", "Polaris", "CBS"]

# MoleculeNet results are NOT all under one root: each compute wave wrote into its own tree, keyed
# by encoder prefix. MoleculeACE / CBS / Polaris are flat by prefix, MoleculeNet is not — so always
# resolve a prefix through molnet_dir() rather than hard-coding climb_v2_phase2.
#   climb_v2_phase2  mainline arms, anchors, controls, the A2 compute ladder, corrupted objectives
#   climb_v2_h1      the 30 canonical/enumerated data-fraction encoders
#   climb_v2_vocab   the 8 bpe_* / unigram_* tokenizer encoders
MOLNET_ROOTS = ["climb_v2_phase2", "climb_v2_h1", "climb_v2_vocab"]


def molnet_dir(prefix, subdir="moleculenet_cv"):
    """Path to a prefix's CV dir, wherever its wave wrote it, or None.

    Accepts EITHER file: `moleculenet_summary.csv` (per-fold rows, the normal runner) or
    `suite_summary.json` (e2e-style runners like chemeleon_e2e write only this). Requiring the CSV
    made those arms read as "not run" on all 7 MolNet datasets, and since fig_A1 admits arms by
    coverage COUNT, a loader gap was silently deciding which arms appear in the ranking panel.
    """
    for root in MOLNET_ROOTS:
        d = FD / root / prefix / subdir
        if (d / "moleculenet_summary.csv").exists() or (d / "suite_summary.json").exists():
            return d
    return None

MOLNET = {"BACE": ("roc_auc", True), "BBBP": ("roc_auc", True), "HIV": ("roc_auc", True),
          "Tox21": ("roc_auc", True), "ESOL": ("rmse", False), "QM7": ("rmse", False),
          "Lipophilicity": ("rmse", False)}

# Polaris metrics where larger is better; everything else (errors) is smaller-is-better.
HIGHER = {"pearsonr", "spearmanr", "r2", "roc_auc", "pr_auc", "accuracy", "explained_var",
          "f1", "mcc", "cohen_kappa"}


def _polaris_manifest():
    return json.load(open(ROOT / "chemeleon_suite/data/polaris/polaris_manifest.json"))


def _molnet(dirs):
    """{dataset: mean over all pretraining-seed dirs x fold ENSEMBLE rows}.

    Uses the plain `<metric>` fold rows (fold0..foldK-1), never `<metric>_cell`. eval_v2.py
    computes the fold row by averaging the head-seed PREDICTIONS first and then scoring the
    ensembled prediction (`pred = np.mean(np.stack(seed_preds), axis=0)`); `_cell` rows are the
    metric on each individual seed's own un-ensembled prediction, added only so a seed-decomposed
    spread CAN be computed -- they are not a valid substitute for the ensemble row as a point
    estimate. Verified 2026-08-16: averaging `_cell` values directly understates performance by
    0.5-1% AUC on BACE/Tox21, ~1.5 RMSE on QM7, and up to 5-6% NEF1 on CBS (small-count top-1%
    metric, most sensitive to pre- vs post-ensemble scoring). Do not reintroduce the `_cell`
    fallback as the value source.

    Resolved PER DATASET, not per file, for two reasons:
      * QM7 (and ESOL/Lipophilicity) exist in two unit conventions and the native re-eval lives in
        its own subdir -- see figures.sixpanel.NATIVE_SUBDIRS. Ranking is computed across arms
        WITHIN a dataset, so a single arm read in the other convention corrupts that dataset's
        whole ranking column, not just its own cell.
      * All of an arm's dirs must come from ONE subdir. Mixing them is what produced a QM7 mean of
        129.9 elsewhere (10 native folds averaged with 5 z-scored ones).
    """
    out = {}
    for ds, (metric, _) in MOLNET.items():
        # one subdir for this (arm, dataset): the first candidate any of the arm's dirs has
        chosen, usable = None, []
        for sub in NATIVE_SUBDIRS.get(ds, ("moleculenet_cv",)):
            usable = [d for d in dirs if molnet_dir(d, sub) is not None]
            if usable:
                chosen = sub
                break
        if not chosen:
            continue
        vals = []
        for d in usable:
            md = molnet_dir(d, chosen)
            f = md / "moleculenet_summary.csv"
            if f.exists():
                vals += [float(r["main_value"]) for r in csv.DictReader(open(f))
                         if r["dataset"] == ds and r["main_metric"] == metric
                         and r["head_seed"] not in ("MEAN", "STD")
                         and r["main_value"] not in ("", "nan")]
                continue
            j = md / "suite_summary.json"
            if j.exists():
                v = {k.lower(): v for k, v in json.load(open(j)).items()}.get(f"{ds}_mean".lower())
                if v is not None:
                    vals.append(float(v))
        if vals:
            out[ds] = st.mean(vals)
    return out


def _moleculeace(dirs):
    """{target: mean overall RMSE over eval seeds and pretraining-seed dirs}."""
    per = collections.defaultdict(list)
    for d in dirs:
        f = FD / "chemeleon_suite" / "moleculeace" / d / "results.csv"
        if not f.exists():
            continue
        for r in csv.DictReader(open(f)):
            if r["subset"] == "overall" and r["metric"] == "rmse":
                try:
                    per[r["task"]].append(float(r["value"]))
                except ValueError:
                    pass
    return {t: st.mean(v) for t, v in per.items()}


def _polaris_summary():
    """{(model_dir, task, metric): mean} from chemeleon_suite/summaries/polaris_summary.csv.

    Recovery source for the 2026-08-16 clobbering: the hERG top-up REWROTE (not appended) the
    polaris_scores.csv of the five headline arms (skip_dense_8M, skip_dense_plus_sparse_8M,
    skip_sparse_all_8M, unsup_8M, u2s_dense_from8M), leaving only the 3 hERG rows. The summary was
    built 2026-08-13 from the SAME per-seed runs (its hERG mean matches the rewritten files to 7
    decimals), so the other 27 tasks' per-task means are recoverable from it. Per-seed granularity
    is lost for those tasks; every consumer here averages per task anyway.
    """
    f = ROOT / "chemeleon_suite" / "summaries" / "polaris_summary.csv"
    out = {}
    if f.exists():
        for r in csv.DictReader(open(f)):
            if r.get("source", "ours") == "ours":
                out[(r["model"], r["task"], r["metric"])] = float(r["mean"])
    return out


def _polaris(dirs, man):
    """{task: mean of the task's primary metric over eval seeds}."""
    per = collections.defaultdict(lambda: collections.defaultdict(list))
    for d in dirs:
        f = FD / "chemeleon_suite" / "polaris" / d / "polaris_scores.csv"
        if not f.exists():
            continue
        for r in csv.DictReader(open(f)):
            per[r["task"]][r["metric"]].append(float(r["value"]))
    out = {}
    for t, m in per.items():
        pm = man.get(t, {}).get("primary_metric")
        if pm in m:
            out[t] = st.mean(m[pm])
    if len(dirs) == 1 and len(out) < len(man):
        summ = _polaris_summary()          # clobbered-file recovery: fill MISSING tasks only
        for t, meta in man.items():
            pm = meta.get("primary_metric")
            if t not in out and (dirs[0], t, pm) in summ:
                out[t] = summ[(dirs[0], t, pm)]
    return out


def _cbs_value(mol_src):
    """CBS NEF1 for one arm, read from figure_data/cbs_benchmark/<dir>/moleculenet_cv/.

    Takes the arm's `mol` dirs, NOT its `cbs` key: CBS lives in its own tree but under the same
    run-dir names as the MolNet wave, which is exactly how scripts/six_panel_aggregate resolves
    the canonical CBS panel. (`arms.py`'s `cbs=` field holds arm LABELS from the deprecated
    precomputed summary — "sup_only:dense" and so on — not directory names, so it cannot be used
    as a path.) The summary experiment_cbs/cbs_nef1_summary.csv is DEPRECATED and must not be
    reintroduced: its `arm` list silently omits whole waves, so arms added after it was generated
    read as "not run" rather than erroring. Audited by audit_figure_consistency.py check 1.
    """
    if not mol_src:
        return None
    dirs = list(mol_src) if isinstance(mol_src, (list, tuple)) else [mol_src]
    vals = []
    for d in dirs:
        for cand in (d, f"{d}_s0"):
            base = FD / "cbs_benchmark" / cand / "moleculenet_cv"
            f = base / "moleculenet_summary.csv"
            if f.exists():
                vals += [float(r["main_value"]) for r in csv.DictReader(open(f))
                         if r["main_metric"] == "nef1" and r["head_seed"].startswith("fold")]
                break
            # FALLBACK: the CBS e2e runner writes per_fold.csv instead (chemeleon_e2e). Same
            # per-fold NEF1 values, different file -- see the note in _molnet.
            pf = base / "per_fold.csv"
            if pf.exists():
                vals += [float(r["nef1"]) for r in csv.DictReader(open(pf))
                         if r.get("nef1") not in (None, "", "nan")]
                break
    return sum(vals) / len(vals) if vals else None


def _seed_dirs(base):
    """A source dir plus its pretraining-seed replicates, if they exist on disk.

    `base` may already be an explicit LIST, exactly as arms.py's `mol` is — s2u_dense's dirs are
    _s0/_s1/_s2 rather than <base>/_s1/_s2, so arms.py spells them out. Mirrors the same guard in
    scripts/six_panel_aggregate.mace_seed_dirs; without it this raised
    AttributeError: 'list' object has no attribute 'endswith'.
    """
    if isinstance(base, (list, tuple)):
        return list(base)
    if base.endswith("_00"):
        return [base, base[:-3] + "_01", base[:-3] + "_02"]
    return [base, f"{base}_s1", f"{base}_s2"]


def wide_table(arms=None):
    """Returns (scores, meta):
    scores  DataFrame, rows = arms, columns = all 66 datasets (NaN where not run)
    meta    DataFrame indexed by dataset with columns suite, metric, higher_better
    """
    arms = arms or ARM_ORDER
    man = _polaris_manifest()
    rows, meta = {}, {}
    for a in arms:
        src = ARMS[a]["src"]
        vals = {}
        for ds, v in _molnet(src["mol"]).items():
            vals[f"MolNet:{ds}"] = v
            meta[f"MolNet:{ds}"] = ("MoleculeNet", MOLNET[ds][0], MOLNET[ds][1])
        if src.get("mace"):                       # e2e-only arms (chemeleon_e2e) have no
            mace_dirs = _seed_dirs(src["mace"])   # MoleculeACE/Polaris -- mace=None
            for t, v in _moleculeace(mace_dirs).items():
                vals[f"MolACE:{t}"] = v
                meta[f"MolACE:{t}"] = ("MoleculeACE", "rmse", False)
            # Polaris has no pretraining-seed replicates, so the base dir alone -- but `mace` may
            # already be an explicit list (s2u_dense), in which case wrapping it in another list
            # produced a PosixPath / list TypeError.
            pol_dirs = list(src["mace"]) if isinstance(src["mace"], (list, tuple)) else [src["mace"]]
            for t, v in _polaris(pol_dirs, man).items():
                pm = man[t]["primary_metric"]
                vals[f"Polaris:{t.split('/')[-1]}"] = v
                meta[f"Polaris:{t.split('/')[-1]}"] = ("Polaris", pm, pm in HIGHER)
        cbs_v = _cbs_value(src.get("mol"))
        if cbs_v is not None:
            vals["CBS:cbs"] = cbs_v
            meta["CBS:cbs"] = ("CBS", "nef1", True)
        rows[a] = vals
    S = pd.DataFrame(rows).T.reindex(index=arms)
    M = pd.DataFrame(meta, index=["suite", "metric", "higher_better"]).T
    M = M.loc[[c for c in S.columns]]
    order = sorted(S.columns, key=lambda c: (SUITES.index(M.loc[c, "suite"]), c))
    return S[order], M.loc[order]


def effective_n(R, M, max_pairs=4000):
    """Effective number of INDEPENDENT datasets, per suite.

    Datasets inside a suite largely agree with each other about which model is better (MoleculeACE's
    30 ChEMBL targets correlate at rho ~ 0.74), so counting them as 30 independent observations
    understates the uncertainty on a mean rank. Standard design-effect correction:
        n_eff = n / (1 + (n - 1) * rho_bar)
    with rho_bar the mean pairwise Spearman correlation between the datasets' model-rankings.
    See scripts/weighting_sensitivity.py for the full diagnostic.
    """
    import itertools
    from scipy.stats import spearmanr
    out = {}
    for s in SUITES:
        cols = [c for c in R.columns if M.loc[c, "suite"] == s and R[c].notna().sum() >= 8]
        if len(cols) < 2:
            out[s] = float(len(cols))
            continue
        pairs = list(itertools.combinations(cols, 2))
        if len(pairs) > max_pairs:
            pairs = [pairs[i] for i in np.linspace(0, len(pairs) - 1, max_pairs).astype(int)]
        vals = []
        for a, b in pairs:
            d = R[[a, b]].dropna()
            if len(d) >= 8:
                r = spearmanr(d[a], d[b]).statistic
                if np.isfinite(r):
                    vals.append(r)
        rho = float(np.mean(vals)) if vals else 0.0
        n = len(cols)
        out[s] = n / (1 + (n - 1) * max(rho, 0.0))
    return out


def wide_ranks(arms=None, per_suite_equal=False):
    """Per-dataset rank (1 = best), and the mean rank per arm.

    per_suite_equal=False -> every DATASET counts once (MoleculeACE + Polaris dominate, 58 of 66)
    per_suite_equal=True  -> mean rank within each suite first, then average the four suites
    """
    S, M = wide_table(arms)
    N = len(S)
    R = pd.DataFrame(index=S.index, columns=S.columns, dtype=float)
    for c in S.columns:
        col = S[c].dropna()
        if len(col) < 2:
            continue
        r = col.rank(ascending=not M.loc[c, "higher_better"])
        R.loc[r.index, c] = 1 + (N - 1) * (r - 1) / (len(col) - 1)   # rescale to the full field
    out = pd.DataFrame(index=S.index)
    for s in SUITES:
        cols = [c for c in S.columns if M.loc[c, "suite"] == s]
        out[s] = R[cols].mean(axis=1)
        out[s + "_n"] = R[cols].notna().sum(axis=1)
    if per_suite_equal:
        out["mean_rank"] = out[SUITES].mean(axis=1)
        out["se_rank"] = out[SUITES].std(axis=1, ddof=1) / np.sqrt(out[SUITES].notna().sum(axis=1))
        out["n_units"] = out[SUITES].notna().sum(axis=1)
    else:
        out["mean_rank"] = R.mean(axis=1)
        out["se_rank"] = R.std(axis=1, ddof=1) / np.sqrt(R.notna().sum(axis=1))
        out["n_units"] = R.notna().sum(axis=1)
    # Honest SE: inflate by the design effect, so 30 near-duplicate MoleculeACE targets do not buy
    # sqrt(30) worth of precision. Without this the bars are ~3x too tight.
    ne = effective_n(R, M)
    deff = out["n_units"] / max(sum(ne.values()), 1e-9)
    out["se_rank_naive"] = out["se_rank"]
    out["se_rank"] = out["se_rank"] * np.sqrt(deff.clip(lower=1.0))
    out["n_eff"] = sum(ne.values())
    return out.sort_values("mean_rank"), R, M
