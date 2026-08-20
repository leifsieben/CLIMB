"""A2 error bars on ONE estimand: sampling uncertainty over evaluation units.

Why (user decision 2026-08-18): the previous `sd_total` was uniform in FORMULA but not in MEANING.
It measures run-to-run reproducibility on a FIXED dataset, and what it contains differs per panel --
15 cells (3 pretraining seeds x 5 folds) on the MolNet/CBS panels, but only 3 eval seeds of a
PRE-AVERAGED 30-target mean on MoleculeACE, and only head-seed noise on ONE 132-molecule split for
hERG. Identical-looking whiskers therefore encoded different questions, and the panel with the least
information (hERG) drew among the tightest bars.

This computes instead "how much would this number move under a fresh draw of the evaluation units?",
which is the quantity that governs replicability, is the same estimand in every panel, and matches
the paper's OWN A1 rigor protocol (2026-08-05 cluster-bootstrap CI).

  BACE / Tox21 / QM7 / CBS  scaffold cluster bootstrap over per-molecule OOF (Bemis-Murcko clusters)
  MoleculeACE               target cluster bootstrap over the 30 ChEMBL targets
  hERG                      CANNOT be resampled -- Polaris withholds test labels, so
                            polaris_scores.csv has no y_true. Analytic Hanley-McNeil SE instead,
                            flagged `derived` so the caption can say so.

Writes figure_data/six_panel/a2_errorbars.csv:
  arm,panel,metric,value,ci_lo,ci_hi,se,method,n_units,n_dirs
"""
from __future__ import annotations
import csv, math, os, sys, collections, statistics as st
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT); sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from compare_models import _scaffold, _metric_over_cols            # noqa: E402
from figures.arms import ARMS                                       # noqa: E402

N_BOOT = 2000
# Polaris panel: Ames since 2026-08-18 (was hERG). n_test=1457 at the 53.32% train active rate.
POLARIS_PANEL, POLARIS_TASK = "Ames", "tdcommons/ames"
POLARIS_NPOS, POLARIS_NNEG = 777, 680
FD = ROOT / "figure_data"
MOL = {"BACE": ("auc", "climb_v2_phase2"), "Tox21": ("auc", "climb_v2_phase2"),
       "QM7": ("rmse", "climb_v2_phase2"), "HIV": ("nef1", "climb_v2_phase2")}
# chemeleon_frozen added 2026-08-19 (user): showing ONLY the fine-tuned CheMeleon invites the
# reading that CheMeleon beats the classical anchor as a representation. It does not -- frozen, it
# is best on 0 of 30 MoleculeACE targets (0.8256 macro RMSE vs ECFP+desc 0.6757) where fine-tuned
# it is best on 21. Both bars belong in the panel so the gap between them is visible.
A2_ARMS = ["ecfp", "ecfp_desc", "r3fp", "r3fp_desc", "sup_dense", "unsup", "u2s_dense",
           "random_encoder", "e2e_no_pretrain", "chemeleon_frozen", "chemeleon_e2e",
           # 2026-08-20: three arms added for SI fig a, which now takes EVERY error bar from this
           # file so that all six points in a panel carry ONE estimator computed one way, rather
           # than each arm's own replicate spread (pretraining-seed SD for the CLIMB arms,
           # head-seed SD for CheMeleon, which has one pretraining by construction). Those are
           # different estimands and putting them on the same axis invites a comparison neither
           # supports. The cluster bootstrap resamples test scaffolds and is defined identically
           # for every arm, whatever its replicate structure.
           #
           # This list is a SUPERSET of what fig_A2 draws -- that figure filters to its own MODELS
           # -- so adding arms here fills SI fig a without touching the A2 panel.
           "unsup_e2e", "sup_dense_e2e", "chemeleon_frozen_xgb"]


# QM7's phase-2 predictions are z-scored for most runs and native for a few; the native re-eval
# writes to moleculenet_cv_qm7native/ and exists only for the runs that needed it. Same rule as
# scripts/six_panel_aggregate: PREFER the native subdir, and never pool the two -- an arm whose
# CI mixed z-scored and native OOF would produce a meaningless interval, exactly as a mixed point
# estimate produced the bogus 129.9 QM7 mean for `no pretrain, end2end`.
SUBDIRS = {"QM7": ("moleculenet_cv_qm7native", "moleculenet_cv")}

# Tox21 is the opposite shape to QM7. Its point estimate is read from moleculenet_cv_tox21fixed/,
# but its PREDICTIONS stay in moleculenet_cv/ -- scripts/rescore_tox21.py derived the fixed summary
# from exactly that file, so bootstrapping moleculenet_cv reproduces the bar to the last digit and
# no SUBDIRS entry is needed. The exception is a dir whose tox21fixed is a FOREIGN RE-EVAL against
# the checkpoint (marked `reference_scoring.json`, ~0.0075 ROC-AUC of environment drift): there the
# summary was NOT derived from the local predictions, so the two paths describe different
# measurements and cannot be made to agree. six_panel_aggregate drops those dirs from the bar; this
# map lets the CI drop the same ones, which is the whole point of check 8 -- both paths must be
# looking at the same seeds.
VINTAGE_SUBDIRS = {"Tox21": "moleculenet_cv_tox21fixed"}


def _has_rows(root, run, dataset, sub):
    """True iff <root>/<run>/<sub>/ actually holds per-molecule rows for `dataset`."""
    for cand in (run, f"{run}_s0"):
        if any((FD / root / cand / sub / n).exists() for n in DUMP_NAMES):
            return load_oof(root, cand, dataset, sub) is not None
    return False


def _oof_subdir(root, runs, dataset):
    """One subdir for the whole arm: the first candidate any of its dirs has. (chosen, usable)."""
    vsub = VINTAGE_SUBDIRS.get(dataset)
    if vsub:
        import six_panel_aggregate as _spa
        keep = [r for r in runs if _spa._usable_dir(root, r, vsub)]
        # only narrow -- if NO dir survives, the arm has no corrected copy at all and the bar is
        # reading the uncorrected subdir too, so both paths still agree.
        runs = keep or runs
    cands = SUBDIRS.get(dataset, ("moleculenet_cv",))
    for sub in cands:
        # COVERAGE, not existence. `test_predictions.csv` existing says nothing about whether it
        # holds THIS dataset: chemeleon_frozen_s1/_s2 carry a dump pruned to QM7 alone, so on HIV,
        # BACE and Tox21 they passed the existence test, contributed no rows, and the CI silently
        # pooled 1 seed while the bar pooled 3. That is the exact failure the pooling comment on
        # load_oof_all warns about, reintroduced one level up. It only surfaced on HIV because
        # NEF1 is a discrete top-k statistic and seed spread there exceeded audit check 8's
        # tolerance; on BACE and Tox21 the same 1-vs-3 mismatch sat under it, unflagged.
        usable = [r for r in runs if _has_rows(root, r, dataset, sub)]
        if usable:
            return sub, usable
    return cands[-1], list(runs)


# THE PER-MOLECULE DUMP HAS TWO FILENAMES, and reading only the first cost six intervals.
#
# The CLIMB end-to-end arms' MolNet runs were finished in two passes: the original run dumped HIV
# to test_predictions.csv, and a later gap-fill pass dumped BACE/BBBP/ESOL/Lipophilicity/QM7/Tox21
# to test_predictions_gapfill.csv beside it. Both are real predictions from the same wave; only the
# filename differs. Reading one name meant unsup_e2e and sup_dense_e2e reported MISSING_OOF on
# exactly the three panels the gap-fill pass produced, while their SUMMARY numbers were present
# and correct -- a hole that looks like "never run" and is actually "written under another name".
#
# The two are searched in order and NEVER pooled: a dataset lives in exactly one of them, and
# concatenating would double every molecule that appeared in both.
DUMP_NAMES = ("test_predictions.csv", "test_predictions_gapfill.csv")


def load_oof(root, run, dataset, subdir="moleculenet_cv"):
    want = "cbs" if root == "cbs_benchmark" else dataset
    for name in DUMP_NAMES:
        p = FD / root / run / subdir / name
        if not p.exists():
            continue
        d = pd.read_csv(p, low_memory=False)
        d = d[d.dataset == want]
        if not len(d):
            continue
        # The gap-fill dumps are concatenations of per-dataset files and carry REPEATED HEADER
        # ROWS, which the `dataset == want` filter already drops -- but it leaves the numeric
        # columns typed as object, and a silently-object y_true turns every downstream comparison
        # into string comparison. Coerce, then insist nothing was lost to the coercion.
        for c in ("y_true", "y_pred", "mol_index", "output_index"):
            if c in d.columns:
                before = d[c].notna().sum()
                d[c] = pd.to_numeric(d[c], errors="coerce")
                assert d[c].notna().sum() == before, (
                    f"{root}/{run}/{subdir}/{name}: {before - int(d[c].notna().sum())} "
                    f"non-numeric value(s) in {c} for {want} -- the dump is malformed, not just "
                    f"concatenated")
        return d
    return None


# Datasets whose predictions may sit in EITHER subdir, resolved per dir instead of per arm, with
# a VALUE guard that makes the mixing impossible rather than merely unlikely. The one-subdir-per-arm
# rule exists to stop z-scored QM7 pooling with native QM7; it does not apply where a row count
# identifies the vintage outright. Tox21's reference dump is exactly 77,864 masked rows (93,876
# cells - 16,012 with w==0), and both moleculenet_cv/ and moleculenet_cv_tox21fixed/ hold that same
# dump for the runs that have it -- rescore_tox21.py derived the fixed SUMMARY from the plain
# subdir's predictions, while the anchor replicates were dumped straight into the fixed one.
# Requiring a single subdir for the whole arm therefore threw away real seeds: ecfp4_anchor_s1/_s2
# carry Tox21 only under tox21fixed/ while the base carries it only under moleculenet_cv/, so the
# anchor -- the arm that wins the figure -- got a 1-seed interval against a 3-seed bar.
PER_DIR_SUBDIRS = {"Tox21": (("moleculenet_cv_tox21fixed", "moleculenet_cv"), 77_864)}


def load_oof_all(root, runs, dataset):
    """OOF from EVERY pretraining-seed dir, tagged by dir. The bar pools all seeds, so the CI must
    too -- using one dir made the interval describe a different estimator than the bar."""
    out = []
    per_dir = PER_DIR_SUBDIRS.get(dataset) if root != "cbs_benchmark" else None
    if per_dir:
        cands, want_rows = per_dir
        for run in runs:
            for cand in (run, f"{run}_s0"):
                got = None
                for sub in cands:
                    d = load_oof(root, cand, dataset, sub)
                    # the guard, not the path, decides: a dump of any other length is a different
                    # vintage and is refused rather than pooled
                    if d is not None and len(d) == want_rows:
                        got = d
                        break
                if got is not None:
                    got = got.copy(); got["_dir"] = cand; out.append(got); break
        return pd.concat(out, ignore_index=True) if out else None
    subdir, runs = _oof_subdir(root, list(runs), dataset)
    for run in runs:
        for cand in (run, f"{run}_s0"):
            d = load_oof(root, cand, dataset, subdir)
            if d is not None:
                d = d.copy(); d["_dir"] = cand; out.append(d); break
    return pd.concat(out, ignore_index=True) if out else None


def fold_ids(root, smiles, y):
    """Per-molecule fold membership, so a bootstrap draw can recompute the BAR's estimator
    (mean over folds of the per-fold metric) rather than one global ranking over pooled OOF.
    That difference is large for NEF1: CBS unsup was 0.814 globally vs 0.735 as mean-of-per-fold."""
    if root == "cbs_benchmark":
        m = {r["smiles"]: int(r["fold"]) for r in csv.DictReader(open(ROOT / "data" / "cbs.csv"))}
        return np.array([m.get(s, -1) for s in smiles])
    import eval_v2
    folds = eval_v2._scaffold_kfold_indices(list(smiles), 5, 0, labels=y)
    out = np.full(len(smiles), -1)
    for j, idx in enumerate(folds):
        out[np.asarray(idx, dtype=int)] = j
    return out


def pooled_metric(m, kind, dirs, folds):
    """The BAR's estimator: mean over seed dirs of (mean over folds of the per-fold metric)."""
    per_dir = []
    for d in np.unique(dirs):
        sel = dirs == d
        vals = []
        for f in np.unique(folds[sel]):
            if f < 0:
                continue
            sub = m[sel & (folds == f)]
            if len(sub) == 0:
                continue
            v = _metric_over_cols(sub, "y_pred_a", kind)
            if np.isfinite(v):
                vals.append(v)
        if vals:
            per_dir.append(float(np.mean(vals)))
    return float(np.mean(per_dir)) if per_dir else np.nan


# The CI must describe the SAME estimator as the bar, and the bar is now clamped (QM7_SUBDIRS
# prefers moleculenet_cv_qm7clamped/). That dir holds summaries only -- the clamp is a scoring-time
# operation, so the predictions never needed rewriting -- which means the bootstrap has to apply it
# itself, once, before resampling. Applying it INSIDE the resample would refit the band on each
# bootstrap draw's own targets and make the bound a random variable.
def clamp_regression(m, kind, folds):
    """Clip y_pred to each fold's TRAIN target range +-25%; identity for classification metrics."""
    if kind not in ("rmse", "macro_rmse"):
        return m
    import eval_v2
    out = m["y_pred_a"].to_numpy(float).copy()
    y = m["y_true_a"].to_numpy(float)
    for f in np.unique(folds):
        if f < 0:
            continue
        te = folds == f
        tr = (folds != f) & (folds >= 0)
        if not te.any() or not tr.any():
            continue
        out[te] = eval_v2._bound_ood(out[te], y[tr], "regression")
    return m.assign(y_pred_a=out)


def scaffold_ci(d, kind, root, seed=0):
    """95% CI by resampling whole Bemis-Murcko scaffold clusters, recomputing the SAME pooled
    estimator the bar reports (all seed dirs; per-fold then averaged)."""
    m = d.rename(columns={"y_true": "y_true_a", "y_pred": "y_pred_a", "raw_smiles": "raw_smiles_a"})
    dirs = m["_dir"].to_numpy() if "_dir" in m.columns else np.array(["one"] * len(m))
    # Fold assignment must be computed on UNIQUE MOLECULES. Tox21's OOF carries one row per
    # (molecule, output_index) -- 12 rows per molecule -- and feeding that duplicated SMILES list to
    # _scaffold_kfold_indices produced a partition that only approximated the real one (residual
    # ~1.3% on Tox21 after the pooling fix). Deduplicate on mol_index first.
    # ...and on the MOST COMPLETE dir, not dirs[0]. `fold_ids` reconstructs the partition eval_v2
    # computed over the dataset's FULL molecule list, so the reference dump should be the one
    # closest to that list. e2e_no_pretrain's HIV dump is 41,120 rows in e2e_random_00 against
    # 41,127 in _01/_02; taking dirs[0] as the reference mapped those 7 molecules to fold -1 in
    # EVERY dir, so two dirs that are complete were scored on a truncated set and neither matched
    # its own eval-time summary. Ties keep the first dir, so single-dir arms are unaffected.
    key = "mol_index" if "mol_index" in m.columns else "raw_smiles_a"
    ref = max(np.unique(dirs), key=lambda d: m.loc[dirs == d, key].nunique())
    sub = m.loc[dirs == ref]
    uniq = sub.drop_duplicates(subset=[key]).sort_values(key)
    folds_u = fold_ids(root, uniq["raw_smiles_a"].tolist(), uniq["y_true_a"].to_numpy())
    fmap = dict(zip(uniq["raw_smiles_a"], folds_u))
    folds = np.array([fmap.get(s, -1) for s in m["raw_smiles_a"]])
    scaf = m["raw_smiles_a"].map(_scaffold).to_numpy()
    groups = collections.defaultdict(list)
    for pos, s in enumerate(scaf):
        groups[s].append(pos)
    keys = list(groups); idx = {k: np.array(v) for k, v in groups.items()}; K = len(keys)
    rng = np.random.default_rng(seed)
    m = clamp_regression(m, kind, folds)
    obs = pooled_metric(m, kind, dirs, folds)
    vals = []
    for _ in range(N_BOOT):
        rows = np.concatenate([idx[keys[i]] for i in rng.integers(0, K, K)])
        v = pooled_metric(m.iloc[rows], kind, dirs[rows], folds[rows])
        if np.isfinite(v):
            vals.append(v)
    if not vals:
        return obs, np.nan, np.nan, K
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return obs, float(lo), float(hi), K


def _expand(base):
    """<base> + its _s1/_s2 replicates -- unless arms.py already spelled the dirs out as a LIST
    (random_encoder's are _00/_01/_02, s2u_dense's are _s0/_s1/_s2)."""
    if isinstance(base, (list, tuple)):
        return list(base)
    return [base, f"{base}_s1", f"{base}_s2"]


def mace_ci(base, seed=0):
    """95% CI by resampling the 30 targets (pooled over pretraining-seed dirs + eval seeds)."""
    dirs = [d for d in _expand(base)
            if (FD / "chemeleon_suite" / "moleculeace" / d / "results.csv").exists()]
    if not dirs:
        return None
    per = collections.defaultdict(list)
    for d in dirs:
        for r in csv.DictReader(open(FD / "chemeleon_suite" / "moleculeace" / d / "results.csv")):
            if r["metric"] == "rmse" and r["subset"] == "overall":
                per[r["task"]].append(float(r["value"]))
    m = {t: st.mean(v) for t, v in per.items()}
    keys = list(m); rng = np.random.default_rng(seed)
    boots = sorted(float(np.mean([m[keys[i]] for i in rng.integers(0, len(keys), len(keys))]))
                   for _ in range(N_BOOT))
    # len(dirs), not len(_expand(base)): the caller records this as n_dirs and audit check 8
    # compares it to the BAR's n_seeds. Reporting the requested expansion instead of what was
    # found on disk made the check fire on ecfp/MoleculeACE with "bar 1, CI 3" -- the exact
    # inverse of the real situation, and a reporting bug masquerading as a data defect.
    return st.mean(m.values()), boots[int(.025*N_BOOT)], boots[int(.975*N_BOOT)], len(keys), len(dirs)


def herg_se(base):
    """Hanley-McNeil analytic SE for the POLARIS panel. Polaris withholds test labels, so this is
    the one panel that cannot be resampled -- flagged DERIVED so the caption can say so.
    2026-08-18: the panel moved hERG (n=132) -> Ames (n=1457). Ames has ~4x the effective sample and
    ~2.2x the headroom per SE, which is why the swap was made."""
    dirs = [d for d in _expand(base)
            if (FD / "chemeleon_suite" / "polaris" / d / "polaris_scores.csv").exists()]
    vals = []
    for d in dirs:
        for r in csv.DictReader(open(FD / "chemeleon_suite" / "polaris" / d / "polaris_scores.csv")):
            if r["task"] == POLARIS_TASK and r["metric"] == "roc_auc":
                vals.append(float(r["value"]))
    if not vals:
        return None
    A = st.mean(vals)
    # `vals` is one row per (dir, eval seed); n_dirs must count DIRS or the check compares a
    # 9-fit count against the bar's 3-dir count and reports a mismatch that does not exist.
    n_dirs = len(dirs)
    n1, n0 = POLARIS_NPOS, POLARIS_NNEG
    Q1, Q2 = A / (2 - A), 2 * A * A / (1 + A)
    se = math.sqrt((A*(1-A) + (n1-1)*(Q1-A*A) + (n0-1)*(Q2-A*A)) / (n1*n0))
    return A, A - 1.96*se, A + 1.96*se, se, len(vals), n_dirs


def main(only=None):
    """`only` = a subset of A2_ARMS to recompute; their rows REPLACE the matching rows in the
    existing CSV and every other arm is carried through untouched. Added 2026-08-18: the full
    sweep is ~1h, and a single arm's inputs change whenever one more replicate lands (here,
    e2e_random_02's native QM7), so recomputing all eight to refresh one is pure waste.
    """
    rows = []
    for arm in (only or A2_ARMS):
        spec = ARMS.get(arm)
        if not spec:
            continue
        src = spec["src"]
        # MolNet-shaped panels + CBS
        for panel, (kind, root) in MOL.items():
            runs = src.get("mol") or []
            if root == "cbs_benchmark":
                runs = src.get("mol") or []
            got = load_oof_all(root, [r for r in runs if r], panel)
            if got is None:
                rows.append(dict(arm=arm, panel=panel, metric=kind, value="", ci_lo="", ci_hi="",
                                 se="", method="MISSING_OOF", n_units=0, n_dirs=0)); continue
            v, lo, hi, K = scaffold_ci(got, kind, root)
            # n_dirs = how many pretraining-seed dirs the INTERVAL pooled. The bar's own seed count
            # lives in mainline_8M.csv's `extra`; audit check 8 compares the two, because a CI over
            # fewer seeds than the bar is an interval for a different estimator (see _oof_subdir).
            nd = int(got["_dir"].nunique()) if "_dir" in got.columns else 1
            rows.append(dict(arm=arm, panel=panel, metric=kind, value=round(v, 4),
                             ci_lo=round(lo, 4), ci_hi=round(hi, 4), se=round((hi-lo)/3.92, 4),
                             method="scaffold_cluster_bootstrap", n_units=K, n_dirs=nd))
            print(f"  {arm:16s} {panel:12s} {v:.4f} [{lo:.4f},{hi:.4f}] ({K} scaffolds, "
                  f"{nd} seed dir(s))", flush=True)
        # MoleculeACE
        if src.get("mace"):
            r = mace_ci(src["mace"])
            if r:
                v, lo, hi, K, nd = r
                rows.append(dict(arm=arm, panel="MoleculeACE", metric="macro_rmse", value=round(v,4),
                                 ci_lo=round(lo,4), ci_hi=round(hi,4), se=round((hi-lo)/3.92,4),
                                 method="target_cluster_bootstrap", n_units=K, n_dirs=nd))
                print(f"  {arm:16s} {'MoleculeACE':12s} {v:.4f} [{lo:.4f},{hi:.4f}] ({K} targets)", flush=True)
        # hERG
        r = herg_se(src.get("mace") or "")
        if r:
            v, lo, hi, se, n, nd = r
            rows.append(dict(arm=arm, panel=POLARIS_PANEL, metric="roc_auc", value=round(v,4),
                             ci_lo=round(lo,4), ci_hi=round(hi,4), se=round(se,4),
                             method="analytic_hanley_mcneil_DERIVED", n_units=1457, n_dirs=nd))
            print(f"  {arm:16s} {POLARIS_PANEL:12s} {v:.4f} [{lo:.4f},{hi:.4f}] SE={se:.4f} (derived)", flush=True)
    out = FD / "six_panel" / "a2_errorbars.csv"
    if only and out.exists():
        keep = [r for r in csv.DictReader(out.open()) if r["arm"] not in set(only)]
        for r in keep:
            r.setdefault("n_dirs", "")      # rows written before n_dirs existed
        # preserve the canonical A2_ARMS order rather than appending the recomputed arms at the end
        order = {a: i for i, a in enumerate(A2_ARMS)}
        rows = sorted(keep + rows, key=lambda r: order.get(r["arm"], len(order)))
        print(f"  merged: recomputed {len(only)} arm(s), carried {len(keep)} existing rows through")
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["arm","panel","metric","value","ci_lo","ci_hi","se","method","n_units","n_dirs"])
        w.writeheader(); w.writerows(rows)
    print(f"\nwrote {out} ({len(rows)} rows)")


def _acquire_lock(force=False):
    """Refuse to run while another copy of this script is running.

    This is an HOUR-LONG job that ends in a single overwrite of a2_errorbars.csv, so two
    overlapping runs do not merely waste CPU -- whichever finishes LAST wins, and on 2026-08-19
    that was nearly a run started before the r3-counts dirs landed. It would have written a table
    missing two arms with a BRAND-NEW mtime, which is precisely the one thing the audit's freshness
    check cannot see: that check compares timestamps, and a stale writer produces a fresh one.

    The lock stores the pid so a crashed run does not wedge the next one.
    """
    lock = ROOT / "figure_data" / "six_panel" / ".a2_bootstrap.lock"
    if lock.exists():
        try:
            pid = int(lock.read_text().split()[0])
        except (ValueError, IndexError):
            pid = None
        alive = False
        if pid is not None:
            try:
                os.kill(pid, 0)
                alive = True
            except (OSError, ProcessLookupError):
                alive = False
        if alive and not force:
            raise SystemExit(
                f"another a2_bootstrap_errorbars.py is running (pid {pid}, lock {lock}).\n"
                f"Two runs race on a2_errorbars.csv and the LAST writer wins, which can silently "
                f"publish an older data vintage under a fresh timestamp. Wait for it, or pass "
                f"--force if you are certain that pid is not this job.")
        if not alive:
            print(f"  stale lock from dead pid {pid}; taking it")
    lock.parent.mkdir(parents=True, exist_ok=True)
    lock.write_text(f"{os.getpid()} {time.strftime('%Y-%m-%dT%H:%M:%S')}\n")
    return lock


if __name__ == "__main__":
    import argparse, atexit, os, time
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", help="comma-separated subset of A2_ARMS to recompute and merge")
    ap.add_argument("--force", action="store_true", help="take the lock even if another run holds it")
    a = ap.parse_args()
    _lock = _acquire_lock(force=a.force)
    atexit.register(lambda: _lock.exists() and _lock.unlink())
    main(only=[x for x in a.arms.split(",") if x] if a.arms else None)
