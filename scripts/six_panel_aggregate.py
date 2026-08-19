"""Aggregate every result on disk into the canonical 6-panel benchmark tables.

The 6 panels (see notes/six-panel-migration.md):
  1 MoleculeACE  macro-mean RMSE over 30 ChEMBL targets (overall + cliff subset)
  2 CBS          NEF1%
  3 BACE         ROC-AUC
  4 hERG         ROC-AUC (Polaris provided split; replaced BBBP 2026-08-16)
  5 Tox21        mean ROC-AUC over 12 subtasks
  6 QM7          RMSE (atomization energy)

Arm names, colours and panel definitions come from ONE place: figures/arms.py. This script is
pure re-aggregation -- no new compute. It writes everything the figure scripts read:

  figure_data/six_panel/mainline_8M.csv            arm x panel point estimates
  figure_data/six_panel/mainline_8M_long.csv       replicate-level values (per target / per fold-seed)
  figure_data/six_panel/mainline_8M_bootstrap.csv  MoleculeACE target-cluster bootstrap 95% CI
  figure_data/six_panel/STATUS.md                  coverage board: what exists, what is still missing

Run:  python3 scripts/six_panel_aggregate.py
"""
from __future__ import annotations
import csv, sys, statistics as st, collections, random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from figures.arms import ARMS, ARM_ORDER, PANELS, PANEL_ORDER  # noqa: E402

FD = ROOT / "figure_data"
OUT = FD / "six_panel"
random.seed(0)  # deterministic bootstrap

# HIV joined on 2026-08-19, replacing CBS in the rare-active-screen slot: it lives in the SAME
# MoleculeNet CV tree as BACE/Tox21/QM7, so it needs no special case at all -- which is itself part
# of why the swap simplifies things (CBS required its own tree, its own reader and its own
# fallbacks). CBS is still aggregated below for the all-suites table and the SI panel.
MOL_PANELS = {"BACE": "roc_auc", "Tox21": "roc_auc", "QM7": "rmse", "HIV": "nef1"}

# hERG replaced BBBP on 2026-08-16 and comes from Polaris (benchmark-provided split), not from our
# scaffold CV -- so it is read from polaris_scores.csv, one value per eval seed.
POLARIS_PANELS = {"Ames": ("tdcommons/ames", "roc_auc")}


# ---------------------------------------------------------------- MoleculeACE ----------------
def mace_seed_dirs(base):
    """<base> plus its pretraining-seed replicates <base>_s1/_s2, whichever exist on disk.

    The MoleculeACE seed top-up (box i-092c1d745d9a9f04e, 2026-08-16) writes those two dirs for
    every mainline arm; until they land this silently returns just the base dir, and the moment
    they appear they are pooled in with no code change.
    """
    # `base` may be an explicit LIST, exactly as arms.py's `mol` already is. Needed because the
    # controls' replicates are named _00/_01/_02 rather than <base>/_s1/_s2, and renaming real
    # result dirs to fit a resolver would be the tail wagging the dog (compute session, 2026-08-18).
    cands = list(base) if isinstance(base, (list, tuple)) else [base, f"{base}_s1", f"{base}_s2"]
    return [d for d in cands
            if (FD / "chemeleon_suite" / "moleculeace" / d / "results.csv").exists()]


def mace_per_target(base):
    """{subset: {target: mean RMSE over eval seeds AND pretraining-seed dirs}}, or None."""
    dirs = mace_seed_dirs(base)
    if not dirs:
        return None
    per = collections.defaultdict(list)
    for d in dirs:
        for r in csv.DictReader(open(FD / "chemeleon_suite" / "moleculeace" / d / "results.csv")):
            if r["metric"] != "rmse":
                continue
            try:
                per[(r["subset"], r["task"])].append(float(r["value"]))
            except ValueError:
                pass
    out = collections.defaultdict(dict)
    for (subset, task), vals in per.items():
        out[subset][task] = st.mean(vals)
    return out or None


def mace_seed_macros(base):
    """One macro-mean (30 targets, overall rmse) per (pretraining-seed dir, eval seed).

    The panel's sd_total is the SD across these macro-means -- the same "one replicate evaluation
    of the whole panel" estimand as the MolNet panels: 3 values today (1 dir x 3 eval seeds),
    9 once the _s1/_s2 pretraining top-up lands (mace_seed_dirs picks the new dirs up unchanged).
    """
    macros = []
    for d in mace_seed_dirs(base):
        per_seed = collections.defaultdict(list)
        for r in csv.DictReader(open(FD / "chemeleon_suite" / "moleculeace" / d / "results.csv")):
            if r["subset"] == "overall" and r["metric"] == "rmse":
                try:
                    per_seed[r["seed"]].append(float(r["value"]))
                except ValueError:
                    pass
        macros += [st.mean(v) for v in per_seed.values() if v]
    return macros


def cluster_bootstrap(by_target, n=2000):
    """95% CI on the macro-mean by resampling the 30 targets (cluster bootstrap)."""
    keys = list(by_target)
    if len(keys) < 2:
        return None, None
    boots = sorted(st.mean([by_target[random.choice(keys)] for _ in keys]) for _ in range(n))
    return boots[int(0.025 * n)], boots[int(0.975 * n)]


# ---------------------------------------------------------------- MoleculeNet / CBS -----------
# QM7 was evaluated with standardize=zscore in the phase-2 wave and the runner reported the
# NORMALIZED rmse (~0.85) for MOST runs but native kcal/mol (~200) for a few, so the panel silently
# mixed two units (see check_panel_units). The re-eval writes native values to a SEPARATE subdir,
# moleculenet_cv_qm7native/, and it exists only for the runs that needed fixing -- the runs that
# were already native were (correctly) not re-run. So the QM7 panel must PREFER the native subdir
# and fall back to the ordinary one, which is what this tuple expresses. Every other panel reads
# moleculenet_cv/ only.
# CLAMPED FIRST (user 2026-08-19). moleculenet_cv_qm7clamped/ is the SAME predictions with
# eval_v2._bound_ood applied -- regression predictions clipped to each fold's TRAIN target
# range +-25%, the bound scripts/chemeleon_suite_run.py has always applied on the suite
# tracks and eval_v2 did not. Without it CheMeleon-frozen's QM7 reads 268.8 because ONE
# molecule is predicted at -15,012 kcal/mol; with it, 208.8. Near-noop elsewhere (<=4.3).
QM7_SUBDIRS = ("moleculenet_cv_qm7clamped", "moleculenet_cv_qm7native", "moleculenet_cv")
# Tox21's 2026-08-05 missing-label fix reached the predictions but only fold0 of the summary rows
# (interrupted re-run); moleculenet_cv_tox21fixed/ is re-scored from each run's own predictions by
# scripts/rescore_tox21.py. Same all-or-nothing rule as QM7: one subdir per arm, never a mix.
TOX21_SUBDIRS = ("moleculenet_cv_tox21fixed", "moleculenet_cv")
DEFAULT_SUBDIRS = ("moleculenet_cv",)


def _usable_dir(root, d, sub):
    """Does `<root>/<d>/<sub>` exist AND come from the reference environment?

    `reference_scoring.json` marks a re-evaluation done on a foreign box (see _pick_subdir). Its
    PRESENCE disqualifies the dir: the number is a fresh measurement, not a restoration of the
    original one, and mixing the two inside a single arm is invisible in the output.
    """
    for cand in (d, f"{d}_s0"):
        dd = FD / root / cand / sub
        if dd.exists():
            return not (dd / "reference_scoring.json").exists()
    return False


def _skip_reason(root, d, sub):
    """WHY a dir was dropped, checked rather than assumed.

    The message used to say "no <sub>/ yet" for every skip, which is only one of the two reasons and
    was the wrong one for e2e_random_01/_02: those dirs HAVE moleculenet_cv_tox21fixed/, they are
    refused because it carries reference_scoring.json. Reporting a missing directory that is sitting
    right there sends the reader to re-run something that already ran.
    """
    # Same candidate resolution as _usable_dir (`<d>` or `<d>_s0`) -- building the path a second,
    # slightly different way is how a diagnostic ends up confidently reporting the wrong reason.
    for cand in (d, f"{d}_s0"):
        dd = FD / root / cand / sub
        if dd.exists():
            return ("foreign re-eval (reference_scoring.json)"
                    if (dd / "reference_scoring.json").exists() else "unusable")
    return "not written yet"


def _pick_subdir(root, dirs, subdirs):
    """Choose ONE subdir for the whole arm, so its dirs can never be read in mixed units.

    THIS IS THE FIX FOR A BUG THAT REACHED A FIGURE. `moleculenet_cv_qm7native/` is written run by
    run, so mid-re-eval some of an arm's 3 pretraining dirs have it and some do not. Resolving each
    dir independently then pooled 2 native folds (~194 kcal/mol) with 5 z-scored ones (~0.85) into
    one "mean" of 129.9 for `no pretrain, end2end` -- a number that looked like a spectacular
    result rather than a unit error, and that check_panel_units could not catch because it compares
    ACROSS arms and every other arm was internally fine.

    Rule: take the first subdir in `subdirs` that ANY of the arm's dirs has, and then use only the
    dirs that have it. Fewer seeds is a visible, honest degradation (n_seeds drops); mixed units is
    an invisible, wrong one. Returns (chosen_subdir, usable_dirs, skipped_dirs).

    A dir is NOT usable if its corrected subdir carries `reference_scoring.json`. That marker means
    the copy is a FRESH RE-EVALUATION against the checkpoint on a box whose RDKit/DeepChem parses
    7,831 Tox21 molecules where the reference environment parses 7,823 -- it is scored on the
    shared molecule set (rows_kept 77,214 of 77,864) but carries a ~0.0075 ROC-AUC offset from
    environment drift. figures/sixpanel.py::_usable has refused those since 2026-08-19; this
    function did not, and the gap put `e2e_no_pretrain` on the Tox21 panel as ONE re-scored dir
    (e2e_random_00) averaged with TWO foreign re-evals (e2e_random_01/_02) -- three seeds of two
    different measurements. It surfaced as audit check 8: the bar (3 mixed dirs, 0.7709) and its
    own whisker (the 2 dirs that have predictions there, 0.7678) described different estimators.
    A 0.0075 offset is 15-40% of the Tox21 differences these figures measure, so it is not
    absorbable; one honest seed beats three mixed ones.
    """
    for sub in subdirs:
        usable = [d for d in dirs if _usable_dir(root, d, sub)]
        if usable:
            return sub, usable, [d for d in dirs if d not in usable]
    return subdirs[-1], list(dirs), []


def _cv_dir(root, src, subdirs=DEFAULT_SUBDIRS):
    """<root>/<src>/<subdir>, falling back to <root>/<src>_s0/<subdir>.

    `subdirs` is tried in order and the FIRST that exists wins, so a per-panel override
    (QM7_SUBDIRS) can prefer a re-evaluated copy without disturbing any other panel.

    The CBS e2e runner numbered its seeds <arm>_s0.._s2 while every other tree uses
    <arm>, <arm>_s1, <arm>_s2 -- so under cbs_benchmark the base name of an e2e arm does not
    exist and its seed-0 lives at <arm>_s0. Only fires when the plain name is absent, so it can
    never double-count a dir that IS present.
    """
    for sub in subdirs:
        d = FD / root / src / sub
        if d.exists():
            return d, src
        d0 = FD / root / f"{src}_s0" / sub
        if d0.exists():
            return d0, f"{src}_s0"
    return None, src


def mol_fold_values(dirs, dataset, metric, root="climb_v2_phase2", subdirs=DEFAULT_SUBDIRS):
    """Per-(pretraining-seed-dir, fold) ENSEMBLE values for one MolNet-shaped panel.

    Reads the plain `<metric>` fold rows (fold0..foldK-1) -- NEVER `<metric>_cell`. eval_v2.py's
    CV loop fits one head per (fold, head-seed), then AVERAGES THE 3 HEAD-SEEDS' PREDICTIONS
    before scoring the fold (`pred = np.mean(np.stack(seed_preds), axis=0)`); that ensembled score
    is what "main_metric" (the plain, non-_cell row) means, and it is the intended point estimate
    -- confirmed against the original protocol in notebook_cells/06.py (`arm_value`/`arm_err`,
    which read exactly these fold rows and are the authority this pipeline inherited) and
    notebook_cells/16.py (`d.main_metric.isin(["roc_auc","rmse"])`, same exclusion of `_cell`).
    `_cell` rows exist ONLY to expose a seed-decomposed view (eval_v2.py's own comment: "so a
    3 seeds x 5 folds error bar is ... computable"); averaging them directly is a DIFFERENT,
    strictly worse estimator (ensembling-before-scoring reduces variance) and must never be used
    as the value. Bug found + fixed 2026-08-16 (user caught it): averaging `_cell` values instead
    of ensemble rows had understated every BACE/Tox21 arm by 0.5-1% AUC, QM7 by ~1.5 RMSE, and CBS
    (small-count top-1% metric, most sensitive to pre- vs post-ensemble scoring) by up to 5-6% NEF1.

    Every arm was pretrained 3x (dirs <base>, <base>_s1, <base>_s2 for CLIMB arms; _00/_01/_02 for
    the two controls) EXCEPT the deterministic/external arms -- ECFP, ECFP+desc, CheMeleon -- which
    have exactly ONE dir because there is no pretraining stage to replicate (XGBoost on a fixed
    classical featurization; a frozen externally-supplied model). That is a fact about those three
    models, not a data gap: do not read "n_seeds=1" for them as missing compute.

    `root` selects the data tree ("climb_v2_phase2" for mainline MolNet, "cbs_benchmark" for CBS
    -- CBS is NOT a special case, it has the identical dir-per-pretraining-seed x 5-fold structure).

    Fallback: where no moleculenet_summary.csv exists, a `per_fold.csv` (columns
    fold,<metric>,... -- written by the e2e CBS runner) supplies the same per-fold values.
    Returns [(unit, value), ...] with unit = "<dir>:<fold>" -- ONE value per (dir, fold), 5 per dir.
    """
    out = []
    if len(subdirs) > 1:
        sub, dirs, skipped = _pick_subdir(root, list(dirs), subdirs)
        if skipped:
            why = "; ".join(f"{d}: {_skip_reason(root, d, sub)}" for d in skipped)
            print(f"  SUBDIR SKIP  {dataset}: using {sub}/ for {len(dirs)} dir(s); "
                  f"dropped {len(skipped)} ({why}) -- pooling them would MIX UNITS")
        subdirs = (sub,)
    for src in dirs:
        d, resolved = _cv_dir(root, src, subdirs)
        if d is None:
            continue
        f = d / "moleculenet_summary.csv"
        if f.exists():
            rows = list(csv.DictReader(open(f)))
            got = [(f"{resolved}:{r['head_seed']}", float(r["main_value"])) for r in rows
                   if r["dataset"] == dataset and r["main_metric"] == metric
                   and r["head_seed"] not in ("MEAN", "STD") and r["main_value"] not in ("", "nan")]
            out += got
            continue
        pf = d / "per_fold.csv"
        if pf.exists():
            for r in csv.DictReader(open(pf)):
                if metric in r and r[metric] not in ("", "nan"):
                    out.append((f"{resolved}:fold{r['fold']}", float(r[metric])))
    return out


# suite_summary.json key for a (dataset, metric) panel: `<DS>_MEAN` for the dataset's main metric,
# EXCEPT CBS -- our panel metric there is nef1, and `cbs_MEAN` is roc_auc. Lookup is
# case-insensitive (the e2e runner writes "BACE_MEAN" but "cbs_nef1_MEAN").
def _suite_key(dataset, metric):
    """suite_summary.json key for a (dataset, metric) panel.

    `<DS>_MEAN` holds the dataset's PRIMARY metric, so any panel scored on a non-primary metric
    needs the explicit `<DS>_<metric>_MEAN` key. This was hardcoded to CBS until 2026-08-19 and
    silently returned ROC-AUC when HIV took CBS's rare-active-screen slot: chemeleon_e2e, the one
    arm with no per-fold CSV, came out at 0.7967 (its ROC-AUC) against every other arm's NEF1% of
    0.43-0.71 -- a value that would have looked like a strong result rather than a wrong metric.
    Keyed on the METRIC now, so it cannot mis-fire for the next panel either.
    """
    return f"{dataset}_MEAN" if metric in ("roc_auc", "rmse") else f"{dataset}_{metric}_MEAN"


def mol_dir_summaries(dirs, dataset, metric, root="climb_v2_phase2", subdirs=DEFAULT_SUBDIRS):
    """[(dir, mean, fold_sd)] from suite_summary.json -- for e2e-style arms whose runner wrote no
    per-fold CSV (chemeleon_e2e on MolNet, s2u_dense on CBS). The json's STD is the POPULATION sd
    (ddof=0) over the 5 CV folds (verified 2026-08-17 against per_fold.csv: 0.12097 == ddof=0);
    callers needing a sample variance must rescale. Protocol tag in the json: scaffold-5fold-CV,
    same fold rule as eval_v2 (`eval_v2._scaffold_kfold_indices(seed=0)`).
    """
    import json
    out = []
    if len(subdirs) > 1:
        sub, dirs, skipped = _pick_subdir(root, list(dirs), subdirs)
        if skipped:
            why = "; ".join(f"{d}: {_skip_reason(root, d, sub)}" for d in skipped)
            print(f"  SUBDIR SKIP  {dataset}: using {sub}/ for {len(dirs)} dir(s); "
                  f"dropped {len(skipped)} ({why}) -- pooling them would MIX UNITS")
        subdirs = (sub,)
    for src in dirs:
        d, resolved = _cv_dir(root, src, subdirs)
        if d is None:
            continue
        f = d / "suite_summary.json"
        if not f.exists():
            continue
        j = json.load(open(f))
        key = _suite_key(dataset, metric)
        lut = {k.lower(): v for k, v in j.items()}
        mean = lut.get(key.lower())
        sd = lut.get(key.replace("_MEAN", "_STD").lower())   # replace BEFORE lowering
        if mean is None or sd is None:
            continue
        out.append((resolved, float(mean), float(sd)))
    return out


def panel_stats(cells=None, dir_summaries=None):
    """(value, sd_total, sd_seeds, n_seeds, n_cells) for one arm x panel -- ONE error-bar estimand
    for the whole suite (user decision 2026-08-17, notes/a2-errorbar-unification-2026-08-17.md):

        sd_total^2 = var_between(dir means) + mean(within-dir fold variance)

    i.e. the spread of ONE (pretraining-seed x fold) evaluation of the panel -- "how much does a
    single replicate run move". Pre-2026-08-17 the figure plotted sd_seeds for multi-dir arms but
    fold-SD for single-dir arms, which made CLIMB bars look ~20x tighter than the anchors' on
    Tox21/QM7 purely by definition (sup_dense Tox21: drawn 0.0023 vs its own fold-SD 0.0477).
    For single-dir arms (anchors, chemeleon_frozen -- no pretraining stage to replicate) the
    between term is undefined and sd_total reduces to the within-dir fold SD, as before.

    cells          [(unit, value)], unit "<dir>:<fold>" -- ensemble fold rows / per_fold.csv
    dir_summaries  [(dir, mean, fold_sd)] from suite_summary.json (ddof=0 fold sd, n=5 folds --
                   rescaled to ddof=1 here so both paths estimate the same quantity)
    value          grand mean (pooled cells; equal 5 folds per dir, so == mean of dir means)
    sd_seeds       SD across dir means only -- kept for transparency, no longer plotted
    """
    if cells:
        by_dir = collections.defaultdict(list)
        for unit, v in cells:
            by_dir[unit.split(":")[0]].append(v)
        dirs = sorted(by_dir)
        dir_means = [st.mean(by_dir[d]) for d in dirs]
        dir_vars = [st.variance(by_dir[d]) if len(by_dir[d]) > 1 else 0.0 for d in dirs]
        value = st.mean([v for _, v in cells])
        n_cells = len(cells)
    elif dir_summaries:
        dir_means = [m for _, m, _ in dir_summaries]
        dir_vars = [sd ** 2 * 5 / 4 for _, _, sd in dir_summaries]   # ddof=0 -> ddof=1 (n=5 folds)
        value = st.mean(dir_means)
        n_cells = len(dir_summaries)          # dirs, not folds -- the folds are not recoverable
    else:
        return None
    n_seeds = len(dir_means)
    sd_seeds = st.stdev(dir_means) if n_seeds > 1 else 0.0
    var_between = st.variance(dir_means) if n_seeds > 1 else 0.0
    sd_total = (var_between + st.mean(dir_vars)) ** 0.5
    return value, sd_total, sd_seeds, n_seeds, n_cells


# ---------------------------------------------------------------- Polaris -------------------
def polaris_cells(base, task, metric):
    """[(unit, value), ...] for one Polaris task, pooled over PRETRAINING-SEED dirs x EVAL seeds.

    Seed expansion is the same rule as mace_seed_dirs: <base> plus <base>_s1/_s2 where they exist.
    Until 2026-08-18 this read ONLY <base>, so 11 of the mainline arms had their Ames panel built
    from 1 of the 3 pretraining seeds sitting on disk -- and STATUS.md duly reported "Ames
    n_seeds=1" as a FACT about the data, when it was a reader gap. Caught by audit check 8: the
    a2 bootstrap expands the seed dirs, so its CI centre disagreed with the bar it was drawn
    around (unsup 0.7987 vs 0.8006, u2s_dense 0.8157 vs 0.8135).
    """
    # `base` may be an explicit LIST (see mace_seed_dirs); pool every dir that exists, tagging the
    # replicate so two dirs cannot collide on the same seed label.
    dirs = (list(base) if isinstance(base, (list, tuple))
            else [base, f"{base}_s1", f"{base}_s2"])
    out = []
    for d in dirs:
        f = FD / "chemeleon_suite" / "polaris" / d / "polaris_scores.csv"
        if not f.exists():
            continue
        out += [(f"{d}:seed{r['seed']}", float(r["value"])) for r in csv.DictReader(open(f))
                if r["task"] == task and r["metric"] == metric and r["value"] not in ("", "nan")]
    return out


# ---------------------------------------------------------------- main ----------------------
def check_panel_units(rows):
    """Fail loudly when one panel's arms are not all in the same UNIT.

    Added 2026-08-18 after QM7 shipped mixed: the frozen MolNet path evaluates QM7 with
    standardize=zscore and reports the NORMALIZED rmse (~0.85), while the e2e/CheMeleon runners
    report native kcal/mol (~199). The aggregator takes each arm's number as-is, so the panel
    inherited both conventions and CheMeleon's bar came out ~230x its neighbours'. Nothing in the
    pipeline noticed, because every arm was internally self-consistent — the defect only exists
    BETWEEN arms.

    Heuristic, deliberately crude: within a panel, if the largest value is more than 25x the
    smallest, these are not the same units. That is far outside any real model-quality spread and
    has no false positives on the current suite.
    """
    import collections
    by_panel = collections.defaultdict(list)
    for r in rows:
        try:
            v = abs(float(r["value"]))
        except (TypeError, ValueError):
            continue
        if v > 0:
            by_panel[r["panel"]].append((v, r["arm"]))
    bad = []
    for panel, vals in by_panel.items():
        if len(vals) < 2:
            continue
        lo, hi = min(vals), max(vals)
        if hi[0] / lo[0] > 25:
            bad.append((panel, lo, hi))
    for panel, lo, hi in bad:
        print(f"  UNIT WARNING  {panel}: values span {hi[0]/lo[0]:.0f}x "
              f"({lo[1]}={lo[0]:.4g} vs {hi[1]}={hi[0]:.4g}) — almost certainly a unit mismatch "
              f"between arms, NOT a quality difference. Do not plot this panel until it is fixed.")
    return bad


def check_cell_units(long_rows):
    """The same 25x test WITHIN one (arm, panel) -- across its replicate cells.

    check_panel_units() compares arms and therefore cannot see an arm whose OWN folds are in two
    units, which is exactly what a partially-completed re-eval produces. That is not hypothetical:
    it happened to `no pretrain, end2end` on QM7 (2 native dirs + 1 z-scored dir -> a mean of
    129.9). _pick_subdir now prevents it upstream; this is the belt-and-braces detector, because
    the next unit split will not necessarily arrive through a subdir.
    """
    import collections
    by = collections.defaultdict(list)
    for r in long_rows:
        try:
            v = abs(float(r["value"]))
        except (TypeError, ValueError):
            continue
        if v > 0:
            by[(r["arm"], r["panel"])].append((v, r["unit"]))
    bad = []
    for (arm, panel), vals in by.items():
        if len(vals) < 2:
            continue
        lo, hi = min(vals), max(vals)
        if hi[0] / lo[0] > 25:
            bad.append((arm, panel, lo, hi))
            print(f"  CELL UNIT WARNING  {arm}/{panel}: this ARM'S OWN cells span "
                  f"{hi[0]/lo[0]:.0f}x ({lo[1]}={lo[0]:.4g} vs {hi[1]}={hi[0]:.4g}). Its point "
                  f"estimate is an average of two units and is meaningless.")
    return bad


def main():
    rows, long_rows, boot_rows = [], [], []
    have = collections.defaultdict(dict)  # arm -> panel -> True/False

    for arm in ARM_ORDER:
        src = ARMS[arm]["src"]

        # --- MoleculeACE -----------------------------------------------------------------
        pt = mace_per_target(src["mace"]) if src.get("mace") else None
        if pt and pt.get("overall"):
            macro = {s: st.mean(pt[s].values()) for s in ("overall", "cliff", "noncliff") if pt.get(s)}
            lo, hi = cluster_bootstrap(pt["overall"])
            macros = mace_seed_macros(src["mace"])
            sd_total = st.stdev(macros) if len(macros) > 1 else 0.0
            rows.append(dict(arm=arm, panel="MoleculeACE", metric="macro_rmse",
                             value=round(macro["overall"], 4),
                             extra=f"cliff={round(macro.get('cliff', float('nan')), 4)};"
                                   f"noncliff={round(macro.get('noncliff', float('nan')), 4)};"
                                   f"sd_total={round(sd_total, 4)};n_seeds={len(mace_seed_dirs(src['mace']))};"
                                   f"n_cells={len(macros)}"))
            for subset in ("overall", "cliff", "noncliff"):
                for task, v in sorted(pt.get(subset, {}).items()):
                    long_rows.append(dict(arm=arm, panel="MoleculeACE", metric="rmse",
                                          unit=task, subset=subset, value=round(v, 6)))
            boot_rows.append(dict(arm=arm, panel="MoleculeACE", metric="macro_rmse_overall",
                                  value=round(macro["overall"], 4),
                                  ci_lo=round(lo, 4) if lo else "", ci_hi=round(hi, 4) if hi else "",
                                  n_targets=len(pt["overall"])))
            have[arm]["MoleculeACE"] = True
        else:
            have[arm]["MoleculeACE"] = False

        # --- CBS -------------------------------------------------------------------------
        # Sourced from figure_data/cbs_benchmark/<dir>/moleculenet_cv/, the SAME fold-ENSEMBLE
        # rows as BACE/Tox21/QM7 below -- CBS is not structurally different from those three
        # panels (verified 2026-08-16: every arm, including the anchors and CheMeleon, has the
        # full pretraining-dir x 5-fold structure on disk). Previously this read a separately
        # precomputed summary (experiment_cbs/cbs_nef1_summary.csv); do not reintroduce that file.
        # e2e-style arms fall back to per_fold.csv, then to suite_summary.json (panel_stats).
        cbs_folds = mol_fold_values(src["mol"], "cbs", "nef1", root="cbs_benchmark") if src.get("mol") else []
        cbs_dirs = mol_dir_summaries(src["mol"], "cbs", "nef1", root="cbs_benchmark") \
            if src.get("mol") and not cbs_folds else None
        stats = panel_stats(cells=cbs_folds or None, dir_summaries=cbs_dirs)
        if stats:
            value, sd_total, sd_seeds, n, n_cells = stats
            rows.append(dict(arm=arm, panel="CBS", metric="nef1", value=round(value, 4),
                             extra=f"sd_total={round(sd_total, 4)};sd_seeds={round(sd_seeds, 4)};"
                                   f"n_seeds={n};n_cells={n_cells}"))
            for unit, v in cbs_folds:
                long_rows.append(dict(arm=arm, panel="CBS", metric="nef1", unit=unit,
                                      subset="overall", value=round(v, 6)))
            for d, m, s in (cbs_dirs or []):
                long_rows.append(dict(arm=arm, panel="CBS", metric="nef1", unit=f"{d}:summary",
                                      subset="overall", value=round(m, 6)))
            have[arm]["CBS"] = True
        else:
            have[arm]["CBS"] = False

        # --- Polaris-sourced panels (hERG) -----------------------------------------------
        for panel, (task, metric) in POLARIS_PANELS.items():
            cells = polaris_cells(src["mace"], task, metric) if src.get("mace") else []
            if cells:
                vals = [v for _, v in cells]
                sd = st.stdev(vals) if len(vals) > 1 else 0.0
                # n_seeds is the number of PRETRAINING-seed dirs actually pooled -- it was hardcoded
                # to 1 until 2026-08-18, which made a reader gap look like a property of the data.
                n_dirs = len({u.split(":")[0] for u, _ in cells})
                rows.append(dict(arm=arm, panel=panel, metric=metric,
                                 value=round(st.mean(vals), 4),
                                 extra=f"sd_total={round(sd, 4)};sd_evalseeds={round(sd, 4)};"
                                       f"n_seeds={n_dirs};n_cells={len(vals)}"))
                for unit, v in cells:
                    long_rows.append(dict(arm=arm, panel=panel, metric=metric, unit=unit,
                                          subset="overall", value=round(v, 6)))
                have[arm][panel] = True
            else:
                have[arm][panel] = False

        # --- MoleculeNet panels ----------------------------------------------------------
        for ds, metric in MOL_PANELS.items():
            subs = (QM7_SUBDIRS if ds == "QM7" else
                    TOX21_SUBDIRS if ds == "Tox21" else DEFAULT_SUBDIRS)
            folds = mol_fold_values(src["mol"], ds, metric, subdirs=subs) if src.get("mol") else []
            dirs = mol_dir_summaries(src["mol"], ds, metric, subdirs=subs) \
                if src.get("mol") and not folds else None
            stats = panel_stats(cells=folds or None, dir_summaries=dirs)
            if stats:
                value, sd_total, sd_seeds, n, n_cells = stats
                rows.append(dict(arm=arm, panel=ds, metric=metric, value=round(value, 4),
                                 extra=f"sd_total={round(sd_total, 4)};sd_seeds={round(sd_seeds, 4)};"
                                       f"n_seeds={n};n_cells={n_cells}"))
                for unit, v in folds:
                    long_rows.append(dict(arm=arm, panel=ds, metric=metric, unit=unit,
                                          subset="overall", value=round(v, 6)))
                for d, m, s in (dirs or []):
                    long_rows.append(dict(arm=arm, panel=ds, metric=metric, unit=f"{d}:summary",
                                          subset="overall", value=round(m, 6)))
                have[arm][ds] = True
            else:
                have[arm][ds] = False

    OUT.mkdir(parents=True, exist_ok=True)
    _write(OUT / "mainline_8M.csv", ["arm", "panel", "metric", "value", "extra"], rows)
    _write(OUT / "mainline_8M_long.csv", ["arm", "panel", "metric", "unit", "subset", "value"], long_rows)
    _write(OUT / "mainline_8M_bootstrap.csv",
           ["arm", "panel", "metric", "value", "ci_lo", "ci_hi", "n_targets"], boot_rows)
    write_status(have)

    missing = [(a, p) for a in ARM_ORDER for p in PANEL_ORDER if not have[a][p]]
    print(f"wrote {OUT/'mainline_8M.csv'}            {len(rows):4d} rows")
    print(f"wrote {OUT/'mainline_8M_long.csv'}       {len(long_rows):4d} rows")
    print(f"wrote {OUT/'mainline_8M_bootstrap.csv'}  {len(boot_rows):4d} rows")
    print(f"wrote {OUT/'STATUS.md'}")
    check_panel_units(rows)
    check_cell_units(long_rows)
    print(f"\ncoverage: {len(ARM_ORDER)*len(PANEL_ORDER)-len(missing)}/{len(ARM_ORDER)*len(PANEL_ORDER)}"
          f" arm x panel cells filled; {len(missing)} missing")
    for a, p in missing:
        print(f"  MISSING  {ARMS[a]['label']:38s} {p}")


def _write(path, fields, rows):
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def write_status(have):
    """Human-readable coverage board -- the one page to check 'do we have the numbers yet?'."""
    waves = [
        ("Wave 1 · mainline re-aggregation (local, no compute)", FD / "six_panel" / "mainline_8M.csv",
         "arm x 6-panel table at the 8M budget"),
        ("Wave 2 · scaling frozen re-eval (GPU)", FD / "SIX_PANEL_W2_DONE",
         "61 scaling encoders x MoleculeACE (+CBS if data/cbs.csv staged) -> Fig B, SI-b"),
        ("Wave 3 · e2e crossover grid (GPU)", FD / "SIX_PANEL_W3_DONE",
         "best-two arms x {BACE,BBBP,Tox21,QM7} x 5 fractions x 3 seeds -> Fig F, SI-a"),
    ]
    L = ["# Six-panel results — status board",
         "",
         "Auto-generated by `scripts/six_panel_aggregate.py`. This is the one page to check before",
         "plotting: which (model × benchmark) numbers exist, and which waves are still running.",
         "",
         "## Panels",
         "",
         "| # | Panel | Task type | Metric | Direction |",
         "|---|---|---|---|---|"]
    for i, p in enumerate(PANEL_ORDER, 1):
        d = PANELS[p]
        L.append(f"| {i} | **{d['label']}** | {d['group']} | {d['metric_label']} | "
                 f"{'higher better' if d['higher_better'] else 'lower better'} |")

    L += ["", "## Coverage — 8M mainline (`mainline_8M.csv`)", "",
          "| Model | " + " | ".join(PANEL_ORDER) + " |",
          "|---|" + "---|" * len(PANEL_ORDER)]
    for a in ARM_ORDER:
        cells = " | ".join("✓" if have[a][p] else "—" for p in PANEL_ORDER)
        L.append(f"| {ARMS[a]['label']} | {cells} |")
    n_missing = sum(1 for a in ARM_ORDER for p in PANEL_ORDER if not have[a][p])
    L += ["", f"✓ = on disk, — = not run. Missing cells: **{n_missing}**"
              f" of {len(ARM_ORDER)*len(PANEL_ORDER)}.", ""]

    L += ["## What each panel averages over", "",
          "Replication is NOT uniform across the suite — check this before quoting an error bar.",
          "",
          "| Panel | Source tree | Pretraining seeds | Eval replicates | Pooled cells |",
          "|---|---|---|---|---|",
          "| MoleculeACE | `chemeleon_suite/moleculeace/` | 3 for CLIMB arms and both controls; "
          "1 for the anchors and CheMeleon | 3 eval seeds x 30 targets | 9 macro-means (3 for 1-seed arms) |",
          "| CBS | `cbs_benchmark/` | 3 (`<arm>`, `_s1`, `_s2`; controls `_00/_01/_02`) | "
          "3 head seeds ensembled, then 5 provided UMAP folds | 15 (5 for 1-seed arms) |",
          "| BACE / Tox21 / QM7 | `climb_v2_phase2/` | 3 | 3 head seeds ensembled, then 5 scaffold "
          "folds | 15 (5 for 1-seed arms) |",
          "| Ames | `chemeleon_suite/polaris/` | **1** (Polaris run once per arm) | 3 eval seeds, "
          "benchmark-provided split | 3 (9 where an arm has 3 dirs) |",
          "",
          "**1-seed arms are a FACT, not a gap.** ECFP, ECFP+desc and CheMeleon have no pretraining "
          "stage to replicate: XGBoost on a fixed classical featurization, and a frozen "
          "externally-supplied model. Do not read `n_seeds=1` for those three as missing compute.",
          "",
          "**Error bar — the plotted estimand is a SAMPLING CI of the evaluation units,** not a "
          "run-to-run SD. fig_A2/fig_A draw `a2_errorbars.csv`: a scaffold-cluster bootstrap for "
          "BACE/Tox21/QM7/CBS, a target-cluster bootstrap over the 30 targets for MoleculeACE, and "
          "an analytic Hanley-McNeil SE for Ames (Polaris withholds the test labels, so that one "
          "panel cannot be resampled and is flagged DERIVED). The `sd_total` column below is "
          "RETAINED for reference and for figures that legitimately want run-to-run spread "
          "(fig_B, SI figs), but it is NOT what the headline whiskers show: it answers "
          "\"how much does a rerun move\", where a reviewer is asking \"how much does the "
          "benchmark sample move\".",
          "",
          "**Two data paths.** Arms with per-fold files contribute real cells (15 = 3 dirs x 5",
          "folds). e2e-style arms whose runner wrote only `suite_summary.json` (chemeleon_e2e on",
          "MolNet) contribute per-dir (mean, fold-sd) summaries — the json's",
          "fold sd is ddof=0 and is rescaled to ddof=1 in `panel_stats`. The CBS e2e runner",
          "numbered its seeds `<arm>_s0.._s2`; `_cv_dir` maps the base name onto `<arm>_s0`.",
          "",
          "**QM7 is stored in two unit conventions** — z-scored rmse (~0.85) for most phase-2 runs, "
          "native kcal/mol (~200) for the rest — and the re-eval writes native values to "
          "`moleculenet_cv_qm7native/`, one run at a time. `_pick_subdir` therefore chooses ONE "
          "subdir for a whole arm and drops the dirs that lack it, printing a `SUBDIR SKIP` line. "
          "Never pool the two: resolving per-dir instead of per-arm once averaged 10 native folds "
          "with 5 z-scored ones and produced a QM7 mean of 129.9 for `no pretrain, end2end`, which "
          "reads as a spectacular result rather than an error. `check_cell_units` is the backstop.",
          "",
          "Some runs stored only per-fold rows rather than per-(head-seed, fold) cells; those "
          "contribute 5 values per seed dir instead of 15 (see `n_cells` in `mainline_8M.csv`). The "
          "`MEAN`/`STD` rows in the source summaries are never treated as data.",
          "",
                    "## Compute waves", "", "| Wave | Marker / output | Status | Feeds |", "|---|---|---|---|"]
    for name, marker, feeds in waves:
        state = "**done**" if marker.exists() else "*pending*"
        L.append(f"| {name} | `{marker.relative_to(ROOT)}` | {state} | {feeds} |")
    L += ["", "## Files", "",
          "| File | What |", "|---|---|",
          "| `mainline_8M.csv` | one row per (model, panel): point estimate + spread |",
          "| `mainline_8M_long.csv` | replicate level — per target (MoleculeACE), per seed×fold (MolNet) |",
          "| `mainline_8M_bootstrap.csv` | MoleculeACE macro-mean + 95% target-cluster-bootstrap CI |",
          "| `scaling_*.csv` | Wave 2 output (pending) |",
          "| `labeleff_fractions.csv` | Wave 3 output (pending) |",
          "",
          "Model names, colours and panel definitions live in `figures/arms.py` — the only place they",
          "are defined. Figure scripts live in `figures/fig_*.py` and write to `figures_v2/`.",
          ""]
    (OUT / "STATUS.md").write_text("\n".join(L))


if __name__ == "__main__":
    main()
