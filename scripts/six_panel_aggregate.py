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

MOL_PANELS = {"BACE": "roc_auc", "Tox21": "roc_auc", "QM7": "rmse"}      # from MoleculeNet CV

# hERG replaced BBBP on 2026-08-16 and comes from Polaris (benchmark-provided split), not from our
# scaffold CV -- so it is read from polaris_scores.csv, one value per eval seed.
POLARIS_PANELS = {"hERG": ("tdcommons/herg", "roc_auc")}


# ---------------------------------------------------------------- MoleculeACE ----------------
def mace_seed_dirs(base):
    """<base> plus its pretraining-seed replicates <base>_s1/_s2, whichever exist on disk.

    The MoleculeACE seed top-up (box i-092c1d745d9a9f04e, 2026-08-16) writes those two dirs for
    every mainline arm; until they land this silently returns just the base dir, and the moment
    they appear they are pooled in with no code change.
    """
    cands = [base, f"{base}_s1", f"{base}_s2"]
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
def _cv_dir(root, src):
    """<root>/<src>/moleculenet_cv, falling back to <root>/<src>_s0/moleculenet_cv.

    The CBS e2e runner numbered its seeds <arm>_s0.._s2 while every other tree uses
    <arm>, <arm>_s1, <arm>_s2 -- so under cbs_benchmark the base name of an e2e arm does not
    exist and its seed-0 lives at <arm>_s0. Only fires when the plain name is absent, so it can
    never double-count a dir that IS present.
    """
    d = FD / root / src / "moleculenet_cv"
    if d.exists():
        return d, src
    d0 = FD / root / f"{src}_s0" / "moleculenet_cv"
    if d0.exists():
        return d0, f"{src}_s0"
    return None, src


def mol_fold_values(dirs, dataset, metric, root="climb_v2_phase2"):
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
    for src in dirs:
        d, resolved = _cv_dir(root, src)
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
    return f"{dataset}_nef1_MEAN" if (dataset, metric) == ("cbs", "nef1") else f"{dataset}_MEAN"


def mol_dir_summaries(dirs, dataset, metric, root="climb_v2_phase2"):
    """[(dir, mean, fold_sd)] from suite_summary.json -- for e2e-style arms whose runner wrote no
    per-fold CSV (chemeleon_e2e on MolNet, s2u_dense on CBS). The json's STD is the POPULATION sd
    (ddof=0) over the 5 CV folds (verified 2026-08-17 against per_fold.csv: 0.12097 == ddof=0);
    callers needing a sample variance must rescale. Protocol tag in the json: scaffold-5fold-CV,
    same fold rule as eval_v2 (`eval_v2._scaffold_kfold_indices(seed=0)`).
    """
    import json
    out = []
    for src in dirs:
        d, resolved = _cv_dir(root, src)
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
    """[(seed, value), ...] for one Polaris task. Polaris was run once per arm (no pretraining-seed
    replicates yet), so the replicates here are the 3 EVAL seeds of that single encoder."""
    f = FD / "chemeleon_suite" / "polaris" / base / "polaris_scores.csv"
    if not f.exists():
        return []
    return [(f"seed{r['seed']}", float(r["value"])) for r in csv.DictReader(open(f))
            if r["task"] == task and r["metric"] == metric and r["value"] not in ("", "nan")]


# ---------------------------------------------------------------- main ----------------------
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
                rows.append(dict(arm=arm, panel=panel, metric=metric,
                                 value=round(st.mean(vals), 4),
                                 extra=f"sd_total={round(sd, 4)};sd_evalseeds={round(sd, 4)};"
                                       f"n_seeds=1;n_cells={len(vals)}"))
                for unit, v in cells:
                    long_rows.append(dict(arm=arm, panel=panel, metric=metric, unit=unit,
                                          subset="overall", value=round(v, 6)))
                have[arm][panel] = True
            else:
                have[arm][panel] = False

        # --- MoleculeNet panels ----------------------------------------------------------
        for ds, metric in MOL_PANELS.items():
            folds = mol_fold_values(src["mol"], ds, metric) if src.get("mol") else []
            dirs = mol_dir_summaries(src["mol"], ds, metric) \
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
          "| Panel | Pretraining seeds | Eval replicates | Pooled n |",
          "|---|---|---|---|",
          "| MoleculeACE | **1** (seed-0 encoder only) | 3 eval seeds × 30 targets | 90 |",
          "| CBS / BACE / Tox21 / QM7 | 3 (`<arm>`, `_s1`, `_s2`; controls `_00/_01/_02`; "
          "anchors + CheMeleon-frozen have 1, no pretraining seed to replicate) | "
          "3 head seeds × 5 folds (CBS: the 5 provided UMAP folds) | 45 (15 for the anchors/CheMeleon) |",
          "| hERG | **1** (Polaris run once per arm) | 3 eval seeds, benchmark-provided split | 3 |",
          "",
          "**Error bar — ONE estimand for every panel (user decision 2026-08-17):** `sd_total`,",
          "the total single-run SD: `sd_total² = var_between(seed-dir means) + mean(within-dir",
          "fold variance)` — the spread of one (pretraining-seed × fold) evaluation of the panel.",
          "Single-dir arms (anchors, chemeleon_frozen) reduce to the within-dir fold SD. MoleculeACE",
          "uses the SD across per-(dir, eval-seed) macro-means; hERG the SD across its 3 eval seeds.",
          "The pre-2026-08-17 mix (seed-SD for CLIMB, fold-SD for anchors, bootstrap CI for",
          "MoleculeACE) made whisker lengths incomparable across bars and is retired.",
          "",
          "**Two data paths.** Arms with per-fold files contribute real cells (15 = 3 dirs × 5",
          "folds). e2e-style arms whose runner wrote only `suite_summary.json` (chemeleon_e2e on",
          "MolNet, s2u_dense on CBS) contribute per-dir (mean, fold-sd) summaries — the json's",
          "fold sd is ddof=0 and is rescaled to ddof=1 in `panel_stats`. The CBS e2e runner",
          "numbered its seeds `<arm>_s0.._s2`; `_cv_dir` maps the base name onto `<arm>_s0`.",
          "",
          "**Known asymmetry:** MoleculeACE and hERG still rest on a single pretraining",
          "seed for the mainline arms — the `_s1`/`_s2` encoders exist on S3 but were never scored on",
          "MoleculeACE. A frozen-probe top-up (Wave-2 machinery, ~14 arms × 30 targets) would close it.",
          "",
          "**s2u_dense (forgetting arm):** its MolNet landed in `moleculenet/` (single hold-out),",
          "NOT `moleculenet_cv/` — deliberately NOT pooled here (a 5-fold CV re-run is in flight).",
          "Its CBS (suite_summary path) and MoleculeACE are on the correct protocol.",
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
