from __future__ import annotations
import csv, sys, statistics as st, collections, random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from figures.arms import ARMS, ARM_ORDER, PANELS, PANEL_ORDER

FD = ROOT / "figure_data"
OUT = FD / "six_panel"
random.seed(0)


MOL_PANELS = {"BACE": "roc_auc", "Tox21": "roc_auc", "QM7": "rmse", "HIV": "nef1"}


POLARIS_PANELS = {"Ames": ("tdcommons/ames", "roc_auc")}


def mace_seed_dirs(base):


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


QM7_SUBDIRS = ("moleculenet_cv_qm7clamped", "moleculenet_cv_qm7native", "moleculenet_cv")


TOX21_SUBDIRS = ("moleculenet_cv_tox21fixed", "moleculenet_cv")
DEFAULT_SUBDIRS = ("moleculenet_cv",)


def _usable_dir(root, d, sub):

    for cand in (d, f"{d}_s0"):
        dd = FD / root / cand / sub
        if dd.exists():
            return not (dd / "reference_scoring.json").exists()
    return False


def _skip_reason(root, d, sub):


    for cand in (d, f"{d}_s0"):
        dd = FD / root / cand / sub
        if dd.exists():
            return ("foreign re-eval (reference_scoring.json)"
                    if (dd / "reference_scoring.json").exists() else "unusable")
    return "not written yet"


def _pick_subdir(root, dirs, subdirs):

    for sub in subdirs:
        usable = [d for d in dirs if _usable_dir(root, d, sub)]
        if usable:
            return sub, usable, [d for d in dirs if d not in usable]
    return subdirs[-1], list(dirs), []


def _cv_dir(root, src, subdirs=DEFAULT_SUBDIRS):

    for sub in subdirs:
        d = FD / root / src / sub
        if d.exists():
            return d, src
        d0 = FD / root / f"{src}_s0" / sub
        if d0.exists():
            return d0, f"{src}_s0"
    return None, src


def mol_fold_values(dirs, dataset, metric, root="climb_v2_phase2", subdirs=DEFAULT_SUBDIRS):

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


def _suite_key(dataset, metric):

    return f"{dataset}_MEAN" if metric in ("roc_auc", "rmse") else f"{dataset}_{metric}_MEAN"


def mol_dir_summaries(dirs, dataset, metric, root="climb_v2_phase2", subdirs=DEFAULT_SUBDIRS):

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
        sd = lut.get(key.replace("_MEAN", "_STD").lower())
        if mean is None or sd is None:
            continue
        out.append((resolved, float(mean), float(sd)))
    return out


def panel_stats(cells=None, dir_summaries=None):

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
        dir_vars = [sd ** 2 * 5 / 4 for _, _, sd in dir_summaries]
        value = st.mean(dir_means)
        n_cells = len(dir_summaries)
    else:
        return None
    n_seeds = len(dir_means)
    sd_seeds = st.stdev(dir_means) if n_seeds > 1 else 0.0
    var_between = st.variance(dir_means) if n_seeds > 1 else 0.0
    sd_total = (var_between + st.mean(dir_vars)) ** 0.5
    return value, sd_total, sd_seeds, n_seeds, n_cells


def polaris_cells(base, task, metric):


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


def check_panel_units(rows):

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
    have = collections.defaultdict(dict)

    for arm in ARM_ORDER:
        src = ARMS[arm]["src"]


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


        for panel, (task, metric) in POLARIS_PANELS.items():
            cells = polaris_cells(src["mace"], task, metric) if src.get("mace") else []
            if cells:
                vals = [v for _, v in cells]
                sd = st.stdev(vals) if len(vals) > 1 else 0.0


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

    missing = [(a, p) for a in ARM_ORDER for p in PANEL_ORDER if not have[a][p]]
    print(f"wrote {OUT/'mainline_8M.csv'}            {len(rows):4d} rows")
    print(f"wrote {OUT/'mainline_8M_long.csv'}       {len(long_rows):4d} rows")
    print(f"wrote {OUT/'mainline_8M_bootstrap.csv'}  {len(boot_rows):4d} rows")
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




if __name__ == "__main__":
    main()
