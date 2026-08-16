"""Wave 1 of the six-panel migration (see notes/six-panel-migration.md).

Re-aggregates the 8M CLIMB-arm + anchor results ALREADY on disk into the canonical 6-panel suite:
  1 MoleculeACE  macro-mean RMSE over 30 targets (overall + cliff) + target-cluster bootstrap CI
  2 CBS          NEF1%
  3 BACE         ROC-AUC
  4 BBBP         ROC-AUC
  5 Tox21        mean ROC-AUC (12 subtasks, already aggregated in the summary)
  6 QM7          RMSE

No new compute — pure re-aggregation. Writes:
  figure_data/six_panel/mainline_8M.csv            (arm x panel point estimates)
  figure_data/six_panel/mainline_8M_bootstrap.csv  (MoleculeACE target-cluster bootstrap CI)
  figure_data/six_panel/README.md                  (schema/provenance)
Also prints a COVERAGE GAPS list (arm x panel cells with no source) -> the Wave-2 top-up list.

CheMeleon is intentionally excluded from the arm set (curiosity comparator only).
"""
from __future__ import annotations
import csv, os, statistics as st, collections, random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FD = ROOT / "figure_data"
random.seed(0)  # deterministic bootstrap

# canonical arm -> source dir/label per suite. None => not expected to exist for that arm.
# MoleculeACE dirs: chemeleon_suite/moleculeace/<name>/results.csv
# MolNet dirs:      climb_v2_phase2/<name>/moleculenet_cv/moleculenet_summary.csv
# CBS:              experiment_cbs/cbs_nef1_summary.csv  (arm column)
ARMS = {
    # anchors
    "ecfp4":                      dict(mace="ecfp4",                     mol="ecfp4_anchor",   cbs="ecfp4"),
    "fp_desc":                    dict(mace="fp_desc",                   mol="fp_desc_anchor", cbs="fp_desc"),
    "no_pretrain":                dict(mace="random_baseline_00",        mol="random_baseline_00", cbs="no_pretrain"),
    # CLIMB frozen arms
    "unsup_only":                 dict(mace="unsup_8M",                  mol="unsup_8M",                  cbs="unsup_only"),
    "sup_only:dense":             dict(mace="skip_dense_8M",             mol="skip_dense_8M",             cbs="sup_only:dense"),
    "sup_only:dense_plus_sparse": dict(mace="skip_dense_plus_sparse_8M", mol="skip_dense_plus_sparse_8M", cbs="sup_only:dense_plus_sparse"),
    "sup_only:sparse_all":        dict(mace="skip_sparse_all_8M",        mol="skip_sparse_all_8M",        cbs="sup_only:sparse_all"),
    "sup_only:mixed":             dict(mace="skip_mixed_8M",             mol="skip_mixed_8M",             cbs=None),
    "sup_only:minimol_full":      dict(mace="skip_minimol_full_8M",      mol="skip_minimol_full_8M",      cbs=None),
    "unsup2sup:dense":            dict(mace="u2s_dense_from8M",          mol="u2s_dense_from8M",          cbs="unsup2sup:dense"),
    "unsup2sup:dense_plus_sparse":dict(mace="u2s_dense_plus_sparse_from8M", mol="u2s_dense_plus_sparse_from8M", cbs="unsup2sup:dense_plus_sparse"),
    "unsup2sup:sparse_all":       dict(mace="u2s_sparse_all_from8M",     mol="u2s_sparse_all_from8M",     cbs="unsup2sup:sparse_all"),
    "unsup2sup:mixed":            dict(mace="u2s_mixed_from8M",          mol="u2s_mixed_from8M",          cbs=None),
    "unsup2sup:minimol_full":     dict(mace="u2s_minimol_full_from8M",   mol="u2s_minimol_full_from8M",   cbs=None),
}

MOL_PANELS = {"BACE": "roc_auc", "BBBP": "roc_auc", "Tox21": "roc_auc", "QM7": "rmse"}


def mace_per_target(src):
    """Return {task: {subset: mean_rmse_over_seeds}} for a MoleculeACE arm, or None."""
    f = FD / "chemeleon_suite" / "moleculeace" / src / "results.csv"
    if not f.exists():
        return None
    per = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in csv.DictReader(open(f)):
        if r["metric"] != "rmse":
            continue
        try:
            per[(r["subset"], r["task"])]["v"].append(float(r["value"]))
        except ValueError:
            pass
    out = collections.defaultdict(dict)  # subset -> task -> mean
    for (subset, task), d in per.items():
        out[subset][task] = st.mean(d["v"])
    return out


def mace_macro(src):
    """macro-mean RMSE (overall, cliff, noncliff) + per-target overall list for bootstrap."""
    pt = mace_per_target(src)
    if not pt:
        return None
    res = {}
    for subset in ("overall", "cliff", "noncliff"):
        if subset in pt and pt[subset]:
            res[subset] = st.mean(pt[subset].values())
    res["_overall_by_target"] = dict(pt.get("overall", {}))
    return res


def mol_metric(src, dataset, metric):
    """mean over s*_fold* cells of the given dataset/metric from a MolNet summary; or None."""
    f = FD / "climb_v2_phase2" / src / "moleculenet_cv" / "moleculenet_summary.csv"
    if not f.exists():
        return None
    allrows = list(csv.DictReader(open(f)))
    # prefer per-fold-seed cells; fall back to aggregate rows (some arms stored only those)
    for key in (f"{metric}_cell", metric):
        vals = [float(r["main_value"]) for r in allrows
                if r["dataset"] == dataset and r["main_metric"] == key and r["main_value"] not in ("", "nan")]
        if vals:
            return st.mean(vals)
    return None


def load_cbs():
    f = ROOT / "experiment_cbs" / "cbs_nef1_summary.csv"
    out = {}
    if f.exists():
        for r in csv.DictReader(open(f)):
            if r["metric"] == "nef1":
                out[r["arm"]] = float(r["mean"])
    return out


def cluster_bootstrap(by_target, n=2000):
    """95% CI on the macro-mean by resampling the targets (cluster bootstrap)."""
    keys = list(by_target)
    if len(keys) < 2:
        return (None, None)
    boots = []
    for _ in range(n):
        samp = [by_target[random.choice(keys)] for _ in keys]
        boots.append(st.mean(samp))
    boots.sort()
    lo = boots[int(0.025 * len(boots))]
    hi = boots[int(0.975 * len(boots))]
    return (lo, hi)


def main():
    cbs = load_cbs()
    rows, boot_rows, gaps = [], [], []
    for arm, srcs in ARMS.items():
        # MoleculeACE
        m = mace_macro(srcs["mace"]) if srcs["mace"] else None
        if m:
            lo, hi = cluster_bootstrap(m["_overall_by_target"])
            rows.append(dict(arm=arm, panel="MoleculeACE", metric="macro_rmse",
                             value=round(m.get("overall", float("nan")), 4),
                             extra=f"cliff={round(m.get('cliff', float('nan')),4)};noncliff={round(m.get('noncliff', float('nan')),4)}"))
            boot_rows.append(dict(arm=arm, panel="MoleculeACE", metric="macro_rmse_overall",
                                  value=round(m.get("overall", float("nan")), 4),
                                  ci_lo=round(lo, 4) if lo else "", ci_hi=round(hi, 4) if hi else "",
                                  n_targets=len(m["_overall_by_target"])))
        else:
            gaps.append((arm, "MoleculeACE"))
        # CBS
        cval = cbs.get(srcs["cbs"]) if srcs["cbs"] else None
        if cval is not None:
            rows.append(dict(arm=arm, panel="CBS", metric="nef1", value=round(cval, 4), extra=""))
        else:
            gaps.append((arm, "CBS"))
        # MolNet panels
        for ds, metric in MOL_PANELS.items():
            v = mol_metric(srcs["mol"], ds, metric) if srcs["mol"] else None
            if v is not None:
                rows.append(dict(arm=arm, panel=ds, metric=metric, value=round(v, 4), extra=""))
            else:
                gaps.append((arm, ds))

    out = FD / "six_panel"
    out.mkdir(exist_ok=True)
    with open(out / "mainline_8M.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["arm", "panel", "metric", "value", "extra"])
        w.writeheader()
        w.writerows(rows)
    with open(out / "mainline_8M_bootstrap.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["arm", "panel", "metric", "value", "ci_lo", "ci_hi", "n_targets"])
        w.writeheader()
        w.writerows(boot_rows)

    print(f"wrote {out/'mainline_8M.csv'}  ({len(rows)} rows)")
    print(f"wrote {out/'mainline_8M_bootstrap.csv'}  ({len(boot_rows)} rows)")
    print(f"\nCOVERAGE GAPS (arm x panel cells with no source) -> Wave-2 top-up list: {len(gaps)}")
    for arm, panel in gaps:
        print(f"  MISSING  {arm:30s} {panel}")


if __name__ == "__main__":
    main()
