"""Audit what the figure layer actually reads: provenance, seed/fold completeness, consistency.

Checks, per arm x panel:
  (a) CURRENT  — does the source exist, and is it stale relative to the newest data on disk?
  (b) COMPLETE — 3 pretraining seeds; each seed 3 head seeds x 5 folds (=15 cells, 45 pooled)
  (c) CONSISTENT — label/probe vs the directory actually read; metric sanity vs a constant predictor

Read-only. Run: python3 scripts/audit_six_panel_sources.py
"""
from __future__ import annotations
import csv, json, sys, statistics as st, collections
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from figures.arms import ARMS, ARM_ORDER  # noqa: E402

FD = ROOT / "figure_data"
MOLNET_ROOTS = ["climb_v2_phase2", "climb_v2_h1", "climb_v2_vocab"]
MOL_PANELS = {"BACE": "roc_auc", "Tox21": "roc_auc", "QM7": "rmse"}
# "predict the training mean" reference: RMSE ~= sigma of the target
CONST_PREDICTOR = {"QM7": 228.7}
issues = []


def note(sev, arm, panel, msg):
    issues.append((sev, arm, panel, msg))


def molnet_dir(prefix):
    for r in MOLNET_ROOTS:
        p = FD / r / prefix / "moleculenet_cv" / "moleculenet_summary.csv"
        if p.exists():
            return p
    return None


def audit_molnet(arm, spec):
    dirs = spec["src"].get("mol") or []
    found, cells = [], collections.defaultdict(list)
    for d in dirs:
        p = molnet_dir(d)
        if p is None:
            note("MISSING", arm, "molnet", f"seed dir '{d}' has no moleculenet_summary.csv in any root")
            continue
        found.append(d)
        for r in csv.DictReader(open(p)):
            ds, mm, hs = r["dataset"], r["main_metric"], r["head_seed"]
            if hs in ("MEAN", "STD") or r["main_value"] in ("", "nan"):
                continue
            for panel, metric in MOL_PANELS.items():
                if ds == panel and mm == f"{metric}_cell":
                    cells[(panel, d)].append((hs, float(r["main_value"])))
    if dirs and len(found) < 3 and not arm.startswith(("ecfp", "chemeleon")):
        note("SEEDS", arm, "molnet", f"only {len(found)}/3 pretraining-seed dirs present: {found}")
    for panel in MOL_PANELS:
        per_seed = {d: cells.get((panel, d), []) for d in found}
        tot = sum(len(v) for v in per_seed.values())
        if tot == 0:
            note("MISSING", arm, panel, "no _cell rows in any seed dir")
            continue
        for d, v in per_seed.items():
            folds = {h.split("_fold")[-1] for h, _ in v if "_fold" in h}
            if len(v) != 15 or len(folds) != 5:
                note("FOLDS", arm, panel, f"{d}: {len(v)} cells / {len(folds)} distinct folds (want 15 / 5)")
        vals = [x for v in per_seed.values() for _, x in v]
        mean = st.mean(vals)
        if panel in CONST_PREDICTOR and mean > CONST_PREDICTOR[panel]:
            note("SANITY", arm, panel,
                 f"mean {mean:.1f} WORSE than constant predictor ({CONST_PREDICTOR[panel]:.1f})")
        # per-fold divergence within the panel
        byfold = collections.defaultdict(list)
        for v in per_seed.values():
            for h, x in v:
                if "_fold" in h:
                    byfold[h.split("_fold")[-1]].append(x)
        if len(byfold) >= 2:
            fm = {f: st.mean(x) for f, x in byfold.items()}
            lo, hi = min(fm.values()), max(fm.values())
            if lo > 0 and hi / lo > 1.5:
                worst = max(fm, key=fm.get)
                note("DIVERGE", arm, panel,
                     f"fold{worst} = {hi:.1f} vs best {lo:.1f} ({hi/lo:.1f}x spread) — " +
                     ", ".join(f"f{f}={v:.0f}" for f, v in sorted(fm.items())))


def audit_mace(arm, spec):
    src = spec["src"].get("mace")
    if not src:
        return
    p = FD / "chemeleon_suite" / "moleculeace" / src / "results.csv"
    if not p.exists():
        note("MISSING", arm, "MoleculeACE", f"'{src}/results.csv' absent")
        return
    rows = [r for r in csv.DictReader(open(p)) if r["metric"] == "rmse" and r["subset"] == "overall"]
    tasks = {r["task"] for r in rows}
    seeds = {r["seed"] for r in rows}
    if len(tasks) != 30:
        note("TASKS", arm, "MoleculeACE", f"{len(tasks)}/30 targets")
    if len(seeds) != 3:
        note("SEEDS", arm, "MoleculeACE", f"{len(seeds)} eval seeds (want 3): {sorted(seeds)}")
    # pretraining-seed replicates (the top-up)
    reps = [d for d in (FD / "chemeleon_suite" / "moleculeace").glob(f"{src}_s*") if d.is_dir()]
    if reps:
        note("INFO", arm, "MoleculeACE", f"{len(reps)} pretraining-seed replicate dir(s) exist but src reads only '{src}'")


def audit_cbs(arm, spec):
    src = spec["src"].get("cbs")
    if not src:
        return
    f = ROOT / "experiment_cbs" / "cbs_nef1_summary.csv"
    if not f.exists():
        note("MISSING", arm, "CBS", "cbs_nef1_summary.csv absent")
        return
    hit = [r for r in csv.DictReader(open(f)) if r["arm"] == src and r["metric"] == "nef1"]
    if not hit:
        note("MISSING", arm, "CBS", f"arm '{src}' not in cbs_nef1_summary.csv")
        return
    n = int(hit[0]["n_seeds"])
    if n < 3 and not arm.startswith(("ecfp", "chemeleon")):
        note("SEEDS", arm, "CBS", f"n_seeds={n} (want 3)")


def main():
    print("=" * 100)
    print("AUDIT: what the figure layer reads (figures/arms.py -> figure_data)")
    print("=" * 100)
    for arm in ARM_ORDER:
        spec = ARMS[arm]
        # (c) label/probe vs actual source consistency
        srcs = " ".join(str(v) for v in spec["src"].values())
        if spec.get("probe") == "frozen" and "e2e" in spec.get("label", "").lower():
            note("LABEL", arm, "-", f"label='{spec['label']}' says end2end but probe='frozen' and src={spec['src']}")
        if "e2e" in srcs and spec.get("probe") == "frozen" and "frozen" not in srcs:
            note("LABEL", arm, "-", f"probe='frozen' but sources look like e2e: {spec['src']}")
        audit_molnet(arm, spec)
        audit_mace(arm, spec)
        audit_cbs(arm, spec)

    order = {"LABEL": 0, "SANITY": 1, "DIVERGE": 2, "MISSING": 3, "SEEDS": 4, "FOLDS": 5, "TASKS": 6, "INFO": 7}
    issues.sort(key=lambda x: (order.get(x[0], 9), x[1]))
    cur = None
    for sev, arm, panel, msg in issues:
        if sev != cur:
            print(f"\n----- {sev} -----"); cur = sev
        print(f"  {arm:18s} {panel:12s} {msg}")
    print(f"\n{len(issues)} findings across {len(ARM_ORDER)} arms")


if __name__ == "__main__":
    main()
