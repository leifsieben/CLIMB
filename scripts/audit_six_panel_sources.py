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
    """Delegate to the FIGURE's resolver instead of reimplementing it.

    This required moleculenet_summary.csv. figures.allsuites.molnet_dir accepts EITHER that or
    suite_summary.json, because e2e-style runners (chemeleon_e2e) write only the JSON -- its
    docstring records that requiring the CSV made those arms read as "not run" on all 7 MolNet
    datasets, and fig_A1 admits arms by coverage COUNT, so a loader gap silently decided which
    arms appear in the ranking. This audit was still reproducing exactly that fixed bug and
    reporting 6 phantom MISSING rows for an arm the figures read at full depth.

    An audit that reimplements the loader can only ever check its own copy of the rule. Delegate,
    and return the summary CSV path when there is one so the row-level checks below still work.
    """
    from figures.allsuites import molnet_dir as _fig_molnet_dir
    d = _fig_molnet_dir(prefix)
    if d is None:
        return None
    csv_path = d / "moleculenet_summary.csv"
    return csv_path if csv_path.exists() else None


def molnet_dir_exists(prefix):
    """True when the FIGURE can read this dir at all, CSV or JSON."""
    from figures.allsuites import molnet_dir as _fig_molnet_dir
    return _fig_molnet_dir(prefix) is not None


def audit_molnet(arm, spec):
    dirs = spec["src"].get("mol") or []
    found, cells = [], collections.defaultdict(list)
    for d in dirs:
        p = molnet_dir(d)
        if p is None:
            if molnet_dir_exists(d):
                # Readable by the figure via suite_summary.json; only the per-fold CSV that the
                # row-level checks below need is absent. Not a data gap -- state the limit rather
                # than reporting the arm as missing, which is what this audit did until 2026-08-21.
                note("INFO", arm, "molnet",
                     f"seed dir '{d}' has suite_summary.json but no per-fold CSV "
                     f"-- figure reads it; fold/seed checks below cannot")
            else:
                note("MISSING", arm, "molnet",
                     f"seed dir '{d}' is unreadable by the figure resolver in any root")
            continue
        found.append(d)
        # READ EACH PANEL FROM THE SUBDIR THE FIGURE READS IT FROM, not from moleculenet_cv for
        # everything. Tox21 resolves through moleculenet_cv_tox21fixed and QM7 through
        # qm7clamped/qm7native; scanning only moleculenet_cv reported "0 cells / 0 distinct folds"
        # for dirs that carry the data in the corrected subdir -- 10 phantom FOLDS findings
        # against ecfp, ecfp_desc and chemeleon_frozen, all of which the figures read at depth 3.
        from figures.allsuites import NATIVE_SUBDIRS, molnet_dir as _fig_dir
        for panel, metric in MOL_PANELS.items():
            for sub in NATIVE_SUBDIRS.get(panel, ("moleculenet_cv",)):
                fd = _fig_dir(d, sub)
                if fd is None:
                    continue
                f = fd / "moleculenet_summary.csv"
                if not f.exists():
                    continue
                hit = False
                for r in csv.DictReader(f.open()):
                    if r["dataset"] != panel or r["main_metric"] != f"{metric}_cell":
                        continue
                    hs = r["head_seed"]
                    if hs in ("MEAN", "STD") or r["main_value"] in ("", "nan"):
                        continue
                    cells[(panel, d)].append((hs, float(r["main_value"])))
                    hit = True
                if hit:
                    break
    if dirs and len(found) < 3 and not arm.startswith(("ecfp", "chemeleon")):
        note("SEEDS", arm, "molnet", f"only {len(found)}/3 pretraining-seed dirs present: {found}")
    for panel in MOL_PANELS:
        per_seed = {d: cells.get((panel, d), []) for d in found}
        tot = sum(len(v) for v in per_seed.values())
        if tot == 0:
            # NOT a data gap. `_cell` rows are the per-head-seed un-ensembled metrics, written
            # only so a seed-decomposed spread CAN be computed; allsuites._molnet takes the
            # ENSEMBLE fold row as the point estimate and explicitly refuses `_cell` as a
            # substitute. No published interval comes from them either -- those are the cluster
            # bootstrap in a2_errorbars.csv. So their absence limits THIS audit's fold check and
            # nothing a figure draws.
            note("INFO", arm, panel, "no _cell rows -- per-seed spread not checkable here; "
                                     "point estimate and interval are unaffected")
            continue
        for d, v in per_seed.items():
            folds = {h.split("_fold")[-1] for h, _ in v if "_fold" in h}
            if len(v) != 15 or len(folds) != 5:
                note("FOLDS", arm, panel, f"{d}: {len(v)} cells / {len(folds)} distinct folds (want 15 / 5)")
        vals = [x for v in per_seed.values() for _, x in v]
        mean = st.mean(vals)
        if panel in CONST_PREDICTOR:
            # Compare the PUBLISHED value, not the `_cell` mean. They are different quantities:
            # `_cell` rows are per-head-seed and un-ensembled, live only in moleculenet_cv, and
            # allsuites._molnet documents that averaging them understates performance -- while the
            # figure takes the ENSEMBLE fold row from the corrected subdir (qm7clamped for QM7).
            # Judging the published bar by the `_cell` mean called chemeleon_frozen QM7 276.0
            # "worse than a constant predictor" when the number actually drawn is 211.5, better
            # than the 228.7 baseline.
            from figures.allsuites import _molnet as _fig_molnet
            pub = _fig_molnet(found).get(panel)
            if pub is not None and pub > CONST_PREDICTOR[panel]:
                note("SANITY", arm, panel,
                     f"published {pub:.1f} WORSE than constant predictor "
                     f"({CONST_PREDICTOR[panel]:.1f})")
            elif pub is not None and mean > CONST_PREDICTOR[panel]:
                note("INFO", arm, panel,
                     f"_cell mean {mean:.1f} exceeds the constant predictor "
                     f"({CONST_PREDICTOR[panel]:.1f}) but the PUBLISHED value is {pub:.1f} -- "
                     f"un-ensembled cells in the uncorrected subdir, not what the figure draws")
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
    """Validate EVERY MoleculeACE dir the figure actually pools, not just the base.

    This crashed with `PosixPath / list` for the whole life of the list-valued `mace` (s2u_dense
    names _s0/_s1/_s2 rather than base/_s1/_s2), which meant the entire audit aborted on the first
    such arm and reported nothing about any arm after it. An audit that cannot run is not a
    weaker audit, it is no audit -- and this one carried the note that would have flagged unread
    replicate dirs. Resolve through _seed_dirs so the audit follows the figure rather than
    reimplementing its rule.
    """
    src = spec["src"].get("mace")
    if not src:
        return
    from figures.allsuites import _seed_dirs
    dirs = _seed_dirs(src)
    present = 0
    for d in dirs:
        p = FD / "chemeleon_suite" / "moleculeace" / d / "results.csv"
        if not p.exists():
            continue
        present += 1
        rows = [r for r in csv.DictReader(open(p))
                if r["metric"] == "rmse" and r["subset"] == "overall"]
        tasks = {r["task"] for r in rows}
        seeds = {r["seed"] for r in rows}
        if len(tasks) != 30:
            note("TASKS", arm, "MoleculeACE", f"{d}: {len(tasks)}/30 targets")
        if len(seeds) != 3:
            note("SEEDS", arm, "MoleculeACE",
                 f"{d}: {len(seeds)} eval seeds (want 3): {sorted(seeds)}")
    if not present:
        note("MISSING", arm, "MoleculeACE", f"none of {dirs} has results.csv")
    elif present < 3:
        note("SEEDS", arm, "MoleculeACE",
             f"{present}/3 pretraining-seed dir(s) present: {dirs}")


def audit_cbs(arm, spec):
    # cbs_legacy_label, NOT a path: this audit checks the DEPRECATED
    # experiment_cbs/cbs_nef1_summary.csv, which no figure reads. The CBS panel the
    # figures draw comes from allsuites._cbs_value via the arm's `mol` dir names.
    src = spec["src"].get("cbs_legacy_label")
    if not src:
        return
    f = ROOT / "experiment_cbs" / "cbs_nef1_summary.csv"
    if not f.exists():
        note("MISSING", arm, "CBS", "cbs_nef1_summary.csv absent")
        return
    hit = [r for r in csv.DictReader(open(f)) if r["arm"] == src and r["metric"] == "nef1"]
    if not hit:
        # The file being checked is DEPRECATED and read by no figure -- this function's own
        # docstring says so, and arms.py records that its `arm` list silently omits whole waves.
        # The CBS panel the figures draw comes from allsuites._cbs_value via the arm's `mol`
        # dirs. Reporting absence from a deprecated artefact as MISSING put 4 permanent false
        # positives in front of every real finding.
        note("INFO", arm, "CBS", f"arm '{src}' absent from the DEPRECATED cbs_nef1_summary.csv "
                                 f"-- no figure reads it; CBS comes from the mol dirs")
        return
    n = int(hit[0]["n_seeds"])
    if n < 3 and not arm.startswith(("ecfp", "chemeleon")):
        # Counted from the DEPRECATED cbs_nef1_summary.csv, which no figure reads -- the CBS
        # panel comes from allsuites._cbs_value over the arm's `mol` dirs, where audit check 19
        # confirms depth 3. Reported as INFO so a real CBS gap would not be buried beside it.
        note("INFO", arm, "CBS", f"deprecated cbs_nef1_summary.csv shows n_seeds={n}; the CBS "
                                 f"panel the figures draw comes from the mol dirs")


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
