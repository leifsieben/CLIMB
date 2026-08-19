"""Cross-figure consistency audit — the check a reviewer would run.

Every figure in this paper can be individually correct and the SET still be indefensible: two
panels in different units, two figures reporting different error-bar estimands, one script reading
a superseded data wave. Those defects are invisible from inside any single script, because each one
is internally self-consistent. This audit looks across the set.

Seven checks, each of which has caught a real defect in this repo:

  1 SUPERSEDED ROOTS   a script reading a data wave that has been replaced. Cost us two wrong
                       findings: SI d was reported as 2/6 panels because it read climb_v2 (the
                       round-1 wave, single hold-out) instead of climb_v2_h1, and CBS was reported
                       missing because it read the deprecated experiment_cbs summary instead of
                       cbs_benchmark/.
  2 UNITS WITHIN PANEL arms in one panel not on one scale. QM7 shipped with 15 arms normalized
                       (~0.85) and 3 native (~199) — every arm internally consistent, the panel
                       wrong, nothing complained.
  3 REPLICATION        arms in one panel resting on different numbers of pretraining seeds. Four
                       CBS arms sat at 1 seed beside neighbours at 3; when the replicates landed,
                       sup_minimol moved 0.760 -> 0.693 and its SD doubled.
  4 ESTIMAND           what each figure DRAWS as an error bar, in one table. The set previously
                       mixed sd_total, seed SD, fold SD and a bootstrap CI without saying so.
  5 PANEL SET          which figures are on the canonical six and which are not, with the reason.
  6 PAGE GEOMETRY      authored width vs the A4 text block, so nothing is silently rescaled by
                       LaTeX into a different on-page font size.
  7 COMPARATOR SCOPE   CheMeleon is an external comparator, not one of our arms. User decision
                       2026-08-18: it appears in the headline figure ONLY (fig_A, i.e. its A1/A2
                       components); every other figure must stand on CLIMB arms alone. Without
                       this check the rule erodes one plausible-looking addition at a time.

Exit code is 0 always: this reports, it does not gate. Read it before shipping.

Run:  python3 scripts/audit_figure_consistency.py
"""
from __future__ import annotations

import sys
import re
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
FIGDIR = ROOT / "figures"
FD = ROOT / "figure_data"
OUT = ROOT / "figures_v2"

CANONICAL = ["MoleculeACE", "CBS", "BACE", "Ames", "Tox21", "QM7"]

# (bad token, what to use instead, why it matters)
SUPERSEDED = [
    ("figure_data/climb_v2/", "figure_data/climb_v2_h1/",
     "climb_v2 is the ROUND-1 wave: it never saved encoders and only has a single hold-out"),
    ("climb_v2_ablation/", "climb_v2_ablation_dedup/",
     "the pre-dedup ablation wave has no encoders and carries eval-test leakage"),
    ("experiment_cbs/cbs_nef1_summary.csv", "figure_data/cbs_benchmark/<arm>/moleculenet_cv/",
     "the precomputed CBS summary is deprecated; its ARMS list silently omits whole waves"),
    ("moleculenet_cv/suite_summary.json\", \"QM7", "moleculenet_cv_qm7native/",
     "QM7 in moleculenet_cv/ is the stale z-scored artifact predating the eval_v2 scaler fix"),
]

# what each figure is DECLARED to draw; check 4 verifies the script actually references it
ESTIMAND = {
    "fig_A":     ("sampling CI of the evaluation units (bootstrap; Ames analytic)", "a2_errorbars"),
    "fig_A1":    ("+-1 SE of the mean rank, design-effect corrected", "se_rank"),
    "fig_A2":    ("sampling CI of the evaluation units (bootstrap; Ames analytic)", "a2_errorbars"),
    "fig_B":     ("none drawn (sd_total available in scaling_ladders.csv)", None),
    "fig_C1":    ("95% bootstrap CI over molecules", "boot"),
    "fig_C2":    ("none drawn (scatter of per-cell lifts)", None),
    "fig_D":     ("none drawn (matrix of per-cell lifts)", None),
    "fig_E":     ("+-1 SD across 3 pretraining seeds, propagated through the lift", "lift_sd_pct"),
    "fig_F":     ("+-1 SD across the seeds of that (task, feature set) cell", "std"),
    "SI_fig_a":  ("+-1 SD of the panel's replicate unit", "sd"),
    "SI_fig_b":  ("+-1 SD of the panel's replicate unit (5 CV folds / 3 eval seeds)", "sd"),
    "SI_fig_d":  ("+-1 SD across 3 pretraining seeds", "sd"),
    "SI_fig_e":  ("none drawn (absolute performance vs labelled size)", None),
}


def hdr(n, t):
    print(f"\n{'='*94}\n{n}. {t}\n{'='*94}")


def check_superseded():
    hdr(1, "SUPERSEDED DATA ROOTS")
    bad = 0
    for p in sorted(list(FIGDIR.glob("*.py")) + list((ROOT / "scripts").glob("build_*.py"))):
        txt = p.read_text()
        code = "\n".join(l for l in txt.split("\n") if not l.strip().startswith("#"))
        # ignore the module docstring, which legitimately DISCUSSES the superseded roots
        code = re.sub(r'""".*?"""', "", code, flags=re.S)
        for token, better, why in SUPERSEDED:
            # A line may opt out with a trailing `# AUDIT-OK: superseded-root <reason>`. The one
            # legitimate case is a script that reads the OLD wave as its INPUT in order to rebuild
            # the new one (build_ablation_dedup_manifests.py reads climb_v2_ablation/manifest.json
            # to re-run it deduped). Opting out is per LINE, so it cannot silently cover a second,
            # unintended read added later in the same file.
            hits = [l for l in code.split("\n")
                    if token in l and "AUDIT-OK: superseded-root" not in l]
            if hits:
                print(f"  FAIL  {p.name}: reads {token}\n        use {better}\n        ({why})")
                bad += 1
    print("  OK — no figure or builder reads a superseded root" if not bad
          else f"  {bad} occurrence(s)")
    return bad


def check_units():
    hdr(2, "UNIT CONSISTENCY WITHIN EACH PANEL")
    bad = 0
    for f in sorted(FD.rglob("*.csv")):
        try:
            d = pd.read_csv(f)
        except Exception:
            continue
        if "panel" not in d.columns:
            continue
        col = next((c for c in ("value", "mean") if c in d.columns), None)
        if col is None:
            continue
        keys = ["panel"] + [c for c in ("dataset", "metric") if c in d.columns]
        for panel, g in d.groupby(keys):
            v = pd.to_numeric(g[col], errors="coerce").abs()
            v = v[np.isfinite(v) & (v > 0)]
            if len(v) < 2:
                continue
            if v.max() / v.min() > 25:
                lo = g.loc[v.idxmin()]
                hi = g.loc[v.idxmax()]
                who = "arm" if "arm" in d.columns else ("label" if "label" in d.columns else None)
                names = (f"{lo[who]}={v.min():.4g} vs {hi[who]}={v.max():.4g}") if who else \
                        f"{v.min():.4g} vs {v.max():.4g}"
                tag = panel if isinstance(panel, str) else "/".join(map(str, panel))
                print(f"  FAIL  {f.relative_to(ROOT)} [{tag}]: spans {v.max()/v.min():.0f}x "
                      f"({names})")
                bad += 1
    print("  OK — every panel is on one scale" if not bad else f"  {bad} panel(s) mixing units")
    return bad


def check_replication():
    hdr(3, "REPLICATION WITHIN EACH PANEL")
    f = FD / "six_panel" / "mainline_8M.csv"
    if not f.exists():
        print("  SKIP — mainline_8M.csv absent")
        return 0
    d = pd.read_csv(f)
    d["n_seeds"] = d.extra.map(lambda x: int(m.group(1)) if (m := re.search(r"n_seeds=(\d+)", str(x))) else np.nan)
    d["n_cells"] = d.extra.map(lambda x: int(m.group(1)) if (m := re.search(r"n_cells=(\d+)", str(x))) else np.nan)
    bad = 0
    # NO-PRETRAINING arms: deterministic featurizers and the fixed external model. n_seeds counts
    # PRETRAINING-seed directories, of which they have exactly one BY CONSTRUCTION -- there is no
    # pretraining stage to replicate. That is not a data gap and must not be read as one. What they
    # DO replicate is the head / fine-tuning seed, and that lives INSIDE the single directory: on
    # MoleculeACE both CheMeleon variants carry 3 distinct fine-tuning runs (e2e 0.6503/0.6547/
    # 0.6526, sd 0.0022; frozen 0.8377/0.8212/0.8180, sd 0.0106), which a directory counter cannot
    # see. So they are exempted from the seed-count comparison and instead REQUIRED to carry >= 3
    # replicate cells -- the honest version of the same demand.
    NO_PRETRAIN_STAGE = ["ecfp", "ecfp_desc", "chemeleon_frozen", "chemeleon_e2e"]
    thin = d[d.arm.isin(NO_PRETRAIN_STAGE) & (d.n_cells < 3)]
    for _, r in thin.iterrows():
        print(f"  THIN  {r.arm} / {r.panel}: only {int(r.n_cells)} replicate cell(s); these arms have "
              f"no pretraining stage, so the head/fine-tuning seed is the only replicate they can have")
        bad += 1
    print(f"  {'panel':<13}{'n_seeds seen':<18}verdict")
    for panel, g in d.groupby("panel"):
        clm = g[~g.arm.isin(NO_PRETRAIN_STAGE)]
        seen = sorted({int(x) for x in clm.n_seeds.dropna()})
        ok = len(seen) <= 1
        if not ok:
            odd = clm[clm.n_seeds == min(seen)].arm.tolist()
            print(f"  {panel:<13}{str(seen):<18}MIXED — {', '.join(odd[:5])} at {min(seen)}")
            bad += 1
        else:
            print(f"  {panel:<13}{str(seen):<18}ok")
    print("  OK — every panel's CLIMB arms share a pretraining-seed count, and every "
          "no-pretraining arm carries >=3 replicate cells" if not bad else
          f"  {bad} item(s) mixing replication")
    return bad


def check_estimand():
    hdr(4, "ERROR-BAR ESTIMAND PER FIGURE")
    bad = 0
    print(f"  {'figure':<12}{'declared estimand':<62}code")
    for name, (desc, token) in ESTIMAND.items():
        p = FIGDIR / f"{name}.py"
        if not p.exists():
            print(f"  {name:<12}{desc:<62}NO SCRIPT")
            bad += 1
            continue
        txt = p.read_text()
        # assembled figures (fig_A, fig_C_D) draw nothing themselves — they compose components,
        # so look through the modules they import rather than calling it a mismatch
        for imported in re.findall(r"import figures\.(\w+)", txt):
            q = FIGDIR / f"{imported}.py"
            if q.exists():
                txt += q.read_text()
        state = "-" if token is None else ("ok" if token in txt else "TOKEN MISSING")
        if state == "TOKEN MISSING":
            bad += 1
        print(f"  {name:<12}{desc:<62}{state}")
    print("  OK — each figure's drawn quantity is declared and present in its code" if not bad
          else f"  {bad} mismatch(es)")
    return bad


def check_panelset():
    hdr(5, "PANEL SET")
    bad = 0
    for p in sorted(FIGDIR.glob("*.py")):
        txt = p.read_text()
        if "OFF-SUITE" in txt:
            why = re.search(r"blocked on data, not\s+on code: (.{0,90})", txt, re.S)
            print(f"  OFF-SUITE  {p.stem}: {' '.join(why.group(1).split()) if why else 'see banner'}...")
            bad += 1
    print("  OK — every figure is on the canonical six" if not bad
          else f"  {bad} figure(s) still off-suite (each carries a banner)")
    return bad


def check_geometry():
    hdr(6, "PAGE GEOMETRY")
    from figures.style import A4_TEXT, _pdf_width_in
    wide = {"fig_A", "fig_C_D"}
    bad = 0
    for pdf in sorted(OUT.glob("*.pdf")):
        w = _pdf_width_in(pdf)
        if w is None:
            continue
        if pdf.stem in wide:
            print(f"  {pdf.stem:<12}{w:6.2f}in   landscape by design (exempt)")
        elif abs(w - A4_TEXT) / A4_TEXT > 0.05:
            print(f"  FAIL  {pdf.stem:<12}{w:6.2f}in vs {A4_TEXT:.2f}in text block "
                  f"({(w/A4_TEXT-1)*100:+.0f}%)")
            bad += 1
    print("  OK — every non-exempt figure is within 5% of the A4 text block" if not bad
          else f"  {bad} figure(s) off-width")
    return bad


def check_comparator_scope():
    hdr(7, "COMPARATOR SCOPE (CheMeleon in the headline figure only)")
    allowed = {"fig_A", "fig_A1", "fig_A2"}
    bad = 0
    for p in sorted(list(FIGDIR.glob("fig_*.py")) + list(FIGDIR.glob("SI_fig_*.py"))):
        if p.stem in allowed:
            continue
        txt = p.read_text()
        code = re.sub(r'\"\"\".*?\"\"\"', '', txt, flags=re.S)          # strip module docstring
        code = "\n".join(l for l in code.split("\n") if not l.strip().startswith("#"))
        # Flag USE of the comparator, not mere presence of the string. `chemeleon_suite/` is the
        # name of the MoleculeACE/Polaris BENCHMARK tree — every canonical panel reads it — and
        # `figure_data/chemeleon_suite/...` in a path says nothing about whether the CheMeleon ARM
        # is plotted. Likewise the retained-but-unplotted comparator files
        # (concat_panels_chemeleon.csv, featurization_timing_chemeleon.json) exist on disk by
        # design and must not fail this check. So: strip the tree name first, then look.
        code = re.sub(r"chemeleon_suite", "BENCHTREE", code, flags=re.I)
        if re.search(r"chemeleon", code, re.I):
            hits = [l.strip() for l in code.split("\n") if re.search(r"chemeleon", l, re.I)]
            print(f"  FAIL  {p.stem} references CheMeleon outside the headline figure:")
            for h in hits[:3]:
                print(f"          {h[:86]}")
            bad += 1
    print("  OK — CheMeleon appears only in fig_A and its components" if not bad
          else f"  {bad} figure(s) leaking the comparator")
    return bad


def check_bar_vs_ci():
    """A bar and its error bar must be the SAME number, computed two ways.

    fig_A's bars come from moleculenet_summary.csv (the eval runner's own per-fold scores) while its
    whiskers come from a bootstrap that recomputes the metric from that run's per-molecule
    test_predictions.csv. Those are two independent paths to one quantity, so a mismatch means one
    of the two artefacts in a single run directory is a different vintage from the other.

    Added 2026-08-18, when it immediately found one: every Tox21 arm's CI centre sat 2-4% above its
    bar (e.g. random_encoder bar 0.7519 vs centre 0.7701), which is the documented +0.015...0.020
    signature of the 2026-08-05 missing-label fix — i.e. summary and predictions disagree about
    whether the fix has been applied. Nothing else in this audit could see it: each artefact is
    internally consistent, on one scale, with the right replication.
    """
    print(f"\n{'='*94}\n8. BAR vs ERROR-BAR CENTRE (two independent paths to one number)\n{'='*94}")
    import csv as _csv
    bars, cis = ROOT / "figure_data/six_panel/mainline_8M.csv", ROOT / "figure_data/six_panel/a2_errorbars.csv"
    if not (bars.exists() and cis.exists()):
        print("  SKIP - one of the two tables is missing")
        return 0
    bar = {(r["arm"], r["panel"]): r["value"] for r in _csv.DictReader(bars.open())}
    bad = []
    for r in _csv.DictReader(cis.open()):
        v, b = r["value"], bar.get((r["arm"], r["panel"]))
        if not v or not b:
            continue
        v, b = float(v), float(b)
        if abs(v - b) / max(abs(b), 1e-9) > 0.002:
            bad.append((r["arm"], r["panel"], b, v, 100 * abs(v - b) / abs(b)))
    for arm, panel, b, v, pct in bad:
        print(f"  FAIL  {arm:16s} {panel:12s} bar={b:.4f} vs CI centre={v:.4f}  ({pct:.2f}% apart)")
    if not bad:
        print("  OK - every drawn bar equals the centre of its own error bar")
    else:
        print(f"  {len(bad)} bar(s) whose whisker is centred somewhere else - the two artefacts in "
              f"that run dir are different vintages; do NOT ship the affected panel")
    return len(bad)


def check_qm7_convention():
    """Every QM7 value a figure draws must be in ONE convention, tested by VALUE not by path.

    QM7 is stored z-scored (~0.85) or native kcal/mol (~200), and on 2026-08-18 the directory name
    stopped being a reliable indicator: corrupt_mtr_8M_s1/_s2 landed with native values written
    into moleculenet_cv/ next to a z-scored seed 0, and the `standardize` column said "zscore" for
    all three, so BOTH the path and the metadata lied. Only the value separates them.

    This matters beyond that one arm. Sweeping the waves by content: climb_v2_h1 (30 runs) and
    climb_v2_vocab (8) are 100% NATIVE inside moleculenet_cv/, while climb_v2_phase2 is 80
    z-scored against 13 native. figures/allsuites.py spans all three roots, so the all-suites
    ranking table could mix conventions in a single column -- and a rank column is computed ACROSS
    arms, so one arm in the wrong unit corrupts the whole column, not just its own cell. It happens
    to be clean today only because every phase-2 arm has a moleculenet_cv_qm7native/ for the
    resolver to prefer. That is luck, not design, so it is checked here.
    """
    print(f"\n{'='*94}\n9. QM7 UNIT CONVENTION (checked by VALUE -- the path and metadata both lie)\n{'='*94}")
    import csv as _csv
    bad = 0
    # (label, rows) where each row yields a QM7 value
    sources = []
    m = ROOT / "figure_data/six_panel/mainline_8M.csv"
    if m.exists():
        sources.append(("mainline_8M.csv", [(r["arm"], float(r["value"]))
                        for r in _csv.DictReader(m.open())
                        if r["panel"] == "QM7" and r["value"] not in ("", "nan")]))
    sc = ROOT / "figure_data/six_panel/scaling_ladders.csv"
    if sc.exists():
        sources.append(("scaling_ladders.csv", [(r["rung"], float(r["value"]))
                        for r in _csv.DictReader(sc.open())
                        if r["panel"] == "QM7" and r["value"] not in ("", "nan")]))
    try:
        import figures.allsuites as A
        from figures.arms import ARM_ORDER
        S, _ = A.wide_table(ARM_ORDER)
        col = [c for c in S.columns if "QM7" in str(c)]
        if col:
            v = S[col[0]].dropna()
            sources.append(("allsuites ranking table", list(zip(v.index, v.to_numpy(float)))))
    except Exception as e:
        print(f"  (allsuites table unavailable: {e})")
    for label, rows in sources:
        if not rows:
            continue
        zs = [k for k, v in rows if v < 10]
        nat = [k for k, v in rows if v > 50]
        gap = [k for k, v in rows if 10 <= v <= 50]
        if gap:
            print(f"  FAIL  {label}: {len(gap)} value(s) between 10 and 50 -- neither convention: "
                  f"{', '.join(map(str, gap[:4]))}")
            bad += 1
        elif zs and nat:
            print(f"  FAIL  {label}: MIXED -- {len(zs)} z-scored ({', '.join(map(str, zs[:3]))}) "
                  f"vs {len(nat)} native")
            bad += 1
        else:
            conv = "native kcal/mol" if nat else "z-scored"
            print(f"  OK    {label}: all {len(rows)} value(s) {conv}")
    if not bad:
        print("  OK — every QM7 value a figure draws is in one convention")
    return bad


def main():
    print("CROSS-FIGURE CONSISTENCY AUDIT")
    total = sum([check_superseded(), check_units(), check_replication(),
                 check_estimand(), check_panelset(), check_geometry(),
                 check_comparator_scope(), check_bar_vs_ci(),
                 check_qm7_convention()])
    print(f"\n{'='*94}\n{'CLEAN' if not total else str(total) + ' ITEM(S) NEED ATTENTION'}\n{'='*94}")


if __name__ == "__main__":
    main()
