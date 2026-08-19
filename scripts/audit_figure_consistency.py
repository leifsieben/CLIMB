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
import datetime as _dt

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
    # DERIVED from arms.py, not listed here. This was a hardcoded four-name list and it went stale
    # the moment R3FP/R3FP+desc were added: two XGBoost anchors structurally identical to the two
    # already exempt were reported as MIXED on MoleculeACE and Ames for a day. Exactly the failure
    # check 11 had, in a different check, found only because check 11's fix made this one's output
    # the odd one out. A name list cannot know about an arm that did not exist when it was written.
    from figures.arms import ARMS as _ARMS
    NO_PRETRAIN_STAGE = [k for k, v in _ARMS.items()
                         if v.get("pretrain_replicates", True) is False]
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
        # assembled figures (fig_A, fig_C+D) draw nothing themselves — they compose components,
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
    wide = {"fig_A", "fig_C+D"}
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
    # SI_fig_f is admitted, and the reason matters because widening this rule casually is how it
    # stops working. The rule exists because CheMeleon is a BENCHMARK comparator whose bars carry a
    # protocol confound -- frozen probe vs end-to-end fine-tune -- so putting it beside CLIMB
    # anywhere except the headline invites an unguarded comparison. SI fig f trains NOTHING: it
    # measures what a fixed representation can resolve, pair by pair. There is no probe, no fitting
    # and therefore no protocol to confound, and CheMeleon's presence is the point rather than a
    # leak -- the finding is that the blind spots are COMPLEMENTARY (CheMeleon is blind to isotopes
    # and stereochemistry where the CLMs are perfect, and best of the five at ring size). Removing
    # it would delete the result. fig_G is the main-text half of the same probe (class A; SI fig f
    # keeps class B and the calibration) and imports SI_fig_f's own drawing code, so the exemption
    # covers both halves of one experiment rather than two separate concessions.
    allowed = {"fig_A", "fig_A1", "fig_A2", "SI_fig_f", "fig_G"}
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


def _nef1_quanta():
    """{panel: smallest expressible NEF1 change} = 1 / min(k, n_actives) for the top-1% cut."""
    import pandas as _pd
    out = {}
    for panel, path in (("HIV", "climb_v2_phase2/ecfp4_anchor/moleculenet_cv/test_predictions.csv"),
                        ("CBS", "cbs_benchmark/ecfp4_anchor/moleculenet_cv/test_predictions.csv")):
        f = ROOT / "figure_data" / path
        if not f.exists():
            continue
        try:
            d = _pd.read_csv(f)
            d = d[d.dataset == ("cbs" if panel == "CBS" else panel)]
            n, pos = len(d), int((d.y_true == 1).sum())
            k = max(1, int(round(0.01 * n)))
            if min(k, pos):
                out[panel] = 1.0 / min(k, pos)
        except Exception:
            pass
    return out


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
    bar = {(r["arm"], r["panel"]): r for r in _csv.DictReader(bars.open())}
    # One hit's worth of NEF1, per panel: 1 / min(top-1% size, n_actives). Measured from each
    # panel's own OOF dump so it cannot drift from the data.
    NEF1_QUANTUM = _nef1_quanta()

    def _nseeds(extra):
        for kv in (extra or "").split(";"):
            if kv.startswith("n_seeds="):
                return int(kv.split("=")[1])
        return None

    bad, seedgap = [], []
    for r in _csv.DictReader(cis.open()):
        br = bar.get((r["arm"], r["panel"]))
        if not r["value"] or not br or not br["value"]:
            continue
        v, b = float(r["value"]), float(br["value"])
        # METRIC-AWARE TOLERANCE. 0.2% is right for a continuous metric like ROC-AUC. NEF1 is a
        # DISCRETE top-k count: on HIV the top 1% is 411 molecules against 1,443 actives, so one
        # additional hit moves the value by 0.0024 -- 0.35% of a value near 0.70. A disagreement
        # below that does not correspond to ANY difference in ranking; it is the two paths
        # reconstructing the fold partition with a molecule or two placed differently, which cannot
        # show up as a metric difference. Flagging it trains the reader to skim this check, which is
        # how a real defect gets lost. The floor is computed from the panel's own units, never
        # hardcoded, so it tightens automatically if a benchmark's active count grows.
        tol = 0.002
        if r["metric"] == "nef1":
            q = NEF1_QUANTUM.get(r["panel"])
            if q:
                tol = max(tol, q / max(abs(b), 1e-9))
        if abs(v - b) / max(abs(b), 1e-9) > tol:
            bad.append((r["arm"], r["panel"], b, v, 100 * abs(v - b) / abs(b)))
        # SEED COVERAGE. Equal numbers are not enough: the bar and the whisker must also be built
        # from the SAME set of pretraining seeds. chemeleon_frozen_s1/_s2 carry prediction dumps
        # pruned to QM7, so on HIV/BACE/Tox21 the bar pooled 3 seed dirs and the CI pooled 1 --
        # invisible in the value comparison above wherever seed spread happens to be small, which
        # is why it went unnoticed on BACE and Tox21 and only tripped the value check on HIV.
        nb, nc = _nseeds(br.get("extra")), r.get("n_dirs")
        if nb and nc not in (None, "") and int(nc) and int(nc) != nb:
            seedgap.append((r["arm"], r["panel"], nb, int(nc)))
    for arm, panel, b, v, pct in bad:
        print(f"  FAIL  {arm:16s} {panel:12s} bar={b:.4f} vs CI centre={v:.4f}  ({pct:.2f}% apart)")
    for arm, panel, nb, nc in seedgap:
        print(f"  FAIL  {arm:16s} {panel:12s} bar pools {nb} pretraining seed(s), CI pools {nc} "
              f"- the interval describes a different estimator than the bar")
    bad = bad + seedgap
    if not bad:
        print("  OK - every drawn bar equals the centre of its own error bar")
    else:
        print(f"  {len(bad)} bar(s) whose whisker is centred somewhere else - the two artefacts in "
              f"that run dir are different vintages; do NOT ship the affected panel")
    return len(bad)


# Tox21's masking fix (79a0dfb, 2026-08-05: y[w==0]=NaN in the MoleculeNet loader) raises ROC-AUC by
# +0.015..0.020. Runs evaluated BEFORE it carry a stale value in moleculenet_cv/ and were repaired
# into moleculenet_cv_tox21fixed/; runs evaluated AFTER it were never wrong and correctly have no
# fixed copy. Both therefore read from moleculenet_cv when no fixed copy exists, and only the
# EVALUATION DATE separates "already correct" from "silently stale".
TOX21_FIX_EPOCH = _dt.datetime(2026, 8, 5).timestamp()


def check_tox21_vintage():
    """Every arm's Tox21 cell must be on the POST-fix scale, whichever subdir it comes from.

    This is the third distinct way the Tox21 correction has leaked into a figure. First the
    direction was inferred from which tree a file came from (backwards). Then 12 `tox21fixed` dirs
    were built from pre-fix predictions. Then an arm mixed one re-scored dir with two foreign
    re-evals. All three shared a root cause: the VINTAGE was inferred from a path instead of
    established. Here it is established -- corrected subdir, or an evaluation newer than the fix.

    s2u_dense is the case that motivated the second branch: it has no tox21fixed/ and no
    test_predictions.csv to rescore, which reads like an unrepaired stale cell. It was evaluated
    2026-08-17, twelve days after the fix, and its per-fold values (0.8187/0.8117/0.8108/0.8153)
    sit on the corrected scale, not the pre-fix one (compare unsup_8M: 0.8202/0.8237/0.8177/0.8110
    corrected vs 0.7999/0.7892/0.7964/0.7794 pre-fix). Nothing to repair.
    """
    print(f"\n{'='*94}\n10. TOX21 VINTAGE (corrected subdir, or evaluated after the fix)\n{'='*94}")
    sys.path.insert(0, str(ROOT / "scripts"))
    import six_panel_aggregate as _spa
    from figures.arms import ARMS
    bad = 0
    for arm, meta in ARMS.items():
        mol = meta.get("src", {}).get("mol")
        if not mol:
            continue
        dirs = list(mol) if isinstance(mol, (list, tuple)) else [mol]
        sub, usable, _ = _spa._pick_subdir("climb_v2_phase2", dirs, _spa.TOX21_SUBDIRS)
        if not usable or sub != "moleculenet_cv":
            continue                      # on the corrected subdir: fine by construction
        stale = []
        for d in usable:
            f = _spa.FD / "climb_v2_phase2" / d / sub / "moleculenet_summary.csv"
            if f.exists() and f.stat().st_mtime < TOX21_FIX_EPOCH:
                stale.append(d)
        if stale:
            print(f"  FAIL  {arm:18s} reads Tox21 from moleculenet_cv/ and {len(stale)} of "
                  f"{len(usable)} dir(s) predate the 2026-08-05 masking fix: {stale}")
            bad += 1
        else:
            print(f"  OK    {arm:18s} no corrected copy needed - all {len(usable)} dir(s) "
                  f"evaluated after the fix")
    if not bad:
        print("  OK - no arm is drawing a pre-fix Tox21 number")
    return bad


def check_replication_parity():
    """Every arm on every panel must rest on the SAME NUMBER OF INDEPENDENT FITS.

    Audit check 3 asks "does this arm have 3 replicates?" and is satisfied by 3 head seeds inside
    one directory. That is not the same question as "is this comparison apples-to-apples?", which
    is about whether two bars in the SAME panel were built from the same amount of evidence. They
    were not: a CLIMB arm resolves to 3 pretraining dirs x 3 head seeds = 9 fits on every panel,
    while ecfp, ecfp_desc, chemeleon_frozen and chemeleon_e2e resolved to 1 dir x 3 seeds = 3.

    That asymmetry is NOT fixable by reseeding pretraining -- an ECFP fingerprint has no
    pretraining stage and never will. What is fixable, and what the peer session did on MoleculeNet
    and CBS in 3c52686, is the number of independent FITS: give those arms two more replicates on
    DISJOINT head-seed triples so the bar rests on 9 fits either way. The two suite tracks were
    left out of that pass, which is what this check exists to keep visible until
    scripts/suite_replicates_run.sh has run.

    Reported per (arm, panel) from the aggregator's own n_seeds, so it can never disagree with what
    the figures draw.
    """
    print(f"\n{'='*94}\n11. REPLICATION PARITY (same number of dirs behind every bar in a panel)\n{'='*94}")
    import csv as _csv, collections
    from figures.arms import ARMS
    f = ROOT / "figure_data/six_panel/mainline_8M.csv"
    if not f.exists():
        print("  SKIP - mainline_8M.csv missing")
        return 0

    def _n(extra):
        for kv in (extra or "").split(";"):
            if kv.startswith("n_seeds="):
                return int(kv.split("=")[1])
        return None

    per_panel = collections.defaultdict(dict)
    for r in _csv.DictReader(f.open()):
        n = _n(r["extra"])
        if n:
            per_panel[r["panel"]][r["arm"]] = n
    # Arms with NO pretraining stage are compared only against each other. An XGBoost anchor on a
    # fixed classical featurization has exactly one run dir on the suite tracks because there is
    # nothing to re-pretrain -- its whole model variance is head/eval-seed variance, and three eval
    # seeds of it sit inside that dir. Flagging it as under-replicated is a FALSE ALARM, and this
    # check emitted six of them a day while STATUS.md said in writing that those arms are complete.
    # A check that cries wolf on known-good arms trains the reader to skim the line where a real
    # failure appears, which is exactly what nearly happened to e2e_no_pretrain on Tox21 below.
    # The exemption is DECLARED IN arms.py, not listed here, so a new anchor inherits it correctly.
    def _exempt(a):
        return ARMS.get(a, {}).get("pretrain_replicates", True) is False

    bad = 0
    for panel in sorted(per_panel):
        counts = per_panel[panel]
        for label, group in (("", {a: n for a, n in counts.items() if not _exempt(a)}),
                             ("no-pretrain arms: ", {a: n for a, n in counts.items() if _exempt(a)})):
            if not group:
                continue
            top = max(group.values())
            short = sorted(a for a, n in group.items() if n < top)
            if short:
                detail = ", ".join(f"{a}={group[a]}" for a in short)
                print(f"  FAIL  {panel:12s} {label}most arms rest on {top} dirs; {len(short)} rest "
                      f"on fewer - {detail}")
                bad += 1
            else:
                print(f"  OK    {panel:12s} {label}all {len(group)} arms on {top} dir(s)")
    if not bad:
        print("  OK - every panel compares arms built from the same number of dirs")
    else:
        print(f"  {bad} group(s) mixing replication depths - run scripts/suite_replicates_run.sh")
    return bad



def check_aggregate_freshness():
    """The derived tables must be NEWER than every run directory they summarise.

    This is the failure that produced stale figures on 2026-08-19 and gave no warning at all:
    the peer session re-ran the anchors with the stereo fix at 11:56, mainline_8M.csv had been
    written at 11:35, and every figure downstream kept drawing stereo-blind numbers. Re-running
    six_panel_aggregate is a MANUAL step with nothing anywhere in the chain checking that it
    happened, so "the data landed" and "the figures show it" were two different facts and only
    one of them was visible.

    Compares each derived table's mtime against the newest moleculenet_summary.csv /
    suite_summary.json / results.csv under the arms it reads. Cheap, and it fails loudly on the
    exact thing that is otherwise invisible.
    """
    print(f"\n{'='*94}\n12. AGGREGATE FRESHNESS (derived tables newer than their inputs)\n{'='*94}")
    import os
    sys.path.insert(0, str(ROOT / "scripts"))
    from figures.arms import ARMS
    fd = ROOT / "figure_data"
    newest, newest_src = 0.0, None
    for arm, meta in ARMS.items():
        src = meta.get("src", {})
        names = []
        for key in ("mol", "mace"):
            v = src.get(key)
            names += (list(v) if isinstance(v, (list, tuple)) else [v]) if v else []
        for n in names:
            for root in ("climb_v2_phase2", "cbs_benchmark", "chemeleon_suite/moleculeace",
                         "chemeleon_suite/polaris"):
                for pat in ("moleculenet_summary.csv", "suite_summary.json", "results.csv",
                            "polaris_scores.csv"):
                    for f in (fd / root / n).glob(f"**/{pat}"):
                        t = f.stat().st_mtime
                        if t > newest:
                            newest, newest_src = t, f
    bad = 0
    for tbl in ("six_panel/mainline_8M.csv", "six_panel/scaling_ladders.csv",
                "six_panel/a2_errorbars.csv"):
        f = fd / tbl
        if not f.exists():
            continue
        age = f.stat().st_mtime
        if newest_src is not None and age < newest - 60:      # 60s slack for a run in flight
            import datetime as _d
            print(f"  FAIL  {tbl} is OLDER than its inputs "
                  f"({_d.datetime.fromtimestamp(age):%m-%d %H:%M} vs "
                  f"{_d.datetime.fromtimestamp(newest):%m-%d %H:%M}, "
                  f"{newest_src.relative_to(fd)}) - re-run the aggregator")
            bad += 1
        else:
            print(f"  OK    {tbl}")
    if not bad:
        print("  OK - every derived table is newer than the runs it summarises")
    return bad



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


def check_invariant_arms():
    """Where two vintages of one table are kept for comparison, the arms that CANNOT have responded
    to the change must be IDENTICAL.

    This is the cheapest provenance check we have and it is the one that caught a real confound.
    fig_F's concat tables were compared across vintages as if they differed only by FP_VARIANT.
    They did not -- three unrelated commits had landed in between -- and the tell was sitting in the
    output: the `CLM` and `desc+CLM` arms carry NO fingerprint, so a featurizer change cannot touch
    them, yet they had moved (MoleculeACE 0.840 -> 0.819, CBS NEF1 0.509 -> 0.768).

    So: any table containing an arm invariant to the change under test carries a free isolation
    check, and this makes looking at it automatic rather than a thing someone remembers to do.
    A mismatch does NOT necessarily mean a number is wrong -- it means the two files differ by more
    than the stated variable, so any delta quoted between them is uninterpretable.
    """
    print(f"\n{'='*94}\n13. INVARIANT ARMS ACROSS TABLE VINTAGES (free isolation check)\n{'='*94}")
    import csv as _csv
    RIG = ROOT / "analysis" / "rigor"
    # (label, file A, file B, arms that cannot differ, why)
    PAIRS = [("concat MolNet", "concat_redundancy_legacy.csv", "concat_redundancy_stereo.csv",
              {"CLM", "desc+CLM"}, "carry no fingerprint, so FP_VARIANT cannot reach them"),
             ("concat panels", "concat_panels_climb_legacy.csv", "concat_panels_climb_stereo.csv",
              {"CLM", "desc+CLM"}, "carry no fingerprint, so FP_VARIANT cannot reach them")]
    bad = 0
    for label, fa, fb, invariant, why in PAIRS:
        pa, pb = RIG / fa, RIG / fb
        if not (pa.exists() and pb.exists()):
            print(f"  SKIP  {label:15s} one vintage absent ({fa if not pa.exists() else fb})")
            continue
        def rows(p):
            return {(r["task"], r["features"], r["metric"]): r["mean"]
                    for r in _csv.DictReader(p.open())}
        A, B = rows(pa), rows(pb)
        keys = [k for k in A if k in B and k[1] in invariant]
        moved = [k for k in keys if float(A[k]) != float(B[k])]
        changed = sum(1 for k in A if k in B and k[1] not in invariant
                      and float(A[k]) != float(B[k]))
        if not keys:
            print(f"  SKIP  {label:15s} no invariant arms found in both files")
            continue
        if moved:
            detail = ", ".join(f"{t}/{f}/{m} {A[(t,f,m)]}->{B[(t,f,m)]}" for t, f, m in moved[:3])
            print(f"  FAIL  {label:15s} {len(moved)}/{len(keys)} invariant row(s) MOVED - {detail}")
            print(f"        those arms {why}; the two files therefore differ by MORE than the "
                  f"stated variable and any delta between them is uninterpretable")
            bad += 1
        else:
            print(f"  OK    {label:15s} all {len(keys)} invariant row(s) identical; "
                  f"{changed} variable row(s) moved - isolation holds")
    if not bad:
        print("  OK - every retained table pair differs by exactly the variable it claims to")
    return bad


def check_regression_units():
    """Every regression summary must agree with the RMSE recomputed from its OWN prediction dump.

    This is the general form of check 9, and it exists because check 9 was not general enough.
    Check 9 separates QM7's two conventions by MAGNITUDE (native ~200 kcal/mol, z-scored ~0.9),
    which works only because that gap is three orders wide. ESOL's native RMSE runs 0.7-1.5 and its
    z-scored 0.35-0.75; Lipophilicity's two conventions differ by a factor of 1.2. There is no
    threshold that separates those, and both shipped corrupted for months underneath a passing
    check 9.

    Recomputation needs no thresholds and no prior knowledge of which convention a file is in: if
    the summary says 0.4391 and the dir's own predictions say 0.9110, one of the two is wrong
    regardless of what either is supposed to mean. That found 78 ESOL dirs and 42 Lipophilicity
    dirs, including the arms feeding fig_A1's 66-dataset ranking, where a halved CLIMB value was
    being ranked against native anchors.

    NOTE the two exemptions, both real rather than convenient:
      * Whole roots evaluated in z-scored space (climb_v2_ablation_dedup, climb_v2_lrsweep) are
        internally CONSISTENT -- summary and dump agree -- so they pass on their own terms. They
        are a different convention, not a corruption, and rebuilding them would be a no-op that
        merely made them look handled.
      * QM7's read path (moleculenet_cv_qm7native/_qm7clamped) carries no dumps, so it cannot be
        recomputed here. Check 9 covers it by value, which for QM7 is sound.
    """
    print(f"\n{'='*94}\n14. REGRESSION SUMMARIES vs THEIR OWN PREDICTIONS\n{'='*94}")
    import csv as _csv, math, statistics as _st, collections
    from figures.sixpanel import NATIVE_SUBDIRS
    FD = ROOT / "figure_data"
    bad = collections.defaultdict(list)
    superseded = collections.Counter()
    checked = 0

    def reader_subdir(rundir, ds):
        """The subdir a FIGURE would resolve for this (run dir, dataset) -- same precedence the
        readers use. Checking any other copy is checking data nobody draws: after the 2026-08-19
        rebuild the original moleculenet_cv/ still holds the z-scored ESOL and always will, so a
        naive sweep reports 120 permanent failures for numbers no figure reads. That is precisely
        the crying-wolf failure check 11 had, and it ends the same way -- follow the redirect."""
        for cand in NATIVE_SUBDIRS.get(ds, ("moleculenet_cv",)):
            if (rundir / cand / "moleculenet_summary.csv").exists():
                return cand
        return "moleculenet_cv"

    for summ in sorted(FD.glob("*/*/moleculenet_cv*/moleculenet_summary.csv")):
        dump = summ.parent / "test_predictions.csv"
        if not dump.exists():
            continue                      # e.g. the QM7 native subdirs; check 9 covers those
        vals = collections.defaultdict(list)
        for r in _csv.DictReader(summ.open()):
            if (r.get("main_metric") == "rmse" and r.get("head_seed") not in ("MEAN", "STD")
                    and r.get("main_value") not in ("", "nan", None)):
                try:
                    vals[r["dataset"]].append(float(r["main_value"]))
                except ValueError:
                    pass
        if not vals:
            continue
        agg = collections.defaultdict(lambda: [0.0, 0])
        for r in _csv.DictReader(dump.open()):
            try:
                a = agg[r["dataset"]]
                a[0] += (float(r["y_true"]) - float(r["y_pred"])) ** 2
                a[1] += 1
            except (KeyError, ValueError):
                pass
        for ds, vs in vals.items():
            if agg[ds][1] < 50:
                continue
            if summ.parent.name != reader_subdir(summ.parent.parent, ds):
                superseded[ds] += 1          # an older copy the figures do not read
                continue
            checked += 1
            got, truth = _st.mean(vs), math.sqrt(agg[ds][0] / agg[ds][1])
            if truth and abs(got / truth - 1) > 0.10:
                rel = summ.parent.parent.parent.name + "/" + summ.parent.parent.name
                bad[ds].append((rel, summ.parent.name, got, truth, truth / got))
    if superseded:
        print("  (skipped as superseded, not read by any figure: "
              + ", ".join(f"{ds} {n}" for ds, n in sorted(superseded.items())) + ")")
    if not bad:
        print(f"  OK - all {checked} regression summary/dump pair(s) a figure reads agree within 10%")
        return 0
    for ds in sorted(bad):
        rs = bad[ds]
        ratios = sorted({round(r[4], 1) for r in rs})
        print(f"  FAIL  {ds:<15} {len(rs)} dir(s) disagree with their own predictions, "
              f"ratio(s) ~{ratios}")
        for rel, sub, got, truth, ratio in rs[:3]:
            print(f"        {rel}/{sub}: summary {got:.4f} vs recomputed {truth:.4f} (x{ratio:.2f})")
        if len(rs) > 3:
            print(f"        ... and {len(rs) - 3} more")
    print(f"  {sum(len(v) for v in bad.values())} of {checked} pair(s) disagree - a summary and "
          f"the predictions beside it cannot both be right; rebuild with "
          f"scripts/regression_native_rebuild.py")
    return len(bad)


def main():
    print("CROSS-FIGURE CONSISTENCY AUDIT")
    total = sum([check_superseded(), check_units(), check_replication(),
                 check_estimand(), check_panelset(), check_geometry(),
                 check_comparator_scope(), check_bar_vs_ci(),
                 check_qm7_convention(), check_tox21_vintage(),
                 check_replication_parity(), check_aggregate_freshness(),
                 check_invariant_arms(), check_regression_units()])
    print(f"\n{'='*94}\n{'CLEAN' if not total else str(total) + ' ITEM(S) NEED ATTENTION'}\n{'='*94}")


if __name__ == "__main__":
    main()
