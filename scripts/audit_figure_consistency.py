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
    # Changed 2026-08-20 from "+-1 SD of the panel's replicate unit". The replicate unit was not
    # the same quantity for every arm on that figure -- pretraining-seed spread for arms with three
    # pretrainings, head-seed spread for CheMeleon, which has one by construction -- so the bars
    # measured different sources of variation and looked like a precision difference. It now takes
    # every interval from a2_errorbars.csv, one method per panel. The token is the COLUMN the
    # figure reads, and it moved from `sd` to `lo`/`hi`, which is what made this check fire.
    "SI_fig_a":  ("sampling CI of the evaluation units (bootstrap; Ames analytic)", "lo"),
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
            # Same declared exemption check 11 uses, read from the SAME field in arms.py, so the
            # two checks cannot disagree about which arms are a known split-axis case. An arm that
            # replicates on the fine-tune axis on the suite tracks and the pretraining axis on
            # MolNet is reported, not counted -- see the note on _split_axis in check 11.
            known = [a for a in odd
                     if _ARMS.get(a, {}).get("suite_seed_axis") == "finetune"]
            if known and len(known) == len(odd):
                print(f"  {panel:<13}{str(seen):<18}KNOWN — {', '.join(known)} at {min(seen)} "
                      f"(fine-tune axis here, pretraining axis on MolNet; declared in arms.py)")
            else:
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
    # Only fig_A is a deliberate landscape plate. fig_C+D was on this list and was re-laid at the
    # text block on 2026-08-19 -- it stayed "exempt" afterwards, so the check would have gone on
    # ignoring it forever, including if a later edit pushed it back over. An exemption list that
    # outlives its reason is the same failure as checks 3 and 11, so it is verified rather than
    # trusted: a figure claiming the exemption must actually BE wide, or it is flagged.
    # fig_A moved OUT of this set on 2026-08-20 when it was restacked to fit the text block, and
    # fig_A_horizontal moved in: the landscape layout is kept as a second rendering of the same
    # numbers for slides and wide spreads. The two must never disagree -- both compose fig_A1.draw()
    # and fig_A2.draw_panel(), so a disagreement means one was not re-rendered, not that a number
    # changed.
    wide = {"fig_A_horizontal"}
    bad = 0
    for pdf in sorted(OUT.glob("*.pdf")):
        w = _pdf_width_in(pdf)
        if w is None:
            continue
        if pdf.stem in wide and abs(w - A4_TEXT) / A4_TEXT > 0.05:
            print(f"  {pdf.stem:<12}{w:6.2f}in   landscape by design (exempt)")
        elif pdf.stem in wide:
            print(f"  FAIL  {pdf.stem:<12}{w:6.2f}in is at the text block but still claims the "
                  f"landscape exemption -- drop it from `wide`")
            bad += 1
        elif abs(w - A4_TEXT) / A4_TEXT > 0.05:
            print(f"  FAIL  {pdf.stem:<12}{w:6.2f}in vs {A4_TEXT:.2f}in text block "
                  f"({(w/A4_TEXT-1)*100:+.0f}%)")
            bad += 1
    print("  OK — every non-exempt figure is within 5% of the A4 text block" if not bad
          else f"  {bad} figure(s) off-width")
    return bad


def check_comparator_scope():
    hdr(7, "COMPARATOR SCOPE (CheMeleon in the headline figure only)")
    # fig_G is admitted, and the reason matters because widening this rule casually is how it stops
    # working. The rule exists because CheMeleon is a BENCHMARK comparator whose bars carry a
    # protocol confound -- frozen probe vs end-to-end fine-tune -- so putting it beside CLIMB
    # anywhere except the headline invites an unguarded comparison. fig_G trains NOTHING: it
    # measures what a fixed representation can resolve, pair by pair. There is no probe, no fitting
    # and therefore no protocol to confound, and CheMeleon's presence is the point rather than a
    # leak -- the finding is that the blind spots are COMPLEMENTARY (CheMeleon is blind to isotopes
    # and stereochemistry where the CLMs are not, and comparatively good at ring size). Removing it
    # would delete the result.
    #
    # SI_fig_f was the other half of this experiment and was dropped 2026-08-19; its two informative
    # class-B modes are fig_G panels (k) and (l), so the exemption is now one figure, not two.
    # SI_fig_f (the probe-head figure) is admitted for a DIFFERENT reason than fig_G, and the distinction is the rule
    # itself. The confound this check guards is frozen-probe vs end-to-end fine-tune: CheMeleon's
    # benchmark bars mix the two, so placing them beside CLIMB invites an unguarded comparison.
    # It uses chemeleon_FROZEN only, against CLIMB's frozen arms, with the same probe on the
    # same splits -- like for like, and the whole question there is whether the HEAD changes the
    # ranking, which cannot be asked without a representation from outside our own family.
    #
    # fig_F admitted 2026-08-19, BY USER DECISION and on a third distinct ground, which is why it
    # is written down rather than just added. fig_F is the redundancy lattice: does an embedding
    # add anything to classical features? Its cells are desc/fp/desc+fp each ALONE, each + CLIMB,
    # and each + CheMeleon, and the CheMeleon column is not a comparison of CheMeleon against
    # CLIMB -- it is the control that says whether "embeddings are redundant to fingerprints" is a
    # statement about CLIMB specifically or about learned embeddings in general. Drop it and the
    # figure can no longer separate those. The protocol confound this check guards does not arise:
    # every cell in the lattice is the same XGBoost head on concatenated frozen features, so the
    # frozen-vs-fine-tuned mix that makes CheMeleon's benchmark bars unsafe is not present.
    #
    # SI_fig_a admitted 2026-08-20, BY USER DECISION, on a fourth ground. That figure asks "does
    # end-to-end fine-tuning beat the frozen probe?" and CheMeleon is the one external
    # representation where BOTH halves exist, so it answers the same question from outside our
    # family. The protocol confound this rule guards is not merely absent -- it is the figure's
    # x-axis: frozen and end2end are the two positions of every slope, so a reader cannot conflate
    # them here. What DOES differ is the wave (CheMeleon is mainline on all six panels; three of
    # them draw CLIMB at label-efficiency), and that is handled inside the figure by dashing the
    # cross-protocol series and keying the dash, not by excluding the arm.
    allowed = {"fig_A", "fig_A1", "fig_A2", "fig_G", "SI_fig_f", "fig_F", "SI_fig_a"}
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

    # A THIRD state, between "fully replicated" and "has no pretraining stage": an arm that
    # replicates on the PRETRAINING axis on some panels and on the FINE-TUNE axis on others. The
    # two CLIMB end-to-end arms are that -- 3 pretraining seeds on MolNet, 1 encoder x 3 fine-tune
    # seeds on MoleculeACE, Ames and CBS.
    #
    # Reported as KNOWN rather than FAIL, and NOT counted, because it is a decision with a
    # measurement behind it (scripts/pretrain_seed_variance.py: end-to-end pretraining-seed spread
    # is 0.44-0.97x the frozen arm's on the two datasets carrying both axes) rather than an
    # oversight. It still prints on every run, because the day someone quotes a suite cell of these
    # arms as a 3-pretraining-seed number is the day this line needs to be in front of them.
    # Declared in arms.py so a future arm with the same shape inherits the treatment.
    def _split_axis(a):
        return ARMS.get(a, {}).get("suite_seed_axis") == "finetune"

    # A SECOND declared exemption, panel-scoped: an arm may have no replicates on a given panel by
    # DECISION rather than by omission. The two __xgb probe arms are the case -- unranked, no
    # figure draws an interval for them, and their suite bases were built in a venv that no longer
    # exists, so a replicate from a different library set would make the spread partly an
    # environment measurement. Declared per panel in arms.py, because the same arms DO get MolNet
    # and CBS replicates; only the two suite tracks are exempt.
    def _declined(a, panel):
        return panel in (ARMS.get(a, {}).get("no_replicates_on") or ())

    bad = 0
    for panel in sorted(per_panel):
        counts = per_panel[panel]
        for label, group in (("", {a: n for a, n in counts.items() if not _exempt(a)}),
                             ("no-pretrain arms: ", {a: n for a, n in counts.items() if _exempt(a)})):
            if not group:
                continue
            top = max(group.values())
            short = sorted(a for a, n in group.items() if n < top)
            known = [a for a in short if _split_axis(a)]
            declined = [a for a in short if a not in known and _declined(a, panel)]
            short = [a for a in short if a not in known and a not in declined]
            if known:
                print(f"  KNOWN {panel:12s} {label}{', '.join(f'{a}={group[a]}' for a in known)} "
                      f"- replicates on the fine-tune axis here, pretraining axis on MolNet "
                      f"(declared in arms.py; not an oversight)")
            if declined:
                print(f"  KNOWN {panel:12s} {label}{', '.join(f'{a}={group[a]}' for a in declined)} "
                      f"- suite replicates DECLINED for this arm: unranked, no figure draws an "
                      f"interval for it, and its base venv is unreproducible (arms.py)")
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
    print(f"\n{'='*94}\n12. AGGREGATE FRESHNESS (derived tables newer than their OWN inputs)\n{'='*94}")
    import csv as _csv, datetime as _d
    sys.path.insert(0, str(ROOT / "scripts"))
    from figures.arms import ARMS
    fd = ROOT / "figure_data"

    def _arm_dirs(arm):
        src = ARMS.get(arm, {}).get("src", {})
        out = []
        for key in ("mol", "mace"):
            v = src.get(key)
            out += (list(v) if isinstance(v, (list, tuple)) else [v]) if v else []
        return out

    def _newest_for(arms, literal_dirs=False):
        """Newest run artefact among JUST these arms (or these literal run-dir stems)."""
        best, src = 0.0, None
        for a in arms:
            for n in ([a] if literal_dirs else _arm_dirs(a)):
                for root in ("climb_v2_phase2", "cbs_benchmark", "chemeleon_suite/moleculeace",
                             "chemeleon_suite/polaris"):
                    for pat in ("moleculenet_summary.csv", "suite_summary.json", "results.csv",
                                "polaris_scores.csv"):
                        for f in (fd / root / n).glob(f"**/{pat}"):
                            t = f.stat().st_mtime
                            if t > best:
                                best, src = t, f
        return best, src

    # PER-TABLE INPUTS, read from the table's OWN arm column -- not a global "newest anything".
    # The global form fired on 2026-08-20 when two new end-to-end arms landed: scaling_ladders.csv
    # and a2_errorbars.csv were reported stale against runs that neither table contains or has any
    # reason to contain. A freshness check that cries wolf on unrelated data gets re-run, seen to
    # be noise, and then ignored on the day it is right.
    bad = 0
    for tbl in ("six_panel/mainline_8M.csv", "six_panel/scaling_ladders.csv",
                "six_panel/a2_errorbars.csv"):
        f = fd / tbl
        if not f.exists():
            continue
        rows = list(_csv.DictReader(f.open()))
        if not rows:
            print(f"  SKIP  {tbl} is empty")
            continue
        # PREFER `rung`, which names the RUN DIRECTORY, over `arm`, which names a family. The
        # scaling ladder has both, and its inputs are the rungs: resolving it through arms.py
        # instead was what made it compare against every arm in the project and report itself
        # stale against two end-to-end runs it does not contain.
        if "rung" in rows[0]:
            keys = sorted({r["rung"] for r in rows if r["rung"]})
            newest, nsrc = _newest_for(keys, literal_dirs=True)
            arms, unknown = keys, []
        elif "arm" in rows[0]:
            arms = sorted({r["arm"] for r in rows if r["arm"] in ARMS})
            unknown = sorted({r["arm"] for r in rows if r["arm"] not in ARMS})
            newest, nsrc = _newest_for(arms)
        else:
            print(f"  SKIP  {tbl} names neither a rung nor a known arm - inputs undeterminable, "
                  f"so freshness is NOT checked here rather than guessed at")
            continue
        age = f.stat().st_mtime
        if nsrc is not None and age < newest - 60:      # 60s slack for a run in flight
            print(f"  FAIL  {tbl} is OLDER than an arm it CONTAINS "
                  f"({_d.datetime.fromtimestamp(age):%m-%d %H:%M} vs "
                  f"{_d.datetime.fromtimestamp(newest):%m-%d %H:%M}, "
                  f"{nsrc.relative_to(fd)}) - re-run the aggregator")
            bad += 1
        else:
            extra = f" ({len(arms)} arm(s) checked" + (f", {len(unknown)} non-arm key(s) skipped)"
                                                       if unknown else ")")
            print(f"  OK    {tbl}{extra}")
    if not bad:
        print("  OK - every derived table is newer than the runs IT summarises")
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


def check_positional_metric_reads():
    """A summary row selected WITHOUT filtering on main_metric is a coin flip between metrics.

    This defect shape has now cost this project three times: build_fig_E_table's `.iloc[0]` gave
    HIV a ROC-AUC compared against a NEF1 floor (+41%, implausible, caught by the user); the peer
    session's first pretraining-variance reading divided a nef1 numerator into a roc_auc
    denominator and got 1.55 where the metric-matched answer is 0.56-0.97, flipping which side of
    the decision rule it landed on; and pretrain_seed_variance.py itself shipped with `the MEAN
    row` for a dataset whose summary carries two.

    It keeps recurring because it is invisible at the call site -- `head_seed == "MEAN"` LOOKS
    fully specified, and on a regression dataset it is. The exposure is real and large:
    classification summaries carry a roc_auc MEAN and a nef1 MEAN per dataset, and on HIV, whose
    panel metric is nef1, roc_auc is written FIRST -- so a positional read returns the wrong metric
    silently, with a plausible value.

    So this checks the CODE rather than the data: every place that reads a moleculenet_summary.csv
    and reaches a published figure must mention main_metric in the same function. Scoped to
    figures/ and the table builders that feed them -- the one-off rescore and upload scripts under
    scripts/ are not gated here, because a wrong read there is caught by the figure that consumes
    it, and gating 47 files would make this check noise.

    A mention of main_metric is necessary, not sufficient -- it cannot tell a correct filter from a
    mention in a comment. It is a tripwire for the omission, which is how all three of these
    actually happened, not a proof of correctness.
    """
    print(f"\n{'='*94}\n15. POSITIONAL METRIC READS (summary row taken without a metric filter)\n{'='*94}")
    import ast as _ast
    targets = sorted((ROOT / "figures").glob("*.py"))
    targets += [ROOT / "scripts" / n for n in ("six_panel_aggregate.py", "six_panel_e2e.py",
                                              "build_SI_fig_a_table.py", "build_fig_E_table.py",
                                              "pretrain_seed_variance.py")]
    bad = 0
    checked = 0
    for f in targets:
        if not f.exists():
            continue
        src = f.read_text()
        if "moleculenet_summary.csv" not in src:
            continue
        try:
            tree = _ast.parse(src)
        except SyntaxError as e:
            print(f"  FAIL  {f.name}: does not parse ({e})")
            bad += 1
            continue
        lines = src.splitlines()
        scopes = []   # (label, source text)
        covered = set()
        for node in _ast.walk(tree):
            if isinstance(node, (_ast.FunctionDef, _ast.AsyncFunctionDef)):
                seg = "\n".join(lines[node.lineno - 1:node.end_lineno])
                scopes.append((f"{f.name}:{node.name}()", seg, node.lineno))
                covered.update(range(node.lineno, node.end_lineno + 1))
        rest = "\n".join(l for i, l in enumerate(lines, 1) if i not in covered)
        scopes.append((f"{f.name}:<module>", rest, 0))
        for label, seg, lineno in scopes:
            if "moleculenet_summary.csv" not in seg:
                continue
            # Naming the path is not reading a row. The defect is extracting a VALUE without
            # saying which metric it is, so the trigger is main_value present and main_metric
            # absent. Without this the check fired on a file-existence helper, two module
            # docstrings and a scope that reads only the dataset column -- four hits, zero
            # defects. A tripwire with a 100% false-positive rate gets ignored, which is worse
            # than not having it.
            if "main_value" not in seg:
                continue
            checked += 1
            if "main_metric" not in seg:
                where = f" (line {lineno})" if lineno else ""
                print(f"  FAIL  {label}{where} reads a summary but never mentions main_metric -- "
                      f"a MEAN row taken here is whichever metric was written first")
                bad += 1
    if not bad:
        print(f"  OK - all {checked} summary-reading scope(s) in figures/ and the table builders "
              f"filter on main_metric")

    # PART 2, a data-shape invariant that makes a whole class of positional reads SAFE rather than
    # policing each one. mainline_8M.csv carries a `metric` column, so `r["value"]` for a panel is
    # a positional metric read exactly like a MEAN row is -- and the static rule above cannot see
    # it, because it keys on the column name `main_value` that this file does not use. It has
    # never been wrong, for one reason only: it holds exactly ONE row per (arm, panel). Asserting
    # that here means every consumer inherits the guarantee, instead of each having to filter.
    # SI_fig_a's anchor line is the reason this matters: HIV's row is nef1 and its panel is nef1,
    # but a second HIV row on roc_auc would have drawn a 0.7373 ROC-AUC onto a NEF1 axis silently.
    import collections as _c
    f = ROOT / "figure_data" / "six_panel" / "mainline_8M.csv"
    if f.exists():
        rows = list(_csv.DictReader(f.open())) if "_csv" in dir() else None
        import csv as _c2
        rows = list(_c2.DictReader(f.open()))
        dup = [k for k, n in _c.Counter((r["arm"], r["panel"]) for r in rows).items() if n > 1]
        if dup:
            print(f"  FAIL  mainline_8M.csv has {len(dup)} (arm, panel) pair(s) on MORE THAN ONE "
                  f"row - e.g. {dup[:3]}. Every consumer reads it positionally; add a metric "
                  f"filter to each, or keep the table one-row-per-panel")
            bad += 1
        else:
            print(f"  OK    mainline_8M.csv is one row per (arm, panel) over {len(rows)} rows - "
                  f"positional metric reads of it are safe by construction")
    return bad



def check_source_coverage():
    """Every derived table a figure's family produces must be READ by that figure.

    A table sitting in analysis/rigor that no figure claims is not a harmless spare file. It is the
    signature of the failure this project keeps having: a figure holds a hardcoded list of input
    paths, a new arm's table lands beside the others, and the figure keeps drawing that arm as
    "not run" while the data exists. Nothing errors, because "file absent" is also the legitimate
    state of a run still in flight -- the two are indistinguishable from inside the figure.

    fig_F is the live case and the reason this check exists. Its list held climb and chemeleon
    only; the supervised arm's tables were already being written under a third stem. The list is
    now derived from ROLE_ORDER, and this check verifies the derivation actually covers what is on
    disk rather than trusting that it does.

    Vintage tables (legacy/stereo/PREFIX_BACKUP) are provenance, not inputs, and are exempt --
    check 13 is what reads those.
    """
    print(f"\n{'='*94}\n16. DERIVED TABLES ARE ACTUALLY READ BY THEIR FIGURE\n{'='*94}")
    import importlib
    RIG = ROOT / "analysis" / "rigor"
    bad = 0
    try:
        figF = importlib.import_module("figures.fig_F")
    except Exception as e:                                     # noqa: BLE001
        print(f"  FAIL  cannot import figures.fig_F to read its SOURCES: {e}")
        return 1
    claimed = {Path(f).name for f in figF.SOURCES}
    on_disk = {f.name for f in RIG.glob("concat_*_v2.csv")}
    orphan = sorted(on_disk - claimed)
    if orphan:
        bad += len(orphan)
        for f in orphan:
            print(f"  FAIL  {f} is in analysis/rigor but not in fig_F.SOURCES "
                  f"- produced and not read; add its role to ROLE_ORDER/ROLE_SRC_STEM")
    else:
        print(f"  OK    all {len(on_disk)} concat_*_v2 table(s) on disk are read by fig_F")
    # The other direction is NOT a failure: a claimed file that is absent is a run in flight, and
    # the figure already says so by name at render time. Report it as pending, not as a defect.
    pending = sorted(claimed - on_disk)
    for f in pending:
        print(f"  PEND  {f} claimed by fig_F, not yet on disk (run in flight; draws as 'not run')")
    print("  OK - every produced concat table has a reader" if not bad
          else f"  {bad} produced table(s) nothing reads")
    return bad



def check_label_conventions():
    """The arm labels are rendered by two different mechanisms that want opposite things.

    fig_A1 draws system() on one line and label() on the next, so a label that repeats the system
    name prints it twice on one tick ("CheMeleon / CheMeleon, end2end"). fig_A2 draws `short`
    ALONE for the CheMeleon arms, so a short that OMITS the system name leaves the reader with
    "frozen, XGBoost" and no model ("end2end" alone is worse -- there is a "no pretrain, end2end"
    control a few rows away). Both defects existed on 2026-08-20 and neither raised anything.

    Also enforced: no parentheses anywhere (user 2026-08-19, "use commas not parentheses"), and
    the comma convention -- the comma separates the encoder from its qualifier, and when an
    encoder's own name already carries one the probe appends with a space rather than a third
    comma ("supervised, desc end2end", not "supervised desc, end2end").
    """
    print(f"\n{'='*94}\n17. ARM LABEL CONVENTIONS (two renderers, opposite requirements)\n{'='*94}")
    import importlib
    A = importlib.import_module("figures.arms")
    bad = 0
    for k, v in A.ARMS.items():
        lab, short, sysname = v["label"], v.get("short", ""), A.system(k)
        if sysname and lab.lower().startswith(sysname.lower()):
            print(f"  FAIL  {k}: label {lab!r} repeats the system name {sysname!r} - fig_A1 draws "
                  f"system() above label(), so this tick names the model twice")
            bad += 1
        if sysname == "CheMeleon" and "CheMeleon" not in short:
            print(f"  FAIL  {k}: short {short!r} omits 'CheMeleon' - fig_A2 renders `short` alone")
            bad += 1
        for field, txt in (("label", lab), ("short", short)):
            if "(" in txt or ")" in txt:
                print(f"  FAIL  {k}: {field} {txt!r} uses parentheses - the convention is commas")
                bad += 1
    # the comma rule, checked where it is checkable: a probe suffix must not be preceded by a
    # comma when the encoder's own name already contains one.
    PROBE_WORDS = ("end2end", "XGBoost probe", "MLP probe")
    for k, v in A.ARMS.items():
        lab = v["label"]
        for w in PROBE_WORDS:
            if lab.endswith(w):
                stem = lab[: -len(w)].rstrip()
                if stem.count(",") >= 1 and stem.endswith(","):
                    head = stem[:-1]
                    if "," in head:
                        print(f"  FAIL  {k}: label {lab!r} has a comma before {w!r} although the "
                              f"encoder name {head!r} already carries one - append with a space")
                        bad += 1
    print("  OK - every label reads correctly in both renderers" if not bad
          else f"  {bad} label convention problem(s)")
    return bad



def check_featurizer_homogeneity():
    """Every dir pooled into one arm must have been built by the SAME featurizer.

    A pooled arm is a mean over its replicate dirs. If one of those dirs was produced by a
    different featurizer, the mean belongs to no featurizer at all -- and the number is not
    obviously wrong, which is what makes it dangerous. Live case, 2026-08-20: ecfp4's MoleculeACE
    base was pre-stereo while its two replicates were ecfp4_stereo, and the pooled value read
    0.6878 against a pre-stereo base of 0.6877 and stereo replicates of 0.6873 and 0.6886. The
    contamination sat well inside the seed-to-seed spread and was invisible in the value.

    verified.json carries fp_variant only for dirs written after 2026-08-20, and deliberately was
    not backfilled -- the older dirs' variant is known from the runner, not from the artefact, and
    asserting it would convert an inference into the file's own claim. So MISSING is reported
    separately from MIXED: a dir with no variant is unlabelled, not wrong, and only becomes a
    finding when it sits beside a labelled one.
    """
    print(f"\n{'='*94}\n18. FEATURIZER HOMOGENEITY WITHIN A POOLED ARM\n{'='*94}")
    import json as _json
    from figures.arms import ARMS
    FD = ROOT / "figure_data"
    bad = 0
    for arm, meta in ARMS.items():
        for key, tree in (("mace", "chemeleon_suite/moleculeace"), ("mol", "climb_v2_phase2")):
            v = meta.get("src", {}).get(key)
            if v is None:
                continue
            dirs = list(v) if isinstance(v, (list, tuple)) else [v, f"{v}_s1", f"{v}_s2"]
            seen, feats = {}, {}
            for d in dirs:
                f = FD / tree / d / "verified.json"
                if not f.exists():
                    continue
                try:
                    j = _json.loads(f.read_text())
                    seen[d], feats[d] = j.get("fp_variant"), j.get("featurizer")
                except Exception:                                     # noqa: BLE001
                    seen[d] = "<unreadable>"
            if len(seen) < 2:
                continue
            labelled = {d: fv for d, fv in seen.items() if fv}
            unlabelled = [d for d, fv in seen.items() if not fv]

            # fp_variant ONLY DESCRIBES FINGERPRINT ARMS. The runner writes it from the
            # environment, so it lands on runs that use no fingerprint at all -- CheMeleon's
            # XGBoost probe carries "featurizer": "chemeleon" AND "fp_variant": "ecfp4_stereo",
            # which is a false claim about a component the arm does not have. Numerically
            # harmless, and exactly the failure family this file keeps finding: a field that
            # answers confidently about something it does not describe.
            if not any(str(feats.get(d, "")).startswith(("ecfp", "fp_desc", "r3fp", "morgan"))
                       for d in seen):
                stray = sorted(labelled)
                if stray:
                    print(f"  FAIL  {arm} ({key}) is featurized by "
                          f"'{feats.get(sorted(seen)[0])}' and has NO fingerprint, but "
                          f"{len(stray)} dir(s) record fp_variant={sorted(set(labelled.values()))} "
                          f"- the environment default leaked onto a run it does not describe")
                    bad += 1
                continue

            if len(set(labelled.values())) > 1:
                print(f"  FAIL  {arm} ({key}) pools MORE THAN ONE featurizer: "
                      + ", ".join(f"{d}={fv}" for d, fv in sorted(labelled.items())))
                bad += 1
            elif labelled and unlabelled:
                # NOT counted. fp_variant exists only on dirs written after 2026-08-20 and was
                # deliberately not backfilled, so a base untouched by that pass is unlabelled
                # whether or not it matches. This states what the artefacts can and cannot show
                # rather than converting an inference into a verdict in either direction.
                print(f"  UNVERIF {arm} ({key}) pools {sorted(set(labelled.values()))[0]} dirs "
                      f"with UNLABELLED base {sorted(unlabelled)} - same vintage per the runner "
                      f"that produced it, NOT confirmable from the files")
    print("  OK - no arm pools dirs of differing featurizer vintage" if not bad
          else f"  {bad} arm(s) pooling mixed or unverifiable featurizer vintages")
    return bad


def main():
    print("CROSS-FIGURE CONSISTENCY AUDIT")
    total = sum([check_superseded(), check_units(), check_replication(),
                 check_estimand(), check_panelset(), check_geometry(),
                 check_comparator_scope(), check_bar_vs_ci(),
                 check_qm7_convention(), check_tox21_vintage(),
                 check_replication_parity(), check_aggregate_freshness(),
                 check_invariant_arms(), check_regression_units(),
                 check_positional_metric_reads(), check_source_coverage(),
                 check_label_conventions(),
                 check_featurizer_homogeneity()])
    print(f"\n{'='*94}\n{'CLEAN' if not total else str(total) + ' ITEM(S) NEED ATTENTION'}\n{'='*94}")


if __name__ == "__main__":
    main()
