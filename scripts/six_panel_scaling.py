"""Aggregate the pretraining-scaling ladders onto the canonical 6 panels -> Fig B.

Five ladders (the primary regimes; mixed/minimol are peripheral and stay out, matching the old
notebook's A2 selection):

  supervised, dense            2M / 8M / 24M / 48M / 96M FP
  supervised, dense+sparse     2M / 8M / 24M / 48M FP
  supervised, sparse           2M / 8M / 24M / 48M FP
  unsupervised (MLM)           2M / 8M / 24M / 48M / 50M / 100M FP
  unsup->sup, dense            from 2M / 8M / 24M / 48M base (+2M FP SFT stage)

Two things the reader must know (both go in the caption):
  * X = TOKENS ACTUALLY PROCESSED (trainer's own non-padding `tokens_seen`, last line of each
    run's metrics.jsonl) -- NOT forward passes x a constant; the tok/FP ratio varies by corpus
    (12M corpus ~42.9, RDKit-canonical ~40.4). unsup->sup counts its TRUE total: MLM-base tokens
    + SFT-stage tokens (the warm start must not hide its spend).
  * unsup_50M / unsup_100M are trained on the LARGER RDKit-canonical corpus (~124M molecules),
    not the 12M corpus of the 2M-48M rungs -- same ladder, different corpus at the top two rungs.
    unsup_48M has no MoleculeACE score (Wave 2 scored the 50M/100M successors instead), so the
    MoleculeACE ladder jumps 24M -> 50M.

Every rung is evaluated seed-0 ONLY (pretraining-seed replicates exist at the 8M rung and are
deliberately ignored -- including them would make the 8M error bar a different quantity from every
other point on the same line). Error bar = sd_total from panel_stats, which for a single dir is
the within-dir fold SD (5 folds; MoleculeACE: SD across the 3 eval-seed macro-means; hERG: SD
across the 3 eval seeds) -- the SAME estimand at every rung.

Writes: figure_data/six_panel/scaling_ladders.csv
Run:  python3 scripts/six_panel_scaling.py
"""
from __future__ import annotations
import csv, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from figures.arms import ARMS  # noqa: E402
from scripts.six_panel_aggregate import (  # noqa: E402
    FD, MOL_PANELS, POLARIS_PANELS, QM7_SUBDIRS, TOX21_SUBDIRS, DEFAULT_SUBDIRS, mol_fold_values,
    mol_dir_summaries, panel_stats, mace_per_target, mace_seed_macros, polaris_cells,
    cluster_bootstrap)
import statistics as st  # noqa: E402

OUT = FD / "six_panel" / "scaling_ladders.csv"

# ladder key -> (arms.py arm whose label/colour it inherits, rung dirs in ascending budget)
LADDERS = {
    # skip_dense_48M IS DELIBERATELY ABSENT (Leif 2026-08-26: "just drop it everywhere").
    # It began 2026-07-16T09:34:03Z, 2h43m before the canonical descriptor-stats object was first
    # written, so it took pretrain_v2's refit branch and fit its own normalizer on a 20k sample --
    # on a box whose venv carried a shadowed rdkit-pypi 2022.9.5 exposing 208 of the 217
    # descriptors. Its config is also the only one in this ladder with no descriptor_precompute_dir,
    # so it computed those 208 live. Three differences in one rung: descriptor set, normalizer,
    # pathway. It is not a noisy point, it is a point answering a different question.
    #
    # It was invisible in the values -- MoleculeACE 0.7674 between 24M's 0.7687 and 96M's 0.7748 --
    # which is why nothing caught it for six weeks and why dropping it beats footnoting it.
    # The _c124 rungs are the SAME supervised-dense regime trained on the 124M RDKit-canonical
    # corpus instead of the 12M one, so they belong on this line with open markers -- exactly the
    # convention unsup_50M/unsup_100M already established on the unsup line. skip_dense_8M_c124 is
    # the bridge: same 8M forward passes as skip_dense_8M, different corpus, so the corpus effect
    # reads directly as the vertical gap between two points at the same budget.
    # skip_dense_100M_c124 is still training (ETA 2026-08-29). It is listed anyway: run_tokens()
    # returns None without a metrics.jsonl, main() prints "SKIP <rung>: no metrics.jsonl" and moves
    # on, and the rung joins the line the moment its data lands. A listed-but-absent rung is
    # announced every run; a rung nobody listed is silent forever.
    # ONE POINT PER BUDGET (Leif 2026-08-28: "only include skip dense c124 drop the other one").
    # skip_dense_8M and skip_dense_8M_c124 are the SAME 8M forward passes on the 12M and the 124M
    # corpus, so they land 0.330B and 0.343B apart on a log axis spanning 0.09B-4B -- effectively
    # the same x. Sorted by tokens the line joined them, and the corpus difference (up to 0.039
    # macro-RMSE on MoleculeACE, larger than the whole line's span) was drawn as a near-vertical
    # zigzag that reads as instability rather than as a result. The c124 rung is the one kept: it
    # is on the same corpus as the 50M/100M rungs above it, which makes this line parallel in
    # construction to the unsup line (small corpus low, 124M corpus high).
    #
    # skip_dense_8M IS NOT DELETED ANYWHERE ELSE. It remains the mainline supervised 8M arm in
    # mainline_8M.csv and the manifest template for every _c124 clone; this drops it from ONE
    # line of ONE figure, and the bridge comparison it anchors is a caption number, not a point.
    "sup_dense":        ("sup_dense",
                         ["skip_dense_2M", "skip_dense_8M_c124",
                          "skip_dense_24M", "skip_dense_96M",
                          "skip_dense_50M_c124", "skip_dense_100M_c124"]),
    "sup_dense_sparse": ("sup_dense_sparse",
                         ["skip_dense_plus_sparse_2M", "skip_dense_plus_sparse_8M",
                          "skip_dense_plus_sparse_24M", "skip_dense_plus_sparse_48M"]),
    "sup_sparse":       ("sup_sparse",
                         ["skip_sparse_all_2M", "skip_sparse_all_8M", "skip_sparse_all_24M", "skip_sparse_all_48M"]),
    "unsup":            ("unsup",
                         ["unsup_2M", "unsup_8M", "unsup_24M", "unsup_48M", "unsup_50M", "unsup_100M"]),
    "u2s_dense":        ("u2s_dense",
                         ["u2s_dense_from2M", "u2s_dense_from8M", "u2s_dense_from24M",
                          "u2s_dense_from48M", "u2s_dense_from50M", "u2s_dense_from100M"]),
}
# warm-start bases: u2s_dense_from{B} continues unsup_{B}; its true token count is base + SFT.
# Each u2s rung's own metrics.jsonl counts ONLY its 2M-FP SFT stage: u2s_dense_from50M and
# u2s_dense_from100M both read 82,438,904 tokens despite continuing very different bases. Without
# an entry here a rung plots at the SFT spend alone, which for the two new ones would put a
# 2.1B-token and a 4.1B-token model at the same x as the 2M rung. The warm start must not hide its
# spend.
U2S_BASE = {"u2s_dense_from2M": "unsup_2M", "u2s_dense_from8M": "unsup_8M",
            "u2s_dense_from24M": "unsup_24M", "u2s_dense_from48M": "unsup_48M",
            "u2s_dense_from50M": "unsup_50M", "u2s_dense_from100M": "unsup_100M"}
# rungs on the larger RDKit-canonical corpus (marked in the figure, stated in the caption)
# Rungs trained on the larger RDKit-canonical corpus (~124M molecules) rather than the 12M one.
# fig_B draws these with open markers and treats them as escaping the 12M unique-molecule cap, so
# a rung missing from this set is not merely mis-styled -- it is plotted as if it re-read a corpus
# it never touched. The u2s rungs inherit the corpus of the base they continue.
BIG_CORPUS = {"unsup_50M", "unsup_100M",
              "skip_dense_8M_c124", "skip_dense_50M_c124", "skip_dense_100M_c124",
              "u2s_dense_from50M", "u2s_dense_from100M"}


def run_tokens(run):
    """Last tokens_seen in metrics.jsonl (trainer's own non-padding count), or None.

    THE LAST LINE IS NOT ALWAYS A TRAINING STEP. A run that finishes through the top-up loop ends
    with a record of the top-up itself --
        {"step": 7813, "topup_steps": 1, "forward_passes_seen": 2000128, ...}
    -- which carries forward_passes_seen but NO tokens_seen. Reading the final line alone then
    returned None for a complete 41 KB metrics file, and both u2s_dense_from50M and
    u2s_dense_from100M vanished from fig_B while their data sat on disk. Scan for the last line
    that actually HAS the field instead of assuming the last line is a step record.
    """
    m = FD / "climb_v2_phase2" / run / "metrics.jsonl"
    if not m.exists():
        return None
    tok = None
    for line in m.open():
        if not line.strip():
            continue
        try:
            v = json.loads(line).get("tokens_seen")
        except Exception:
            continue
        if v is not None:
            tok = v
    return None if tok is None else float(tok)


def ladder_subdir(all_rungs, subdirs, panel, mixing_cost):
    """ONE subdir for the ENTIRE ladder set — all-or-nothing, never per rung.

    THE HAZARD. `_pick_subdir` resolves one subdir per ARM and drops the dirs that lack it, which
    is right when the dropped dir is one replicate of many. It is WRONG for a ladder: a rung is a
    point on a curve, not a replicate, so dropping it puts a hole in the line. Worse, this file
    calls the readers with a SINGLE-rung list, so `_pick_subdir` only ever sees one dir and
    resolves per rung — and when it falls through to the last subdir it reports nothing, because
    `skipped` is empty. Two rungs scored under different protocols then land on the same line with
    no warning anywhere. So: use the preferred subdir only if EVERY rung has it, otherwise the
    fallback everywhere, and say which.

    TEST THE DIRECTORY THAT IS ACTUALLY RETURNED. The QM7 version of this checked for
    `moleculenet_cv_qm7native` and then returned `moleculenet_cv_qm7clamped` — so a rung carrying
    native but not clamped passed the gate, and `_cv_dir` then found nothing and dropped that
    rung's QM7 row silently. Correct only while every rung happened to have both. The two are not
    interchangeable: on unsup_2M clamped reads 197.98 against native's 201.17.

    `mixing_cost` states, in the panel's own units, what a silent mix would draw as a scaling
    effect. It goes in the message because a warning that does not say what it costs gets ignored.
    """
    preferred, fallback = subdirs[0], subdirs[-1]
    have = [r for r in all_rungs if (FD / "climb_v2_phase2" / r / preferred).exists()]
    if len(have) == len(all_rungs):
        print(f"  {panel}: {preferred}/ for all {len(all_rungs)} rungs — one protocol, "
              f"same convention as fig_A")
        return (preferred,)
    missing = [r for r in all_rungs if r not in set(have)]
    print(f"  {panel}: falling back to {fallback}/ for ALL {len(all_rungs)} rungs — "
          f"{len(missing)} rung(s) have no {preferred}/ ({', '.join(missing[:6])}"
          f"{'...' if len(missing) > 6 else ''}). {mixing_cost} "
          f"This figure's {panel} panel is then NOT comparable to fig_A's.")
    return (fallback,)


# What a silent mix would cost, measured on the 14 rungs that had both on 2026-08-28. These are
# the numbers that make the all-or-nothing rule non-negotiable rather than tidy-minded.
TOX21_MIX_COST = ("Mixing them would draw a +0.0233 AUC step (measured, range +0.0199..+0.0278) "
                  "as a scaling effect — roughly 75% of the whole Tox21 ladder's range, so one "
                  "mis-scored rung does not add noise, it invents a result.")
QM7_MIX_COST = ("Mixing them would draw a 230x unit step (z-scored ~0.85 vs native ~195 kcal/mol) "
                "as a scaling effect.")


def main():
    rows = []
    listed = [r for _, rungs in LADDERS.values() for r in rungs]
    # ONLY RUNGS THAT WILL ACTUALLY BE DRAWN GET A VOTE ON THE PROTOCOL. A rung listed ahead of its
    # data contributes no row at all, so it cannot mix anything -- but ladder_subdir sees only a
    # missing directory and cannot tell "absent because it is still training" from "present and
    # scored the old way". Passing the raw list let five not-yet-evaluated rungs drag BOTH panels
    # to the stale protocol: every Tox21 value in the table fell by ~0.023, the exact step the
    # guard exists to prevent, caused by the guard. Filtered on the same condition the loop below
    # skips on, so the decision is made over precisely the set that gets plotted.
    all_rungs = [r for r in listed if run_tokens(r) is not None]
    _absent = [r for r in listed if r not in set(all_rungs)]
    if _absent:
        print(f"  {len(_absent)} listed rung(s) have no data yet and are excluded from the "
              f"subdir decision: {', '.join(_absent)}")
    qm7_sub = ladder_subdir(all_rungs, QM7_SUBDIRS, "QM7", QM7_MIX_COST)
    tox21_sub = ladder_subdir(all_rungs, TOX21_SUBDIRS, "Tox21", TOX21_MIX_COST)
    for ladder, (arm, rungs) in LADDERS.items():
        for rung in rungs:
            tok = run_tokens(rung)
            if tok is None:
                # Say WHICH of the two it is. "no metrics.jsonl" was printed for both, and for a
                # run whose file is present but unparsed that sends the reader to re-run a job
                # that already finished.
                _m = FD / "climb_v2_phase2" / rung / "metrics.jsonl"
                why = ("no metrics.jsonl" if not _m.exists() else
                       f"metrics.jsonl present ({_m.stat().st_size:,} bytes) but no tokens_seen "
                       f"in any record -- NOT a missing run")
                print(f"  SKIP {rung}: {why}")
                continue
            if rung in U2S_BASE:
                base = run_tokens(U2S_BASE[rung]) or 0.0
                tok = tok + base
            base_row = dict(ladder=ladder, arm=arm, rung=rung, tokens=round(tok, 1),
                            big_corpus=int(rung in BIG_CORPUS))

            # --- MoleculeACE ---------------------------------------------------------
            # ONE pretraining dir per rung, passed as an explicit [rung] so mace_seed_dirs does
            # NOT expand to _s1/_s2. The ladder is deliberately single-seed -- replicating every
            # rung is too expensive (user, 2026-08-19) -- but only the 8M rung HAS _s1/_s2 on
            # disk, so the default expansion silently made that one point a mean over 3
            # pretraining runs while its neighbours were means over 1. On a scaling curve that is
            # not extra rigour, it is one point estimated differently from the rest: better
            # averaged and with a spuriously different sd, at exactly the rung the mainline
            # figures also report. Consistency ACROSS RUNGS is what a ladder needs.
            pt = mace_per_target([rung])
            if pt and pt.get("overall"):
                macros = mace_seed_macros([rung])
                rows.append(dict(**base_row, panel="MoleculeACE", metric="macro_rmse",
                                 value=round(st.mean(pt["overall"].values()), 4),
                                 sd_total=round(st.stdev(macros) if len(macros) > 1 else 0.0, 4),
                                 n_cells=len(macros)))
            # --- CBS -----------------------------------------------------------------
            folds = mol_fold_values([rung], "cbs", "nef1", root="cbs_benchmark")
            dirs = mol_dir_summaries([rung], "cbs", "nef1", root="cbs_benchmark") if not folds else None
            stats = panel_stats(cells=folds or None, dir_summaries=dirs)
            if stats:
                value, sd_total, _, _, n_cells = stats
                rows.append(dict(**base_row, panel="CBS", metric="nef1",
                                 value=round(value, 4), sd_total=round(sd_total, 4), n_cells=n_cells))
            # --- hERG ----------------------------------------------------------------
            cells = polaris_cells([rung], *POLARIS_PANELS["Ames"])   # single dir, see above
            if cells:
                vals = [v for _, v in cells]
                rows.append(dict(**base_row, panel="Ames", metric="roc_auc",
                                 value=round(st.mean(vals), 4),
                                 sd_total=round(st.stdev(vals) if len(vals) > 1 else 0.0, 4),
                                 n_cells=len(vals)))
            # --- MoleculeNet ---------------------------------------------------------
            for ds, metric in MOL_PANELS.items():
                subs = (qm7_sub if ds == "QM7" else
                        tox21_sub if ds == "Tox21" else DEFAULT_SUBDIRS)
                folds = mol_fold_values([rung], ds, metric, subdirs=subs)
                dirs = mol_dir_summaries([rung], ds, metric, subdirs=subs) if not folds else None
                stats = panel_stats(cells=folds or None, dir_summaries=dirs)
                if stats:
                    value, sd_total, _, _, n_cells = stats
                    rows.append(dict(**base_row, panel=ds, metric=metric,
                                     value=round(value, 4), sd_total=round(sd_total, 4), n_cells=n_cells))

    fields = ["ladder", "arm", "rung", "tokens", "big_corpus", "panel", "metric", "value",
              "sd_total", "n_cells"]
    with open(OUT, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    # coverage board
    print(f"wrote {OUT}  {len(rows)} rows\n")
    panels = ["MoleculeACE", "HIV", "BACE", "Ames", "Tox21", "QM7"]
    have = {(r["ladder"], r["rung"], r["panel"]) for r in rows}
    print(f"{'rung':32s} " + " ".join(f"{p[:6]:>6s}" for p in panels))
    for ladder, (_, rungs) in LADDERS.items():
        for rung in rungs:
            cells = " ".join(f"{'ok' if (ladder, rung, p) in have else '--':>6s}" for p in panels)
            print(f"{rung:32s} {cells}")


if __name__ == "__main__":
    main()
