from __future__ import annotations
import csv, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from figures.arms import ARMS
from scripts.six_panel_aggregate import (
    FD, MOL_PANELS, POLARIS_PANELS, QM7_SUBDIRS, TOX21_SUBDIRS, DEFAULT_SUBDIRS, mol_fold_values,
    mol_dir_summaries, panel_stats, mace_per_target, mace_seed_macros, polaris_cells,
    cluster_bootstrap)
import statistics as st

OUT = FD / "six_panel" / "scaling_ladders.csv"


LADDERS = {


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


U2S_BASE = {"u2s_dense_from2M": "unsup_2M", "u2s_dense_from8M": "unsup_8M",
            "u2s_dense_from24M": "unsup_24M", "u2s_dense_from48M": "unsup_48M",
            "u2s_dense_from50M": "unsup_50M", "u2s_dense_from100M": "unsup_100M"}


BIG_CORPUS = {"unsup_50M", "unsup_100M",
              "skip_dense_8M_c124", "skip_dense_50M_c124", "skip_dense_100M_c124",
              "u2s_dense_from50M", "u2s_dense_from100M"}


def run_tokens(run):

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


TOX21_MIX_COST = ("Mixing them would draw a +0.0233 AUC step (measured, range +0.0199..+0.0278) "
                  "as a scaling effect — roughly 75% of the whole Tox21 ladder's range, so one "
                  "mis-scored rung does not add noise, it invents a result.")
QM7_MIX_COST = ("Mixing them would draw a 230x unit step (z-scored ~0.85 vs native ~195 kcal/mol) "
                "as a scaling effect.")


def main():
    rows = []
    listed = [r for _, rungs in LADDERS.values() for r in rungs]


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


            pt = mace_per_target([rung])
            if pt and pt.get("overall"):
                macros = mace_seed_macros([rung])
                rows.append(dict(**base_row, panel="MoleculeACE", metric="macro_rmse",
                                 value=round(st.mean(pt["overall"].values()), 4),
                                 sd_total=round(st.stdev(macros) if len(macros) > 1 else 0.0, 4),
                                 n_cells=len(macros)))

            folds = mol_fold_values([rung], "cbs", "nef1", root="cbs_benchmark")
            dirs = mol_dir_summaries([rung], "cbs", "nef1", root="cbs_benchmark") if not folds else None
            stats = panel_stats(cells=folds or None, dir_summaries=dirs)
            if stats:
                value, sd_total, _, _, n_cells = stats
                rows.append(dict(**base_row, panel="CBS", metric="nef1",
                                 value=round(value, 4), sd_total=round(sd_total, 4), n_cells=n_cells))

            cells = polaris_cells([rung], *POLARIS_PANELS["Ames"])
            if cells:
                vals = [v for _, v in cells]
                rows.append(dict(**base_row, panel="Ames", metric="roc_auc",
                                 value=round(st.mean(vals), 4),
                                 sd_total=round(st.stdev(vals) if len(vals) > 1 else 0.0, 4),
                                 n_cells=len(vals)))

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
