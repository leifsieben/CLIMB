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
    FD, MOL_PANELS, POLARIS_PANELS, mol_fold_values, mol_dir_summaries, panel_stats,
    mace_per_target, mace_seed_macros, polaris_cells, cluster_bootstrap)
import statistics as st  # noqa: E402

OUT = FD / "six_panel" / "scaling_ladders.csv"

# ladder key -> (arms.py arm whose label/colour it inherits, rung dirs in ascending budget)
LADDERS = {
    "sup_dense":        ("sup_dense",
                         ["skip_dense_2M", "skip_dense_8M", "skip_dense_24M", "skip_dense_48M", "skip_dense_96M"]),
    "sup_dense_sparse": ("sup_dense_sparse",
                         ["skip_dense_plus_sparse_2M", "skip_dense_plus_sparse_8M",
                          "skip_dense_plus_sparse_24M", "skip_dense_plus_sparse_48M"]),
    "sup_sparse":       ("sup_sparse",
                         ["skip_sparse_all_2M", "skip_sparse_all_8M", "skip_sparse_all_24M", "skip_sparse_all_48M"]),
    "unsup":            ("unsup",
                         ["unsup_2M", "unsup_8M", "unsup_24M", "unsup_48M", "unsup_50M", "unsup_100M"]),
    "u2s_dense":        ("u2s_dense",
                         ["u2s_dense_from2M", "u2s_dense_from8M", "u2s_dense_from24M", "u2s_dense_from48M"]),
}
# warm-start bases: u2s_dense_from{B} continues unsup_{B}; its true token count is base + SFT.
U2S_BASE = {"u2s_dense_from2M": "unsup_2M", "u2s_dense_from8M": "unsup_8M",
            "u2s_dense_from24M": "unsup_24M", "u2s_dense_from48M": "unsup_48M"}
# rungs on the larger RDKit-canonical corpus (marked in the figure, stated in the caption)
BIG_CORPUS = {"unsup_50M", "unsup_100M"}


def run_tokens(run):
    """Last tokens_seen in metrics.jsonl (trainer's own non-padding count), or None."""
    m = FD / "climb_v2_phase2" / run / "metrics.jsonl"
    if not m.exists():
        return None
    last = ""
    for line in m.open():
        if line.strip():
            last = line
    try:
        return float(json.loads(last).get("tokens_seen"))
    except Exception:
        return None


def main():
    rows = []
    for ladder, (arm, rungs) in LADDERS.items():
        for rung in rungs:
            tok = run_tokens(rung)
            if tok is None:
                print(f"  SKIP {rung}: no metrics.jsonl")
                continue
            if rung in U2S_BASE:
                base = run_tokens(U2S_BASE[rung]) or 0.0
                tok = tok + base
            base_row = dict(ladder=ladder, arm=arm, rung=rung, tokens=round(tok, 1),
                            big_corpus=int(rung in BIG_CORPUS))

            # --- MoleculeACE ---------------------------------------------------------
            pt = mace_per_target(rung)
            if pt and pt.get("overall"):
                macros = mace_seed_macros(rung)
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
            cells = polaris_cells(rung, *POLARIS_PANELS["Ames"])
            if cells:
                vals = [v for _, v in cells]
                rows.append(dict(**base_row, panel="Ames", metric="roc_auc",
                                 value=round(st.mean(vals), 4),
                                 sd_total=round(st.stdev(vals) if len(vals) > 1 else 0.0, 4),
                                 n_cells=len(vals)))
            # --- MoleculeNet ---------------------------------------------------------
            for ds, metric in MOL_PANELS.items():
                folds = mol_fold_values([rung], ds, metric)
                dirs = mol_dir_summaries([rung], ds, metric) if not folds else None
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
    panels = ["MoleculeACE", "CBS", "BACE", "Ames", "Tox21", "QM7"]
    have = {(r["ladder"], r["rung"], r["panel"]) for r in rows}
    print(f"{'rung':32s} " + " ".join(f"{p[:6]:>6s}" for p in panels))
    for ladder, (_, rungs) in LADDERS.items():
        for rung in rungs:
            cells = " ".join(f"{'ok' if (ladder, rung, p) in have else '--':>6s}" for p in panels)
            print(f"{rung:32s} {cells}")


if __name__ == "__main__":
    main()
