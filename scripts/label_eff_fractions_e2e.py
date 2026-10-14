from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")
import sys, json, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np, pandas as pd
from rdkit import RDLogger; RDLogger.DisableLog("rdApp.*")

from evaluate import _load_moleculenet
from finetune import evaluate_finetuned, FT_HPARAMS
from config import MOLECULENET_TASKS

import os
ENC = "figure_data/climb_v2_phase2/random_baseline_00/encoder"
TOK = "figure_data/_tokenizer"

TASKS = os.environ.get("LE_TASKS", "QM7 BACE Tox21 HIV").split()
TYPE = dict(MOLECULENET_TASKS)
FRACTIONS = [0.05, 0.10, 0.25, 0.50, 1.00]
FT_SEEDS = list(range(int(os.environ.get("LE_SEEDS", "3"))))
EPOCHS = FT_HPARAMS["epochs"]

CELLROOT = Path(os.environ.get("LE_CELLROOT", "figure_data/climb_v2_labeleff_v2_frac_e2e"))
CELLROOT.mkdir(parents=True, exist_ok=True)
OUT = Path("analysis/rigor"); OUT.mkdir(parents=True, exist_ok=True)
LONG = Path(os.environ.get("LE_LONG", str(OUT / "label_efficiency_fractions_e2e.csv")))
SUMM = Path(os.environ.get("LE_SUMM", str(OUT / "label_efficiency_fractions_e2e_summary.csv")))


def n_for(task: str, frac: float) -> int:
    tr_s, *_ = _load_moleculenet(task)
    n_full = len(tr_s)
    return n_full if frac >= 1.0 else max(1, round(frac * n_full))


def collect_cell(cell_dir: Path, arm, task, tt, frac, pct):
    """Read a fine-tuned cell's summary into tidy long rows (per fine-tune seed + MEAN/STD)."""
    f = cell_dir / "moleculenet" / "moleculenet_summary.csv"
    if not f.exists():
        return []
    d = pd.read_csv(f); d = d[d.dataset == task]
    prim = "roc_auc" if tt == "classification" else "rmse"
    out = []
    wanted = [(prim, "test"), (f"{prim}_train", "train")] + ([("nef1", "test")] if tt == "classification" else [])
    for mm, split in wanted:
        rows = d[d.main_metric == mm]
        for _, r in rows.iterrows():
            hs = str(r.head_seed)
            if hs in ("MEAN", "STD"):
                continue
            out.append(dict(arm=arm, task=task, task_type=tt, fraction=frac, pct=pct,
                            n_train=int(r.n_train), subsample_seed=None, head_seed=int(hs),
                            metric=("nef1" if mm == "nef1" else prim), split=split,
                            value=float(r.main_value)))
    return out


def main():
    rows = []
    if LONG.exists():
        rows = pd.read_csv(LONG).to_dict("records")
    t_start = time.time()
    for task in TASKS:
        tt = TYPE[task]
        for frac in FRACTIONS:
            pct = int(round(frac * 100))
            sub_seeds = [0] if frac >= 1.0 else [0, 1, 2]
            n = n_for(task, frac)
            for ss in sub_seeds:
                cell = f"e2e_{task}_f{pct:03d}_s{ss}"
                cell_dir = CELLROOT / cell
                marker = cell_dir / "verified.json"
                if marker.exists():
                    print(f"[skip] {cell} (done)", flush=True)

                    if not any(r for r in rows if r.get("task") == task and r.get("pct") == pct
                               and r.get("subsample_seed") in (ss, None) and r.get("arm") == "e2e"):
                        rows += [dict(r, subsample_seed=ss) for r in collect_cell(cell_dir, "e2e", task, tt, frac, pct)]
                    continue
                budget = None if frac >= 1.0 else n
                print(f"\n===== {cell}  budget={budget} (n={n}) ft_seeds={FT_SEEDS} epochs={EPOCHS} =====", flush=True)
                t0 = time.time()
                evaluate_finetuned(
                    encoder_path=ENC, tokenizer_path=TOK,
                    output_dir=str(cell_dir / "moleculenet"), seeds=FT_SEEDS,
                    datasets=[(task, tt)], train_subsample=budget, subsample_seed=ss,
                    epochs=EPOCHS, heartbeat_path=str(cell_dir / "heartbeat.json"),
                )
                new = collect_cell(cell_dir, "e2e", task, tt, frac, pct)
                for r in new:
                    r["subsample_seed"] = ss

                ok = (len([r for r in new if r["split"] == "test" and r["metric"] in ("roc_auc", "rmse")]) >= 1
                      and len([r for r in new if r["split"] == "train"]) >= 1
                      and all(np.isfinite(r["value"]) for r in new))
                if ok:
                    marker.write_text(json.dumps({"cell": cell, "n": n, "pct": pct,
                                                  "seconds": round(time.time() - t0, 1)}, indent=2))
                    print(f"[ok] {cell} in {(time.time()-t0)/60:.1f} min", flush=True)
                else:
                    print(f"[NOT VERIFIED] {cell}", flush=True)
                rows += new
                pd.DataFrame(rows).to_csv(LONG, index=False)

    df = pd.DataFrame(rows)
    df.to_csv(LONG, index=False)
    g = (df.groupby(["arm", "task", "task_type", "metric", "split", "fraction", "pct"], as_index=False)
           .agg(n_train=("n_train", "max"), mean=("value", "mean"),
                std=("value", "std"), n_cells=("value", "size")))
    g["std"] = g["std"].fillna(0.0)
    g.sort_values(["arm", "task", "metric", "split", "fraction"]).to_csv(SUMM, index=False)
    print(f"\nDONE in {(time.time()-t_start)/60:.1f} min -> {LONG} ({len(df)} rows), {SUMM} ({len(g)} points)", flush=True)


if __name__ == "__main__":
    main()
