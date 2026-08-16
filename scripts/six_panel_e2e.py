"""Wave 3 of the six-panel migration (see notes/six-panel-migration.md).

End-to-end fine-tune the BEST-TWO CLIMB encoders (unsup_only, sup_only:dense) across the
label-efficiency FRACTION grid on the MoleculeNet panels {BACE, BBBP, Tox21, QM7}. This is the
"does training e2e after pre-training surpass the best CLIMB frozen model, and where?" crossover.

Scope rationale: the crossover can only exist where data is plentiful (Tox21 7.8k, QM7 6.8k);
MoleculeACE is entirely small-data (<=3.7k) so e2e won't overtake there, and its full-data e2e
already exists (unsup_8M_e2e / skip_dense_8M_e2e). CBS e2e is a separate follow-up (custom-task
fine-tune path). This driver reuses the PROVEN label_eff_fractions_e2e pattern verbatim, only
generalized to loop the two encoders and sync them from S3.

Runs ON THE BOX with the CLIMB venv. Idempotent per (encoder, task, fraction, subsample_seed)
cell via verified.json. Writes tidy long rows to analysis/rigor/six_panel_e2e.csv.
"""
from __future__ import annotations
import json, time, subprocess, sys, os
from pathlib import Path
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))          # repo root on path (script runs from scripts/)
os.chdir(_ROOT)                          # run from repo root (relative data paths)
import numpy as np
import pandas as pd
from rdkit import RDLogger; RDLogger.DisableLog("rdApp.*")

from eval_v2 import _load_moleculenet
from finetune_e2e_v2 import evaluate_finetuned, FT_HPARAMS
from config_v2 import MOLECULENET_TASKS_V2

S3B = "s3://climb-s3-bucket/experiments/climb_v2_phase2"
TOK = "figure_data/_tokenizer"
# best-two CLIMB arms -> their 8M encoder prefix
ENCODERS = {"unsup_only": "unsup_8M", "sup_only:dense": "skip_dense_8M"}
TASKS = ["BACE", "BBBP", "Tox21", "QM7"]
TYPE = dict(MOLECULENET_TASKS_V2)
FRACTIONS = [0.05, 0.10, 0.25, 0.50, 1.00]
FT_SEEDS = [0, 1, 2]
EPOCHS = FT_HPARAMS["epochs"]

ROOT = Path(".")
CELLROOT = Path("figure_data/six_panel_e2e")
CELLROOT.mkdir(parents=True, exist_ok=True)
OUT = Path("analysis/rigor"); OUT.mkdir(parents=True, exist_ok=True)
LONG = OUT / "six_panel_e2e.csv"


def sh(cmd):
    return subprocess.run(cmd, check=False)


def stage_encoder(prefix):
    enc = ROOT / "figure_data" / "climb_v2_phase2" / prefix / "encoder"
    if not (enc / "model.safetensors").exists():
        enc.mkdir(parents=True, exist_ok=True)
        sh(["aws", "s3", "sync", f"{S3B}/{prefix}/encoder", str(enc), "--only-show-errors"])
    return str(enc)


def n_for(task, frac):
    tr_s, *_ = _load_moleculenet(task)
    n_full = len(tr_s)
    return n_full if frac >= 1.0 else max(1, round(frac * n_full))


def collect_cell(cell_dir, arm, task, tt, frac, pct):
    f = cell_dir / "moleculenet" / "moleculenet_summary.csv"
    if not f.exists():
        return []
    d = pd.read_csv(f); d = d[d.dataset == task]
    prim = "roc_auc" if tt == "classification" else "rmse"
    out = []
    wanted = [(prim, "test"), (f"{prim}_train", "train")] + ([("nef1", "test")] if tt == "classification" else [])
    for mm, split in wanted:
        for _, r in d[d.main_metric == mm].iterrows():
            hs = str(r.head_seed)
            if hs in ("MEAN", "STD"):
                continue
            out.append(dict(arm=arm, task=task, task_type=tt, fraction=frac, pct=pct,
                            n_train=int(r.n_train), head_seed=int(hs),
                            metric=("nef1" if mm == "nef1" else prim), split=split,
                            value=float(r.main_value)))
    return out


def main():
    rows = pd.read_csv(LONG).to_dict("records") if LONG.exists() else []
    total = len(ENCODERS) * len(TASKS) * (1 + (len(FRACTIONS) - 1) * len(FT_SEEDS))  # full=1 sub-seed
    done = 0
    for arm, prefix in ENCODERS.items():
        enc = stage_encoder(prefix)
        if not Path(enc, "model.safetensors").exists():
            print(f"[w3] ERROR {arm}: encoder missing after sync", flush=True); continue
        for task in TASKS:
            tt = TYPE[task]
            for frac in FRACTIONS:
                pct = int(round(frac * 100))
                sub_seeds = [0] if frac >= 1.0 else [0, 1, 2]
                n = n_for(task, frac)
                for ss in sub_seeds:
                    cell = f"{prefix}_{task}_f{pct:03d}_s{ss}"
                    cell_dir = CELLROOT / arm.replace(":", "_") / cell
                    marker = cell_dir / "verified.json"
                    if marker.exists():
                        done += 1
                        if not any(r for r in rows if r.get("arm") == arm and r.get("task") == task
                                   and r.get("pct") == pct):
                            rows += [dict(r, subsample_seed=ss) for r in collect_cell(cell_dir, arm, task, tt, frac, pct)]
                        continue
                    budget = None if frac >= 1.0 else n
                    print(f"\n===== {arm} {cell} budget={budget} (n={n}) epochs={EPOCHS} =====", flush=True)
                    t0 = time.time()
                    evaluate_finetuned(
                        encoder_path=enc, tokenizer_path=TOK,
                        output_dir=str(cell_dir / "moleculenet"), seeds=FT_SEEDS,
                        datasets=[(task, tt)], train_subsample=budget, subsample_seed=ss,
                        epochs=EPOCHS, heartbeat_path=str(cell_dir / "heartbeat.json"),
                    )
                    new = collect_cell(cell_dir, arm, task, tt, frac, pct)
                    for r in new:
                        r["subsample_seed"] = ss
                    ok = (any(r["split"] == "test" and r["metric"] in ("roc_auc", "rmse") for r in new)
                          and any(r["split"] == "train" for r in new)
                          and all(np.isfinite(r["value"]) for r in new))
                    if ok:
                        marker.write_text(json.dumps({"cell": cell, "n": n, "pct": pct,
                                                      "seconds": round(time.time() - t0, 1)}))
                        rows += new
                        pd.DataFrame(rows).to_csv(LONG, index=False)
                        done += 1
                        print(f"[w3 ok] {cell} in {(time.time()-t0)/60:.1f} min", flush=True)
                        # sync partial results to S3 as we go
                        sh(["aws", "s3", "cp", str(LONG), "s3://climb-s3-bucket/experiments/six_panel/six_panel_e2e.csv", "--only-show-errors"])
                    else:
                        print(f"[w3 FAIL] {cell} — left unverified", flush=True)
    pd.DataFrame(rows).to_csv(LONG, index=False)
    print(f"[w3] DONE {done}/{total} cells", flush=True)
    if done >= total:
        Path("figure_data/SIX_PANEL_W3_DONE").write_text("all best-two e2e fraction cells done\n")
        print("[w3] SIX_PANEL_W3_DONE written", flush=True)


if __name__ == "__main__":
    main()
