import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); os.chdir(ROOT)

from finetune import finetune_predict
from evaluate_benchmarks import load_task, task_list, reg_metrics, clf_metrics, _rmse


def _flush(out_dir, tasks, rows, pred_rows, track, model, suffix, seeds, encoder_path):

    with (out_dir / "results.csv").open("w", newline="") as f:
        w = csv.writer(f); w.writerow(["task", "seed", "subset", "metric", "value", "n_test"]); w.writerows(rows)
    with (out_dir / "test_predictions.csv").open("w", newline="") as f:
        w = csv.writer(f); w.writerow(["task", "seed", "test_index", "smiles", "y_pred"]); w.writerows(pred_rows)
    n_done = len({r[0] for r in pred_rows})
    vj = out_dir / "verified.json"
    if n_done == len(tasks):
        vj.write_text(json.dumps({"track": track, "model": f"{model}{suffix}", "mode": "e2e",
                                  "seeds": seeds, "n_tasks": n_done,
                                  "encoder": str(encoder_path)}))
    elif vj.exists():
        vj.unlink()
    return n_done


def run(track, model, encoder_path, tokenizer_path, seeds, lr=None, epochs=None, patience=None,
        only_tasks=None, suffix="_e2e"):
    from transformers import PreTrainedTokenizerFast
    tok = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)
    out_dir = ROOT / "figure_data" / "chemeleon_suite" / track / f"{model}{suffix}"
    out_dir.mkdir(parents=True, exist_ok=True)
    fp_kw = {k: v for k, v in (("lr", lr), ("epochs", epochs), ("patience", patience)) if v is not None}
    tasks = task_list(track)
    if only_tasks:
        tasks = [t for t in tasks if t in only_tasks]
    rows, pred_rows = [], []
    for task in tasks:
        smi, y, split, cliff, ttype = load_task(track, task)
        idx = np.arange(len(smi))
        tr = idx[[s == "train" for s in split]]
        te = idx[[s == "test" for s in split]]


        if TRAIN_FRACTION is not None and TRAIN_FRACTION < 1.0:
            n_keep = max(1, int(round(TRAIN_FRACTION * len(tr))))
            if n_keep < len(tr):
                tr = np.sort(np.random.default_rng(SUBSAMPLE_SEED).choice(tr, size=n_keep, replace=False))
        te_smi = [smi[i] for i in te]
        yte = y[te]
        has_labels = bool(np.isfinite(yte).any())
        for seed in seeds:
            rng = np.random.default_rng(seed)
            perm = rng.permutation(len(tr))
            n_va = max(1, int(0.1 * len(tr)))
            va = tr[perm[:n_va]]; trs = tr[perm[n_va:]]
            (pred,) = finetune_predict(
                encoder_path, tok, [smi[i] for i in trs], y[trs], [smi[i] for i in va], y[va],
                [te_smi], ttype, seed=seed, **fp_kw)
            pred = np.asarray(pred, dtype=np.float64)
            if ttype == "regression":
                ylo, yhi = float(np.nanmin(y[trs])), float(np.nanmax(y[trs])); m = 0.25 * (yhi - ylo + 1e-9)
                pred = np.clip(pred, ylo - m, yhi + m)
            pv = pred.ravel()
            for i in range(len(te_smi)):
                pred_rows.append([task, seed, i, te_smi[i], float(pv[i])])
            if has_labels:
                mets = reg_metrics(yte, pred) if ttype == "regression" else clf_metrics(yte, pred)
                for k, v in mets.items():
                    rows.append([task, seed, "overall", k, v, len(te)])
                if track == "moleculeace":
                    ct = cliff[te]
                    if ct.any():
                        rows.append([task, seed, "cliff", "rmse", _rmse(yte[ct], pred[ct]), int(ct.sum())])
                    if (~ct).any():
                        rows.append([task, seed, "noncliff", "rmse", _rmse(yte[~ct], pred[~ct]), int((~ct).sum())])
            print(f"[e2e] {track}/{model} {task} seed{seed}: done ({len(te)} test)", flush=True)
        _flush(out_dir, tasks, rows, pred_rows, track, model, suffix, seeds, encoder_path)

    n_done = _flush(out_dir, tasks, rows, pred_rows, track, model, suffix, seeds, encoder_path)
    print(f"[e2e] wrote {out_dir} ({n_done}/{len(tasks)} tasks)", flush=True)


TRAIN_FRACTION = None
SUBSAMPLE_SEED = 0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--track", required=True, choices=["moleculeace", "polaris"])
    p.add_argument("--model", required=True, help="output label (unsup_8M, skip_dense_8M, no_pretrain_e2e)")
    p.add_argument("--encoder", required=True, help="encoder weights to fine-tune FROM")
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 117, 709])
    p.add_argument("--train_fraction", type=float, default=None,
                   help="Label-efficiency sweep: keep this FRACTION of each task's train split.")
    p.add_argument("--subsample_seed", type=int, default=0)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--patience", type=int, default=None)
    p.add_argument("--only_tasks", default=None, help="comma-separated task subset")
    p.add_argument("--suffix", default="_e2e", help="output dir suffix (e.g. _e2e_tuned)")
    a = p.parse_args()
    global TRAIN_FRACTION, SUBSAMPLE_SEED
    TRAIN_FRACTION, SUBSAMPLE_SEED = a.train_fraction, a.subsample_seed
    only = a.only_tasks.split(",") if a.only_tasks else None
    run(a.track, a.model, a.encoder, a.tokenizer, a.seeds, lr=a.lr, epochs=a.epochs,
        patience=a.patience, only_tasks=only, suffix=a.suffix)


if __name__ == "__main__":
    main()
