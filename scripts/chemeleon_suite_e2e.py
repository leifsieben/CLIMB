"""END-TO-END fine-tune runner for the CheMeleon suite (our ModernBERT encoders). Fine-tunes the encoder
on each task's fixed train split and predicts the held-out test set — same artifacts as the frozen runner
(results.csv for MoleculeACE incl cliff/non-cliff; test_predictions.csv for Polaris predict->evaluate()).

3 e2e arms x 3 seeds (per METHODOLOGY): unsup_8M, skip_dense_8M (best supervised), no_pretrain_e2e
(finetune from random_baseline_00). Reuses finetune_e2e_v2.finetune_predict (the exact ModernBERT
end-to-end recipe used elsewhere in the paper). Output dirs are suffixed to never collide with frozen:
  figure_data/chemeleon_suite/<track>/<model>_e2e/

Run from repo root with ~/venvs/climb/bin/python (torch+transformers). GPU strongly recommended.
Usage: python scripts/chemeleon_suite_e2e.py --track moleculeace --model unsup_8M \
         --encoder figure_data/climb_v2_phase2/unsup_8M/encoder --tokenizer figure_data/_tokenizer --seeds 42 117 709
"""
import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); os.chdir(ROOT)

from finetune_e2e_v2 import finetune_predict  # noqa: E402
from chemeleon_suite_run import load_task, task_list, reg_metrics, clf_metrics, _rmse  # noqa: E402


def _flush(out_dir, tasks, rows, pred_rows, track, model, suffix, seeds, encoder_path):
    """Write the dir's artifacts from what is finished SO FAR, and return the completed-task count.

    Called after every task, not only at the end. The original wrote both CSVs once the whole
    grid was done, so an interruption at task 57 of 58 -- hours of GPU -- left nothing on disk
    at all. Rewriting the two files each task is milliseconds against a fine-tune.

    verified.json is written ONLY at full coverage, and it names the dir it actually is. It used
    to hardcode f"{model}_e2e" regardless of --suffix, so every replicate dir would have claimed
    to be the base: unsup_8M_e2e_s1 stamped "model": "unsup_8M_e2e". The encoder is recorded too,
    because for these arms the replicate axis IS the encoder (pretraining seed) and nothing else
    in the dir says which one produced it.
    """
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
        vj.unlink()          # never let a stale full-coverage claim outlive a partial rerun
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
        # Label-efficiency sweep (SI fig e): reduce the TRAIN budget only, on this benchmark's own
        # native split, with the same per-task fraction and without-replacement draw the frozen
        # runner and eval_v2._subsample_train use -- so the e2e arm's x-axis means exactly what the
        # frozen arms' does. Test is never touched.
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
            if ttype == "regression":  # bound OOD extrapolation, uniform with the frozen runner
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
