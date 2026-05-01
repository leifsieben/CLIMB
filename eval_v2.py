"""v2 frozen-encoder MoleculeNet evaluation.

For each (encoder, dataset) pair:
  1. Load the encoder + tokenizer.
  2. Forward-pass every train/val/test molecule once (cache CLS hidden state).
  3. Train a small head from the cached features 3x (different seeds), pick the
     best by val loss with early stopping, evaluate on test.
  4. Report mean ± sd across the 3 head seeds.

This runs the 9 pre-registered MoleculeNet tasks from config_v2.MOLECULENET_TASKS_V2.

Output: <output_dir>/moleculenet_summary.csv with one row per (dataset, head_seed)
plus an aggregate mean/sd row per dataset.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import (
    PreTrainedTokenizerFast,
    RobertaConfig,
    RobertaModel,
    set_seed,
)

from config_v2 import MOLECULENET_TASKS_V2


# ---------- DeepChem dataset loaders ----------

def _load_moleculenet(name: str) -> Tuple[List[str], np.ndarray, List[str], np.ndarray, List[str], np.ndarray]:
    """Returns (train_smiles, train_y, val_smiles, val_y, test_smiles, test_y)."""
    import deepchem as dc

    loaders = {
        "BACE": dc.molnet.load_bace_classification,
        "BBBP": dc.molnet.load_bbbp,
        "ClinTox": dc.molnet.load_clintox,
        "Tox21": dc.molnet.load_tox21,
        "ToxCast": dc.molnet.load_toxcast,
        "SIDER": dc.molnet.load_sider,
        "ESOL": dc.molnet.load_delaney,
        "FreeSolv": dc.molnet.load_freesolv,
        "Lipophilicity": dc.molnet.load_lipo,
    }
    if name not in loaders:
        raise ValueError(f"Unknown MoleculeNet dataset: {name}")
    tasks, datasets, _ = loaders[name](featurizer="Raw", splitter="scaffold")
    train_ds, val_ds, test_ds = datasets
    return (
        list(train_ds.ids), np.asarray(train_ds.y, dtype=np.float32),
        list(val_ds.ids), np.asarray(val_ds.y, dtype=np.float32),
        list(test_ds.ids), np.asarray(test_ds.y, dtype=np.float32),
    )


# ---------- tokenize + forward-pass-cache ----------

def _tokenize(tokenizer, smiles: List[str], max_length: int) -> Tuple[torch.Tensor, torch.Tensor]:
    enc = tokenizer(
        smiles, padding="max_length", truncation=True, max_length=max_length, return_tensors="pt",
    )
    return enc["input_ids"], enc["attention_mask"]


def _forward_features(
    encoder: RobertaModel, input_ids: torch.Tensor, attention_mask: torch.Tensor,
    device: torch.device, batch_size: int = 64,
) -> torch.Tensor:
    encoder.eval()
    feats = []
    with torch.no_grad():
        for i in range(0, input_ids.shape[0], batch_size):
            ids = input_ids[i : i + batch_size].to(device)
            mask = attention_mask[i : i + batch_size].to(device)
            out = encoder(input_ids=ids, attention_mask=mask)
            cls = out.last_hidden_state[:, 0, :]  # [B, H]
            feats.append(cls.detach().cpu())
    return torch.cat(feats, dim=0)


# ---------- head training ----------

def _train_head(
    train_x: torch.Tensor, train_y: np.ndarray,
    val_x: torch.Tensor, val_y: np.ndarray,
    test_x: torch.Tensor, test_y: np.ndarray,
    task_type: str,
    n_outputs: int,
    seed: int,
    num_epochs: int = 50,
    early_stopping_patience: int = 10,
    lr: float = 1e-3,
    batch_size: int = 64,
) -> Tuple[float, np.ndarray]:
    """Train a 1-layer head with early stopping. Return (test_metric, test_predictions)."""
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hidden = train_x.shape[1]
    head = nn.Linear(hidden, n_outputs).to(device)
    optimizer = torch.optim.Adam(head.parameters(), lr=lr)

    is_clf = task_type == "classification"
    train_y_t = torch.from_numpy(train_y).to(torch.float32)
    val_y_t = torch.from_numpy(val_y).to(torch.float32)
    test_y_t = torch.from_numpy(test_y).to(torch.float32)

    if train_y_t.ndim == 1:
        train_y_t = train_y_t.unsqueeze(1)
        val_y_t = val_y_t.unsqueeze(1)
        test_y_t = test_y_t.unsqueeze(1)

    train_ds = TensorDataset(train_x, train_y_t)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    best_val = float("inf")
    best_state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
    patience = 0

    for epoch in range(num_epochs):
        head.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            preds = head(x)
            valid = ~torch.isnan(y)
            if valid.sum() == 0:
                continue
            if is_clf:
                loss = nn.functional.binary_cross_entropy_with_logits(preds[valid], y[valid])
            else:
                loss = nn.functional.mse_loss(preds[valid], y[valid])
            loss.backward()
            optimizer.step()

        # Val
        head.eval()
        with torch.no_grad():
            v = head(val_x.to(device))
            valid = ~torch.isnan(val_y_t.to(device))
            if valid.sum() == 0:
                val_loss = float("inf")
            elif is_clf:
                val_loss = nn.functional.binary_cross_entropy_with_logits(
                    v[valid], val_y_t.to(device)[valid]
                ).item()
            else:
                val_loss = nn.functional.mse_loss(
                    v[valid], val_y_t.to(device)[valid]
                ).item()

        if val_loss < best_val - 1e-5:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= early_stopping_patience:
                break

    # Test
    head.load_state_dict(best_state)
    head.to(device)
    head.eval()
    with torch.no_grad():
        test_preds = head(test_x.to(device)).detach().cpu().numpy()

    metric = _metric(test_preds, test_y, task_type)
    return metric, test_preds


def _metric(preds: np.ndarray, labels: np.ndarray, task_type: str) -> float:
    if task_type == "classification":
        from sklearn.metrics import roc_auc_score
        # multitask: average AUC across columns where both classes present and non-NaN
        if labels.ndim == 1:
            mask = ~np.isnan(labels)
            if mask.sum() == 0 or len(np.unique(labels[mask])) < 2:
                return float("nan")
            return float(roc_auc_score(labels[mask], preds[mask]))
        scores = []
        for j in range(labels.shape[1]):
            mask = ~np.isnan(labels[:, j])
            if mask.sum() == 0 or len(np.unique(labels[mask, j])) < 2:
                continue
            scores.append(roc_auc_score(labels[mask, j], preds[mask, j]))
        return float(np.mean(scores)) if scores else float("nan")
    else:
        # regression: RMSE averaged over columns
        if labels.ndim == 1:
            mask = ~np.isnan(labels)
            return float(np.sqrt(np.mean((preds[mask] - labels[mask]) ** 2)))
        rmses = []
        for j in range(labels.shape[1]):
            mask = ~np.isnan(labels[:, j])
            if mask.sum() == 0:
                continue
            rmses.append(np.sqrt(np.mean((preds[mask, j] - labels[mask, j]) ** 2)))
        return float(np.mean(rmses)) if rmses else float("nan")


# ---------- main ----------

def evaluate(
    encoder_path: str,
    tokenizer_path: str,
    output_dir: str,
    head_seeds: List[int],
    datasets: List[Tuple[str, str]],
    max_length: int = 512,
) -> Path:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)
    encoder = RobertaModel.from_pretrained(encoder_path).to(device)
    encoder.eval()

    rows = []

    for ds_name, task_type in datasets:
        t0 = time.time()
        print(f"[eval_v2] {ds_name} ({task_type}) ...")
        try:
            tr_s, tr_y, va_s, va_y, te_s, te_y = _load_moleculenet(ds_name)
        except Exception as exc:
            print(f"  failed to load: {exc}")
            continue

        # Tokenize + forward once (cached).
        tr_ids, tr_mask = _tokenize(tokenizer, tr_s, max_length)
        va_ids, va_mask = _tokenize(tokenizer, va_s, max_length)
        te_ids, te_mask = _tokenize(tokenizer, te_s, max_length)
        tr_x = _forward_features(encoder, tr_ids, tr_mask, device)
        va_x = _forward_features(encoder, va_ids, va_mask, device)
        te_x = _forward_features(encoder, te_ids, te_mask, device)

        n_outputs = tr_y.shape[1] if tr_y.ndim > 1 else 1

        per_seed = []
        for seed in head_seeds:
            metric, _ = _train_head(
                tr_x, tr_y, va_x, va_y, te_x, te_y,
                task_type=task_type, n_outputs=n_outputs, seed=seed,
            )
            per_seed.append(metric)
            rows.append({
                "dataset": ds_name,
                "task_type": task_type,
                "main_metric": "roc_auc" if task_type == "classification" else "rmse",
                "head_seed": seed,
                "main_value": metric,
                "elapsed_seconds": time.time() - t0,
            })

        per_seed = np.array(per_seed, dtype=np.float64)
        rows.append({
            "dataset": ds_name,
            "task_type": task_type,
            "main_metric": "roc_auc" if task_type == "classification" else "rmse",
            "head_seed": "MEAN",
            "main_value": float(np.nanmean(per_seed)),
            "elapsed_seconds": time.time() - t0,
        })
        rows.append({
            "dataset": ds_name,
            "task_type": task_type,
            "main_metric": "roc_auc" if task_type == "classification" else "rmse",
            "head_seed": "STD",
            "main_value": float(np.nanstd(per_seed)),
            "elapsed_seconds": time.time() - t0,
        })

    summary_path = out / "moleculenet_summary.csv"
    with summary_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["dataset", "task_type", "main_metric", "head_seed", "main_value", "elapsed_seconds"])
        w.writeheader()
        for r in rows:
            w.writerow(r)

    suite_path = out / "suite_summary.json"
    summary = {}
    for r in rows:
        if r["head_seed"] in ("MEAN", "STD"):
            key = f"{r['dataset']}_{r['head_seed']}"
            summary[key] = r["main_value"]
    suite_path.write_text(json.dumps(summary, indent=2))

    print(f"[eval_v2] wrote {summary_path} and {suite_path}")
    return summary_path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--encoder", required=True, help="Path to a saved RobertaModel encoder")
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--head_seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--datasets", nargs="+", default=None,
                   help="Override the 9-task subset; pass dataset names")
    args = p.parse_args()

    if args.datasets is not None:
        # Look up task type from MOLECULENET_TASKS_V2
        type_map = dict(MOLECULENET_TASKS_V2)
        ds_list = [(n, type_map.get(n, "classification")) for n in args.datasets]
    else:
        ds_list = MOLECULENET_TASKS_V2

    evaluate(
        encoder_path=args.encoder,
        tokenizer_path=args.tokenizer,
        output_dir=args.output_dir,
        head_seeds=args.head_seeds,
        datasets=ds_list,
    )


if __name__ == "__main__":
    main()
