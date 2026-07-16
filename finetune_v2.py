"""v2 end-to-end fine-tuning (secondary robustness check).

The PRIMARY protocol is frozen-featurizer eval (eval_v2) — that is how molecular
foundation models are actually deployed. This module unfreezes the encoder and
fine-tunes it end-to-end with a small head, purely to confirm the frozen-featurizer
conclusions are not an artifact of freezing. Kept deliberately small: the 3
single-task core datasets (BBBP, BACE, ESOL — no NaN labels, so no masking needed),
single seed, small LR, early stopping on val.

Usage:
    python finetune_v2.py --encoder <dir> --tokenizer <dir> --datasets BBBP BACE ESOL \
        --output_dir <out>
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn as nn

from eval_v2 import _load_moleculenet
from featurize_v2 import pool
from heads_v2 import compute_metric

FINETUNE_TASKS = ["BBBP", "BACE", "ESOL"]  # single-task, NaN-free
TASK_TYPE = {"BBBP": "classification", "BACE": "classification", "ESOL": "regression",
             "HIV": "classification"}  # HIV = harder/larger (~41k) ceiling-test task


class EncoderWithHead(nn.Module):
    def __init__(self, encoder, hidden_size: int):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = pool(out.last_hidden_state, attention_mask, "mean")
        return self.head(pooled)  # [B, 1]


def _batch_smiles(tokenizer, smiles: List[str], max_length: int, device):
    enc = tokenizer(smiles, truncation=True, max_length=max_length,
                    padding="longest", return_tensors="pt")
    return enc["input_ids"].to(device), enc["attention_mask"].to(device)


def finetune_one(encoder_path, tokenizer_path, ds_name, task_type, *, seed=0,
                 lr=2e-5, epochs=20, patience=5, batch_size=32, max_length=256):
    from transformers import ModernBertModel, PreTrainedTokenizerFast, set_seed
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)
    encoder = ModernBertModel.from_pretrained(
        encoder_path, attn_implementation="sdpa", reference_compile=False
    )
    model = EncoderWithHead(encoder, encoder.config.hidden_size).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    is_clf = task_type == "classification"
    loss_fn = nn.functional.binary_cross_entropy_with_logits if is_clf else nn.functional.mse_loss

    tr_s, tr_y, va_s, va_y, te_s, te_y = _load_moleculenet(ds_name)
    tr_y = tr_y.reshape(-1).astype(np.float32)
    va_y = va_y.reshape(-1).astype(np.float32)
    te_y = te_y.reshape(-1).astype(np.float32)

    best_val, best_state, wait = float("inf"), None, 0
    order = np.arange(len(tr_s))
    rng = np.random.default_rng(seed)
    for _ in range(epochs):
        model.train()
        rng.shuffle(order)
        for i in range(0, len(order), batch_size):
            idx = order[i:i + batch_size]
            ids, mask = _batch_smiles(tokenizer, [tr_s[j] for j in idx], max_length, device)
            y = torch.tensor(tr_y[idx], device=device).unsqueeze(1)
            opt.zero_grad()
            loss = loss_fn(model(ids, mask), y)
            loss.backward()
            opt.step()
        # validation
        model.eval()
        with torch.no_grad():
            vloss_tot, n = 0.0, 0
            for i in range(0, len(va_s), batch_size):
                ids, mask = _batch_smiles(tokenizer, va_s[i:i + batch_size], max_length, device)
                y = torch.tensor(va_y[i:i + batch_size], device=device).unsqueeze(1)
                vloss_tot += loss_fn(model(ids, mask), y).item() * len(y)
                n += len(y)
            vloss = vloss_tot / max(n, 1)
        if vloss < best_val - 1e-5:
            best_val, wait = vloss, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)

    # test
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(te_s), batch_size):
            ids, mask = _batch_smiles(tokenizer, te_s[i:i + batch_size], max_length, device)
            preds.append(model(ids, mask).cpu().numpy())
    preds = np.concatenate(preds, axis=0).reshape(-1)
    return compute_metric(preds, te_y, task_type)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--encoder", required=True)
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--datasets", nargs="+", default=FINETUNE_TASKS)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for ds in args.datasets:
        tt = TASK_TYPE.get(ds, "classification")
        t0 = time.time()
        metric = finetune_one(args.encoder, args.tokenizer, ds, tt, seed=args.seed)
        mm = "roc_auc" if tt == "classification" else "rmse"
        print(f"[finetune_v2] {ds}: {mm} = {metric:.4f}")
        rows.append({"dataset": ds, "task_type": tt, "main_metric": mm,
                     "seed": args.seed, "main_value": metric,
                     "elapsed_seconds": round(time.time() - t0, 1)})

    csv_path = out / "finetune_summary.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["dataset", "task_type", "main_metric", "seed",
                                          "main_value", "elapsed_seconds"])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    (out / "finetune_summary.json").write_text(
        json.dumps({r["dataset"]: r["main_value"] for r in rows}, indent=2))
    print(f"[finetune_v2] wrote {csv_path}")


if __name__ == "__main__":
    main()
