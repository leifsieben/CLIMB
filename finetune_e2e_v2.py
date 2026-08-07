"""End-to-end fine-tuned MoleculeNet evaluation, scored under the *frozen-arm* protocol.

`eval_v2` is the primary protocol (freeze the encoder, fit a small head on pooled
embeddings). `finetune_v2` unfreezes the encoder but reports a single scalar per task
and cannot produce anything a figure can consume: no per-molecule predictions, no
scaffold-CV scheme, no NEF1%, and only the three NaN-free single-task datasets.

This module closes that gap. It runs **end-to-end fine-tuning** (encoder unfrozen +
linear head) but emits the *identical artifacts* the frozen arms emit — the same
`moleculenet_summary.csv` schema, the same `suite_summary.json` keys, and a
`test_predictions.csv` whose `(dataset, mol_index, output_index)` rows pair
molecule-for-molecule with every frozen run. That pairing is the whole point: the
headline comparison table (README §8.1) merges arms on exactly those three columns,
so an end-to-end arm that re-implements its own splits is worthless for significance
testing no matter how good its numbers are.

Everything split-shaped is therefore *imported* from `eval_v2` rather than
reimplemented — `_load_moleculenet`, `_load_moleculenet_full`, `_scaffold_kfold_indices`
and `_dump_test_predictions`. The fold partition, the 10% validation carve-out and its
RNG draw order are reproduced call-for-call so the fold assignment is bit-identical to
the frozen runs.

What it adds over `finetune_v2`:
  - **multi-output tasks with sparse labels** (Tox21: 12 columns, mostly NaN) via
    per-column masked BCE — the same masking `heads_v2._TorchHead` uses, so the two
    arms optimize the same objective on the same entries;
  - **regression tasks** (ESOL, QM7, Lipophilicity) — `finetune_v2.TASK_TYPE` silently
    defaulted unknown datasets to classification, which would have trained QM7 with
    BCE on unbounded targets;
  - **NEF1%** (top-1% enrichment) alongside ROC-AUC, so HIV's headline metric matches
    `TASKS["HIV"]["suite_key"] == "HIV_nef1"`;
  - **both split schemes** — single DeepChem scaffold hold-out and 5-fold scaffold CV.

Regression targets ARE standardized here (fit on the train split, predictions unscaled back
to native units), via the same eval_v2._fit_target_scaler the frozen arm uses. This was added
after `_load_moleculenet` switched to `transformers=[]` (native units): its earlier behavior
returned mean-0/sd-1 targets, so the e2e head trained fine without scaling, but on native units
QM7's ~-1531 offset made the MSE head collapse to predicting ~0 (RMSE ~1500, worse than the
mean). Unscaling on return keeps the reported RMSE in native units, comparable to the frozen arm.

Usage:
    python finetune_e2e_v2.py --encoder <dir> --tokenizer <dir> --output_dir <out> \
        --datasets ESOL BBBP BACE Tox21 QM7 HIV --seeds 0 1 2
    python finetune_e2e_v2.py ... --cv_folds 5 --seeds 0
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

from config_v2 import MOLECULENET_TASKS_V2
from eval_v2 import (_dump_test_predictions, _fit_target_scaler, _load_moleculenet,
                     _load_moleculenet_full, _scale_targets, _scaffold_kfold_indices,
                     _subsample_train, _unscale_preds)
from featurize_v2 import pool
from heads_v2 import compute_metric, compute_nef

# Fine-tuning hyperparameters. Inherited verbatim from finetune_v2 so the end-to-end arm
# is not silently a differently-tuned model; recorded into metadata.json by the runner.
FT_HPARAMS = {"lr": 2e-5, "weight_decay": 0.01, "epochs": 20, "patience": 5,
              "batch_size": 32, "max_length": 256}


class EncoderWithHead(nn.Module):
    """Encoder + linear readout over masked-mean-pooled token states.

    Same pooling as the frozen arm (`featurize_v2.pool(..., "mean")`), so the only
    difference between the two arms is whether the encoder receives gradient.
    `n_outputs > 1` for multitask sets (Tox21); the head is one shared linear layer,
    matching `heads_v2` which also emits all columns from one readout.
    """

    def __init__(self, encoder, hidden_size: int, n_outputs: int):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(hidden_size, n_outputs)

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = pool(out.last_hidden_state, attention_mask, "mean")
        return self.head(pooled.float())  # [B, n_outputs]


class _TokenizedSet:
    """SMILES tokenized ONCE, padded per batch.

    HIV is ~41k molecules and is seen up to 20 times; re-tokenizing every epoch made
    the CPU the bottleneck on a g5.2xlarge whose vCPUs are also shared with the
    dataloader. Tokenizing up front and padding per batch is a pure speed change —
    identical ids reach the model either way.
    """

    def __init__(self, tokenizer, smiles: Sequence[str], max_length: int):
        enc = tokenizer(list(smiles), truncation=True, max_length=max_length)
        self.ids = enc["input_ids"]
        self.pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    def __len__(self):
        return len(self.ids)

    def batch(self, idx: Sequence[int], device) -> Tuple[torch.Tensor, torch.Tensor]:
        rows = [self.ids[i] for i in idx]
        width = max(len(r) for r in rows)
        ids = np.full((len(rows), width), self.pad_id, dtype=np.int64)
        mask = np.zeros((len(rows), width), dtype=np.int64)
        for r, row in enumerate(rows):
            ids[r, :len(row)] = row
            mask[r, :len(row)] = 1
        return (torch.from_numpy(ids).to(device), torch.from_numpy(mask).to(device))


def _masked_loss(logits: torch.Tensor, y: torch.Tensor, is_clf: bool):
    """Loss over non-NaN label entries only.

    Tox21 labels are ~17% populated across 12 columns; a naive loss over the dense
    tensor would backpropagate NaN through the whole batch. This is byte-for-byte the
    masking rule in `heads_v2._TorchHead.fit`, so the frozen and fine-tuned arms
    optimize the same quantity on the same entries.
    """
    valid = ~torch.isnan(y)
    if not bool(valid.any()):
        return None
    fn = (nn.functional.binary_cross_entropy_with_logits if is_clf
          else nn.functional.mse_loss)
    return fn(logits[valid], y[valid])


def finetune_predict(
    encoder_path: str,
    tokenizer,
    tr_s: Sequence[str], tr_y: np.ndarray,
    va_s: Sequence[str], va_y: np.ndarray,
    predict_sets: Sequence[Sequence[str]],
    task_type: str,
    *,
    seed: int = 0,
    lr: float = FT_HPARAMS["lr"],
    weight_decay: float = FT_HPARAMS["weight_decay"],
    epochs: int = FT_HPARAMS["epochs"],
    patience: int = FT_HPARAMS["patience"],
    batch_size: int = FT_HPARAMS["batch_size"],
    max_length: int = FT_HPARAMS["max_length"],
    bf16: bool = True,
    progress_cb=None,
) -> List[np.ndarray]:
    """Fine-tune a fresh copy of `encoder_path` on (tr, va) and predict every set in
    `predict_sets`. Returns one [N, n_outputs] array per requested set.

    A fresh encoder is loaded per call on purpose: each head seed / CV fold must start
    from the SAME initial weights, and reusing an in-memory model would leak the
    previous fold's training into the next one.

    Outputs are raw logits for classification (never sigmoid), matching
    `heads_v2._TorchHead.predict`. ROC-AUC, NEF1% and DeLong are all rank-based, so
    logits and probabilities score identically — but mixing the two in one
    `test_predictions.csv` column would make the dumped values incomparable.
    """
    from transformers import ModernBertModel, set_seed

    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = ModernBertModel.from_pretrained(
        encoder_path, attn_implementation="sdpa", reference_compile=False
    )
    tr_y = np.asarray(tr_y, dtype=np.float32)
    va_y = np.asarray(va_y, dtype=np.float32)
    if tr_y.ndim == 1:
        tr_y = tr_y[:, None]
    if va_y.ndim == 1:
        va_y = va_y[:, None]
    n_outputs = tr_y.shape[1]

    # Standardize regression targets (fit on TRAIN only), exactly as the frozen arm does via
    # eval_v2._fit_target_scaler. This is NOT optional: _load_moleculenet now returns NATIVE-unit
    # targets (transformers=[]), so QM7's ~-1531 mean is fed raw to an MSE head that cannot reach
    # that offset in a few epochs and collapses to predicting ~0 (RMSE ~1500, worse than the mean).
    # The docstring's old "molnet loaders pre-normalize" assumption no longer holds. Predictions are
    # unscaled back to native units before returning, so the reported RMSE stays comparable to the
    # frozen arm and the rest of the paper (ysc is None for classification -> a no-op there).
    ysc = _fit_target_scaler(tr_y, task_type)
    if ysc is not None:
        tr_y = _scale_targets(tr_y, ysc).astype(np.float32)
        va_y = _scale_targets(va_y, ysc).astype(np.float32)

    model = EncoderWithHead(encoder, encoder.config.hidden_size, n_outputs).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    is_clf = task_type == "classification"

    # bf16 autocast on the A10G: ~2x throughput at no accuracy cost for a 41M encoder,
    # and it is the precision the models were pretrained in (TrainingConfigV2.bf16).
    use_amp = bool(bf16 and device.type == "cuda" and torch.cuda.is_bf16_supported())
    amp = (torch.autocast(device_type="cuda", dtype=torch.bfloat16) if use_amp
           else torch.autocast(device_type="cpu", enabled=False))

    tr_tok = _TokenizedSet(tokenizer, tr_s, max_length)
    va_tok = _TokenizedSet(tokenizer, va_s, max_length)
    tr_y_t = torch.from_numpy(tr_y).to(device)
    va_y_t = torch.from_numpy(va_y).to(device)

    def _eval_loss(tok, y_t) -> float:
        model.eval()
        tot, n = 0.0, 0
        with torch.no_grad():
            for i in range(0, len(tok), batch_size):
                sl = list(range(i, min(i + batch_size, len(tok))))
                ids, mask = tok.batch(sl, device)
                with amp:
                    logits = model(ids, mask)
                loss = _masked_loss(logits.float(), y_t[sl], is_clf)
                if loss is not None:
                    tot += float(loss) * len(sl)
                    n += len(sl)
        return tot / max(n, 1)

    best_val, best_state, wait = float("inf"), None, 0
    order = np.arange(len(tr_tok))
    rng = np.random.default_rng(seed)
    for ep in range(epochs):
        model.train()
        rng.shuffle(order)
        for i in range(0, len(order), batch_size):
            idx = order[i:i + batch_size]
            ids, mask = tr_tok.batch(idx, device)
            with amp:
                logits = model(ids, mask)
            loss = _masked_loss(logits.float(), tr_y_t[idx], is_clf)
            if loss is None:      # an all-NaN Tox21 batch contributes no gradient
                continue
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        vloss = _eval_loss(va_tok, va_y_t)
        if progress_cb is not None:
            progress_cb(len(order), ep + 1, vloss)
        if vloss < best_val - 1e-5:
            best_val, wait = vloss, 0
            best_state = {k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)

    outs = []
    model.eval()
    with torch.no_grad():
        for smiles in predict_sets:
            tok = _TokenizedSet(tokenizer, smiles, max_length)
            chunks = []
            for i in range(0, len(tok), batch_size):
                sl = list(range(i, min(i + batch_size, len(tok))))
                ids, mask = tok.batch(sl, device)
                with amp:
                    logits = model(ids, mask)
                chunks.append(logits.float().cpu().numpy())
            outs.append(np.concatenate(chunks, axis=0) if chunks
                        else np.zeros((0, n_outputs), dtype=np.float32))
    del model, encoder
    if device.type == "cuda":
        torch.cuda.empty_cache()
    # Back to native units for regression, so the caller's compute_metric(pred, te_y) is a
    # native-unit RMSE matching the frozen arm (no-op when ysc is None, i.e. classification logits).
    if ysc is not None:
        outs = [_unscale_preds(o, ysc) for o in outs]
    return outs


def evaluate_finetuned(
    encoder_path: str,
    tokenizer_path: str,
    output_dir: str,
    seeds: List[int],
    datasets: List[Tuple[str, str]],
    *,
    max_length: int = FT_HPARAMS["max_length"],
    lr: float = FT_HPARAMS["lr"],
    epochs: int = FT_HPARAMS["epochs"],
    patience: int = FT_HPARAMS["patience"],
    batch_size: int = FT_HPARAMS["batch_size"],
    bf16: bool = True,
    train_subsample: Optional[int] = None,
    subsample_seed: int = 0,
    cv_folds: Optional[int] = None,
    heartbeat_path: Optional[str] = None,
) -> Path:
    """End-to-end analogue of `eval_v2.evaluate`, writing the same three artifacts.

    Deliberately mirrors `eval_v2.evaluate`'s control flow (including the CV RNG draw
    order) rather than sharing a refactored core: the frozen path is the protocol of
    record for every published bar and must not be touched to add this arm.

    `seeds` plays the role `head_seeds` plays in the frozen arm — the hold-out MEAN/STD
    is over seeds, the CV MEAN/STD is over folds with per-fold predictions averaged
    over seeds.

    `train_subsample` is the label-efficiency knob (Fig B1p1) and reuses `eval_v2`'s own
    `_subsample_train`, so at a given (n, subsample_seed) the fine-tuned arm trains on
    the *same molecules* the frozen arms did — otherwise the two label-efficiency curves
    would differ by their draw as well as by frozen-vs-fine-tuned. It subsamples TRAIN
    only: the validation set used for early stopping stays full size, again matching the
    frozen path, which passes the untouched `va_x` to the head.
    """
    from transformers import PreTrainedTokenizerFast

    if cv_folds and train_subsample:
        # The CV branch has no train/val/test hold-out to subsample and no consumer that
        # wants one; silently ignoring the flag would report a full-data number under a
        # small-n directory name.
        raise ValueError("train_subsample is only defined for the hold-out scheme, not CV")

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    # _dump_test_predictions appends; clear any prior run so re-running cannot double rows.
    (out / "test_predictions.csv").unlink(missing_ok=True)
    tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)

    state = {"samples": 0}

    def _beat(n_samples, epoch, vloss, ctx=""):
        """Monotone progress counter for the throughput watchdog. Completion is judged
        from this achieved work, never from an output file existing."""
        state["samples"] += int(n_samples)
        if heartbeat_path:
            Path(heartbeat_path).write_text(json.dumps({
                "updated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "samples_seen": state["samples"], "context": ctx,
                "epoch": int(epoch), "val_loss": float(vloss),
            }))

    def _row(ds_name, task_type, main_metric, tag, n_train, val, t0):
        # Same fieldnames as eval_v2 so every existing consumer parses this file unchanged.
        return {
            "dataset": ds_name, "task_type": task_type, "featurizer": "encoder_finetune",
            "pool": "mean", "standardize": "none", "head": "linear_e2e",
            "main_metric": main_metric, "head_seed": tag, "n_train": n_train,
            "main_value": val, "elapsed_seconds": round(time.time() - t0, 1),
        }

    rows = []
    for ds_name, task_type in datasets:
        t0 = time.time()
        main_metric = "roc_auc" if task_type == "classification" else "rmse"
        mode = f"scaffold-{cv_folds}fold-CV" if cv_folds else "scaffold-holdout"
        print(f"[ft_e2e] {ds_name} ({task_type}) split={mode} seeds={seeds}", flush=True)

        # ---------------- scaffold k-fold cross-validation ----------------
        if cv_folds:
            try:
                s_all, y_all = _load_moleculenet_full(ds_name)
            except Exception as exc:
                print(f"  failed to load: {exc}", flush=True)
                continue
            folds = _scaffold_kfold_indices(s_all, cv_folds, subsample_seed)
            # The RNG is created once and drawn per fold, exactly as in eval_v2 — the
            # val carve-out depends on the draw ORDER, so creating it per fold would
            # give a different (still valid, but unpaired) split.
            rng = np.random.default_rng(subsample_seed)
            fold_metrics, fold_nefs = [], []
            oof = np.full(y_all.shape, np.nan, dtype=np.float64)
            for j, test_idx in enumerate(folds):
                test_idx = np.asarray(test_idx, dtype=int)
                if len(test_idx) == 0:
                    continue
                train_pool = np.asarray([i for t, f in enumerate(folds) if t != j for i in f], dtype=int)
                perm = rng.permutation(len(train_pool))
                n_va = max(1, int(0.1 * len(train_pool)))
                va_idx = train_pool[perm[:n_va]]
                tr_idx = train_pool[perm[n_va:]]
                seed_preds = []
                for seed in seeds:
                    cb = lambda n, e, v, _c=f"{ds_name}/cv{j}/seed{seed}": _beat(n, e, v, _c)
                    (pred,) = finetune_predict(
                        encoder_path, tokenizer,
                        [s_all[i] for i in tr_idx], y_all[tr_idx],
                        [s_all[i] for i in va_idx], y_all[va_idx],
                        [[s_all[i] for i in test_idx]], task_type, seed=seed, lr=lr,
                        epochs=epochs, patience=patience, batch_size=batch_size,
                        max_length=max_length, bf16=bf16, progress_cb=cb)
                    seed_preds.append(np.asarray(pred, dtype=np.float64))
                pred = np.mean(np.stack(seed_preds, axis=0), axis=0)
                oof[test_idx] = pred
                m = compute_metric(pred, y_all[test_idx], task_type)
                fold_metrics.append(m)
                rows.append(_row(ds_name, task_type, main_metric, f"fold{j}", len(tr_idx), m, t0))
                if task_type == "classification":
                    nef = compute_nef(pred, y_all[test_idx])
                    fold_nefs.append(nef)
                    rows.append(_row(ds_name, task_type, "nef1", f"fold{j}", len(tr_idx), nef, t0))
                print(f"  fold{j}: {main_metric}={m:.4f} n_train={len(tr_idx)} "
                      f"({time.time()-t0:.0f}s elapsed)", flush=True)
            arr = np.array(fold_metrics, dtype=np.float64)
            rows.append(_row(ds_name, task_type, main_metric, "MEAN", len(s_all), float(np.nanmean(arr)), t0))
            rows.append(_row(ds_name, task_type, main_metric, "STD", len(s_all), float(np.nanstd(arr)), t0))
            print(f"  {ds_name}: {main_metric} = {np.nanmean(arr):.4f} ± {np.nanstd(arr):.4f} "
                  f"(scaffold {cv_folds}-fold CV, n={len(s_all)})", flush=True)
            if fold_nefs:
                narr = np.array(fold_nefs, dtype=np.float64)
                rows.append(_row(ds_name, task_type, "nef1", "MEAN", len(s_all), float(np.nanmean(narr)), t0))
                rows.append(_row(ds_name, task_type, "nef1", "STD", len(s_all), float(np.nanstd(narr)), t0))
                print(f"  {ds_name}: NEF1% = {np.nanmean(narr):.4f} ± {np.nanstd(narr):.4f}", flush=True)
            try:
                _dump_test_predictions(out, ds_name, task_type, s_all, y_all, oof)
            except Exception as exc:
                print(f"  [warn] OOF prediction dump failed for {ds_name}: {exc}", flush=True)
            _write_outputs(out, rows)
            continue

        # ---------------- single scaffold hold-out split ----------------
        try:
            tr_s, tr_y, va_s, va_y, te_s, te_y = _load_moleculenet(ds_name)
        except Exception as exc:
            print(f"  failed to load: {exc}", flush=True)
            continue
        tr_s, tr_y = _subsample_train(tr_s, tr_y, train_subsample, subsample_seed)
        n_train = len(tr_s)

        per_seed, per_seed_train, per_seed_nef, seed_preds = [], [], [], []
        for seed in seeds:
            cb = lambda n, e, v, _c=f"{ds_name}/holdout/seed{seed}": _beat(n, e, v, _c)
            te_pred, tr_pred = finetune_predict(
                encoder_path, tokenizer, tr_s, tr_y, va_s, va_y, [te_s, tr_s],
                task_type, seed=seed, lr=lr, epochs=epochs, patience=patience,
                batch_size=batch_size, max_length=max_length, bf16=bf16, progress_cb=cb)
            seed_preds.append(np.asarray(te_pred, dtype=np.float64))
            metric = compute_metric(te_pred, te_y, task_type)
            per_seed.append(metric)
            rows.append(_row(ds_name, task_type, main_metric, seed, n_train, metric, t0))
            # TRAIN-set score for the same fitted model, emitted as "<metric>_train" —
            # the frozen arm emits it and the label-efficiency panel reads it, so an
            # arm missing the column would silently drop out of that figure.
            tr_metric = compute_metric(tr_pred, tr_y, task_type)
            per_seed_train.append(tr_metric)
            rows.append(_row(ds_name, task_type, f"{main_metric}_train", seed, n_train, tr_metric, t0))
            if task_type == "classification":
                nef = compute_nef(te_pred, te_y)
                per_seed_nef.append(nef)
                rows.append(_row(ds_name, task_type, "nef1", seed, n_train, nef, t0))
            print(f"  seed{seed}: {main_metric}={metric:.4f} train={tr_metric:.4f} "
                  f"({time.time()-t0:.0f}s elapsed)", flush=True)

        arr = np.array(per_seed, dtype=np.float64)
        rows.append(_row(ds_name, task_type, main_metric, "MEAN", n_train, float(np.nanmean(arr)), t0))
        rows.append(_row(ds_name, task_type, main_metric, "STD", n_train, float(np.nanstd(arr)), t0))
        if per_seed_train:
            tarr = np.array(per_seed_train, dtype=np.float64)
            rows.append(_row(ds_name, task_type, f"{main_metric}_train", "MEAN", n_train, float(np.nanmean(tarr)), t0))
            rows.append(_row(ds_name, task_type, f"{main_metric}_train", "STD", n_train, float(np.nanstd(tarr)), t0))
        print(f"  {ds_name}: {main_metric} = {np.nanmean(arr):.4f} ± {np.nanstd(arr):.4f} "
              f"(n_train={n_train})", flush=True)
        if per_seed_nef:
            narr = np.array(per_seed_nef, dtype=np.float64)
            rows.append(_row(ds_name, task_type, "nef1", "MEAN", n_train, float(np.nanmean(narr)), t0))
            rows.append(_row(ds_name, task_type, "nef1", "STD", n_train, float(np.nanstd(narr)), t0))
            print(f"  {ds_name}: NEF1% = {np.nanmean(narr):.4f} ± {np.nanstd(narr):.4f}", flush=True)

        try:
            if seed_preds:
                _dump_test_predictions(out, ds_name, task_type, te_s, te_y,
                                       np.mean(np.stack(seed_preds, axis=0), axis=0))
        except Exception as exc:
            print(f"  [warn] per-molecule prediction dump failed for {ds_name}: {exc}", flush=True)
        _write_outputs(out, rows)

    return _write_outputs(out, rows)


def _write_outputs(out: Path, rows: List[dict]) -> Path:
    """Rewrite summary CSV + suite JSON from the rows accumulated so far.

    Called after EVERY dataset, not once at the end: a box that dies on HIV (the last
    and longest task) would otherwise take the five completed tasks down with it. Both
    files are rewritten whole, so a partial file is always internally consistent —
    completeness is judged from `verified.json`, never from these existing.
    """
    fieldnames = ["dataset", "task_type", "featurizer", "pool", "standardize", "head",
                  "main_metric", "head_seed", "n_train", "main_value", "elapsed_seconds"]
    summary_path = out / "moleculenet_summary.csv"
    tmp = summary_path.with_suffix(".csv.tmp")
    with tmp.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    tmp.replace(summary_path)

    suite = {}
    for r in rows:
        if r["head_seed"] in ("MEAN", "STD"):
            primary = "roc_auc" if r["task_type"] == "classification" else "rmse"
            if r["main_metric"] == primary:
                suite[f"{r['dataset']}_{r['head_seed']}"] = r["main_value"]
            else:
                suite[f"{r['dataset']}_{r['main_metric']}_{r['head_seed']}"] = r["main_value"]
    tmp = out / "suite_summary.json.tmp"
    tmp.write_text(json.dumps(suite, indent=2))
    tmp.replace(out / "suite_summary.json")
    return summary_path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--encoder", required=True)
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--datasets", nargs="+", default=None)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2],
                   help="fine-tuning seeds; the end-to-end analogue of eval_v2's head seeds")
    p.add_argument("--cv_folds", type=int, default=None)
    p.add_argument("--train_subsample", type=int, default=None,
                   help="label budget: keep this many training molecules (hold-out scheme "
                        "only). Same name/semantics as eval_v2's flag.")
    p.add_argument("--subsample_seed", type=int, default=0,
                   help="MUST match the frozen runs (0) or the fold partition will not pair")
    p.add_argument("--lr", type=float, default=FT_HPARAMS["lr"])
    p.add_argument("--epochs", type=int, default=FT_HPARAMS["epochs"])
    p.add_argument("--patience", type=int, default=FT_HPARAMS["patience"])
    p.add_argument("--batch_size", type=int, default=FT_HPARAMS["batch_size"])
    p.add_argument("--max_length", type=int, default=FT_HPARAMS["max_length"])
    p.add_argument("--no_bf16", action="store_true")
    p.add_argument("--heartbeat", default=None)
    args = p.parse_args()

    type_map = dict(MOLECULENET_TASKS_V2)
    if args.datasets is not None:
        unknown = [d for d in args.datasets if d not in type_map]
        if unknown:
            # finetune_v2 defaulted unknown datasets to "classification", which would have
            # trained QM7 (unbounded regression targets) with BCE and reported ROC-AUC.
            raise SystemExit(f"Unknown dataset(s) {unknown}; task type must be declared in "
                             f"config_v2.MOLECULENET_TASKS_V2, not defaulted.")
        ds_list = [(n, type_map[n]) for n in args.datasets]
    else:
        ds_list = MOLECULENET_TASKS_V2

    evaluate_finetuned(
        encoder_path=args.encoder, tokenizer_path=args.tokenizer,
        output_dir=args.output_dir, seeds=args.seeds, datasets=ds_list,
        max_length=args.max_length, lr=args.lr, epochs=args.epochs,
        patience=args.patience, batch_size=args.batch_size, bf16=not args.no_bf16,
        train_subsample=args.train_subsample,
        subsample_seed=args.subsample_seed, cv_folds=args.cv_folds,
        heartbeat_path=args.heartbeat,
    )


if __name__ == "__main__":
    main()
