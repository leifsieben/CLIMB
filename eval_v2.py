"""v2 frozen-featurizer MoleculeNet evaluation.

How molecular foundation models are actually deployed: freeze the encoder, extract
one embedding per molecule, train a small downstream head on those embeddings. This
runs that protocol for the 5 pre-registered tasks (config_v2.MOLECULENET_TASKS_V2)
and reports ABSOLUTE per-task metrics — no z-scores, no pooled aggregate.

Three featurizers share the identical head pipeline so comparisons are fair:
  - encoder : frozen ModernBERT, masked-mean pooled, standardized (fixes the v1
              CLS-linear-probe pathology)
  - ecfp4   : Morgan fingerprint (the classical "how bad is our CLM?" anchor;
              pair with --head xgb)
  - (random-encoder floor is produced by random_baseline_v2 → featurizer=encoder)

Downstream head ∈ {linear, mlp, xgb} from heads_v2, trained 3× (head seeds), plus
optional train-set subsampling for the label-efficiency curve (Exp D).

Output: <output_dir>/moleculenet_summary.csv (one row per dataset×featurizer×head_seed
+ MEAN/STD rows) and suite_summary.json.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch

from config_v2 import MOLECULENET_TASKS_V2
from featurize_v2 import apply_standardizer, ecfp4_features, fit_standardizer, pool
from heads_v2 import compute_metric, make_head


# ---------- DeepChem dataset loaders ----------

def _load_moleculenet(name: str):
    """Returns (train_smiles, train_y, val_smiles, val_y, test_smiles, test_y)."""
    import deepchem as dc

    loaders = {
        "ESOL": dc.molnet.load_delaney,
        "BBBP": dc.molnet.load_bbbp,
        "BACE": dc.molnet.load_bace_classification,
        "Tox21": dc.molnet.load_tox21,
        "QM7": dc.molnet.load_qm7,
        # optional later extensions
        "HIV": dc.molnet.load_hiv,
        "QM9": dc.molnet.load_qm9,
    }
    if name not in loaders:
        raise ValueError(f"Unknown MoleculeNet dataset: {name}")
    tasks, datasets, _ = loaders[name](featurizer="Raw", splitter="scaffold")
    train_ds, val_ds, test_ds = datasets
    return (
        [str(s) for s in train_ds.ids], np.asarray(train_ds.y, dtype=np.float32),
        [str(s) for s in val_ds.ids], np.asarray(val_ds.y, dtype=np.float32),
        [str(s) for s in test_ds.ids], np.asarray(test_ds.y, dtype=np.float32),
    )


# ---------- encoder featurization ----------

def _encoder_features(encoder, tokenizer, smiles: List[str], device, pool_mode: str,
                      max_length: int, batch_size: int = 128) -> np.ndarray:
    encoder.eval()
    feats = []
    with torch.no_grad():
        for i in range(0, len(smiles), batch_size):
            chunk = smiles[i:i + batch_size]
            enc = tokenizer(chunk, truncation=True, max_length=max_length,
                            padding="longest", return_tensors="pt")
            ids = enc["input_ids"].to(device)
            mask = enc["attention_mask"].to(device)
            out = encoder(input_ids=ids, attention_mask=mask)
            pooled = pool(out.last_hidden_state, mask, pool_mode).float().cpu().numpy()
            feats.append(pooled)
    return np.concatenate(feats, axis=0)


def _subsample_train(smiles: List[str], y: np.ndarray, n: Optional[int], seed: int):
    if n is None or n >= len(smiles):
        return smiles, y
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(smiles), size=n, replace=False)
    return [smiles[i] for i in idx], y[idx]


# ---------- main ----------

def evaluate(
    encoder_path: Optional[str],
    tokenizer_path: Optional[str],
    output_dir: str,
    head_seeds: List[int],
    datasets: List[Tuple[str, str]],
    *,
    featurizer: str = "encoder",
    pool_mode: str = "mean",
    standardize: str = "zscore",
    head: str = "mlp",
    max_length: int = 256,
    train_subsample: Optional[int] = None,
    subsample_seed: int = 0,
) -> Path:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    encoder = tokenizer = None
    if featurizer == "encoder":
        from transformers import ModernBertModel, PreTrainedTokenizerFast
        tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)
        # reference_compile=False: ModernBERT otherwise triggers torch.compile/triton,
        # which needs a working gcc/CUDA toolchain the workers don't have. SDPA is plenty.
        encoder = ModernBertModel.from_pretrained(
            encoder_path, attn_implementation="sdpa", reference_compile=False
        ).to(device)
        encoder.eval()
    elif featurizer != "ecfp4":
        raise ValueError(f"Unknown featurizer: {featurizer!r} (expected encoder|ecfp4)")

    # ECFP bit vectors are already 0/1; standardizing them is pointless (trees are
    # scale-invariant) and can destabilize PCA, so force 'none' for the fingerprint.
    std_method = "none" if featurizer == "ecfp4" else standardize

    rows = []
    for ds_name, task_type in datasets:
        t0 = time.time()
        main_metric = "roc_auc" if task_type == "classification" else "rmse"
        print(f"[eval_v2] {ds_name} ({task_type}) featurizer={featurizer} head={head}")
        try:
            tr_s, tr_y, va_s, va_y, te_s, te_y = _load_moleculenet(ds_name)
        except Exception as exc:
            print(f"  failed to load: {exc}")
            continue

        tr_s, tr_y = _subsample_train(tr_s, tr_y, train_subsample, subsample_seed)
        n_train = len(tr_s)
        n_outputs = tr_y.shape[1] if tr_y.ndim > 1 else 1

        # Featurize
        if featurizer == "ecfp4":
            tr_x = ecfp4_features(tr_s); va_x = ecfp4_features(va_s); te_x = ecfp4_features(te_s)
        else:
            tr_x = _encoder_features(encoder, tokenizer, tr_s, device, pool_mode, max_length)
            va_x = _encoder_features(encoder, tokenizer, va_s, device, pool_mode, max_length)
            te_x = _encoder_features(encoder, tokenizer, te_s, device, pool_mode, max_length)

        # Standardize (fit on train only)
        std_params = fit_standardizer(tr_x, std_method)
        tr_x = apply_standardizer(tr_x, std_params)
        va_x = apply_standardizer(va_x, std_params)
        te_x = apply_standardizer(te_x, std_params)

        per_seed = []
        for seed in head_seeds:
            hd = make_head(head, task_type, n_outputs, seed).fit(tr_x, tr_y, va_x, va_y)
            metric = compute_metric(hd.predict(te_x), te_y, task_type)
            per_seed.append(metric)
            rows.append({
                "dataset": ds_name, "task_type": task_type, "featurizer": featurizer,
                "pool": pool_mode if featurizer == "encoder" else "-",
                "standardize": std_method, "head": head, "main_metric": main_metric,
                "head_seed": seed, "n_train": n_train, "main_value": metric,
                "elapsed_seconds": round(time.time() - t0, 1),
            })

        arr = np.array(per_seed, dtype=np.float64)
        for tag, val in (("MEAN", float(np.nanmean(arr))), ("STD", float(np.nanstd(arr)))):
            rows.append({
                "dataset": ds_name, "task_type": task_type, "featurizer": featurizer,
                "pool": pool_mode if featurizer == "encoder" else "-",
                "standardize": std_method, "head": head, "main_metric": main_metric,
                "head_seed": tag, "n_train": n_train, "main_value": val,
                "elapsed_seconds": round(time.time() - t0, 1),
            })
        print(f"  {ds_name}: {main_metric} = {np.nanmean(arr):.4f} ± {np.nanstd(arr):.4f} (n_train={n_train})")

    fieldnames = ["dataset", "task_type", "featurizer", "pool", "standardize", "head",
                  "main_metric", "head_seed", "n_train", "main_value", "elapsed_seconds"]
    summary_path = out / "moleculenet_summary.csv"
    with summary_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    suite = {}
    for r in rows:
        if r["head_seed"] in ("MEAN", "STD"):
            suite[f"{r['dataset']}_{r['head_seed']}"] = r["main_value"]
    (out / "suite_summary.json").write_text(json.dumps(suite, indent=2))
    print(f"[eval_v2] wrote {summary_path}")
    return summary_path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--encoder", default=None, help="Path to a saved ModernBertModel encoder")
    p.add_argument("--tokenizer", default=None)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--head_seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--datasets", nargs="+", default=None, help="Override the 5-task subset")
    p.add_argument("--featurizer", choices=["encoder", "ecfp4"], default="encoder")
    p.add_argument("--pool", choices=["cls", "mean", "cls_mean"], default="mean")
    p.add_argument("--standardize", choices=["zscore", "pca_whiten", "none"], default="zscore")
    p.add_argument("--head", choices=["linear", "mlp", "xgb"], default="mlp")
    p.add_argument("--max_length", type=int, default=256)
    p.add_argument("--train_subsample", type=int, default=None)
    p.add_argument("--subsample_seed", type=int, default=0)
    args = p.parse_args()

    if args.datasets is not None:
        type_map = dict(MOLECULENET_TASKS_V2)
        ds_list = [(n, type_map.get(n, "classification")) for n in args.datasets]
    else:
        ds_list = MOLECULENET_TASKS_V2

    evaluate(
        encoder_path=args.encoder, tokenizer_path=args.tokenizer, output_dir=args.output_dir,
        head_seeds=args.head_seeds, datasets=ds_list, featurizer=args.featurizer,
        pool_mode=args.pool, standardize=args.standardize, head=args.head,
        max_length=args.max_length, train_subsample=args.train_subsample,
        subsample_seed=args.subsample_seed,
    )


if __name__ == "__main__":
    main()
