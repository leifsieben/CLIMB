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
from heads_v2 import compute_metric, compute_nef, make_head


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
        "HIV": dc.molnet.load_hiv,            # large (~41k) drug-discovery classification
        "Lipophilicity": dc.molnet.load_lipo,  # ~4.2k ADMET/physchem regression (healthy signal)
        # optional later extensions
        "QM9": dc.molnet.load_qm9,
    }
    if name not in loaders:
        raise ValueError(f"Unknown MoleculeNet dataset: {name}")
    # reload=False: skip DeepChem's on-disk featurization cache. Loading several datasets
    # in one process otherwise collides on a shared cache dir ("No Metadata found ...") and
    # returns a corrupt/ragged dataset. Raw featurizer is cheap, so recomputing is fine.
    tasks, datasets, _ = loaders[name](featurizer="Raw", splitter="scaffold", reload=False)
    train_ds, val_ds, test_ds = datasets
    return (
        [str(s) for s in train_ds.ids], np.asarray(train_ds.y, dtype=np.float32),
        [str(s) for s in val_ds.ids], np.asarray(val_ds.y, dtype=np.float32),
        [str(s) for s in test_ds.ids], np.asarray(test_ds.y, dtype=np.float32),
    )


def _load_moleculenet_full(name: str):
    """The full labelled dataset (train+val+test unioned), for scaffold k-fold CV."""
    tr_s, tr_y, va_s, va_y, te_s, te_y = _load_moleculenet(name)
    smiles = list(tr_s) + list(va_s) + list(te_s)
    y = np.concatenate([tr_y, va_y, te_y], axis=0)
    return smiles, y


def _scaffold_kfold_indices(smiles: List[str], k: int, seed: int = 0):
    """Partition molecules into k SCAFFOLD-DISJOINT folds (Bemis–Murcko). Every scaffold
    lives entirely in one fold, so each fold's test set contains only scaffolds unseen in
    that fold's training data — the small-data cross-validation analogue of the single
    scaffold split, giving an honest error bar from the fold spread instead of one noisy
    draw. Largest scaffold groups are placed first into the currently-smallest fold to keep
    folds balanced in size."""
    try:
        from rdkit.Chem.Scaffolds import MurckoScaffold
        from rdkit import RDLogger
        RDLogger.DisableLog("rdApp.*")
    except Exception:
        # No RDKit → fall back to a deterministic shuffled k-fold (not scaffold-aware).
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(smiles))
        return [list(idx[j::k]) for j in range(k)]

    scaffolds = {}
    for i, s in enumerate(smiles):
        try:
            scaf = MurckoScaffold.MurckoScaffoldSmiles(smiles=s, includeChirality=False)
        except Exception:
            scaf = None
        # Ring-less molecules have an empty Murcko scaffold and share no real scaffold, so
        # treat each as its own singleton — they distribute across folds (no leakage, since
        # there is no scaffold to share) and keep folds balanced on acyclic-heavy sets (ESOL).
        key = scaf if scaf else f"__noscaffold_{i}"
        scaffolds.setdefault(key, []).append(i)

    groups = sorted(scaffolds.values(), key=lambda g: len(g), reverse=True)
    folds = [[] for _ in range(k)]
    for g in groups:
        j = min(range(k), key=lambda t: len(folds[t]))
        folds[j].extend(g)
    return folds


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


def _canonical_keys(smiles_list: List[str]) -> List[str]:
    """Exact-identity canonical SMILES (isomeric, NO salt-strip) for H9's 'seen' axis
    and to join per-molecule predictions against pretrain-overlap / fingerprints (C16/C17).
    This is deliberately the STRICT key (keeps stereo + salts); it is NOT the fuzzy
    largest-fragment dedup key. Falls back to the raw string if RDKit can't parse."""
    try:
        from rdkit import Chem
        from rdkit import RDLogger
        RDLogger.DisableLog("rdApp.*")
    except Exception:
        return list(smiles_list)
    out = []
    for s in smiles_list:
        try:
            m = Chem.MolFromSmiles(s)
            out.append(Chem.MolToSmiles(m) if m is not None else s)
        except Exception:
            out.append(s)
    return out


def _dump_test_predictions(out: Path, ds_name: str, task_type: str, te_s: List[str],
                           te_y: np.ndarray, te_pred_mean: np.ndarray) -> None:
    """Per-molecule TEST predictions (C16): (canonical_key, output_index, y_true, y_pred),
    averaged over head seeds. Required to build the H9 panels and to bin any residual/rank
    analysis without re-running eval. Multi-output tasks (e.g. Tox21) get one row per
    (molecule, output_index). Appended to <out>/test_predictions.csv."""
    keys = _canonical_keys(te_s)
    yt = np.asarray(te_y, dtype=np.float64)
    yp = np.asarray(te_pred_mean, dtype=np.float64)
    if yt.ndim == 1:
        yt = yt[:, None]
    if yp.ndim == 1:
        yp = yp[:, None]
    n_out = yt.shape[1]
    pred_path = out / "test_predictions.csv"
    write_header = not pred_path.exists()
    import csv as _csv
    with pred_path.open("a", newline="") as f:
        w = _csv.writer(f)
        if write_header:
            w.writerow(["dataset", "task_type", "split", "mol_index", "canonical_key",
                        "raw_smiles", "output_index", "y_true", "y_pred"])
        for i in range(len(te_s)):
            for o in range(n_out):
                yt_v = yt[i, o] if o < yt.shape[1] else ""
                yp_v = yp[i, o] if o < yp.shape[1] else ""
                # skip NaN labels (Tox21 sparse) — nothing to compare against
                if isinstance(yt_v, float) and np.isnan(yt_v):
                    continue
                w.writerow([ds_name, task_type, "test", i, keys[i], te_s[i], o,
                            f"{yt_v:.6g}" if yt_v != "" else "",
                            f"{yp_v:.6g}" if yp_v != "" else ""])


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
    cv_folds: Optional[int] = None,
) -> Path:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    # per-molecule dump is append-mode within a run (one call per dataset); clear any prior
    # run's file first so re-evaluating the same output_dir cannot accumulate duplicate rows.
    (out / "test_predictions.csv").unlink(missing_ok=True)
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
    elif featurizer not in ("ecfp4", "rdkit_desc", "fp_desc"):
        raise ValueError(f"Unknown featurizer: {featurizer!r} (expected encoder|ecfp4|rdkit_desc|fp_desc)")

    # ECFP bit vectors are 0/1; RDKit descriptors are heterogeneous-scale with NaNs. For
    # both, the intended head is XGBoost (scale-invariant, native NaN handling), so skip
    # standardization. This makes rdkit_desc the classical control for the dense-MTR arm:
    # the SAME 217 descriptors used directly in a tree model vs learned by the CLM.
    std_method = "none" if featurizer in ("ecfp4", "rdkit_desc", "fp_desc") else standardize

    def _featurize(smiles):
        if featurizer == "ecfp4":
            return np.asarray(ecfp4_features(smiles))
        if featurizer == "rdkit_desc":
            from descriptors_v2 import rdkit_descriptors
            x = np.asarray(rdkit_descriptors(list(smiles)), dtype=np.float32)
            x[~np.isfinite(x)] = np.nan  # XGBoost treats NaN as missing but ERRORS on inf
            return x
        if featurizer == "fp_desc":
            # Tougher classical baseline: Morgan ECFP4 bits ++ 217 RDKit descriptors → XGBoost.
            # Gives the tree both substructure presence AND computed physchem properties.
            from descriptors_v2 import rdkit_descriptors
            fp = np.asarray(ecfp4_features(smiles), dtype=np.float32)
            d = np.asarray(rdkit_descriptors(list(smiles)), dtype=np.float32)
            d[~np.isfinite(d)] = np.nan
            return np.concatenate([fp, d], axis=1)
        return _encoder_features(encoder, tokenizer, smiles, device, pool_mode, max_length)

    def _row(ds_name, task_type, main_metric, tag, n_train, val, t0):
        return {
            "dataset": ds_name, "task_type": task_type, "featurizer": featurizer,
            "pool": pool_mode if featurizer == "encoder" else "-",
            "standardize": std_method, "head": head, "main_metric": main_metric,
            "head_seed": tag, "n_train": n_train, "main_value": val,
            "elapsed_seconds": round(time.time() - t0, 1),
        }

    rows = []
    for ds_name, task_type in datasets:
        t0 = time.time()
        main_metric = "roc_auc" if task_type == "classification" else "rmse"
        mode = f"scaffold-{cv_folds}fold-CV" if cv_folds else "scaffold-holdout"
        print(f"[eval_v2] {ds_name} ({task_type}) featurizer={featurizer} head={head} split={mode}")

        # ---------------- scaffold k-fold cross-validation ----------------
        # Error bar (the STD row) = spread ACROSS folds — the honest small-data uncertainty
        # the single hold-out split cannot show. Encoder features are computed ONCE for the
        # whole dataset and sliced per fold, so CV costs only k extra small head-fits.
        if cv_folds:
            try:
                s_all, y_all = _load_moleculenet_full(ds_name)
            except Exception as exc:
                print(f"  failed to load: {exc}")
                continue
            n_outputs = y_all.shape[1] if y_all.ndim > 1 else 1
            X_all = _featurize(s_all)
            folds = _scaffold_kfold_indices(s_all, cv_folds, subsample_seed)
            rng = np.random.default_rng(subsample_seed)
            fold_metrics = []
            fold_nefs = []   # NEF1% per fold (classification only) — the VS early-enrichment metric
            oof = np.full(y_all.shape, np.nan, dtype=np.float64)
            for j, test_idx in enumerate(folds):
                test_idx = np.asarray(test_idx, dtype=int)
                if len(test_idx) == 0:
                    continue  # degenerate: too few scaffolds for k folds — skip empty fold
                train_pool = np.asarray([i for t, f in enumerate(folds) if t != j for i in f], dtype=int)
                perm = rng.permutation(len(train_pool))
                n_va = max(1, int(0.1 * len(train_pool)))
                va_idx = train_pool[perm[:n_va]]
                tr_idx = train_pool[perm[n_va:]]
                sp = fit_standardizer(X_all[tr_idx], std_method)
                trx, vax, tex = (apply_standardizer(X_all[ix], sp) for ix in (tr_idx, va_idx, test_idx))
                seed_preds = []
                for seed in head_seeds:
                    hd = make_head(head, task_type, n_outputs, seed).fit(trx, y_all[tr_idx], vax, y_all[va_idx])
                    seed_preds.append(np.asarray(hd.predict(tex), dtype=np.float64))
                pred = np.mean(np.stack(seed_preds, axis=0), axis=0)
                oof[test_idx] = pred
                m = compute_metric(pred, y_all[test_idx], task_type)
                fold_metrics.append(m)
                rows.append(_row(ds_name, task_type, main_metric, f"fold{j}", len(tr_idx), m, t0))
                if task_type == "classification":   # NEF1% is computed PER held-out fold (Truong et al. 2026)
                    nef = compute_nef(pred, y_all[test_idx])
                    fold_nefs.append(nef)
                    rows.append(_row(ds_name, task_type, "nef1", f"fold{j}", len(tr_idx), nef, t0))
            arr = np.array(fold_metrics, dtype=np.float64)
            rows.append(_row(ds_name, task_type, main_metric, "MEAN", len(s_all), float(np.nanmean(arr)), t0))
            rows.append(_row(ds_name, task_type, main_metric, "STD", len(s_all), float(np.nanstd(arr)), t0))
            print(f"  {ds_name}: {main_metric} = {np.nanmean(arr):.4f} ± {np.nanstd(arr):.4f} "
                  f"(scaffold {cv_folds}-fold CV, n={len(s_all)})")
            if fold_nefs:
                narr = np.array(fold_nefs, dtype=np.float64)
                rows.append(_row(ds_name, task_type, "nef1", "MEAN", len(s_all), float(np.nanmean(narr)), t0))
                rows.append(_row(ds_name, task_type, "nef1", "STD", len(s_all), float(np.nanstd(narr)), t0))
                print(f"  {ds_name}: NEF1% = {np.nanmean(narr):.4f} ± {np.nanstd(narr):.4f} (top-1% early enrichment)")
            try:  # OOF: every molecule is in exactly one test fold → a complete prediction set
                _dump_test_predictions(out, ds_name, task_type, s_all, y_all, oof)
            except Exception as exc:
                print(f"  [warn] OOF prediction dump failed for {ds_name}: {exc}")
            continue

        # ---------------- single scaffold hold-out split (default) ----------------
        try:
            tr_s, tr_y, va_s, va_y, te_s, te_y = _load_moleculenet(ds_name)
        except Exception as exc:
            print(f"  failed to load: {exc}")
            continue

        tr_s, tr_y = _subsample_train(tr_s, tr_y, train_subsample, subsample_seed)
        n_train = len(tr_s)
        n_outputs = tr_y.shape[1] if tr_y.ndim > 1 else 1

        tr_x, va_x, te_x = _featurize(tr_s), _featurize(va_s), _featurize(te_s)
        std_params = fit_standardizer(tr_x, std_method)
        tr_x = apply_standardizer(tr_x, std_params)
        va_x = apply_standardizer(va_x, std_params)
        te_x = apply_standardizer(te_x, std_params)

        per_seed = []
        per_seed_nef = []
        seed_preds = []
        for seed in head_seeds:
            hd = make_head(head, task_type, n_outputs, seed).fit(tr_x, tr_y, va_x, va_y)
            te_pred = hd.predict(te_x)
            seed_preds.append(np.asarray(te_pred, dtype=np.float64))
            metric = compute_metric(te_pred, te_y, task_type)
            per_seed.append(metric)
            rows.append(_row(ds_name, task_type, main_metric, seed, n_train, metric, t0))
            if task_type == "classification":
                nef = compute_nef(te_pred, te_y)
                per_seed_nef.append(nef)
                rows.append(_row(ds_name, task_type, "nef1", seed, n_train, nef, t0))

        arr = np.array(per_seed, dtype=np.float64)
        rows.append(_row(ds_name, task_type, main_metric, "MEAN", n_train, float(np.nanmean(arr)), t0))
        rows.append(_row(ds_name, task_type, main_metric, "STD", n_train, float(np.nanstd(arr)), t0))
        print(f"  {ds_name}: {main_metric} = {np.nanmean(arr):.4f} ± {np.nanstd(arr):.4f} (n_train={n_train})")
        if per_seed_nef:
            narr = np.array(per_seed_nef, dtype=np.float64)
            rows.append(_row(ds_name, task_type, "nef1", "MEAN", n_train, float(np.nanmean(narr)), t0))
            rows.append(_row(ds_name, task_type, "nef1", "STD", n_train, float(np.nanstd(narr)), t0))
            print(f"  {ds_name}: NEF1% = {np.nanmean(narr):.4f} ± {np.nanstd(narr):.4f} (top-1% early enrichment)")

        # C16: per-molecule TEST predictions (mean over head seeds) for the mechanism figures.
        try:
            if seed_preds:
                _dump_test_predictions(out, ds_name, task_type, te_s, te_y,
                                       np.mean(np.stack(seed_preds, axis=0), axis=0))
        except Exception as exc:
            print(f"  [warn] per-molecule prediction dump failed for {ds_name}: {exc}")

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
            # Primary metric (roc_auc / rmse) keeps the bare `<ds>_<MEAN|STD>` key so every
            # existing loader is unchanged; secondary metrics (e.g. nef1) are namespaced as
            # `<ds>_<metric>_<MEAN|STD>` to avoid clobbering it.
            primary = "roc_auc" if r["task_type"] == "classification" else "rmse"
            if r["main_metric"] == primary:
                suite[f"{r['dataset']}_{r['head_seed']}"] = r["main_value"]
            else:
                suite[f"{r['dataset']}_{r['main_metric']}_{r['head_seed']}"] = r["main_value"]
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
    p.add_argument("--featurizer", choices=["encoder", "ecfp4", "rdkit_desc", "fp_desc"], default="encoder")
    p.add_argument("--pool", choices=["cls", "mean", "cls_mean"], default="mean")
    p.add_argument("--standardize", choices=["zscore", "pca_whiten", "none"], default="zscore")
    p.add_argument("--head", choices=["linear", "mlp", "xgb"], default="mlp")
    p.add_argument("--max_length", type=int, default=256)
    p.add_argument("--train_subsample", type=int, default=None)
    p.add_argument("--subsample_seed", type=int, default=0)
    p.add_argument("--cv_folds", type=int, default=None,
                   help="If set, use scaffold k-fold cross-validation (error bar = fold spread) "
                        "instead of the single scaffold hold-out split.")
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
        subsample_seed=args.subsample_seed, cv_folds=args.cv_folds,
    )


if __name__ == "__main__":
    main()
