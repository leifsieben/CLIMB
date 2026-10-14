from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch

from config import MOLECULENET_TASKS
from features import apply_standardizer, ecfp4_features, fit_standardizer, pool
from heads import compute_metric, compute_nef, make_head


_CUSTOM_TASKS: dict = {}
_CV_SCHEME: str = "scaffold"
_PROVIDED_FOLDS = None


def register_custom_task(name: str, csv_path: str, smiles_col: str = "smiles", y_col: str = "y",
                         fold_col: str = "fold"):
    _CUSTOM_TASKS[name] = (str(csv_path), smiles_col, y_col, fold_col)


def set_cv_scheme(scheme: str):
    global _CV_SCHEME
    if scheme not in ("scaffold", "random", "provided"):
        raise ValueError(f"cv_scheme must be scaffold|random|provided, got {scheme!r}")
    _CV_SCHEME = scheme


def _load_custom_csv(name: str):

    global _PROVIDED_FOLDS
    path, sc, yc, fc = _CUSTOM_TASKS[name]
    smiles, ys, folds = [], [], []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        has_fold = fc in (reader.fieldnames or [])
        for row in reader:
            s = (row.get(sc) or "").strip()
            if not s:
                continue
            smiles.append(s)
            ys.append(float(row[yc]))
            if has_fold:
                folds.append(str(row[fc]).strip())
    _PROVIDED_FOLDS = folds if (folds and len(folds) == len(smiles)) else None


    return smiles, np.asarray(ys, dtype=np.float32).reshape(-1, 1)


def _load_moleculenet(name: str):
    """Returns (train_smiles, train_y, val_smiles, val_y, test_smiles, test_y)."""
    if name in _CUSTOM_TASKS:


        raise ValueError(f"custom CSV task {name!r} supports --cv_folds (k-fold CV) only")
    import deepchem as dc

    loaders = {
        "ESOL": dc.molnet.load_delaney,
        "BACE": dc.molnet.load_bace_classification,
        "Tox21": dc.molnet.load_tox21,
        "QM7": dc.molnet.load_qm7,
        "HIV": dc.molnet.load_hiv,

        "QM9": dc.molnet.load_qm9,
    }
    if name not in loaders:
        raise ValueError(f"Unknown MoleculeNet dataset: {name}")


    tasks, datasets, _ = loaders[name](featurizer="Raw", splitter="scaffold", reload=False,
                                       transformers=[])
    train_ds, val_ds, test_ds = datasets

    def _y_masked(ds):


        y = np.asarray(ds.y, dtype=np.float32).copy()
        w = np.asarray(ds.w, dtype=np.float32)
        if w.shape == y.shape:
            y[w == 0] = np.nan
        return y

    return (
        [str(s) for s in train_ds.ids], _y_masked(train_ds),
        [str(s) for s in val_ds.ids], _y_masked(val_ds),
        [str(s) for s in test_ds.ids], _y_masked(test_ds),
    )


def _load_moleculenet_full(name: str):
    """The full labelled dataset (train+val+test unioned), for scaffold k-fold CV."""
    if name in _CUSTOM_TASKS:
        return _load_custom_csv(name)
    tr_s, tr_y, va_s, va_y, te_s, te_y = _load_moleculenet(name)
    smiles = list(tr_s) + list(va_s) + list(te_s)
    y = np.concatenate([tr_y, va_y, te_y], axis=0)
    return smiles, y


def _fit_target_scaler(y_train: np.ndarray, task_type: str):
    if task_type != "regression":
        return None
    y = np.asarray(y_train, dtype=np.float64)
    mu = np.nanmean(y, axis=0)
    sd = np.nanstd(y, axis=0)
    sd = np.where(np.isfinite(sd) & (sd > 0), sd, 1.0)
    return (mu, sd)


def _scale_targets(y: np.ndarray, sc):
    if sc is None:
        return y
    return (np.asarray(y, dtype=np.float64) - sc[0]) / sc[1]


def _unscale_preds(p: np.ndarray, sc):
    if sc is None:
        return p
    return np.asarray(p, dtype=np.float64) * sc[1] + sc[0]


def _random_kfold_indices(n: int, k: int, seed: int = 0, labels=None):

    rng = np.random.default_rng(seed)
    folds = [[] for _ in range(k)]
    if labels is not None:
        y = np.asarray(labels).reshape(len(labels), -1)[:, 0]
        strata = np.where(np.isnan(y), -1.0, (y > 0.5).astype(float))
    else:
        strata = np.zeros(n)
    for s in np.unique(strata):
        idx = np.where(strata == s)[0]
        idx = idx[rng.permutation(len(idx))]
        for t, i in enumerate(idx):
            folds[t % k].append(int(i))
    return [sorted(f) for f in folds]


def _provided_kfold_indices(n: int, k: int):

    if _PROVIDED_FOLDS is None or len(_PROVIDED_FOLDS) != n:
        raise ValueError("cv_scheme=provided requires a fold column aligned to the task rows")
    uniq = sorted(set(_PROVIDED_FOLDS))
    if len(uniq) != k:
        raise ValueError(f"cv_scheme=provided: data has {len(uniq)} fold ids {uniq} but --cv_folds={k}")
    pos = {u: i for i, u in enumerate(uniq)}
    folds = [[] for _ in range(k)]
    for idx, fv in enumerate(_PROVIDED_FOLDS):
        folds[pos[fv]].append(idx)
    return folds


def _scaffold_kfold_indices(smiles: List[str], k: int, seed: int = 0, labels=None):

    if _CV_SCHEME == "provided":
        return _provided_kfold_indices(len(smiles), k)
    if _CV_SCHEME == "random":
        return _random_kfold_indices(len(smiles), k, seed, labels)
    try:
        from rdkit.Chem.Scaffolds import MurckoScaffold
        from rdkit import RDLogger
        RDLogger.DisableLog("rdApp.*")
    except Exception:

        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(smiles))
        return [list(idx[j::k]) for j in range(k)]

    scaffolds = {}
    for i, s in enumerate(smiles):
        try:
            scaf = MurckoScaffold.MurckoScaffoldSmiles(smiles=s, includeChirality=False)
        except Exception:
            scaf = None


        key = scaf if scaf else f"__noscaffold_{i}"
        scaffolds.setdefault(key, []).append(i)

    groups = sorted(scaffolds.values(), key=lambda g: len(g), reverse=True)
    folds = [[] for _ in range(k)]
    for g in groups:
        j = min(range(k), key=lambda t: len(folds[t]))
        folds[j].extend(g)
    return folds


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


def _load_feature_table(path: str) -> dict:
    """{canonical SMILES: vector} from an .npz holding `smiles` and `X`."""
    z = np.load(path, allow_pickle=True)
    smi, X = z["smiles"], z["X"]
    if len(smi) != len(X):
        raise ValueError(f"{path}: {len(smi)} smiles but {len(X)} vectors")
    print(f"  [features] {len(smi)} precomputed vectors ({X.shape[1]}d) from {path}", flush=True)
    return {str(s): X[i] for i, s in enumerate(smi)}


def _lookup_features(table: dict, smiles) -> np.ndarray:

    miss = [s for s in smiles if str(s) not in table]
    if miss:
        raise KeyError(f"{len(miss)} SMILES absent from the feature table, e.g. {miss[:3]}")
    return np.asarray([table[str(s)] for s in smiles], dtype=np.float32)


def _subsample_train(smiles: List[str], y: np.ndarray, n: Optional[int], seed: int):
    if n is None or n >= len(smiles):
        return smiles, y
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(smiles), size=n, replace=False)
    return [smiles[i] for i in idx], y[idx]


def _canonical_keys(smiles_list: List[str]) -> List[str]:

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

                if isinstance(yt_v, float) and np.isnan(yt_v):
                    continue
                w.writerow([ds_name, task_type, "test", i, keys[i], te_s[i], o,
                            f"{yt_v:.6g}" if yt_v != "" else "",
                            f"{yp_v:.6g}" if yp_v != "" else ""])


def _impute_stats(x, tree_head):
    """Per-column TRAIN medians for NaN imputation; None for XGBoost, which wants the NaNs."""
    if tree_head:
        return None
    a = np.asarray(x, dtype=np.float64).copy()
    a[~np.isfinite(a)] = np.nan
    med = np.nanmedian(a, axis=0)
    return np.where(np.isfinite(med), med, 0.0)


def _prep(x, med, clip=10.0, post=False):

    if med is None:
        return x
    a = np.asarray(x, dtype=np.float32 if post else np.float64)
    if post:
        return np.clip(a, -clip, clip).astype(np.float32)
    bad = ~np.isfinite(a)
    if bad.any():
        a = np.where(bad, np.broadcast_to(med, a.shape), a)
    return a


def _bound_ood(pred, y_train, task_type, margin=0.25):

    if task_type != "regression":
        return pred
    yt = np.asarray(y_train, dtype=np.float64)
    yt = yt[np.isfinite(yt)]
    if not len(yt):
        return pred
    lo, hi = float(yt.min()), float(yt.max())
    m = margin * (hi - lo + 1e-9)
    return np.clip(pred, lo - m, hi + m)


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
    features_npz: Optional[str] = None,
    train_subsample: Optional[int] = None,
    subsample_seed: int = 0,
    cv_folds: Optional[int] = None,
) -> Path:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)


    (out / "test_predictions.csv").unlink(missing_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available()
                          else "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
                          else "cpu")

    encoder = tokenizer = None


    if featurizer == "encoder" and features_npz:
        print(f"[evaluate] features_npz supplied -- skipping encoder/tokenizer load", flush=True)
    elif featurizer == "encoder":
        from transformers import ModernBertModel, PreTrainedTokenizerFast
        tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)


        encoder = ModernBertModel.from_pretrained(
            encoder_path, attn_implementation="sdpa", reference_compile=False
        ).to(device)
        encoder.eval()
    elif featurizer not in ("ecfp4", "rdkit_desc", "fp_desc"):
        raise ValueError(f"Unknown featurizer: {featurizer!r} (expected encoder|ecfp4|rdkit_desc|fp_desc)")


    tree_head = head == "xgb"
    std_method = ("none" if (featurizer in ("ecfp4", "rdkit_desc", "fp_desc") and tree_head)
                  else standardize)


    _feature_table = _load_feature_table(features_npz) if features_npz else None

    def _featurize(smiles):


        if _feature_table is not None:
            return _lookup_features(_feature_table, smiles)
        if featurizer == "ecfp4":
            return np.asarray(ecfp4_features(smiles))
        if featurizer == "rdkit_desc":
            from descriptors import rdkit_descriptors
            x = np.asarray(rdkit_descriptors(list(smiles)), dtype=np.float32)
            x[~np.isfinite(x)] = np.nan
            return x
        if featurizer == "fp_desc":


            from descriptors import rdkit_descriptors
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
        mode = f"{_CV_SCHEME}-{cv_folds}fold-CV" if cv_folds else "scaffold-holdout"
        print(f"[evaluate] {ds_name} ({task_type}) featurizer={featurizer} head={head} split={mode}")


        if cv_folds:
            try:
                s_all, y_all = _load_moleculenet_full(ds_name)
            except Exception as exc:
                print(f"  failed to load: {exc}")
                continue
            n_outputs = y_all.shape[1] if y_all.ndim > 1 else 1
            X_all = _featurize(s_all)
            folds = _scaffold_kfold_indices(s_all, cv_folds, subsample_seed, labels=y_all)
            rng = np.random.default_rng(subsample_seed)
            fold_metrics = []
            fold_nefs = []
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
                med = _impute_stats(X_all[tr_idx], tree_head)
                sp = fit_standardizer(_prep(X_all[tr_idx], med), std_method)
                trx, vax, tex = (_prep(apply_standardizer(_prep(X_all[ix], med), sp), med,
                                       post=True) for ix in (tr_idx, va_idx, test_idx))
                ysc = _fit_target_scaler(y_all[tr_idx], task_type)
                seed_preds = []
                for seed in head_seeds:
                    hd = make_head(head, task_type, n_outputs, seed).fit(
                        trx, _scale_targets(y_all[tr_idx], ysc), vax, _scale_targets(y_all[va_idx], ysc))
                    sp_pred = _unscale_preds(np.asarray(hd.predict(tex), dtype=np.float64), ysc)
                    sp_pred = _bound_ood(sp_pred, y_all[tr_idx], task_type)
                    seed_preds.append(sp_pred)


                    rows.append(_row(ds_name, task_type, f"{main_metric}_cell", f"s{seed}_fold{j}",
                                     len(tr_idx), compute_metric(sp_pred, y_all[test_idx], task_type), t0))
                    if task_type == "classification":
                        rows.append(_row(ds_name, task_type, "nef1_cell", f"s{seed}_fold{j}",
                                         len(tr_idx), compute_nef(sp_pred, y_all[test_idx]), t0))
                pred = np.mean(np.stack(seed_preds, axis=0), axis=0)
                oof[test_idx] = pred
                m = compute_metric(pred, y_all[test_idx], task_type)
                fold_metrics.append(m)
                rows.append(_row(ds_name, task_type, main_metric, f"fold{j}", len(tr_idx), m, t0))
                if task_type == "classification":
                    nef = compute_nef(pred, y_all[test_idx])
                    fold_nefs.append(nef)
                    rows.append(_row(ds_name, task_type, "nef1", f"fold{j}", len(tr_idx), nef, t0))
            arr = np.array(fold_metrics, dtype=np.float64)
            rows.append(_row(ds_name, task_type, main_metric, "MEAN", len(s_all), float(np.nanmean(arr)), t0))
            rows.append(_row(ds_name, task_type, main_metric, "STD", len(s_all), float(np.nanstd(arr)), t0))
            print(f"  {ds_name}: {main_metric} = {np.nanmean(arr):.4f} ± {np.nanstd(arr):.4f} "
                  f"({_CV_SCHEME} {cv_folds}-fold CV, n={len(s_all)})")
            if fold_nefs:
                narr = np.array(fold_nefs, dtype=np.float64)
                rows.append(_row(ds_name, task_type, "nef1", "MEAN", len(s_all), float(np.nanmean(narr)), t0))
                rows.append(_row(ds_name, task_type, "nef1", "STD", len(s_all), float(np.nanstd(narr)), t0))
                print(f"  {ds_name}: NEF1% = {np.nanmean(narr):.4f} ± {np.nanstd(narr):.4f} (top-1% early enrichment)")
            try:
                _dump_test_predictions(out, ds_name, task_type, s_all, y_all, oof)
            except Exception as exc:
                print(f"  [warn] OOF prediction dump failed for {ds_name}: {exc}")
            continue


        try:
            tr_s, tr_y, va_s, va_y, te_s, te_y = _load_moleculenet(ds_name)
        except Exception as exc:
            print(f"  failed to load: {exc}")
            continue

        tr_s, tr_y = _subsample_train(tr_s, tr_y, train_subsample, subsample_seed)
        n_train = len(tr_s)
        n_outputs = tr_y.shape[1] if tr_y.ndim > 1 else 1

        tr_x, va_x, te_x = _featurize(tr_s), _featurize(va_s), _featurize(te_s)
        med = _impute_stats(tr_x, tree_head)
        std_params = fit_standardizer(_prep(tr_x, med), std_method)
        tr_x = _prep(apply_standardizer(_prep(tr_x, med), std_params), med, post=True)
        va_x = _prep(apply_standardizer(_prep(va_x, med), std_params), med, post=True)
        te_x = _prep(apply_standardizer(_prep(te_x, med), std_params), med, post=True)

        ysc = _fit_target_scaler(tr_y, task_type)
        per_seed = []
        per_seed_nef = []
        seed_preds = []
        per_seed_train = []
        for seed in head_seeds:
            hd = make_head(head, task_type, n_outputs, seed).fit(
                tr_x, _scale_targets(tr_y, ysc), va_x, _scale_targets(va_y, ysc))
            te_pred = _unscale_preds(np.asarray(hd.predict(te_x), dtype=np.float64), ysc)
            te_pred = _bound_ood(te_pred, tr_y, task_type)
            seed_preds.append(np.asarray(te_pred, dtype=np.float64))
            metric = compute_metric(te_pred, te_y, task_type)
            per_seed.append(metric)
            rows.append(_row(ds_name, task_type, main_metric, seed, n_train, metric, t0))


            tr_pred = _unscale_preds(np.asarray(hd.predict(tr_x), dtype=np.float64), ysc)
            tr_metric = compute_metric(tr_pred, tr_y, task_type)
            per_seed_train.append(tr_metric)
            rows.append(_row(ds_name, task_type, f"{main_metric}_train", seed, n_train, tr_metric, t0))
            if task_type == "classification":
                nef = compute_nef(te_pred, te_y)
                per_seed_nef.append(nef)
                rows.append(_row(ds_name, task_type, "nef1", seed, n_train, nef, t0))

        arr = np.array(per_seed, dtype=np.float64)
        rows.append(_row(ds_name, task_type, main_metric, "MEAN", n_train, float(np.nanmean(arr)), t0))
        rows.append(_row(ds_name, task_type, main_metric, "STD", n_train, float(np.nanstd(arr)), t0))
        if per_seed_train:
            tarr = np.array(per_seed_train, dtype=np.float64)
            rows.append(_row(ds_name, task_type, f"{main_metric}_train", "MEAN", n_train,
                             float(np.nanmean(tarr)), t0))
            rows.append(_row(ds_name, task_type, f"{main_metric}_train", "STD", n_train,
                             float(np.nanstd(tarr)), t0))
        print(f"  {ds_name}: {main_metric} = {np.nanmean(arr):.4f} ± {np.nanstd(arr):.4f} (n_train={n_train})")
        if per_seed_nef:
            narr = np.array(per_seed_nef, dtype=np.float64)
            rows.append(_row(ds_name, task_type, "nef1", "MEAN", n_train, float(np.nanmean(narr)), t0))
            rows.append(_row(ds_name, task_type, "nef1", "STD", n_train, float(np.nanstd(narr)), t0))
            print(f"  {ds_name}: NEF1% = {np.nanmean(narr):.4f} ± {np.nanstd(narr):.4f} (top-1% early enrichment)")


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


            primary = "roc_auc" if r["task_type"] == "classification" else "rmse"
            if r["main_metric"] == primary:
                suite[f"{r['dataset']}_{r['head_seed']}"] = r["main_value"]
            else:
                suite[f"{r['dataset']}_{r['main_metric']}_{r['head_seed']}"] = r["main_value"]
    (out / "suite_summary.json").write_text(json.dumps(suite, indent=2))
    print(f"[evaluate] wrote {summary_path}")
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
    p.add_argument("--features_npz", default=None,
                   help="Precomputed {smiles: vector} table (.npz with `smiles` and `X`), used "
                        "instead of running the featurizer in-process.")
    p.add_argument("--train_subsample", type=int, default=None)
    p.add_argument("--subsample_seed", type=int, default=0)
    p.add_argument("--cv_folds", type=int, default=None,
                   help="If set, use k-fold cross-validation (error bar = fold spread) "
                        "instead of the single scaffold hold-out split.")
    p.add_argument("--cv_scheme", choices=["scaffold", "random", "provided"], default="scaffold",
                   help="Fold partition for --cv_folds: scaffold-disjoint (published), "
                        "label-stratified random, or provided (the CSV's own fold column).")
    p.add_argument("--task_csv", default=None,
                   help="Evaluate an arbitrary labelled CSV instead of MoleculeNet (CV-only).")
    p.add_argument("--task_name", default="custom", help="Name/output key for --task_csv.")
    p.add_argument("--task_type", choices=["classification", "regression"], default="classification")
    args = p.parse_args()

    set_cv_scheme(args.cv_scheme)
    if args.task_csv is not None:
        register_custom_task(args.task_name, args.task_csv)
        ds_list = [(args.task_name, args.task_type)]
    elif args.datasets is not None:
        type_map = dict(MOLECULENET_TASKS)
        ds_list = [(n, type_map.get(n, "classification")) for n in args.datasets]
    else:
        ds_list = MOLECULENET_TASKS

    evaluate(
        encoder_path=args.encoder, tokenizer_path=args.tokenizer, output_dir=args.output_dir,
        head_seeds=args.head_seeds, datasets=ds_list, featurizer=args.featurizer,
        features_npz=args.features_npz,
        pool_mode=args.pool, standardize=args.standardize, head=args.head,
        max_length=args.max_length, train_subsample=args.train_subsample,
        subsample_seed=args.subsample_seed, cv_folds=args.cv_folds,
    )


if __name__ == "__main__":
    main()
