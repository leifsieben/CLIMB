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


# ---------- custom (non-MoleculeNet) CSV tasks ----------
# A registry + a CV-scheme toggle let us evaluate an arbitrary labelled CSV (smiles,y) under the
# EXACT frozen/e2e head pipeline the paper uses, without touching the published DeepChem+scaffold
# path. Both are module globals so finetune_e2e_v2 — which imports the loader/fold functions from
# here — inherits custom-task support for free once its main() sets them.
_CUSTOM_TASKS: dict = {}          # name -> (csv_path, smiles_col, y_col, fold_col)
_CV_SCHEME: str = "scaffold"      # scaffold (published) | random (stratified) | provided (CSV fold col)
_PROVIDED_FOLDS = None            # per-row fold ids, set by _load_custom_csv, aligned to the smiles order


def register_custom_task(name: str, csv_path: str, smiles_col: str = "smiles", y_col: str = "y",
                         fold_col: str = "fold"):
    _CUSTOM_TASKS[name] = (str(csv_path), smiles_col, y_col, fold_col)


def set_cv_scheme(scheme: str):
    global _CV_SCHEME
    if scheme not in ("scaffold", "random", "provided"):
        raise ValueError(f"cv_scheme must be scaffold|random|provided, got {scheme!r}")
    _CV_SCHEME = scheme


def _load_custom_csv(name: str):
    """Return (smiles, y) for a registered CSV task. y is float32 [N, 1] (single-output). If the
    CSV carries a fold column it is captured into _PROVIDED_FOLDS (row-aligned) for cv_scheme=provided,
    which lets us reproduce a benchmark's OWN similarity-controlled split rather than re-partitioning."""
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
    # Shape [N, 1] to match MoleculeNet single-task convention (n_outputs=1); the OOF buffer,
    # target scaler and metric helpers all assume a 2D label matrix.
    return smiles, np.asarray(ys, dtype=np.float32).reshape(-1, 1)


# ---------- DeepChem dataset loaders ----------

def _load_moleculenet(name: str):
    """Returns (train_smiles, train_y, val_smiles, val_y, test_smiles, test_y)."""
    if name in _CUSTOM_TASKS:
        # Custom CSV tasks are CV-only: there is no canonical train/val/test hold-out for them,
        # and every caller that wants one is the scaffold-holdout path we deliberately don't use.
        raise ValueError(f"custom CSV task {name!r} supports --cv_folds (k-fold CV) only")
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
    #
    # transformers=[]: return RAW targets in native units. DeepChem's default for the regression
    # sets (ESOL/QM7/Lipo) is a NormalizationTransformer fit on ITS scaffold-train split; combined
    # with our downstream concatenate-and-re-split into custom CV folds, that both (a) leaks the
    # normalization stats across folds and (b) leaves RMSE in standardized units mislabelled as
    # log mol/L / physical. We instead scale targets PER FOLD (fit on that fold's train only) and
    # inverse-transform predictions before scoring, so RMSE is native-unit and leakage-free. For
    # classification this drops the BalancingTransformer, which is a no-op here (the heads never
    # used sample weights, and ROC-AUC/NEF are rank-based hence transform-invariant).
    tasks, datasets, _ = loaders[name](featurizer="Raw", splitter="scaffold", reload=False,
                                       transformers=[])
    train_ds, val_ds, test_ds = datasets

    def _y_masked(ds):
        # DeepChem encodes MISSING multitask labels (e.g. Tox21's sparse 12-column matrix)
        # as y=0 with w=0 -- NOT as NaN. This loader used to return only `.y`, so those
        # missing entries (16,012 for Tox21) were indistinguishable from true inactives and
        # were fed as real negatives into head training, validation early-stopping, ROC-AUC,
        # NEF and the paired tests. The whole downstream pipeline masks missing labels by NaN
        # (heads_v2 / compute_metric / _dump_test_predictions all use ~isnan), so we convert
        # the zero-weight entries to NaN here and every existing mask starts working. Single-
        # task datasets (ESOL/BBBP/BACE/QM7/HIV/Lipo) have all-ones w, so this is a no-op.
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


# ---- per-split target standardization (regression only) ------------------------------------
# Rigorous replacement for DeepChem's global NormalizationTransformer: the scaler is fit ONLY on
# the current split's training labels, targets are standardized for head training (helps the MLP
# optimize; harmless for XGBoost), and predictions are inverse-transformed back to native units
# BEFORE any metric so RMSE is reported in log mol/L (ESOL) / physical units (QM7) with no
# cross-fold leakage. Classification returns None -> identity, so class labels are never scaled.
def _fit_target_scaler(y_train: np.ndarray, task_type: str):
    if task_type != "regression":
        return None
    y = np.asarray(y_train, dtype=np.float64)
    mu = np.nanmean(y, axis=0)
    sd = np.nanstd(y, axis=0)
    sd = np.where(np.isfinite(sd) & (sd > 0), sd, 1.0)   # guard constant/empty columns
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
    """Deterministic RANDOM (non-scaffold) k-fold. When `labels` is a binary vector the split
    is STRATIFIED on the class: each class's shuffled indices are dealt round-robin across the
    folds, so every fold holds a proportional share of the positive class. This matters at low
    prevalence — a plain random split can hand a fold zero actives, making NEF1% / ROC-AUC
    undefined for it. NaN labels form their own stratum; with labels=None it is unstratified."""
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
    """Partition by the benchmark's OWN fold column (_PROVIDED_FOLDS), so each held-out fold is
    exactly the paper's leakage-controlled test set — the only scheme comparable to their reported
    per-fold metrics. Errors loudly if the fold ids are missing or don't number exactly k."""
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
    """Partition molecules into k SCAFFOLD-DISJOINT folds (Bemis–Murcko). Every scaffold
    lives entirely in one fold, so each fold's test set contains only scaffolds unseen in
    that fold's training data — the small-data cross-validation analogue of the single
    scaffold split, giving an honest error bar from the fold spread instead of one noisy
    draw. Largest scaffold groups are placed first into the currently-smallest fold to keep
    folds balanced in size. When the module CV scheme is "random" this returns a label-stratified
    random k-fold; when "provided" it returns the benchmark's own fold column."""
    if _CV_SCHEME == "provided":
        return _provided_kfold_indices(len(smiles), k)
    if _CV_SCHEME == "random":
        return _random_kfold_indices(len(smiles), k, seed, labels)
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


_CHEMELEON_FP = None


def _load_feature_table(path: str) -> dict:
    """{canonical SMILES: vector} from an .npz holding `smiles` and `X`."""
    z = np.load(path, allow_pickle=True)
    smi, X = z["smiles"], z["X"]
    if len(smi) != len(X):
        raise ValueError(f"{path}: {len(smi)} smiles but {len(X)} vectors")
    print(f"  [features] {len(smi)} precomputed vectors ({X.shape[1]}d) from {path}", flush=True)
    return {str(s): X[i] for i, s in enumerate(smi)}


def _lookup_features(table: dict, smiles) -> np.ndarray:
    """Strict lookup. A miss RAISES: silently mean-filling would fabricate a molecule's
    representation and quietly change the score, which is exactly the class of error this
    split-featurization path exists to avoid."""
    miss = [s for s in smiles if str(s) not in table]
    if miss:
        raise KeyError(f"{len(miss)} SMILES absent from the feature table, e.g. {miss[:3]}")
    return np.asarray([table[str(s)] for s in smiles], dtype=np.float32)


def _chemeleon_features(smiles: List[str], device, batch_size: int = 512) -> np.ndarray:
    """Fixed CheMeleon D-MPNN embedding (Burns et al. 2025, descriptor-pretrained graph model) per
    molecule — a frozen EXTERNAL featurizer, standardized then fed to the identical head our own
    encoders use, so it is an apples-to-apples comparator. The released model is loaded once and
    cached. Batched so the ~41k HIV set does not build one giant BatchMolGraph."""
    global _CHEMELEON_FP
    if _CHEMELEON_FP is None:
        from chemeleon_fingerprint import CheMeleonFingerprint
        _CHEMELEON_FP = CheMeleonFingerprint(device=str(device))
    from rdkit.Chem import MolFromSmiles
    # Pre-filter RDKit-unparseable SMILES (e.g. hypervalent N in a few MoleculeNet rows): the
    # vendored CheMeleonFingerprint passes MolFromSmiles(...) straight into chemprop's featurizer,
    # which crashes on None / yields a ragged batch that strict NumPy rejects ("inhomogeneous
    # shape"), aborting the whole dataset. We featurize only the valid molecules and neutral-fill
    # the rest, keeping chemeleon_fingerprint.py pristine. No-op when every SMILES parses (CBS).
    N = len(smiles)
    valid_idx = [i for i, s in enumerate(smiles) if MolFromSmiles(s) is not None]
    valid_smi = [smiles[i] for i in valid_idx]
    feats, dim = [], None
    for i in range(0, len(valid_smi), batch_size):
        m = np.asarray(_CHEMELEON_FP(list(valid_smi[i:i + batch_size])), dtype=np.float32)
        dim = dim or (m.shape[1] if m.ndim == 2 else None)
        feats.append(m)
    dim = dim or 2048
    out = np.full((N, dim), np.nan, dtype=np.float32)
    if feats:
        allv = np.concatenate(feats, axis=0)
        for k, idx in enumerate(valid_idx):
            out[idx] = allv[k]
    bad = ~np.isfinite(out).all(axis=1)          # invalid SMILES → neutral (mean) embedding
    if bad.any():
        col_mean = np.nanmean(out[~bad], axis=0) if (~bad).any() else np.zeros(dim, np.float32)
        out[bad] = col_mean
        print(f"  [chemeleon] {int(bad.sum())}/{N} invalid SMILES -> mean-filled", flush=True)
    return out


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

def _impute_stats(x, tree_head):
    """Per-column TRAIN medians for NaN imputation; None for XGBoost, which wants the NaNs."""
    if tree_head:
        return None
    a = np.asarray(x, dtype=np.float64).copy()
    a[~np.isfinite(a)] = np.nan
    med = np.nanmedian(a, axis=0)
    return np.where(np.isfinite(med), med, 0.0)


def _prep(x, med, clip=10.0, post=False):
    """Make a feature block safe for a gradient-trained head; a no-op for XGBoost (med is None).

    XGBoost reads NaN as "missing" and is invariant to monotone rescaling, so the classical
    featurizers hand it raw descriptors on purpose. An MLP can do neither: a NaN poisons every
    weight it touches and an unbounded column (RDKit's Ipc reaches 3.9e23 on BACE) saturates the
    first layer. So for non-tree heads, non-finite entries are imputed with the TRAIN column
    MEDIAN -- not 0, which would sit at an arbitrary point on an unstandardized column and drag
    that column's mean and sd -- and the standardized values are then clipped to +-`clip` sigma,
    the same clip descriptors_v2.normalize() already uses for the MTR targets.

    `post=True` is the after-standardization pass: only the clip applies there.
    """
    if med is None:
        return x
    a = np.asarray(x, dtype=np.float32 if post else np.float64)
    if post:
        return np.clip(a, -clip, clip).astype(np.float32)
    bad = ~np.isfinite(a)
    if bad.any():
        a = np.where(bad, np.broadcast_to(med, a.shape), a)
    return a


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
    # per-molecule dump is append-mode within a run (one call per dataset); clear any prior
    # run's file first so re-evaluating the same output_dir cannot accumulate duplicate rows.
    (out / "test_predictions.csv").unlink(missing_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available()
                          else "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
                          else "cpu")

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
    elif featurizer not in ("ecfp4", "rdkit_desc", "fp_desc", "chemeleon"):
        raise ValueError(f"Unknown featurizer: {featurizer!r} (expected encoder|ecfp4|rdkit_desc|fp_desc|chemeleon)")

    # ECFP bit vectors are 0/1; RDKit descriptors are heterogeneous-scale with NaNs. For
    # both, the intended head is XGBoost (scale-invariant, native NaN handling), so skip
    # standardization. This makes rdkit_desc the classical control for the dense-MTR arm:
    # the SAME 217 descriptors used directly in a tree model vs learned by the CLM.
    #
    # ...BUT ONLY FOR THAT HEAD (2026-08-19). "Skip standardization" encodes an assumption about
    # the head that stopped holding when the SI head comparison started running these featurizers
    # under the MLP. An MLP on raw fp_desc gets columns spanning 20+ orders of magnitude -- Ipc
    # reaches 3.9e23 on BACE -- plus literal NaNs, and it diverges: ESOL RMSE came back as
    # 4,382,073 and BBBP produced NaN predictions that crashed roc_auc_score. Reporting that as
    # "the MLP probe is worse" would be measuring a broken configuration, not a probe. Non-tree
    # heads therefore get exactly what the encoder path gets -- z-score on TRAIN statistics --
    # plus the finite-clip below, so the comparison isolates the head.
    tree_head = head == "xgb"
    std_method = ("none" if (featurizer in ("ecfp4", "rdkit_desc", "fp_desc") and tree_head)
                  else standardize)

    # CheMeleon needs chemprop>=2.2 (Python >=3.11); the environment that DEFINES our Tox21
    # numbers is Python 3.9 with deepchem 2.8.0, and deepchem 2.8.0 has no 3.12 wheel -- the two
    # cannot coexist. Rather than accept the ~0.008 Tox21 drift that mismatched environments
    # produce (the drift that already cost us a figure column), featurization is split out: a
    # 3.12 box turns SMILES into vectors, and every parsing, fold and scoring decision stays
    # here in the reference environment. --features_npz supplies that precomputed table.
    _feature_table = _load_feature_table(features_npz) if features_npz else None

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
        if featurizer == "chemeleon":
            if _feature_table is not None:
                return _lookup_features(_feature_table, smiles)
            return _chemeleon_features(smiles, device)
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
            folds = _scaffold_kfold_indices(s_all, cv_folds, subsample_seed, labels=y_all)
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
                med = _impute_stats(X_all[tr_idx], tree_head)
                sp = fit_standardizer(_prep(X_all[tr_idx], med), std_method)
                trx, vax, tex = (_prep(apply_standardizer(_prep(X_all[ix], med), sp), med,
                                       post=True) for ix in (tr_idx, va_idx, test_idx))
                ysc = _fit_target_scaler(y_all[tr_idx], task_type)   # regression: fit on THIS fold's train only
                seed_preds = []
                for seed in head_seeds:
                    hd = make_head(head, task_type, n_outputs, seed).fit(
                        trx, _scale_targets(y_all[tr_idx], ysc), vax, _scale_targets(y_all[va_idx], ysc))
                    sp_pred = _unscale_preds(np.asarray(hd.predict(tex), dtype=np.float64), ysc)  # back to native units
                    seed_preds.append(sp_pred)
                    # Per-(seed, fold) metric. The seed-averaged rows below stay exactly as they
                    # were -- this only ADDS the un-averaged cells. Without them the CV scheme can
                    # report at most `cv_folds` values per model, because averaging the seeds'
                    # predictions before scoring destroys the seed axis; a "3 seeds x 5 folds"
                    # error bar is then not computable from the output at all.
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
                if task_type == "classification":   # NEF1% is computed PER held-out fold (Truong et al. 2026)
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
        med = _impute_stats(tr_x, tree_head)
        std_params = fit_standardizer(_prep(tr_x, med), std_method)
        tr_x = _prep(apply_standardizer(_prep(tr_x, med), std_params), med, post=True)
        va_x = _prep(apply_standardizer(_prep(va_x, med), std_params), med, post=True)
        te_x = _prep(apply_standardizer(_prep(te_x, med), std_params), med, post=True)

        ysc = _fit_target_scaler(tr_y, task_type)   # regression: fit on the hold-out TRAIN split only
        per_seed = []
        per_seed_nef = []
        seed_preds = []
        per_seed_train = []
        for seed in head_seeds:
            hd = make_head(head, task_type, n_outputs, seed).fit(
                tr_x, _scale_targets(tr_y, ysc), va_x, _scale_targets(va_y, ysc))
            te_pred = _unscale_preds(np.asarray(hd.predict(te_x), dtype=np.float64), ysc)  # native units
            seed_preds.append(np.asarray(te_pred, dtype=np.float64))
            metric = compute_metric(te_pred, te_y, task_type)
            per_seed.append(metric)
            rows.append(_row(ds_name, task_type, main_metric, seed, n_train, metric, t0))
            # TRAIN-set score for the same fitted head. Emitted as "<metric>_train" so the
            # schema is unchanged and every existing consumer (which filters main_metric to
            # rmse/roc_auc) ignores it. Needed for the label-efficiency train-vs-test panel:
            # test alone cannot separate "the head cannot FIT this many labels" from "it fits
            # and fails to GENERALISE", which is the whole question at small N.
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
    p.add_argument("--featurizer", choices=["encoder", "ecfp4", "rdkit_desc", "fp_desc", "chemeleon"], default="encoder")
    p.add_argument("--pool", choices=["cls", "mean", "cls_mean"], default="mean")
    p.add_argument("--standardize", choices=["zscore", "pca_whiten", "none"], default="zscore")
    p.add_argument("--head", choices=["linear", "mlp", "xgb"], default="mlp")
    p.add_argument("--max_length", type=int, default=256)
    p.add_argument("--features_npz", default=None,
                   help="Precomputed {smiles: vector} table (.npz with `smiles` and `X`), used "
                        "instead of running the featurizer in-process. Lets CheMeleon embeddings "
                        "be produced in a chemprop>=2.2 environment while scoring stays in the "
                        "reference environment that defines the published numbers.")
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
        type_map = dict(MOLECULENET_TASKS_V2)
        ds_list = [(n, type_map.get(n, "classification")) for n in args.datasets]
    else:
        ds_list = MOLECULENET_TASKS_V2

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
