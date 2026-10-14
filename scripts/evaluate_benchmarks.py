import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from features import apply_standardizer, ecfp4_features, fit_standardizer
from heads import make_head
import evaluate

MACE_DIR = ROOT / "benchmarks" / "data" / "moleculeace"
POLARIS_DIR = ROOT / "benchmarks" / "data" / "polaris"
MACE_TARGET = "y [pEC50/pKi]"


def load_task(track, task):
    """Return (smiles, y[N,1], split[list], cliff[bool array or None], task_type)."""
    if track == "moleculeace":
        rows = list(csv.DictReader((MACE_DIR / f"{task}.csv").open()))
        smi = [r["smiles"] for r in rows]
        y = np.array([float(r[MACE_TARGET]) for r in rows], dtype=np.float64).reshape(-1, 1)
        split = [r["split"] for r in rows]
        cliff = np.array([r["cliff_mol"] in ("1", "1.0", "True") for r in rows])
        return smi, y, split, cliff, "regression"
    if track == "polaris":
        man = json.loads((POLARIS_DIR / "polaris_manifest.json").read_text())
        rows = list(csv.DictReader((POLARIS_DIR / f"{task.replace('/', '__')}.csv").open()))
        smi = [r["smiles"] for r in rows]
        split = [r["split"] for r in rows]
        y = np.array([float(r["y"]) if r["y"] not in ("", "nan", "None") else np.nan
                      for r in rows], dtype=np.float64).reshape(-1, 1)
        return smi, y, split, None, man[task]["type"]
    raise ValueError(f"unknown track {track}")


def task_list(track):
    name = {"moleculeace": "moleculeace_tasks.txt", "polaris": "polaris_tasks.txt"}[track]
    return (ROOT / "benchmarks" / "tasks" / name).read_text().split()


def _rmse(yt, yp):
    return float(np.sqrt(np.mean((yt - yp) ** 2)))


def reg_metrics(yt, yp):
    yt = yt.ravel(); yp = yp.ravel()
    ss_res = np.sum((yt - yp) ** 2); ss_tot = np.sum((yt - np.mean(yt)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    def _sp(a, b):
        ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b))
        return float(np.corrcoef(ra, rb)[0, 1])
    return {"rmse": _rmse(yt, yp), "mae": float(np.mean(np.abs(yt - yp))),
            "r2": r2, "spearman": _sp(yt, yp), "pearson": float(np.corrcoef(yt, yp)[0, 1])}


def clf_metrics(yt, yp):
    from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score
    yt = yt.ravel(); yp = yp.ravel()
    m = ~np.isnan(yt)
    yt, yp = yt[m], yp[m]
    out = {}
    if len(np.unique(yt)) >= 2:
        out["roc_auc"] = float(roc_auc_score(yt, yp))
        out["pr_auc"] = float(average_precision_score(yt, yp))
        out["accuracy"] = float(accuracy_score(yt, (yp > 0.5).astype(int)))
    return out


NPZ_META: dict = {}


def prepare_fold(X, tr, head, std_method):

    import numpy as _np
    import evaluate as _E
    if head != "xgb" and std_method == "none":
        std_method = "zscore"
    tree = (head == "xgb")
    Xa = _np.asarray(X)
    if tree:
        return Xa
    med = _np.nanmedian(_np.where(_np.isfinite(Xa[tr]), Xa[tr], _np.nan), axis=0)
    med = _np.where(_np.isfinite(med), med, 0.0)
    Z = _E._prep(Xa, med)
    if std_method == "zscore":
        mu = _np.nanmean(Z[tr], 0)
        sd = _np.nanstd(Z[tr], 0)
        sd[sd == 0] = 1.0
        Z = (Z - mu) / sd
        Z = _E._prep(Z, med, post=True)
    return _np.asarray(Z, dtype=_np.float32)


def make_featurizer(featurizer, encoder_path, tokenizer_path, hf_model=None, hf_revision=None):
    import torch
    device = torch.device("cuda" if torch.cuda.is_available()
                          else "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
                          else "cpu")
    enc = tok = None
    max_len = 256
    if featurizer == "encoder":
        from transformers import ModernBertModel, PreTrainedTokenizerFast
        tok = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)
        enc = ModernBertModel.from_pretrained(encoder_path, attn_implementation="sdpa",
                                              reference_compile=False).to(device).eval()
    elif featurizer == "npz":


        _table = evaluate._load_feature_table(encoder_path)
        _dim = len(next(iter(_table.values())))


        try:
            _z = np.load(encoder_path, allow_pickle=False)
            NPZ_META.update(json.loads(str(_z["meta"])) if "meta" in _z.files else {})
        except Exception as _e:
            print(f"[npz] WARNING: no meta in {encoder_path} ({_e}) -- provenance will be thin",
                  flush=True)
        print(f"[npz] {encoder_path}: {len(_table)} molecules, dim {_dim} "
              f"model={NPZ_META.get('hf_model', 'UNRECORDED')}", flush=True)
    elif featurizer == "hf_encoder":


        from transformers import AutoModel, AutoTokenizer
        kw = {"trust_remote_code": True}
        if hf_revision:
            kw["revision"] = hf_revision
        tok = AutoTokenizer.from_pretrained(hf_model, **kw)
        enc = AutoModel.from_pretrained(hf_model, **kw).to(device).eval()


        cand = [256]
        for attr in ("max_position_embeddings", "n_positions", "max_len"):
            v = getattr(enc.config, attr, None)
            if isinstance(v, int) and 0 < v < 4096:
                cand.append(v)
        mm = getattr(tok, "model_max_length", None)
        if isinstance(mm, int) and 0 < mm < 4096:
            cand.append(mm)
        max_len = min(cand)
        print(f"[hf_encoder] {hf_model} rev={hf_revision or 'main'} "
              f"hidden={enc.config.hidden_size} max_length={max_len}", flush=True)

    def feat(smiles):
        if featurizer == "ecfp4":
            return np.asarray(ecfp4_features(smiles), dtype=np.float32)
        if featurizer == "fp_desc":
            from descriptors import rdkit_descriptors
            fp = np.asarray(ecfp4_features(smiles), dtype=np.float32)
            d = np.asarray(rdkit_descriptors(list(smiles)), dtype=np.float32)
            d[~np.isfinite(d)] = np.nan
            return np.concatenate([fp, d], axis=1)
        if featurizer == "npz":
            missing = [x for x in smiles if x not in _table]
            if missing:
                raise KeyError(f"{len(missing)} molecules absent from {encoder_path}, "
                               f"e.g. {missing[:3]} -- re-extract rather than impute")
            M = np.asarray([_table[x] for x in smiles], dtype=np.float32)


            bad = ~np.isfinite(M).all(axis=1)
            if bad.any():
                good = M[~bad]
                fill = np.nanmean(good, axis=0) if len(good) else np.zeros(M.shape[1], np.float32)
                M[bad] = fill
                print(f"[npz] {int(bad.sum())}/{len(M)} molecules had no representation "
                      f"({100 * bad.mean():.3f}%) -- imputed to the feature mean", flush=True)
                NPZ_META["n_imputed_this_task"] = int(bad.sum())
            return M
        return evaluate._encoder_features(enc, tok, smiles, device, "mean", max_len)

    std = "none" if featurizer in ("ecfp4", "fp_desc") else "zscore"
    return feat, std


def run(track, featurizer, model, head, seeds, encoder_path, tokenizer_path,
        hf_model=None, hf_revision=None):
    out_dir = ROOT / "figure_data" / "chemeleon_suite" / track / model
    out_dir.mkdir(parents=True, exist_ok=True)
    feat, std_method = make_featurizer(featurizer, encoder_path, tokenizer_path,
                                       hf_model=hf_model, hf_revision=hf_revision)


    if head != "xgb" and std_method == "none":
        print(f"[suite] head={head} on featurizer={featurizer}: standardize none -> zscore "
              f"(matches this arm's MolNet cell; unscaled descriptors collapse an MLP)", flush=True)
        std_method = "zscore"
    tasks = task_list(track)
    rows = []
    pred_rows = []
    for task in tasks:
        smi, y, split, cliff, ttype = load_task(track, task)
        X = feat(smi)
        idx = np.arange(len(smi))
        tr = idx[[s == "train" for s in split]]
        te = idx[[s == "test" for s in split]]


        if TRAIN_FRACTION is not None and TRAIN_FRACTION < 1.0:
            n_keep = max(1, int(round(TRAIN_FRACTION * len(tr))))
            if n_keep < len(tr):
                rng_sub = np.random.default_rng(SUBSAMPLE_SEED)
                tr = np.sort(rng_sub.choice(tr, size=n_keep, replace=False))


        if head != "xgb" and not np.isfinite(X).all():
            bad = ~np.isfinite(X)
            with np.errstate(all="ignore"):
                med = np.nanmedian(np.where(np.isfinite(X[tr]), X[tr], np.nan), axis=0)
            med[~np.isfinite(med)] = 0.0
            print(f"[suite] {task}: head={head} cannot consume NaN -- imputing "
                  f"{int(bad.sum())} non-finite cells in {int(bad.any(0).sum())} column(s) "
                  f"from train medians", flush=True)
            X = np.where(bad, med, X)

        sp = fit_standardizer(X[tr], std_method)
        Xtr, Xte = apply_standardizer(X[tr], sp), apply_standardizer(X[te], sp)
        ysc = evaluate._fit_target_scaler(y[tr], ttype)
        n_out = 1
        yte = y[te]
        te_smi = [smi[i] for i in te]
        has_labels = bool(np.isfinite(yte).any())
        for seed in seeds:
            hd = make_head(head, ttype, n_out, seed)


            if head in ("mlp", "linear") and hasattr(hd, "hp"):
                hd.hp = dict(hd.hp); hd.hp.update({"batch_size": 512, "epochs": 60, "patience": 8})
            hd = hd.fit(Xtr, evaluate._scale_targets(y[tr], ysc), Xtr, evaluate._scale_targets(y[tr], ysc))
            pred = evaluate._unscale_preds(np.asarray(hd.predict(Xte), dtype=np.float64), ysc)
            if ttype == "regression":


                ylo, yhi = float(np.nanmin(y[tr])), float(np.nanmax(y[tr])); m = 0.25 * (yhi - ylo + 1e-9)
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
        print(f"[suite] {track}/{model} {task}: done ({len(te)} test, "
              f"{'scored' if has_labels else 'preds-only'})", flush=True)

    res = out_dir / "results.csv"
    with res.open("w", newline="") as f:
        w = csv.writer(f); w.writerow(["task", "seed", "subset", "metric", "value", "n_test"])
        w.writerows(rows)

    with (out_dir / "test_predictions.csv").open("w", newline="") as f:
        w = csv.writer(f); w.writerow(["task", "seed", "test_index", "smiles", "y_pred"])
        w.writerows(pred_rows)
    n_tasks_done = len({r[0] for r in pred_rows})
    if n_tasks_done == len(tasks):
        (out_dir / "verified.json").write_text(json.dumps(
            {"track": track, "model": model, "featurizer": featurizer, "head": head,
             "seeds": seeds, "n_tasks": n_tasks_done,


             **({"hf_model": hf_model} if hf_model else {}),
             **({"hf_revision": hf_revision} if hf_revision else {}),
             **({"features_npz": encoder_path, "npz_provenance": NPZ_META}
                if featurizer == "npz" else {}),


             **({"fp_variant": os.environ.get("FP_VARIANT", "ecfp4_stereo")}
                if featurizer in ("ecfp4", "fp_desc") else {})}))
    print(f"[suite] wrote {res}  ({n_tasks_done}/{len(tasks)} tasks)", flush=True)


TRAIN_FRACTION = None
SUBSAMPLE_SEED = 0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--track", required=True, choices=["moleculeace", "polaris"])
    p.add_argument("--featurizer", required=True,
               choices=["ecfp4", "fp_desc", "encoder", "hf_encoder", "npz"])
    p.add_argument("--hf_model", default=None, help="HF id for --featurizer hf_encoder")
    p.add_argument("--hf_revision", default=None,
                   help="PIN the checkpoint revision. MoLFormer's main-branch remote code "
                        "imports create_bidirectional_mask, absent from the transformers 4.57.3 "
                        "every CLIMB result is pinned to; revision 7b12d946c181 works. Upgrading "
                        "transformers instead would move every other arm -- that is exactly how "
                        "the fig_F v1/v2 mismatch happened.")
    p.add_argument("--train_fraction", type=float, default=None,
                   help="Label-efficiency sweep: keep this FRACTION of each task's own train split "
                        "(test untouched). Per-task, so points cannot collapse on small tasks.")
    p.add_argument("--subsample_seed", type=int, default=0)
    p.add_argument("--model", required=True, help="output label (e.g. unsup_8M, ecfp4)")
    p.add_argument("--head", default="mlp", choices=["mlp", "linear", "xgb"])
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 117, 709])
    p.add_argument("--encoder", default=None)
    p.add_argument("--tokenizer", default=None)
    a = p.parse_args()
    global TRAIN_FRACTION, SUBSAMPLE_SEED
    TRAIN_FRACTION, SUBSAMPLE_SEED = a.train_fraction, a.subsample_seed
    if TRAIN_FRACTION is not None:
        print(f"[suite] label-efficiency: keeping {TRAIN_FRACTION:.0%} of each task's train split "
              f"(seed {SUBSAMPLE_SEED}); test split untouched", flush=True)
    if a.featurizer == "hf_encoder" and not a.hf_model:
        raise SystemExit("--featurizer hf_encoder requires --hf_model")
    run(a.track, a.featurizer, a.model, a.head, a.seeds, a.encoder, a.tokenizer,
        hf_model=a.hf_model, hf_revision=a.hf_revision)


if __name__ == "__main__":
    main()
