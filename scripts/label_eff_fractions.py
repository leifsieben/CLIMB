from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")
import os, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np, pandas as pd, torch
from rdkit import RDLogger; RDLogger.DisableLog("rdApp.*")

import evaluate as E
from evaluate import _load_moleculenet, _encoder_features, _fit_target_scaler, _scale_targets, _unscale_preds
from features import fit_standardizer, apply_standardizer
from heads import make_head, compute_metric, compute_nef
from config import MOLECULENET_TASKS
from transformers import ModernBertModel, PreTrainedTokenizerFast

SRC = "figure_data/climb_v2_phase2"
TOK = "figure_data/_tokenizer"
_ALL_ARMS = {"random": "random_baseline_00", "unsup": "unsup_8M",
             "sup": "skip_dense_8M", "unsup2sup": "u2s_dense_from8M",
             "fp_desc": "__classical__"}
CLASSICAL = {"fp_desc": ("fp_desc", "xgb")}
_arm_sel = os.environ.get("LE_ARMS", "").split()
ARMS = {a: _ALL_ARMS[a] for a in _arm_sel} if _arm_sel else _ALL_ARMS
TASKS = os.environ.get("LE_TASKS", "QM7 BACE Tox21 HIV").split()
TYPE = dict(MOLECULENET_TASKS)
FRACTIONS = [float(x) for x in os.environ.get("LE_FRACTIONS", "0.05 0.10 0.25 0.50 1.00").split()]
HEAD_SEEDS = list(range(int(os.environ.get("LE_SEEDS", "3"))))
POOL, ML, STD, HEAD = "mean", 256, "zscore", "mlp"

OUT = Path("analysis/rigor"); OUT.mkdir(parents=True, exist_ok=True)
LONG = Path(os.environ.get("LE_LONG", str(OUT / "label_efficiency_fractions.csv")))
SUMM = Path(os.environ.get("LE_SUMM", str(OUT / "label_efficiency_fractions_summary.csv")))

device = torch.device("cuda" if torch.cuda.is_available()
                      else "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
                      else "cpu")
print("device:", device, flush=True)
tok = PreTrainedTokenizerFast.from_pretrained(TOK)


def subsample_idx(n_full: int, frac: float, seed: int) -> np.ndarray:
    """Identical draw to evaluate._subsample_train(n=round(frac*n_full), seed)."""
    n = n_full if frac >= 1.0 else max(1, round(frac * n_full))
    if n >= n_full:
        return np.arange(n_full)
    return np.random.default_rng(seed).choice(n_full, size=n, replace=False)


rows = []
for arm, run in ARMS.items():
    classical = arm in CLASSICAL
    if classical:
        feat_name, head_name = CLASSICAL[arm]
        print(f"\n===== arm={arm} (classical {feat_name} + {head_name}) =====", flush=True)
        encoder = None
    else:
        enc_path = f"{SRC}/{run}/encoder"
        if not Path(enc_path, "model.safetensors").exists():
            print(f"[{arm}] MISSING encoder {enc_path} -- skipping", flush=True)
            continue
        print(f"\n===== arm={arm} ({run}) =====", flush=True)
        encoder = ModernBertModel.from_pretrained(enc_path, attn_implementation="sdpa",
                                                  reference_compile=False).to(device)
        encoder.eval()

    for task in TASKS:
        tt = TYPE[task]
        try:
            tr_s, tr_y, va_s, va_y, te_s, te_y = _load_moleculenet(task)
        except Exception as exc:
            print(f"  [{task}] load failed: {exc}", flush=True); continue
        n_full = len(tr_s)
        n_out = tr_y.shape[1] if tr_y.ndim > 1 else 1
        t0 = time.time()

        if classical:
            from features import ecfp4_features
            from descriptors import rdkit_descriptors

            def _classical(sm):
                fp = np.asarray(ecfp4_features(list(sm)), dtype=np.float32)
                d = np.asarray(rdkit_descriptors(list(sm)), dtype=np.float32)
                d[~np.isfinite(d)] = np.nan
                return np.concatenate([fp, d], axis=1)

            tr_x_full, va_x, te_x = _classical(tr_s), _classical(va_s), _classical(te_s)
        else:
            tr_x_full = _encoder_features(encoder, tok, list(tr_s), device, POOL, ML)
            va_x = _encoder_features(encoder, tok, list(va_s), device, POOL, ML)
            te_x = _encoder_features(encoder, tok, list(te_s), device, POOL, ML)
        print(f"  [{task}] n_train={n_full} featurized in {time.time()-t0:.0f}s", flush=True)

        for frac in FRACTIONS:
            sub_seeds = [0] if frac >= 1.0 else [0, 1, 2]
            for ss in sub_seeds:
                idx = subsample_idx(n_full, frac, ss)
                n_train = len(idx)
                trx, try_ = tr_x_full[idx], tr_y[idx]


                if classical:
                    trx_s, vax_s, tex_s = trx, va_x, te_x
                else:
                    sp = fit_standardizer(trx, STD)
                    trx_s = apply_standardizer(trx, sp)
                    vax_s = apply_standardizer(va_x, sp)
                    tex_s = apply_standardizer(te_x, sp)
                ysc = _fit_target_scaler(try_, tt)
                for hs in HEAD_SEEDS:


                    hd = make_head(head_name if classical else HEAD,
                                   tt, n_out, hs).fit(
                        trx_s, _scale_targets(try_, ysc), vax_s, _scale_targets(va_y, ysc))
                    te_pred = _unscale_preds(np.asarray(hd.predict(tex_s), dtype=np.float64), ysc)
                    tr_pred = _unscale_preds(np.asarray(hd.predict(trx_s), dtype=np.float64), ysc)
                    prim = "roc_auc" if tt == "classification" else "rmse"
                    base = dict(arm=arm, task=task, task_type=tt, fraction=frac,
                                pct=int(round(frac * 100)), n_train=n_train,
                                subsample_seed=ss, head_seed=hs)
                    rows.append({**base, "metric": prim, "split": "test",
                                 "value": compute_metric(te_pred, te_y, tt)})
                    rows.append({**base, "metric": prim, "split": "train",
                                 "value": compute_metric(tr_pred, try_, tt)})
                    if tt == "classification":
                        rows.append({**base, "metric": "nef1", "split": "test",
                                     "value": compute_nef(te_pred, te_y)})
            done = [r for r in rows if r["arm"] == arm and r["task"] == task and r["fraction"] == frac
                    and r["metric"] == ("roc_auc" if tt == "classification" else "rmse") and r["split"] == "test"]
            vals = [r["value"] for r in done]
            print(f"    {task} {int(frac*100):>3}%  n={done[-1]['n_train']:>5}  "
                  f"{'roc_auc' if tt=='classification' else 'rmse'}="
                  f"{np.nanmean(vals):.4f}±{np.nanstd(vals):.4f}", flush=True)

        pd.DataFrame(rows).to_csv(LONG, index=False)

df = pd.DataFrame(rows)
df.to_csv(LONG, index=False)

g = (df.groupby(["arm", "task", "task_type", "metric", "split", "fraction", "pct"], as_index=False)
       .agg(n_train=("n_train", "max"), mean=("value", "mean"),
            std=("value", "std"), n_cells=("value", "size")))
g["std"] = g["std"].fillna(0.0)
g.sort_values(["arm", "task", "metric", "split", "fraction"]).to_csv(SUMM, index=False)
print(f"\nDONE -> {LONG}  ({len(df)} rows)\n     -> {SUMM}  ({len(g)} points)", flush=True)
