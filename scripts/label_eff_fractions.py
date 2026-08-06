"""Label-efficiency, re-run at per-task FRACTIONS instead of shared absolute counts.

Why this exists
---------------
`b1_replicates_v2.sh` swept a single absolute budget {100,300,1000,3000,full} across all 7
tasks. On the small tasks that budget caps at the train size, so points *collapse* onto each
other (ESOL n=1000/3000/full are the identical 902 molecules; BACE/BBBP n=3000==full). The
figure then plots the same result two or three times, which a reviewer correctly flagged.

Fix: subsample a FRACTION of each task's own single-hold-out train split -- {5,10,25,50,100%}.
Every point is then a genuine without-replacement subset and lands at that task's own n_train,
so no two points on a task share a subset (verified distinct for all 7 tasks). This also makes
the DATA match the Methods text, which already states 5/10/25/50/100%.

Faithfulness
------------
The scoring path is a bit-for-bit replica of eval_v2.evaluate's single-hold-out branch
(lines ~401-459): same _load_moleculenet split, same _subsample_train rng (np.random.default_rng
+ choice(replace=False)), fit_standardizer('zscore') on the *subsampled* train, _fit_target_scaler
on the subsampled train, make_head('mlp'), native-unit unscale, compute_metric / compute_nef, and
the <metric>_train counterpart for the fit-vs-generalise panel. The ONLY change is that the budget
is per-task (fraction*n_train) rather than a shared integer -- so this output can drop straight in
for the frozen arms of Fig B1p1.

Efficiency: encoder features for a (arm, task) are identical across fractions and seeds, so we
featurize ONCE per (arm, task) and slice indices -- HIV (33k mols) is encoded 4 times (once per
arm), not ~48 times.

Covers the four FROZEN arms only (random, unsup, sup, unsup2sup). The e2e arm needs re-finetuning
and is handled separately.

Writes:
  analysis/rigor/label_efficiency_fractions.csv          (tidy long: every cell)
  analysis/rigor/label_efficiency_fractions_summary.csv  (MEAN/STD over subsample x head seeds)
"""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np, pandas as pd, torch
from rdkit import RDLogger; RDLogger.DisableLog("rdApp.*")

import eval_v2 as E
from eval_v2 import _load_moleculenet, _encoder_features, _fit_target_scaler, _scale_targets, _unscale_preds
from featurize_v2 import fit_standardizer, apply_standardizer
from heads_v2 import make_head, compute_metric, compute_nef
from config_v2 import MOLECULENET_TASKS_V2
from transformers import ModernBertModel, PreTrainedTokenizerFast

SRC = "figure_data/climb_v2_phase2"
TOK = "figure_data/_tokenizer"
ARMS = {"random": "random_baseline_00", "unsup": "unsup_8M",
        "sup": "skip_dense_8M", "unsup2sup": "u2s_dense_from8M"}
TASKS = ["ESOL", "Lipophilicity", "QM7", "BBBP", "BACE", "Tox21", "HIV"]
TYPE = dict(MOLECULENET_TASKS_V2)
FRACTIONS = [0.05, 0.10, 0.25, 0.50, 1.00]
HEAD_SEEDS = [0, 1, 2]
POOL, ML, STD, HEAD = "mean", 256, "zscore", "mlp"

OUT = Path("analysis/rigor"); OUT.mkdir(parents=True, exist_ok=True)
LONG = OUT / "label_efficiency_fractions.csv"
SUMM = OUT / "label_efficiency_fractions_summary.csv"

device = torch.device("cuda" if torch.cuda.is_available()
                      else "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
                      else "cpu")
print("device:", device, flush=True)
tok = PreTrainedTokenizerFast.from_pretrained(TOK)


def subsample_idx(n_full: int, frac: float, seed: int) -> np.ndarray:
    """Identical draw to eval_v2._subsample_train(n=round(frac*n_full), seed)."""
    n = n_full if frac >= 1.0 else max(1, round(frac * n_full))
    if n >= n_full:
        return np.arange(n_full)
    return np.random.default_rng(seed).choice(n_full, size=n, replace=False)


rows = []
for arm, run in ARMS.items():
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
        # featurize ONCE (the expensive encoder pass); reuse across fractions/seeds
        tr_x_full = _encoder_features(encoder, tok, list(tr_s), device, POOL, ML)
        va_x = _encoder_features(encoder, tok, list(va_s), device, POOL, ML)
        te_x = _encoder_features(encoder, tok, list(te_s), device, POOL, ML)
        print(f"  [{task}] n_train={n_full} featurized in {time.time()-t0:.0f}s", flush=True)

        for frac in FRACTIONS:
            sub_seeds = [0] if frac >= 1.0 else [0, 1, 2]   # full: subsample is a no-op
            for ss in sub_seeds:
                idx = subsample_idx(n_full, frac, ss)
                n_train = len(idx)
                trx, try_ = tr_x_full[idx], tr_y[idx]
                sp = fit_standardizer(trx, STD)
                trx_s = apply_standardizer(trx, sp)
                vax_s = apply_standardizer(va_x, sp)
                tex_s = apply_standardizer(te_x, sp)
                ysc = _fit_target_scaler(try_, tt)
                for hs in HEAD_SEEDS:
                    hd = make_head(HEAD, tt, n_out, hs).fit(
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
        # flush after each (arm, task) so a crash never loses completed work
        pd.DataFrame(rows).to_csv(LONG, index=False)

df = pd.DataFrame(rows)
df.to_csv(LONG, index=False)
# aggregate over subsample x head seeds -> the plotted point + error bar
g = (df.groupby(["arm", "task", "task_type", "metric", "split", "fraction", "pct"], as_index=False)
       .agg(n_train=("n_train", "max"), mean=("value", "mean"),
            std=("value", "std"), n_cells=("value", "size")))
g["std"] = g["std"].fillna(0.0)
g.sort_values(["arm", "task", "metric", "split", "fraction"]).to_csv(SUMM, index=False)
print(f"\nDONE -> {LONG}  ({len(df)} rows)\n     -> {SUMM}  ({len(g)} points)", flush=True)
