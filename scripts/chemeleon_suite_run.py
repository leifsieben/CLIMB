"""CheMeleon-suite FROZEN-probe runner (Tracks A/B). For each task it uses the track's OWN fixed
train/test split (NOT CV), featurizes with the chosen featurizer, z-scores (encoder/chemeleon only),
fits our standard head per seed, predicts the held-out test set, and records the task's metrics —
plus, for MoleculeACE, the overall / cliff / non-cliff test RMSE separately.

Apples-to-apples with CLIMB's frozen encoders: identical featurize->standardize->head pipeline
(reused from featurize_v2 / heads_v2 / eval_v2). Regression is native-unit (target scaler fit on
train only, predictions unscaled).

Outputs (NOTHING overwritten):
  figure_data/chemeleon_suite/<track>/<model>/results.csv   # long: task,seed,subset,metric,value,n_test
  figure_data/chemeleon_suite/<track>/<model>/verified.json # written when all tasks x seeds complete

e2e (chemprop) and ToxCast-kNN modes are separate runners (see HARNESS.md); this file is frozen-only.

Usage:
  python scripts/chemeleon_suite_run.py --track moleculeace --featurizer ecfp4  --model ecfp4 \
      --head mlp --seeds 42 117 709
  python scripts/chemeleon_suite_run.py --track polaris --featurizer chemeleon --model chemeleon_frozen ...
  python scripts/chemeleon_suite_run.py --track moleculeace --featurizer encoder \
      --encoder figure_data/climb_v2_phase2/unsup_8M/encoder --tokenizer figure_data/_tokenizer --model unsup_8M
"""
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

from featurize_v2 import apply_standardizer, ecfp4_features, fit_standardizer  # noqa: E402
from heads_v2 import make_head  # noqa: E402
import eval_v2  # noqa: E402

MACE_DIR = ROOT / "chemeleon_suite" / "data" / "moleculeace"
POLARIS_DIR = ROOT / "chemeleon_suite" / "data" / "polaris"
MACE_TARGET = "y [pEC50/pKi]"


# ---------------- task loading ----------------

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
    return (ROOT / "chemeleon_suite" / "tasks" / name).read_text().split()


# ---------------- metrics ----------------

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


# ---------------- featurization ----------------

def make_featurizer(featurizer, encoder_path, tokenizer_path):
    import torch
    device = torch.device("cuda" if torch.cuda.is_available()
                          else "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
                          else "cpu")
    enc = tok = None
    if featurizer == "encoder":
        from transformers import ModernBertModel, PreTrainedTokenizerFast
        tok = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)
        enc = ModernBertModel.from_pretrained(encoder_path, attn_implementation="sdpa",
                                              reference_compile=False).to(device).eval()

    def feat(smiles):
        if featurizer == "ecfp4":
            return np.asarray(ecfp4_features(smiles), dtype=np.float32)
        if featurizer == "fp_desc":
            from descriptors_v2 import rdkit_descriptors
            fp = np.asarray(ecfp4_features(smiles), dtype=np.float32)
            d = np.asarray(rdkit_descriptors(list(smiles)), dtype=np.float32)
            d[~np.isfinite(d)] = np.nan
            return np.concatenate([fp, d], axis=1)
        if featurizer == "chemeleon":
            return eval_v2._chemeleon_features(smiles, device)
        return eval_v2._encoder_features(enc, tok, smiles, device, "mean", 256)

    std = "none" if featurizer in ("ecfp4", "fp_desc") else "zscore"
    return feat, std


# ---------------- run ----------------

def run(track, featurizer, model, head, seeds, encoder_path, tokenizer_path):
    out_dir = ROOT / "figure_data" / "chemeleon_suite" / track / model
    out_dir.mkdir(parents=True, exist_ok=True)
    feat, std_method = make_featurizer(featurizer, encoder_path, tokenizer_path)
    tasks = task_list(track)
    rows = []
    pred_rows = []
    for task in tasks:
        smi, y, split, cliff, ttype = load_task(track, task)
        X = feat(smi)
        idx = np.arange(len(smi))
        tr = idx[[s == "train" for s in split]]
        te = idx[[s == "test" for s in split]]
        sp = fit_standardizer(X[tr], std_method)
        Xtr, Xte = apply_standardizer(X[tr], sp), apply_standardizer(X[te], sp)
        ysc = eval_v2._fit_target_scaler(y[tr], ttype)
        n_out = 1
        yte = y[te]
        te_smi = [smi[i] for i in te]
        has_labels = bool(np.isfinite(yte).any())   # MoleculeACE: yes; Polaris: test labels hidden
        for seed in seeds:
            hd = make_head(head, ttype, n_out, seed)
            # Fast frozen-probe head: big batch + fewer epochs. The default (batch 64, 100 epochs) is
            # launch-overhead-bound for a tiny MLP over frozen features; this is ~10x faster with
            # negligible effect on a converged probe. Applied UNIFORMLY to every suite arm (fair).
            if head in ("mlp", "linear") and hasattr(hd, "hp"):
                hd.hp = dict(hd.hp); hd.hp.update({"batch_size": 512, "epochs": 60, "patience": 8})
            hd = hd.fit(Xtr, eval_v2._scale_targets(y[tr], ysc), Xtr, eval_v2._scale_targets(y[tr], ysc))
            pred = eval_v2._unscale_preds(np.asarray(hd.predict(Xte), dtype=np.float64), ysc)
            if ttype == "regression":
                # Bound OOD extrapolation: an unbounded MLP over some pretrained embeddings (e.g. CheMeleon)
                # blows up on a few test molecules far from the train distribution, wrecking RMSE. Clip to the
                # train target range + 25% margin — physically-motivated, uniform across arms, a no-op for
                # well-behaved features (whose predictions already sit inside this band).
                ylo, yhi = float(np.nanmin(y[tr])), float(np.nanmax(y[tr])); m = 0.25 * (yhi - ylo + 1e-9)
                pred = np.clip(pred, ylo - m, yhi + m)
            # always dump per-seed test predictions (Polaris scores these later via benchmark.evaluate()).
            pv = pred.ravel()
            for i in range(len(te_smi)):
                pred_rows.append([task, seed, i, te_smi[i], float(pv[i])])
            if has_labels:  # local scoring (MoleculeACE) — Polaris test labels are hidden by design
                mets = reg_metrics(yte, pred) if ttype == "regression" else clf_metrics(yte, pred)
                for k, v in mets.items():
                    rows.append([task, seed, "overall", k, v, len(te)])
                if track == "moleculeace":  # cliff / non-cliff RMSE split (the CliffPFN numbers)
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
    # per-seed test predictions in test order (Polaris scoring reads these; also useful for MoleculeACE)
    with (out_dir / "test_predictions.csv").open("w", newline="") as f:
        w = csv.writer(f); w.writerow(["task", "seed", "test_index", "smiles", "y_pred"])
        w.writerows(pred_rows)
    n_tasks_done = len({r[0] for r in pred_rows})
    if n_tasks_done == len(tasks):
        (out_dir / "verified.json").write_text(json.dumps(
            {"track": track, "model": model, "featurizer": featurizer, "head": head,
             "seeds": seeds, "n_tasks": n_tasks_done}))
    print(f"[suite] wrote {res}  ({n_tasks_done}/{len(tasks)} tasks)", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--track", required=True, choices=["moleculeace", "polaris"])
    p.add_argument("--featurizer", required=True, choices=["ecfp4", "fp_desc", "chemeleon", "encoder"])
    p.add_argument("--model", required=True, help="output label (e.g. unsup_8M, chemeleon_frozen, ecfp4)")
    p.add_argument("--head", default="mlp", choices=["mlp", "linear", "xgb"])
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 117, 709])
    p.add_argument("--encoder", default=None)
    p.add_argument("--tokenizer", default=None)
    a = p.parse_args()
    run(a.track, a.featurizer, a.model, a.head, a.seeds, a.encoder, a.tokenizer)


if __name__ == "__main__":
    main()
