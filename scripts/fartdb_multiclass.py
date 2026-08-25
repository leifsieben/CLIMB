"""FartDB 5-class taste classification -- a SEPARATE, SELF-CONTAINED multiclass path.

WHY THIS IS ITS OWN FILE AND ITS OWN HEAD. Every other number in the paper flows through
heads_v2.make_head, whose classification path is binary: an MLP under
binary_cross_entropy_with_logits, and a GBTHead that fits ONE BINARY MODEL PER OUTPUT COLUMN.
Teaching that shared code multiclass -- even behind a flag -- puts every existing arm and every
figure one bug away from moving. The fig_F v1/v2 mismatch is what that costs: an unpinned xgboost
shifted 27 of 30 embedding-free cells by a median 0.38 fold SD, larger than the effects being
drawn, and it was invisible until a duplicate-block check caught it.

So heads_v2.py is NOT IMPORTED HERE AT ALL. This module defines its own softmax MLP and its own
multi:softprob XGBoost. Nothing it does can reach another result.

WHAT IS SHARED, DELIBERATELY: the scaffold-fold construction, imported read-only from eval_v2 so
FartDB is split the same way as every other dataset in the suite. It is a pure function of the
molecule list and mutates nothing.

DATA. FartLabs/FartDB, 15,031 molecules, label `Canonicalized Taste` with five classes:
sweet 9542, undefined 2150, bitter 1676, sour 1605, umami 58. `undefined` is KEPT on Leif's
instruction (2026-08-25). Recording the consequence rather than arguing it: 14.3% of the score
then measures predicting that the label is unknown, so this is not a pure taste-discrimination
number. umami at 58 molecules is ~9 per test fold and will dominate the macro variance.

The shipped train/validation/test split is NOT used: 85 SMILES appear in both train and test and
there are 454 duplicate SMILES overall. We dedup on the standardized SMILES and apply our own
scaffold folds, consistent with the rest of the suite.

METRIC. Macro one-vs-rest ROC-AUC computed FROM THE SOFTMAX PROBABILITIES. The head is the
faithful mutually-exclusive model; the metric stays directly comparable to the binary AUCs used
everywhere else in the paper. Accuracy and per-class AUC are reported alongside.
"""
from __future__ import annotations
import argparse, csv, json, os, sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); os.chdir(ROOT)

CLASSES = ["sweet", "undefined", "bitter", "sour", "umami"]
K_FOLDS = 5


# ------------------------------------------------------------------ data
def load_fartdb() -> tuple[list[str], np.ndarray]:
    import pandas as pd
    from huggingface_hub import hf_hub_download
    p = hf_hub_download("FartLabs/FartDB", "data/full-00000-of-00001.parquet", repo_type="dataset")
    df = pd.read_parquet(p)
    df = df.rename(columns={"Standardized SMILES": "smiles", "Canonicalized Taste": "taste"})
    # DEDUP FIRST. 454 SMILES repeat and 85 straddle the shipped split; a duplicate that lands in
    # both train and test is leakage that every arm would benefit from equally and none would flag.
    before = len(df)
    df = df.drop_duplicates(subset="smiles", keep="first").reset_index(drop=True)
    idx = {c: i for i, c in enumerate(CLASSES)}
    keep = df["taste"].isin(idx)
    df = df[keep].reset_index(drop=True)
    y = np.array([idx[t] for t in df["taste"]], dtype=np.int64)
    print(f"[fartdb] {before} rows -> {len(df)} after dedup; class counts "
          f"{ {c: int((y == i).sum()) for i, c in enumerate(CLASSES)} }", flush=True)
    return df["smiles"].tolist(), y


# ------------------------------------------------------------------ heads (LOCAL, not heads_v2)
class SoftmaxMLP:
    """The same shape as the paper's probe MLP -- one hidden layer, early stopping on a held-out
    split -- but with a softmax output and cross-entropy loss, because the classes are mutually
    exclusive. Written here rather than added to heads_v2 so it cannot affect any other arm."""

    def __init__(self, n_classes: int, seed: int, hidden: int = 512, epochs: int = 200,
                 lr: float = 1e-3, patience: int = 20):
        self.n_classes, self.seed = n_classes, seed
        self.hidden, self.epochs, self.lr, self.patience = hidden, epochs, lr, patience

    def fit(self, xtr, ytr, xva, yva):
        import torch, torch.nn as nn
        torch.manual_seed(self.seed); np.random.seed(self.seed)
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = nn.Sequential(nn.Linear(xtr.shape[1], self.hidden), nn.ReLU(),
                                 nn.Dropout(0.1), nn.Linear(self.hidden, self.n_classes)).to(dev)
        opt = torch.optim.AdamW(self.net.parameters(), lr=self.lr, weight_decay=1e-2)
        Xtr = torch.tensor(np.nan_to_num(xtr), dtype=torch.float32, device=dev)
        Ytr = torch.tensor(ytr, dtype=torch.long, device=dev)
        Xva = torch.tensor(np.nan_to_num(xva), dtype=torch.float32, device=dev)
        Yva = torch.tensor(yva, dtype=torch.long, device=dev)
        best, best_state, bad = float("inf"), None, 0
        for _ in range(self.epochs):
            self.net.train(); opt.zero_grad()
            loss = nn.functional.cross_entropy(self.net(Xtr), Ytr)
            loss.backward(); opt.step()
            self.net.eval()
            with torch.no_grad():
                v = nn.functional.cross_entropy(self.net(Xva), Yva).item()
            if v < best - 1e-5:
                best, bad = v, 0
                best_state = {k: t.detach().clone() for k, t in self.net.state_dict().items()}
            else:
                bad += 1
                if bad >= self.patience:
                    break
        if best_state:
            self.net.load_state_dict(best_state)
        self._dev = dev
        return self

    def predict_proba(self, x):
        import torch
        self.net.eval()
        with torch.no_grad():
            X = torch.tensor(np.nan_to_num(x), dtype=torch.float32, device=self._dev)
            return torch.softmax(self.net(X), dim=1).cpu().numpy()


class SoftmaxXGB:
    """XGBoost's native multi:softprob. The paper's GBTHead deliberately fits one binary model per
    column, which is one-vs-rest; this is the genuinely multiclass objective the data calls for."""

    def __init__(self, n_classes: int, seed: int):
        self.n_classes, self.seed = n_classes, seed

    def fit(self, xtr, ytr, xva, yva):
        import xgboost as xgb
        self.m = xgb.XGBClassifier(objective="multi:softprob", num_class=self.n_classes,
                                   n_estimators=500, learning_rate=0.05, max_depth=6,
                                   subsample=0.8, colsample_bytree=0.8, random_state=self.seed,
                                   early_stopping_rounds=30, eval_metric="mlogloss", n_jobs=8)
        self.m.fit(xtr, ytr, eval_set=[(xva, yva)], verbose=False)
        return self

    def predict_proba(self, x):
        return self.m.predict_proba(x)


# ------------------------------------------------------------------ metric
def macro_ovr_auc(proba: np.ndarray, y: np.ndarray) -> tuple[float, dict]:
    """Macro one-vs-rest AUC from softmax probabilities -- comparable to the binary AUCs used
    everywhere else. A class absent from a fold is SKIPPED and named, not scored as 0.5, so a
    fold that never saw umami cannot quietly drag the macro down."""
    from sklearn.metrics import roc_auc_score
    per = {}
    for i, c in enumerate(CLASSES):
        pos = (y == i)
        if pos.sum() == 0 or pos.sum() == len(y):
            per[c] = float("nan")
            continue
        per[c] = float(roc_auc_score(pos.astype(int), proba[:, i]))
    vals = [v for v in per.values() if v == v]
    return (float(np.mean(vals)) if vals else float("nan")), per


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="output label, e.g. chemberta_mtr")
    ap.add_argument("--featurizer", required=True,
                    choices=["ecfp4", "fp_desc", "chemeleon", "encoder", "hf_encoder"])
    ap.add_argument("--encoder", default=None)
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--hf_model", default=None)
    ap.add_argument("--hf_revision", default=None)
    ap.add_argument("--head", default="mlp", choices=["mlp", "xgb"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 117, 709])
    a = ap.parse_args()

    smiles, y = load_fartdb()

    sys.path.insert(0, str(ROOT))
    import eval_v2 as E                       # read-only: scaffold folds + the shared featurizers
    from scripts.chemeleon_suite_run import make_featurizer  # noqa: E402
    feat, _ = make_featurizer(a.featurizer, a.encoder, a.tokenizer,
                              hf_model=a.hf_model, hf_revision=a.hf_revision)
    X = np.asarray(feat(smiles), dtype=np.float32)
    print(f"[fartdb] features {X.shape}", flush=True)

    folds = E._scaffold_kfold_indices(smiles, K_FOLDS, 0)
    out = ROOT / "figure_data" / "fartdb" / a.model
    out.mkdir(parents=True, exist_ok=True)
    rows, fold_rows = [], []
    for seed in a.seeds:
        for j in range(K_FOLDS):
            te = np.array(folds[j])
            pool = np.array([i for f in range(K_FOLDS) if f != j for i in folds[f]])
            rng = np.random.default_rng(seed); perm = rng.permutation(len(pool))
            nv = max(1, int(0.1 * len(pool)))
            va, tr = pool[perm[:nv]], pool[perm[nv:]]
            # z-score on TRAIN ONLY, matching the suite's treatment of encoder features
            mu, sd = np.nanmean(X[tr], 0), np.nanstd(X[tr], 0)
            sd[sd == 0] = 1.0
            Z = (X - mu) / sd
            head = (SoftmaxMLP(len(CLASSES), seed) if a.head == "mlp"
                    else SoftmaxXGB(len(CLASSES), seed))
            head.fit(Z[tr], y[tr], Z[va], y[va])
            proba = head.predict_proba(Z[te])
            auc, per = macro_ovr_auc(proba, y[te])
            acc = float((proba.argmax(1) == y[te]).mean())
            fold_rows.append(dict(model=a.model, seed=seed, fold=j, metric="macro_ovr_auc",
                                  value=round(auc, 6)))
            fold_rows.append(dict(model=a.model, seed=seed, fold=j, metric="accuracy",
                                  value=round(acc, 6)))
            for c, v in per.items():
                fold_rows.append(dict(model=a.model, seed=seed, fold=j, metric=f"auc_{c}",
                                      value=(round(v, 6) if v == v else "")))
            print(f"  seed {seed} fold {j}: macro_ovr_auc={auc:.4f} acc={acc:.4f}", flush=True)
    for metric in sorted({r["metric"] for r in fold_rows}):
        vals = [r["value"] for r in fold_rows if r["metric"] == metric and r["value"] != ""]
        if vals:
            rows.append(dict(task="FartDB", model=a.model, metric=metric,
                             mean=round(float(np.mean(vals)), 4),
                             std=round(float(np.std(vals)), 4), n=len(vals)))
    with (out / "results.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["task", "model", "metric", "mean", "std", "n"])
        w.writeheader(); w.writerows(rows)
    with (out / "fold_values.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["model", "seed", "fold", "metric", "value"])
        w.writeheader(); w.writerows(fold_rows)
    (out / "verified.json").write_text(json.dumps({
        "task": "FartDB", "model": a.model, "featurizer": a.featurizer,
        "hf_model": a.hf_model, "hf_revision": a.hf_revision,
        "head": f"softmax_{a.head}", "seeds": a.seeds, "n_folds": K_FOLDS,
        "classes": CLASSES, "undefined_kept": True,
        "n_cells": len(a.seeds) * K_FOLDS,
        "note": "separate multiclass path; heads_v2 not imported",
    }, indent=2))
    print(f"[fartdb] wrote {out}/results.csv ({len(rows)} metrics, "
          f"{len(fold_rows)} fold rows)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
