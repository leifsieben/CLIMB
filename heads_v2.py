"""v2 downstream heads + metric.

Three interchangeable heads trained on frozen molecule features (from any of the
encoder / random-encoder / ECFP4 featurizers in featurize_v2):

- LinearHead  : single linear layer (the v1 readout, kept as a diagnostic baseline)
- MLPHead     : 2-layer MLP (ReLU + dropout + weight decay) — the default readout,
                because a single linear layer under-reads frozen embeddings
                ("heads were too small")
- GBTHead     : gradient-boosted trees (XGBoost, sklearn HistGB fallback) — the
                strong classical head; ECFP4 + GBTHead is the "how bad is our CLM
                really?" anchor.

All heads support multitask labels (e.g. Tox21's 12 columns) with per-column NaN
masking. Common interface: ``.fit(train_x, train_y, val_x, val_y)`` then
``.predict(x) -> [N, T]`` (raw logits for classification, values for regression).
Turn predictions into a scalar with ``compute_metric``.
"""

from __future__ import annotations

from typing import List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Pre-registered global head hyperparameters. Tuned ONCE on the dev val split
# (BBBP val for clf, ESOL val for reg) during Exp A, then frozen for all cells.
HEAD_HPARAMS = {
    "linear": {"lr": 1e-3, "weight_decay": 1e-4, "epochs": 100, "patience": 15, "batch_size": 64},
    "mlp": {"hidden": 256, "dropout": 0.2, "lr": 1e-3, "weight_decay": 1e-4,
            "epochs": 100, "patience": 15, "batch_size": 64},
    "xgb": {"n_estimators": 600, "max_depth": 6, "learning_rate": 0.08,
            "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 2,
            "early_stopping_rounds": 40},
}


def _as2d(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=np.float32)
    return y[:, None] if y.ndim == 1 else y


# ---------- metric ----------

def compute_metric(preds: np.ndarray, labels: np.ndarray, task_type: str) -> float:
    """ROC-AUC (classification) or RMSE (regression), averaged over valid label
    columns. NaN labels are ignored; classification columns without both classes
    are skipped."""
    preds = _as2d(preds)
    labels = _as2d(labels)
    if task_type == "classification":
        from sklearn.metrics import roc_auc_score
        scores = []
        for j in range(labels.shape[1]):
            mask = ~np.isnan(labels[:, j])
            if mask.sum() == 0 or len(np.unique(labels[mask, j])) < 2:
                continue
            scores.append(roc_auc_score(labels[mask, j], preds[mask, j]))
        return float(np.mean(scores)) if scores else float("nan")
    # regression: RMSE averaged over columns
    rmses = []
    for j in range(labels.shape[1]):
        mask = ~np.isnan(labels[:, j])
        if mask.sum() == 0:
            continue
        rmses.append(float(np.sqrt(np.mean((preds[mask, j] - labels[mask, j]) ** 2))))
    return float(np.mean(rmses)) if rmses else float("nan")


def compute_nef(preds: np.ndarray, labels: np.ndarray, top_frac: float = 0.01) -> float:
    """Normalized enrichment factor at the top `top_frac` — the virtual-screening
    early-enrichment metric (LIT-PCBA NEF; Tran-Nguyen, Jacquemard & Rognan, JCIM 2020;
    used as NEF1% by Truong et al. 2026, s13321-026-01262-x).

        NEF_x% = EF_x% / EF_x%^max = H_a / min(n, A),   n = ceil(top_frac * N)

    where N = #compounds, A = #actives, n = size of the top-ranked slice, and H_a =
    #actives in that slice. Equivalently precision@top-n, capped by recall when actives
    are scarce. Range [0, 1]; 1 = a perfect ranking put every retrievable active on top.
    Averaged over valid classification columns (per-column for multitask sets like Tox21).
    Only defined for classification (needs discrete actives); NaN if no actives present."""
    preds = _as2d(preds)
    labels = _as2d(labels)
    vals = []
    for j in range(labels.shape[1]):
        mask = ~np.isnan(labels[:, j])
        y = labels[mask, j]
        s = preds[mask, j]
        N = len(y)
        A = int((y > 0.5).sum())
        if N == 0 or A == 0:               # no actives in this column → enrichment undefined
            continue
        n = max(1, int(np.ceil(top_frac * N)))
        order = np.argsort(-s, kind="mergesort")   # descending predicted score; stable
        H_a = int((y[order[:n]] > 0.5).sum())
        vals.append(H_a / min(n, A))
    return float(np.mean(vals)) if vals else float("nan")


# ---------- torch heads (linear / mlp) ----------

class _TorchHead:
    def __init__(self, kind: str, task_type: str, n_outputs: int, seed: int, hparams: dict):
        self.kind = kind
        self.task_type = task_type
        self.n_outputs = n_outputs
        self.seed = seed
        self.hp = hparams
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model: nn.Module = None

    def _build(self, in_dim: int) -> nn.Module:
        if self.kind == "linear":
            return nn.Linear(in_dim, self.n_outputs)
        h = self.hp["hidden"]
        return nn.Sequential(
            nn.Linear(in_dim, h), nn.ReLU(), nn.Dropout(self.hp["dropout"]),
            nn.Linear(h, self.n_outputs),
        )

    def fit(self, train_x, train_y, val_x, val_y):
        try:  # transformers.set_seed when available; identical local fallback otherwise (some
            from transformers import set_seed  # envs, e.g. the chemprop venv, don't ship transformers)
            set_seed(self.seed)
        except ModuleNotFoundError:
            import random as _random
            _random.seed(self.seed); np.random.seed(self.seed); torch.manual_seed(self.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.seed)
        train_x = torch.as_tensor(np.asarray(train_x), dtype=torch.float32)
        val_x = torch.as_tensor(np.asarray(val_x), dtype=torch.float32)
        train_y = torch.as_tensor(_as2d(train_y), dtype=torch.float32)
        val_y = torch.as_tensor(_as2d(val_y), dtype=torch.float32)

        self.model = self._build(train_x.shape[1]).to(self.device)
        opt = torch.optim.Adam(self.model.parameters(), lr=self.hp["lr"],
                               weight_decay=self.hp["weight_decay"])
        is_clf = self.task_type == "classification"
        loss_fn = nn.functional.binary_cross_entropy_with_logits if is_clf else nn.functional.mse_loss

        loader = DataLoader(TensorDataset(train_x, train_y),
                            batch_size=self.hp["batch_size"], shuffle=True)
        best_val, best_state, patience = float("inf"), None, 0
        for _ in range(self.hp["epochs"]):
            self.model.train()
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                valid = ~torch.isnan(yb)
                if valid.sum() == 0:
                    continue
                opt.zero_grad()
                pred = self.model(xb)
                loss = loss_fn(pred[valid], yb[valid])
                loss.backward()
                opt.step()
            # validation
            self.model.eval()
            with torch.no_grad():
                vp = self.model(val_x.to(self.device))
                vy = val_y.to(self.device)
                valid = ~torch.isnan(vy)
                vloss = (loss_fn(vp[valid], vy[valid]).item()
                         if valid.sum() > 0 else float("inf"))
            if vloss < best_val - 1e-5:
                best_val = vloss
                best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
                patience = 0
            else:
                patience += 1
                if patience >= self.hp["patience"]:
                    break
        if best_state is not None:
            self.model.load_state_dict(best_state)
        return self

    def predict(self, x) -> np.ndarray:
        self.model.eval()
        x = torch.as_tensor(np.asarray(x), dtype=torch.float32)
        with torch.no_grad():
            out = self.model(x.to(self.device)).detach().cpu().numpy()
        return _as2d(out)


# ---------- gradient-boosted-trees head (xgboost / hist fallback) ----------

class GBTHead:
    """One GBT model per output column, each trained on that column's non-NaN rows.
    Uses XGBoost when available, else sklearn HistGradientBoosting."""

    def __init__(self, task_type: str, n_outputs: int, seed: int, hparams: dict):
        self.task_type = task_type
        self.n_outputs = n_outputs
        self.seed = seed
        self.hp = hparams
        self.models: List = [None] * n_outputs
        try:
            import xgboost  # noqa: F401
            self._backend = "xgb"
        except Exception:
            self._backend = "hist"

    def _new_model(self):
        is_clf = self.task_type == "classification"
        if self._backend == "xgb":
            import xgboost as xgb
            common = dict(
                n_estimators=self.hp["n_estimators"], max_depth=self.hp["max_depth"],
                learning_rate=self.hp["learning_rate"], subsample=self.hp["subsample"],
                colsample_bytree=self.hp["colsample_bytree"],
                min_child_weight=self.hp.get("min_child_weight", 1),
                random_state=self.seed, n_jobs=0,
                early_stopping_rounds=self.hp.get("early_stopping_rounds"),
            )
            return (xgb.XGBClassifier(eval_metric="logloss", **common) if is_clf
                    else xgb.XGBRegressor(eval_metric="rmse", **common))
        from sklearn.ensemble import (HistGradientBoostingClassifier,
                                       HistGradientBoostingRegressor)
        common = dict(max_depth=self.hp["max_depth"], learning_rate=self.hp["learning_rate"],
                      max_iter=self.hp["n_estimators"], random_state=self.seed,
                      early_stopping=True, validation_fraction=0.15)
        return (HistGradientBoostingClassifier(**common) if is_clf
                else HistGradientBoostingRegressor(**common))

    def fit(self, train_x, train_y, val_x, val_y):
        train_x = np.asarray(train_x, dtype=np.float32)
        val_x = np.asarray(val_x, dtype=np.float32)
        ty = _as2d(train_y)
        vy = _as2d(val_y)
        for j in range(self.n_outputs):
            tm = ~np.isnan(ty[:, j])
            if tm.sum() == 0 or (self.task_type == "classification" and len(np.unique(ty[tm, j])) < 2):
                self.models[j] = None
                continue
            model = self._new_model()
            if self._backend == "xgb":
                vm = ~np.isnan(vy[:, j])
                fit_kw = {}
                if vm.sum() > 0:
                    fit_kw["eval_set"] = [(val_x[vm], vy[vm, j])]
                    fit_kw["verbose"] = False
                else:
                    # No val rows for early stopping: disable it.
                    model.set_params(early_stopping_rounds=None)
                model.fit(train_x[tm], ty[tm, j], **fit_kw)
            else:
                model.fit(train_x[tm], ty[tm, j])
            self.models[j] = model
        return self

    def predict(self, x) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        out = np.full((x.shape[0], self.n_outputs), np.nan, dtype=np.float32)
        is_clf = self.task_type == "classification"
        for j, model in enumerate(self.models):
            if model is None:
                out[:, j] = 0.0
                continue
            if is_clf:
                out[:, j] = model.predict_proba(x)[:, 1]
            else:
                out[:, j] = model.predict(x)
        return out


# ---------- factory ----------

def make_head(head: str, task_type: str, n_outputs: int, seed: int):
    hp = HEAD_HPARAMS[head]
    if head in ("linear", "mlp"):
        return _TorchHead(head, task_type, n_outputs, seed, hp)
    if head == "xgb":
        return GBTHead(task_type, n_outputs, seed, hp)
    raise ValueError(f"Unknown head: {head!r} (expected linear|mlp|xgb)")
