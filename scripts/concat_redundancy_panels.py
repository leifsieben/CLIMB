from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")
import csv, os, sys
from pathlib import Path
import numpy as np, pandas as pd, torch
from rdkit import RDLogger; RDLogger.DisableLog("rdApp.*")

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT); sys.path.insert(0, str(ROOT))
import evaluate as E
from evaluate import ecfp4_features
from descriptors import rdkit_descriptors
from heads import make_head

TAG = os.environ.get("CONCAT_TAG") or "CLM"
PANELS = os.environ.get("CONCAT_PANELS", "MoleculeACE Ames").split()
ENC = os.environ.get("CONCAT_ENC", "figure_data/climb_v2_phase2/unsup_8M/encoder")
TOK = "figure_data/_tokenizer"
SEEDS = [0]
OUT = Path("analysis/rigor"); OUT.mkdir(parents=True, exist_ok=True)


OUTFILE = OUT / os.environ.get("CONCAT_PANEL_OUT", "concat_panels_climb.csv")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device, "| panels:", PANELS, flush=True)
from transformers import ModernBertModel, PreTrainedTokenizerFast
_enc = ModernBertModel.from_pretrained(ENC, attn_implementation="sdpa", reference_compile=False).to(device)
_enc.eval(); _tok = PreTrainedTokenizerFast.from_pretrained(TOK)


def embed(smiles):
    return E._encoder_features(_enc, _tok, list(smiles), device, "mean", 256).astype(np.float32)


def feature_sets(smiles):

    fp = np.asarray(ecfp4_features(smiles), dtype=np.float32)
    d = np.asarray(rdkit_descriptors(list(smiles)), dtype=np.float32)
    d = np.asarray(d, dtype=np.float32); d[~np.isfinite(d)] = np.nan


    emb = None if (_BLOCK_SEL and not any(TAG in b for b in _BLOCK_SEL)) else embed(smiles)
    if emb is None:
        return {"fp": fp, "desc": d, "fp+desc": np.concatenate([fp, d], 1)}
    return {"fp": fp, "desc": d, "fp+desc": np.concatenate([fp, d], 1), TAG: emb,
            f"fp+{TAG}": np.concatenate([fp, emb], 1),
            f"desc+{TAG}": np.concatenate([d, emb], 1),
            f"fp+desc+{TAG}": np.concatenate([fp, d, emb], 1)}


_BLOCK_SEL = [b for b in os.environ.get("CONCAT_BLOCKS", "").split(",") if b]


def _select(F):
    if not _BLOCK_SEL:
        return F
    missing = [b for b in _BLOCK_SEL if b not in F]
    if missing:
        raise SystemExit(f"CONCAT_BLOCKS names unknown blocks {missing}; have {sorted(F)}")
    return {k: v for k, v in F.items() if k in _BLOCK_SEL}


def fit_predict(X, y, tr, te, va, tt, n_out):
    preds = []
    for sd in SEEDS:
        h = make_head("xgb", tt, n_out, sd).fit(X[tr], y[tr], X[va], y[va])
        preds.append(np.asarray(h.predict(X[te]), dtype=np.float64))
    return np.mean(preds, axis=0)


rows = []
fold_rows = []


if "MoleculeACE" in PANELS:
    tasks = (ROOT / "benchmarks" / "tasks" / "moleculeace_tasks.txt").read_text().split()
    per_feat = {}
    for ti, task in enumerate(tasks, 1):
        recs = list(csv.DictReader(open(ROOT / "benchmarks" / "data" / "moleculeace" / f"{task}.csv")))
        smi = [r["smiles"] for r in recs]
        y = np.array([[float(r["y [pEC50/pKi]"])] for r in recs])
        split = [r["split"] for r in recs]
        tr = np.array([i for i, s in enumerate(split) if s == "train"])
        te = np.array([i for i, s in enumerate(split) if s == "test"])
        rng = np.random.default_rng(0); perm = rng.permutation(len(tr))
        nv = max(1, int(0.1 * len(tr))); va, tr2 = tr[perm[:nv]], tr[perm[nv:]]
        F = feature_sets(smi)
        F = _select(F)
        for name, X in F.items():
            p = fit_predict(X, y, tr2, te, va, "regression", 1)
            rmse = float(np.sqrt(np.mean((p - y[te]) ** 2)))
            per_feat.setdefault(name, []).append(rmse)
            fold_rows.append(dict(task="MoleculeACE", features=name, metric="rmse",
                                  fold=task, value=round(rmse, 6)))
        if ti % 5 == 0:
            print(f"  MoleculeACE {ti}/{len(tasks)}", flush=True)
    for name, v in per_feat.items():
        rows.append(dict(task="MoleculeACE", features=name, metric="macro_rmse",
                         mean=round(float(np.mean(v)), 4), std=round(float(np.std(v)), 4)))
        print(f"  MoleculeACE {name:18} macro_rmse={np.mean(v):.4f}", flush=True)


if "Ames" in PANELS:
    recs = list(csv.DictReader(open(ROOT / "benchmarks" / "data" / "polaris" / "tdcommons__ames.csv")))
    smi = [r["smiles"] for r in recs]
    split = [r["split"] for r in recs]
    tr_i = [i for i, s in enumerate(split) if s == "train"]
    te_i = [i for i, s in enumerate(split) if s == "test"]
    y = np.zeros((len(recs), 1))
    for i in tr_i:
        y[i, 0] = float(recs[i]["y"])
    F = feature_sets(smi)
    F = _select(F)
    pred_rows = []
    for name, X in F.items():
        rng = np.random.default_rng(0); perm = rng.permutation(len(tr_i))
        nv = max(1, int(0.1 * len(tr_i)))
        va = np.array(tr_i)[perm[:nv]]; tr = np.array(tr_i)[perm[nv:]]
        p = fit_predict(X, y, tr, np.array(te_i), va, "classification", 1)
        for j, i in enumerate(te_i):
            pred_rows.append(dict(task="tdcommons/ames", seed=name, test_index=j,
                                  smiles=smi[i], y_pred=float(p[j])))
        print(f"  Ames {name:18} predictions written (score off-box via Polaris)", flush=True)


    d = ROOT / "figure_data" / "chemeleon_suite" / "polaris" / f"concat_{Path(OUTFILE).stem}"
    d.mkdir(parents=True, exist_ok=True)
    with (d / "test_predictions.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["task", "seed", "test_index", "smiles", "y_pred"])
        w.writeheader(); w.writerows(pred_rows)

pd.DataFrame(rows).to_csv(OUTFILE, index=False)


FOLDFILE = str(OUTFILE).replace(".csv", "_folds.csv")
pd.DataFrame(fold_rows).to_csv(FOLDFILE, index=False)
print(f"\nDONE -> {OUTFILE}  (+ {FOLDFILE}, {len(fold_rows)} per-fold rows)", flush=True)
