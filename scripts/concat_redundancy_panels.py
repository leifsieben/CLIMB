"""Fig F (main text) extended to the canonical six panels, both embedding arms.

concat_redundancy.py answers "does the embedding add anything on top of fingerprints+descriptors?"
but is hardwired to _load_moleculenet_full + scaffold CV, which only fits BACE/Tox21/QM7. The other
three panels have their own native protocols and must keep them (that is what every other figure
uses for them):
    MoleculeACE  30 tasks, each with a PROVIDED train/test split      -> macro-mean RMSE
    CBS          provided 5-fold UMAP folds                           -> NEF1%
    Ames         Polaris tdcommons/ames, provided split, scored via   -> ROC-AUC
                 benchmark.evaluate() because labels are withheld

Same XGBoost head and seeds as the MoleculeNet run, so the rows drop into the same schema:
    task,features,metric,mean,std
with features in {fp+desc, <TAG>, desc+<TAG>, fp+desc+<TAG>}, TAG = CLM | CheMeleon.

Env: CONCAT_EMB=climb|chemeleon, CONCAT_PANELS="MoleculeACE CBS Ames"
Writes analysis/rigor/concat_panels_<emb>.csv (+ Ames predictions for off-box Polaris scoring).
"""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")
import csv, os, sys
from pathlib import Path
import numpy as np, pandas as pd, torch
from rdkit import RDLogger; RDLogger.DisableLog("rdApp.*")

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT); sys.path.insert(0, str(ROOT))
import eval_v2 as E
from eval_v2 import ecfp4_features
from descriptors_v2 import rdkit_descriptors
from heads_v2 import make_head, compute_metric, compute_nef

EMB = os.environ.get("CONCAT_EMB", "climb")
TAG = {"climb": "CLM", "chemeleon": "CheMeleon"}[EMB]
PANELS = os.environ.get("CONCAT_PANELS", "MoleculeACE CBS Ames").split()
ENC = "figure_data/climb_v2_phase2/unsup_8M/encoder"
TOK = "figure_data/_tokenizer"
SEEDS = [0]
OUT = Path("analysis/rigor"); OUT.mkdir(parents=True, exist_ok=True)
# CONCAT_PANEL_OUT lets a re-run under a different featurizer land BESIDE the existing table
# instead of over it. fig_F is a negative result, so being able to show the old and new side by
# side is the whole point -- and this script writes with to_csv, which is a silent overwrite.
OUTFILE = OUT / os.environ.get("CONCAT_PANEL_OUT", f"concat_panels_{EMB}.csv")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device, "| emb:", EMB, "| panels:", PANELS, flush=True)
if EMB == "climb":
    from transformers import ModernBertModel, PreTrainedTokenizerFast
    _enc = ModernBertModel.from_pretrained(ENC, attn_implementation="sdpa", reference_compile=False).to(device)
    _enc.eval(); _tok = PreTrainedTokenizerFast.from_pretrained(TOK)


def embed(smiles):
    if EMB == "climb":
        return E._encoder_features(_enc, _tok, list(smiles), device, "mean", 256).astype(np.float32)
    return np.asarray(E._chemeleon_features(list(smiles), device), dtype=np.float32)


def feature_sets(smiles):
    fp = np.asarray(ecfp4_features(smiles), dtype=np.float32)
    d = np.asarray(rdkit_descriptors(list(smiles)), dtype=np.float32); d[~np.isfinite(d)] = np.nan
    e = embed(smiles)
    return {"fp+desc": np.concatenate([fp, d], 1), TAG: e,
            f"desc+{TAG}": np.concatenate([d, e], 1),
            f"fp+desc+{TAG}": np.concatenate([fp, d, e], 1)}


def fit_predict(X, y, tr, te, va, tt, n_out):
    preds = []
    for sd in SEEDS:
        h = make_head("xgb", tt, n_out, sd).fit(X[tr], y[tr], X[va], y[va])
        preds.append(np.asarray(h.predict(X[te]), dtype=np.float64))
    return np.mean(preds, axis=0)


rows = []

# ---------------- MoleculeACE: 30 provided-split tasks, macro-mean RMSE ----------------
if "MoleculeACE" in PANELS:
    tasks = (ROOT / "chemeleon_suite" / "tasks" / "moleculeace_tasks.txt").read_text().split()
    per_feat = {}
    for ti, task in enumerate(tasks, 1):
        recs = list(csv.DictReader(open(ROOT / "chemeleon_suite" / "data" / "moleculeace" / f"{task}.csv")))
        smi = [r["smiles"] for r in recs]
        y = np.array([[float(r["y [pEC50/pKi]"])] for r in recs])
        split = [r["split"] for r in recs]
        tr = np.array([i for i, s in enumerate(split) if s == "train"])
        te = np.array([i for i, s in enumerate(split) if s == "test"])
        rng = np.random.default_rng(0); perm = rng.permutation(len(tr))
        nv = max(1, int(0.1 * len(tr))); va, tr2 = tr[perm[:nv]], tr[perm[nv:]]
        F = feature_sets(smi)
        for name, X in F.items():
            p = fit_predict(X, y, tr2, te, va, "regression", 1)
            per_feat.setdefault(name, []).append(float(np.sqrt(np.mean((p - y[te]) ** 2))))
        if ti % 5 == 0:
            print(f"  MoleculeACE {ti}/{len(tasks)}", flush=True)
    for name, v in per_feat.items():
        rows.append(dict(task="MoleculeACE", features=name, metric="macro_rmse",
                         mean=round(float(np.mean(v)), 4), std=round(float(np.std(v)), 4)))
        print(f"  MoleculeACE {name:18} macro_rmse={np.mean(v):.4f}", flush=True)

# ---------------- CBS: provided 5-fold UMAP folds, NEF1% ----------------
if "CBS" in PANELS:
    recs = list(csv.DictReader(open(ROOT / "data" / "cbs.csv")))
    smi = [r["smiles"] for r in recs]
    y = np.array([[float(r["y"])] for r in recs])
    fold = np.array([int(r["fold"]) for r in recs])
    F = feature_sets(smi)
    for name, X in F.items():
        nefs, rocs = [], []
        for f in sorted(set(fold.tolist())):
            te = np.where(fold == f)[0]; pool = np.where(fold != f)[0]
            rng = np.random.default_rng(0); perm = rng.permutation(len(pool))
            nv = max(1, int(0.1 * len(pool))); va, tr = pool[perm[:nv]], pool[perm[nv:]]
            p = fit_predict(X, y, tr, te, va, "classification", 1)
            nefs.append(compute_nef(p.reshape(-1, 1), y[te]))
            rocs.append(compute_metric(p, y[te], "classification"))
        rows.append(dict(task="CBS", features=name, metric="nef1",
                         mean=round(float(np.mean(nefs)), 4), std=round(float(np.std(nefs)), 4)))
        rows.append(dict(task="CBS", features=name, metric="roc_auc",
                         mean=round(float(np.mean(rocs)), 4), std=round(float(np.std(rocs)), 4)))
        print(f"  CBS {name:18} nef1={np.mean(nefs):.4f}", flush=True)

# ---------------- Ames: Polaris provided split; labels withheld -> dump preds ----------------
if "Ames" in PANELS:
    recs = list(csv.DictReader(open(ROOT / "chemeleon_suite" / "data" / "polaris" / "tdcommons__ames.csv")))
    smi = [r["smiles"] for r in recs]
    split = [r["split"] for r in recs]
    tr_i = [i for i, s in enumerate(split) if s == "train"]
    te_i = [i for i, s in enumerate(split) if s == "test"]
    y = np.zeros((len(recs), 1))
    for i in tr_i:
        y[i, 0] = float(recs[i]["y"])
    F = feature_sets(smi)
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
    d = ROOT / "figure_data" / "chemeleon_suite" / "polaris" / f"concat_{EMB}"
    d.mkdir(parents=True, exist_ok=True)
    with (d / "test_predictions.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["task", "seed", "test_index", "smiles", "y_pred"])
        w.writeheader(); w.writerows(pred_rows)

pd.DataFrame(rows).to_csv(OUTFILE, index=False)
print(f"\nDONE -> {OUTFILE}", flush=True)
