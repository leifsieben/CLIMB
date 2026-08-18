"""Independent re-implementation of the XGBoost anchors on BBBP.

Why: BBBP is the one panel where the classical anchors lose badly to every transformer arm
(ECFP+desc 0.904 pooled-OOF ROC-AUC vs 0.95 for the unsupervised encoder) — and where even an
*untrained* random encoder scores 0.941. That pattern is odd enough to be worth reproducing from
scratch rather than trusting the pipeline.

This script shares NOTHING with the evaluation pipeline except the molecules and the fold rule:
  * molecules + labels are read back from the stored out-of-fold prediction file (the exact
    2039 BBBP molecules the pipeline scored),
  * folds are the same deterministic Bemis-Murcko scaffold partition (largest scaffold group
    first into the currently-smallest fold; no seed), re-implemented here in ~10 lines,
  * featurization (Morgan/ECFP4 + RDKit descriptors) and the XGBoost fit are written fresh.

It also fits a few reference models to show what the split itself supports.

Run:  python3 scripts/verify_bbbp_anchor.py
"""
from __future__ import annotations
import numpy as np, pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold
import xgboost as xgb

RDLogger.DisableLog("rdApp.*")
ROOT = Path(__file__).resolve().parent.parent
PRED = ROOT / "figure_data/climb_v2_phase2/fp_desc_anchor/moleculenet_cv/test_predictions.csv"
K = 5


def load_bbbp():
    d = pd.read_csv(PRED)
    d = d[d.dataset == "BBBP"].sort_values("mol_index").reset_index(drop=True)
    assert d.mol_index.is_unique, "expected one pooled out-of-fold row per molecule"
    return d.canonical_key.tolist(), d.y_true.to_numpy(float), d.y_pred.to_numpy(float)


def scaffold_folds(smiles, k=K):
    """Same rule as eval_v2._scaffold_kfold_indices: scaffold-disjoint, largest group first."""
    groups = {}
    for i, s in enumerate(smiles):
        try:
            scaf = MurckoScaffold.MurckoScaffoldSmiles(smiles=s, includeChirality=False)
        except Exception:
            scaf = None
        groups.setdefault(scaf if scaf else f"__noscaffold_{i}", []).append(i)
    folds = [[] for _ in range(k)]
    for g in sorted(groups.values(), key=len, reverse=True):
        folds[min(range(k), key=lambda t: len(folds[t]))].extend(g)
    return folds


DESC = [n for n, _ in Descriptors._descList]


def featurize(smiles):
    """ECFP4 (2048 bit, radius 2) and the full RDKit descriptor block, built independently."""
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fp, ds = [], []
    for s in smiles:
        m = Chem.MolFromSmiles(s)
        if m is None:
            fp.append(np.zeros(2048, np.int8)); ds.append(np.full(len(DESC), np.nan)); continue
        fp.append(np.array(gen.GetFingerprint(m), np.int8))
        v = Descriptors.CalcMolDescriptors(m)
        ds.append(np.array([v.get(n, np.nan) for n in DESC], float))
    # float32 first: a handful of descriptors (Ipc) overflow to ~1e300, which is finite in
    # float64 and would swamp the tree's split search — float32 sends them to inf, and inf
    # then becomes NaN, which XGBoost handles natively as "missing".
    D = np.asarray(ds, np.float32)
    D[~np.isfinite(D)] = np.nan
    return np.asarray(fp, np.int8), D.astype(float)


def cv(X, y, folds, model_fn, needs_scaling=False):
    """Pooled out-of-fold ROC-AUC + per-fold AUCs, over the scaffold-disjoint folds."""
    oof = np.zeros(len(y), float)
    per = []
    for f in folds:
        te = np.array(f)
        tr = np.setdiff1d(np.arange(len(y)), te)
        Xtr, Xte = X[tr], X[te]
        if needs_scaling:
            imp = SimpleImputer(strategy="median").fit(Xtr)
            sc = StandardScaler().fit(imp.transform(Xtr))
            Xtr, Xte = sc.transform(imp.transform(Xtr)), sc.transform(imp.transform(Xte))
        m = model_fn()
        m.fit(Xtr, y[tr])
        p = m.predict_proba(Xte)[:, 1]
        oof[te] = p
        per.append(roc_auc_score(y[te], p))
    return roc_auc_score(y, oof), np.array(per)


def main():
    smiles, y, pipeline_pred = load_bbbp()
    print(f"BBBP: {len(smiles)} molecules, {int(y.sum())} positive ({y.mean():.1%})")
    folds = scaffold_folds(smiles)
    print("fold sizes:", [len(f) for f in folds])
    print(f"\npipeline's stored ECFP+desc predictions -> pooled OOF AUC "
          f"{roc_auc_score(y, pipeline_pred):.4f}\n")

    FP, D = featurize(smiles)
    FPD = np.hstack([FP.astype(float), D])
    print(f"features: ECFP4 {FP.shape}, descriptors {D.shape}\n")

    def xgb_fn(**kw):
        p = dict(n_estimators=500, max_depth=6, learning_rate=0.05, subsample=0.8,
                 colsample_bytree=0.8, reg_lambda=1.0, n_jobs=4, eval_metric="logloss",
                 tree_method="hist")
        p.update(kw)
        return lambda: xgb.XGBClassifier(**p)

    runs = [
        ("XGBoost · ECFP4 only",            FP.astype(float), xgb_fn(), False),
        ("XGBoost · descriptors only",      D,                xgb_fn(), False),
        ("XGBoost · ECFP4 + descriptors",   FPD,              xgb_fn(), False),
        ("XGBoost · ECFP+desc, 2000 trees", FPD,              xgb_fn(n_estimators=2000,
                                                                     learning_rate=0.02), False),
        ("RandomForest · ECFP+desc",        FPD,
         lambda: RandomForestClassifier(n_estimators=500, n_jobs=4, min_samples_leaf=2), False),
        ("LogisticRegression · ECFP+desc",  FPD,
         lambda: LogisticRegression(max_iter=3000, C=1.0), True),
        # The transformer arms are scored with an MLP head on z-scored features, the anchors with
        # XGBoost on raw features. If an MLP on the SAME classical features closes the BBBP gap,
        # the gap is a head/preprocessing artifact rather than a representation difference.
        ("MLP · ECFP+desc (z-scored)",      FPD,
         lambda: MLPClassifier(hidden_layer_sizes=(256, 128), max_iter=400, early_stopping=True,
                               random_state=0), True),
        ("MLP · descriptors only",          D,
         lambda: MLPClassifier(hidden_layer_sizes=(256, 128), max_iter=400, early_stopping=True,
                               random_state=0), True),
    ]
    print(f"{'model':36s} {'pooledOOF':>9s}  {'fold mean':>9s} {'fold sd':>7s}")
    for name, X, fn, sc in runs:
        pooled, per = cv(X, y, folds, fn, needs_scaling=sc)
        print(f"{name:36s} {pooled:9.4f}  {per.mean():9.4f} {per.std(ddof=1):7.4f}")

    # what does the split support with almost no chemistry?
    triv = np.array([[Descriptors.MolWt(Chem.MolFromSmiles(s) or Chem.MolFromSmiles("C")),
                      Descriptors.MolLogP(Chem.MolFromSmiles(s) or Chem.MolFromSmiles("C")),
                      Descriptors.TPSA(Chem.MolFromSmiles(s) or Chem.MolFromSmiles("C"))]
                     for s in smiles])
    pooled, per = cv(triv, y, folds, xgb_fn(n_estimators=300), False)
    print(f"{'XGBoost · MolWt+logP+TPSA only':36s} {pooled:9.4f}  {per.mean():9.4f} "
          f"{per.std(ddof=1):7.4f}")


if __name__ == "__main__":
    main()
