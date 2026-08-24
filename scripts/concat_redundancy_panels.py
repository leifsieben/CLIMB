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
# "CheMel", not "CheMeleon": fig_F's lattice keys on these strings directly, so the tag IS
# the figure's cell name -- a mapping layer here is one more list to go stale.
TAG = os.environ.get("CONCAT_TAG") or {"climb": "CLM", "chemeleon": "CheMel"}[EMB]
PANELS = os.environ.get("CONCAT_PANELS", "MoleculeACE CBS Ames").split()
# CONCAT_DESC picks WHICH descriptor block "desc" means -- the ~200 RDKit descriptors (default,
# every published cell) or the 1,613 Mordred 2D descriptors. Mordred is the sharp comparator for
# CheMeleon, which was pretrained to regress exactly that set. Mirrors concat_redundancy.py so the
# MolNet and panel halves of fig_F cannot drift apart in what "desc" means.
DESC_KIND = os.environ.get("CONCAT_DESC", "rdkit")
if DESC_KIND not in ("rdkit", "mordred"):
    raise SystemExit(f"CONCAT_DESC must be rdkit|mordred, got {DESC_KIND!r}")
MORDRED_NPZ = os.environ.get("CONCAT_MORDRED_NPZ", "figure_data/_mordred_features.npz")
ENC = os.environ.get("CONCAT_ENC", "figure_data/climb_v2_phase2/unsup_8M/encoder")
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


# CONCAT_FEATURES_NPZ: a precomputed {smiles: vector} table, so the CheMeleon arm can run in the
# REFERENCE environment. CheMeleon needs chemprop>=2.2 (python>=3.11) and deepchem 2.8.0 -- which
# defines the Tox21 parse -- has no 3.12 wheel, so the two cannot share an interpreter. A separate
# python3.12 pass produces the table; everything that decides a number stays here.
_FEATURE_TABLE = None


def _chemeleon_from_npz(smiles):
    global _FEATURE_TABLE
    import os as _os
    p = _os.environ.get("CONCAT_FEATURES_NPZ")
    if not p:
        return None
    if _FEATURE_TABLE is None:
        z = np.load(p, allow_pickle=True)
        # Same lazy-npz trap as scripts/concat_redundancy.py: `z["X"]` re-decodes the whole
        # 369 MB member on every iteration. Hoist both arrays before building the dict.
        X, S = z["X"], z["smiles"]
        _FEATURE_TABLE = {str(s): X[i] for i, s in enumerate(S)}
        print(f"[concat] {len(_FEATURE_TABLE)} precomputed CheMeleon vectors from {p}", flush=True)
    miss = [s for s in smiles if str(s) not in _FEATURE_TABLE]
    if miss:
        raise KeyError(f"{len(miss)} SMILES absent from the feature table, e.g. {miss[:2]} -- "
                       "refusing to mean-fill, which would fabricate a molecule's representation")
    return np.asarray([_FEATURE_TABLE[str(s)] for s in smiles], dtype=np.float32)


def embed(smiles):
    if _BLOCK_SEL and not any(TAG in b for b in _BLOCK_SEL):
        emb = None
    elif EMB == "climb":
        return E._encoder_features(_enc, _tok, list(smiles), device, "mean", 256).astype(np.float32)
    pre = _chemeleon_from_npz(list(smiles))
    if pre is not None:
        return pre
    return np.asarray(E._chemeleon_features(list(smiles), device), dtype=np.float32)


_MORDRED_TABLE = None
def _mordred_from_npz(smiles):
    """Strict lookup into the precomputed Mordred table. A miss RAISES rather than mean-filling,
    which would fabricate a molecule's descriptors and quietly change the score."""
    global _MORDRED_TABLE
    if _MORDRED_TABLE is None:
        z = np.load(MORDRED_NPZ, allow_pickle=True)
        # HOIST: npz members decode lazily; z["X"][i] in a comprehension re-reads the whole member
        # per molecule. That wedged this very script's box for 38 minutes once.
        X, S = z["X"], z["smiles"]
        _MORDRED_TABLE = {str(s): X[i] for i, s in enumerate(S)}
        print(f"[panels] {len(_MORDRED_TABLE)} precomputed Mordred vectors "
              f"({X.shape[1]}d) from {MORDRED_NPZ}", flush=True)
    miss = [s for s in smiles if str(s) not in _MORDRED_TABLE]
    if miss:
        raise KeyError(f"{len(miss)} SMILES absent from the Mordred table, e.g. {miss[:2]}")
    return np.asarray([_MORDRED_TABLE[str(s)] for s in smiles], dtype=np.float32)


def feature_sets(smiles):
    """All seven blocks, so a figure can pick CONTROLLED PAIRS rather than a feature-set parade.

    fig_F's redesign (2026-08-19) holds one block fixed and adds one thing, which needs the bare
    bases -- `fp` and `desc` alone -- that the original four-set version never produced. Emitting
    all seven costs one XGBoost fit each on features that are already computed, and it means the
    figure can be re-cut without another box.
    """
    fp = np.asarray(ecfp4_features(smiles), dtype=np.float32)
    if DESC_KIND == "mordred":
        d = _mordred_from_npz(list(smiles))
    else:
        d = np.asarray(rdkit_descriptors(list(smiles)), dtype=np.float32)
    d = np.asarray(d, dtype=np.float32); d[~np.isfinite(d)] = np.nan
    if EMB == "climb":
        emb = embed(smiles)
    else:
        emb = embed(smiles)
    if emb is None:
        return {"fp": fp, "desc": d, "fp+desc": np.concatenate([fp, d], 1)}
    return {"fp": fp, "desc": d, "fp+desc": np.concatenate([fp, d], 1), TAG: emb,
            f"fp+{TAG}": np.concatenate([fp, emb], 1),
            f"desc+{TAG}": np.concatenate([d, emb], 1),
            f"fp+desc+{TAG}": np.concatenate([fp, d, emb], 1)}


# CONCAT_BLOCKS restricts which feature blocks are FIT, without changing which are computed.
# fig_F draws four ticks -- RDKit | Mordred | ECFP4 | RDKit+ECFP4 -- each with and without the
# embedding. That is 6 of the 7 blocks from the RDKit family and only 2 from the Mordred family
# (Mordred's fp+desc combination is not drawn). Each block is 5 XGBoost fits per dataset, so
# fitting only what is drawn gets the figure its data far sooner; the surplus blocks are run
# afterwards into the same files so the CSV record stays complete and the figure can be re-cut.
# Empty/unset = all blocks, so every existing invocation is unchanged.
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

# ---------------- CBS: provided 5-fold UMAP folds, NEF1% ----------------
if "CBS" in PANELS:
    recs = list(csv.DictReader(open(ROOT / "data" / "cbs.csv")))
    smi = [r["smiles"] for r in recs]
    y = np.array([[float(r["y"])] for r in recs])
    fold = np.array([int(r["fold"]) for r in recs])
    F = feature_sets(smi)
    F = _select(F)
    for name, X in F.items():
        nefs, rocs = [], []
        for f in sorted(set(fold.tolist())):
            te = np.where(fold == f)[0]; pool = np.where(fold != f)[0]
            rng = np.random.default_rng(0); perm = rng.permutation(len(pool))
            nv = max(1, int(0.1 * len(pool))); va, tr = pool[perm[:nv]], pool[perm[nv:]]
            p = fit_predict(X, y, tr, te, va, "classification", 1)
            nefs.append(compute_nef(p.reshape(-1, 1), y[te]))
            rocs.append(compute_metric(p, y[te], "classification"))
            fold_rows.append(dict(task="CBS", features=name, metric="nef1", fold=f,
                                  value=round(float(nefs[-1]), 6)))
            fold_rows.append(dict(task="CBS", features=name, metric="roc_auc", fold=f,
                                  value=round(float(rocs[-1]), 6)))
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
    # KEY THE PREDICTION DIR TO THE TABLE, NOT TO THE EMBEDDING FAMILY. It was f"concat_{EMB}",
    # so every run sharing an EMB wrote the same file: CLMunsup and CLMsup, and the RDKit and
    # Mordred families, all landed on concat_climb/test_predictions.csv and silently overwrote
    # each other. Only the last run's Ames predictions survived, with every table still written
    # and every count still passing. The output stem is 1:1 with the table, so this cannot alias.
    d = ROOT / "figure_data" / "chemeleon_suite" / "polaris" / f"concat_{Path(OUTFILE).stem}"
    d.mkdir(parents=True, exist_ok=True)
    with (d / "test_predictions.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["task", "seed", "test_index", "smiles", "y_pred"])
        w.writeheader(); w.writerows(pred_rows)

pd.DataFrame(rows).to_csv(OUTFILE, index=False)
# Ames contributes NO per-fold rows and that is correct, not a gap: Polaris withholds the test
# labels, so there is one held-out evaluation and no fold axis to pair on. The figure carries the
# Hanley-McNeil analytic SE there instead, and must show it as a different kind of interval.
FOLDFILE = str(OUTFILE).replace(".csv", "_folds.csv")
pd.DataFrame(fold_rows).to_csv(FOLDFILE, index=False)
print(f"\nDONE -> {OUTFILE}  (+ {FOLDFILE}, {len(fold_rows)} per-fold rows)", flush=True)
