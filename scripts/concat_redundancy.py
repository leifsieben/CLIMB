"""Redundancy test: does the CLIMB embedding add anything on top of fingerprints+descriptors?

Reviewer's highest-value experiment. Feed the SAME XGBoost four feature sets under the SAME 5-fold
scaffold CV, per task:
    fp+desc            (the classical baseline = the fp_desc anchor)
    CLM                (unsup_8M masked-mean embedding, the headline CLM)
    desc+CLM           (clean ablation: does the embedding beat descriptors alone, no fp dilution)
    fp+desc+CLM        (concat: does the embedding add anything to the full baseline)

If fp+desc+CLM does not beat fp+desc, the 512-d embedding is informationally redundant given
fingerprints+descriptors — a much stronger claim than "the CLM scores lower". XGBoost is
scale-invariant, so raw features are concatenated (no standardization). Native-unit regression
targets; Tox21 missing labels are already NaN-masked by the fixed loader.

Writes analysis/rigor/concat_redundancy.csv.
"""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")
import os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # repo root: import eval_v2 etc.
os.chdir(Path(__file__).resolve().parent.parent)
import numpy as np, pandas as pd, torch
from rdkit import RDLogger; RDLogger.DisableLog("rdApp.*")

import eval_v2 as E
from eval_v2 import ecfp4_features
from descriptors_v2 import rdkit_descriptors
from heads_v2 import make_head, compute_metric, compute_nef

# EMB selects which foundation embedding is tested against the classical features.
# "climb"     -> frozen CLIMB unsup_8M embedding      (labels: CLM / desc+CLM / fp+desc+CLM)
# "chemeleon" -> frozen CheMeleon fingerprint         (labels: CheMeleon / desc+CheMeleon / ...)
# Promoted to main-text Fig F 2026-08-18: if BOTH embeddings are redundant to fp+desc, the claim
# generalises from "CLIMB is redundant" to "these molecular foundation embeddings are redundant".
EMB = os.environ.get("CONCAT_EMB", "climb")
# "CheMel", not "CheMeleon": fig_F's lattice keys on these strings directly, so the tag IS
# the figure's cell name -- a mapping layer here is one more list to go stale.
TAG = os.environ.get("CONCAT_TAG") or {"climb": "CLM", "chemeleon": "CheMel"}[EMB]
OUTFILE = os.environ.get("CONCAT_OUT", f"concat_redundancy{'' if EMB=='climb' else '_chemeleon'}.csv")
# CONCAT_ENC / CONCAT_TAG make the CLIMB arm a PARAMETER. This was hardcoded to unsup_8M, so every
# "CLM" cell fig_F has ever drawn is the unsupervised arm and the supervised one had never been
# through this experiment at all. The default is unchanged so every existing table still reproduces.
ENC = os.environ.get("CONCAT_ENC", "figure_data/climb_v2_phase2/unsup_8M/encoder")
TOK = "figure_data/_tokenizer"
# CONCAT_TASKS limits the run to a subset, for reproducing a single cell cheaply. Added because
# "is this table the same protocol as that one, or a changed recipe?" is unanswerable from
# provenance once two boxes with their own script copies are involved -- but it IS answerable by
# re-running one dataset in the SAME environment and checking the value is bit-identical. That
# separates a recipe change from environment-level float nondeterminism, which is the actual
# question when two tables disagree by less than their own fold spread.
# CONCAT_DESC selects WHICH descriptor block "desc" means: the ~200 RDKit descriptors (default,
# what every published fig_F cell used) or the 1,613 Mordred 2D descriptors.
#
# WHY MORDRED IS THE INTERESTING ONE. CheMeleon is a D-MPNN pretrained to regress exactly these
# 1,613 Mordred descriptors. Against RDKit descriptors, "CheMeleon adds nothing" is a statement
# about two different feature families. Against MORDRED, it is the sharp version: if CheMeleon
# adds nothing on top of its own pretraining target, it compressed that target rather than
# learning structure beyond it. Calculator(descriptors, ignore_3D=True) is 1613 descriptors --
# that set exactly, not an approximation of it.
DESC_KIND = os.environ.get("CONCAT_DESC", "rdkit")
if DESC_KIND not in ("rdkit", "mordred"):
    raise SystemExit(f"CONCAT_DESC must be rdkit|mordred, got {DESC_KIND!r}")
MORDRED_NPZ = os.environ.get("CONCAT_MORDRED_NPZ", "figure_data/_mordred_features.npz")
_TASK_SEL = os.environ.get("CONCAT_TASKS", "").split()
TASKS = [("ESOL", "regression"), ("QM7", "regression"), ("BBBP", "classification"),
         ("BACE", "classification"), ("Tox21", "classification"), ("HIV", "classification")]
if _TASK_SEL:
    TASKS = [t for t in TASKS if t[0] in _TASK_SEL]
    if not TASKS:
        raise SystemExit(f"CONCAT_TASKS={_TASK_SEL} matched no dataset")
SEEDS = [0]           # XGBoost seed(s) per fold; fold spread is the error bar
K, ML = 5, 256
OUT = Path("analysis/rigor"); OUT.mkdir(parents=True, exist_ok=True)

device = torch.device("mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
                      else "cuda" if torch.cuda.is_available() else "cpu")
print("device:", device, flush=True)
if EMB == "climb":
    # imported lazily: the chemeleon venv (chemprop) has no transformers, and the CheMeleon arm
    # does not need it -- a top-level import made that arm fail before it ran a single model.
    from transformers import ModernBertModel, PreTrainedTokenizerFast
    encoder = ModernBertModel.from_pretrained(ENC, attn_implementation="sdpa", reference_compile=False).to(device)
    encoder.eval()
    tok = PreTrainedTokenizerFast.from_pretrained(TOK)
else:
    encoder = tok = None   # CheMeleon featurizer needs no local encoder (chemprop foundation)


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
        # HOIST THE ARRAYS OUT OF THE COMPREHENSION. np.load on an .npz is LAZY: every `z["X"]`
        # decodes the whole member afresh, so `z["X"][i]` inside the loop re-read all 369 MB once
        # per molecule -- ~150k full decompressions, one core pinned, no progress and no error.
        # It wedged the fig_F concat box for 38 minutes on a 1,128-molecule dataset (2026-08-20).
        X, S = z["X"], z["smiles"]
        _FEATURE_TABLE = {str(s): X[i] for i, s in enumerate(S)}
        print(f"[concat] {len(_FEATURE_TABLE)} precomputed CheMeleon vectors from {p}", flush=True)
    miss = [s for s in smiles if str(s) not in _FEATURE_TABLE]
    if miss:
        raise KeyError(f"{len(miss)} SMILES absent from the feature table, e.g. {miss[:2]} -- "
                       "refusing to mean-fill, which would fabricate a molecule's representation")
    return np.asarray([_FEATURE_TABLE[str(s)] for s in smiles], dtype=np.float32)


_MORDRED_TABLE = None
def _mordred_from_npz(smiles):
    """Strict lookup into the precomputed Mordred table. A miss RAISES.

    Mean-filling a missing molecule would fabricate its descriptor vector and quietly change the
    score -- the same failure the CheMeleon table refuses. The table is built by
    scripts/compute_mordred.py from SMILES exported by the reference environment, so a miss means
    the two disagree about the molecule set and that is worth stopping for.
    """
    global _MORDRED_TABLE
    if _MORDRED_TABLE is None:
        z = np.load(MORDRED_NPZ, allow_pickle=True)
        # HOIST. npz members decode lazily; z["X"][i] inside a comprehension re-reads the whole
        # member per molecule. That wedged a box for 38 minutes once.
        X, S = z["X"], z["smiles"]
        _MORDRED_TABLE = {str(s): X[i] for i, s in enumerate(S)}
        print(f"[concat] {len(_MORDRED_TABLE)} precomputed Mordred vectors "
              f"({X.shape[1]}d) from {MORDRED_NPZ}", flush=True)
    miss = [s for s in smiles if str(s) not in _MORDRED_TABLE]
    if miss:
        raise KeyError(f"{len(miss)} SMILES absent from the Mordred table, e.g. {miss[:2]} -- "
                       "refusing to mean-fill, which would fabricate a molecule's descriptors")
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
    # SKIP THE EMBEDDING when nothing selected needs it. fp/desc/fp+desc are identical across all
    # three tags -- proven bit-for-bit over 48 cells -- so they are computed ONCE per family rather
    # than three times, and that shared run should not pay for an encoder forward pass it discards.
    if _BLOCK_SEL and not any(TAG in b for b in _BLOCK_SEL):
        emb = None
    elif EMB == "climb":
        emb = E._encoder_features(encoder, tok, list(smiles), device, "mean", ML).astype(np.float32)
    else:
        emb = _chemeleon_from_npz(list(smiles))
        if emb is None:
            emb = np.asarray(E._chemeleon_features(list(smiles), device), dtype=np.float32)
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



rows = []
fold_rows = []
for ds, tt in TASKS:
    s_all, y_all = E._load_moleculenet_full(ds)
    y_all = np.asarray(y_all, dtype=np.float64)
    if y_all.ndim == 1:
        y_all = y_all[:, None]
    n_out = y_all.shape[1]
    print(f"\n[{ds}] {len(s_all)} molecules, {n_out} output(s) — featurizing…", flush=True)
    F = feature_sets(s_all)
    F = _select(F)
    folds = E._scaffold_kfold_indices(s_all, K, 0)
    for name, X in F.items():
        fm, fn = [], []
        for j in range(K):
            test = np.array(folds[j])
            pool_ = np.array([i for f in range(K) if f != j for i in folds[f]])
            rng = np.random.default_rng(0); perm = rng.permutation(len(pool_))
            nv = max(1, int(0.1 * len(pool_)))
            va, tr = pool_[perm[:nv]], pool_[perm[nv:]]
            preds = []
            for sd in SEEDS:
                h = make_head("xgb", tt, n_out, sd).fit(X[tr], y_all[tr], X[va], y_all[va])
                preds.append(np.asarray(h.predict(X[test]), dtype=np.float64))
            pred = np.mean(preds, axis=0)
            fm.append(compute_metric(pred, y_all[test], tt))
            if tt == "classification":
                fn.append(compute_nef(pred, y_all[test]))
        met = "rmse" if tt == "regression" else "roc_auc"
        rows.append(dict(task=ds, features=name, metric=met,
                         mean=round(float(np.mean(fm)), 4), std=round(float(np.std(fm)), 4)))
        if fn:
            rows.append(dict(task=ds, features=name, metric="nef1",
                             mean=round(float(np.mean(fn)), 4), std=round(float(np.std(fn)), 4)))
        # PER-FOLD ROWS. fig_F's y-axis is now LIFT over a reference block, and the honest error
        # bar on a paired lift is the SD of the per-fold DIFFERENCE: both arms see the same folds,
        # so fold difficulty is a common term that cancels when you pair and does not cancel when
        # you subtract two independently-computed SDs. That is why the aggregate SDs run 2-8x the
        # lifts they sit on. These lists already existed here and were discarded at aggregation;
        # the fold INDEX is what makes them pairable, so it is emitted rather than the order relied
        # upon. Folds are E._scaffold_kfold_indices(s_all, K, 0) -- seeded and deterministic from
        # the molecule list -- so fold j means the same molecules across separate runs, which is
        # what lets a shared-block file pair against a per-tag one.
        for j, v in enumerate(fm):
            fold_rows.append(dict(task=ds, features=name, metric=met, fold=j, value=round(float(v), 6)))
        for j, v in enumerate(fn):
            fold_rows.append(dict(task=ds, features=name, metric="nef1", fold=j, value=round(float(v), 6)))
        print(f"  {name:12} {met}={np.mean(fm):.4f}±{np.std(fm):.4f}", flush=True)

pd.DataFrame(rows).to_csv(OUT / OUTFILE, index=False)
FOLDFILE = OUTFILE.replace(".csv", "_folds.csv")
pd.DataFrame(fold_rows).to_csv(OUT / FOLDFILE, index=False)
print(f"\nDONE -> analysis/rigor/{OUTFILE}  (+ {FOLDFILE}, {len(fold_rows)} per-fold rows)", flush=True)
