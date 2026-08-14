"""Native chemprop END-TO-END CheMeleon arm on the 7 MoleculeNet tasks, scored under the EXACT A1.b
protocol of record so it drops onto our main figures beside the CLIMB arms: pooled 5-fold SCAFFOLD
cross-validation (folds from eval_v2._scaffold_kfold_indices at subsample_seed=0), primary metric =
ROC-AUC (classification) / RMSE native-units (regression) via heads_v2.compute_metric, plus NEF1%.

To stay apples-to-apples with the frozen CheMeleon arm (chemeleon_bench.py) we MIRROR eval_v2's CV:
the fold partition is fixed (seed 0); within each fold we train 3 chemprop models (seeds 0/1/2) and
AVERAGE their test predictions before scoring; the error bar is the spread ACROSS folds. Each model
is a chemprop D-MPNN initialised from the CheMeleon foundation (--from-foundation CHEMELEON) and
fine-tuned end-to-end — i.e. the CheMeleon model itself, e2e.

Output (schema identical to eval_v2's suite_summary.json → picked up by notebook build_table):
  figure_data/climb_v2_phase2/chemeleon_e2e/moleculenet_cv/suite_summary.json   # <DS>_MEAN/_STD (+ _nef1_*)
Incremental + crash-safe: a (dataset) already present in the summary is skipped on re-run; each
dataset syncs to S3 as soon as it lands. Run on the box with the chemeleon venv (chemprop+deepchem):
  ~/venvs/chemeleon/bin/python scripts/molnet_chemprop_e2e.py
Env: MOLNET_SEEDS ("0 1 2"), MOLNET_EPOCHS (50), MOLNET_RUN (chemeleon_e2e), MOLNET_DATASETS.
"""
import csv
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import eval_v2  # noqa: E402  (loader + scaffold folds)
from heads_v2 import compute_metric, compute_nef  # noqa: E402

CHEMPROP = os.environ.get("CHEMPROP_BIN", str(Path(sys.executable).with_name("chemprop")))
SEEDS = [int(s) for s in os.environ.get("MOLNET_SEEDS", "0 1 2").split()]
EPOCHS = int(os.environ.get("MOLNET_EPOCHS", "50"))
RUN = os.environ.get("MOLNET_RUN", "chemeleon_e2e")
FOUNDATION = os.environ.get("MOLNET_FOUNDATION", "CHEMELEON")  # "" → vanilla chemprop from scratch
K = 5
FOLD_SEED = 0  # matches eval_v2 subsample_seed default so folds == the frozen arm's folds

# dataset -> task_type. Small→large; HIV (41k) last so 6/7 land before the long pole.
DATASETS = [("ESOL", "regression"), ("BACE", "classification"), ("BBBP", "classification"),
            ("Lipophilicity", "regression"), ("QM7", "regression"), ("Tox21", "classification"),
            ("HIV", "classification")]
if os.environ.get("MOLNET_DATASETS"):
    keep = set(os.environ["MOLNET_DATASETS"].split(","))
    DATASETS = [d for d in DATASETS if d[0] in keep]

# PER-SEED run dirs (chemeleon_e2e / _s1 / _s2), matching the notebook's _s{n} seed convention so the
# arm aggregates 3 seeds (spread) and the A1-table scaffold cluster-bootstrap can read runs[0]'s OOF.
def run_name(seed):
    return RUN if seed == 0 else f"{RUN}_s{seed}"


def paths(seed):
    out = ROOT / "figure_data" / "climb_v2_phase2" / run_name(seed) / "moleculenet_cv"
    s3 = f"s3://climb-s3-bucket/experiments/climb_v2_phase2/{run_name(seed)}/moleculenet_cv"
    return out, s3


def _chemprop_version():
    try:
        import chemprop
        return chemprop.__version__
    except Exception:
        return "unknown"


def _foundation_md5():
    """MD5 of the cached CheMeleon foundation checkpoint (Zenodo record 15460715), for provenance."""
    import hashlib
    p = Path.home() / ".chemprop" / "chemeleon_mp.pt"
    if not p.exists():
        return None
    h = hashlib.md5(); h.update(p.read_bytes()); return h.hexdigest()


def _write_train(path, smi, Y):
    ncol = Y.shape[1]
    cols = ["y"] if ncol == 1 else [f"t{j}" for j in range(ncol)]
    with open(path, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["smiles"] + cols)
        for i, s in enumerate(smi):
            w.writerow([s] + ["" if not np.isfinite(Y[i, j]) else Y[i, j] for j in range(ncol)])
    return cols


def _write_test(path, smi):
    with open(path, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["smiles"])
        for s in smi:
            w.writerow([s])


def _find_model(out_dir):
    for pat in ("**/best*.pt", "**/model*.pt", "**/*.ckpt"):
        hits = sorted(Path(out_dir).glob(pat))
        if hits:
            return str(hits[0])
    return None


def _read_preds(path, cols):
    rows = list(csv.DictReader(open(path)))
    out = np.full((len(rows), len(cols)), np.nan)
    for i, r in enumerate(rows):
        for j, c in enumerate(cols):
            try:
                out[i, j] = float(r[c])
            except (KeyError, ValueError):
                pass
    return out


def _train_predict(task_type, tr_smi, tr_Y, te_smi, cols, seed, td, save_to=None):
    trp, tep, outp, predp = td / "tr.csv", td / "te.csv", td / f"out{seed}", td / f"pred{seed}.csv"
    _write_train(trp, tr_smi, tr_Y)
    _write_test(tep, te_smi)
    cmd = [CHEMPROP, "train", "--data-path", str(trp), "--task-type", task_type,
           "--smiles-columns", "smiles", "--target-columns", *cols, "--output-dir", str(outp),
           "--epochs", str(EPOCHS), "--patience", "15", "--split-sizes", "0.9", "0.1", "0.0",
           "--pytorch-seed", str(seed), "--data-seed", str(seed), "--num-workers", "0"]
    if task_type == "classification":
        cmd += ["--class-balance"]
    if FOUNDATION:
        cmd += ["--from-foundation", FOUNDATION]
    r = subprocess.run(cmd, capture_output=True, text=True)
    model = _find_model(outp)
    if model is None:
        raise RuntimeError(f"train failed (seed{seed}): {r.stderr[-1200:]}")
    if save_to is not None:                     # persist the trained D-MPNN checkpoint before temp cleanup
        import shutil
        save_to.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(model, save_to)
    pr = subprocess.run([CHEMPROP, "predict", "--test-path", str(tep), "--model-path", model,
                         "--preds-path", str(predp), "--smiles-columns", "smiles"],
                        capture_output=True, text=True)
    if not predp.exists():
        raise RuntimeError(f"predict failed (seed{seed}): {pr.stderr[-1200:]}")
    return _read_preds(predp, cols)


def run_dataset_seed(name, task_type, seed, out, s3, summary):
    """ONE seed's 5-fold scaffold CV for one dataset. Builds the OOF prediction vector (each molecule
    predicted by the fold that held it out) and dumps per-molecule rows via eval_v2._dump_test_predictions
    (byte-identical to every other arm: dataset, task_type, split, mol_index, canonical_key, raw_smiles,
    output_index, y_true, y_pred), so the A1-table scaffold cluster-bootstrap can rank CheMeleon."""
    if f"{name}_MEAN" in summary:
        print(f"[molnet-e2e] SKIP {name} seed{seed}: already in summary", flush=True)
        return summary
    smi, Y = eval_v2._load_moleculenet_full(name)
    Y = Y if Y.ndim > 1 else Y.reshape(-1, 1)
    ncol = Y.shape[1]
    cols = ["y"] if ncol == 1 else [f"t{j}" for j in range(ncol)]
    folds = eval_v2._scaffold_kfold_indices(smi, K, FOLD_SEED, labels=Y)   # folds fixed (seed 0) across seeds
    oof = np.full(Y.shape, np.nan, dtype=np.float64)                       # per-molecule held-out prediction
    fold_m, fold_nef = [], []
    for j, test_idx in enumerate(folds):
        test_idx = np.asarray(test_idx, dtype=int)
        if len(test_idx) == 0:
            continue
        train_idx = np.asarray([i for t, f in enumerate(folds) if t != j for i in f], dtype=int)
        tr_smi = [smi[i] for i in train_idx]
        te_smi = [smi[i] for i in test_idx]
        with tempfile.TemporaryDirectory() as td:
            save_to = (out / "models" / f"{name}_fold{j}_seed{seed}.pt") if os.environ.get("SAVE_MODELS") else None
            pred = _train_predict(task_type, tr_smi, Y[train_idx], te_smi, cols, seed, Path(td), save_to=save_to)
        oof[test_idx] = pred                                              # single-seed prediction, no averaging
        m = compute_metric(pred, Y[test_idx], task_type)
        fold_m.append(m)
        if task_type == "classification":
            fold_nef.append(compute_nef(pred, Y[test_idx]))
        print(f"[molnet-e2e] {name} s{seed} fold{j}: {'roc' if task_type=='classification' else 'rmse'}={m:.4f}"
              f"{'' if task_type=='regression' else f' nef1={fold_nef[-1]:.4f}'} (ntr={len(train_idx)}, nte={len(test_idx)})",
              flush=True)
    arr = np.array(fold_m, dtype=np.float64)
    summary[f"{name}_MEAN"] = float(np.nanmean(arr)); summary[f"{name}_STD"] = float(np.nanstd(arr))
    if fold_nef:
        na = np.array(fold_nef, dtype=np.float64)
        summary[f"{name}_nef1_MEAN"] = float(np.nanmean(na)); summary[f"{name}_nef1_STD"] = float(np.nanstd(na))
    out.mkdir(parents=True, exist_ok=True)
    (out / "suite_summary.json").write_text(json.dumps(summary, indent=2))
    eval_v2._dump_test_predictions(out, name, task_type, smi, Y, oof)     # append per-molecule OOF rows
    subprocess.run(["aws", "s3", "cp", "--recursive", str(out), s3, "--only-show-errors"], check=False)
    print(f"[molnet-e2e] {name} s{seed}: MEAN={summary[f'{name}_MEAN']:.4f} +-{summary[f'{name}_STD']:.4f} -> synced", flush=True)
    return summary


def main():
    eval_v2.set_cv_scheme("scaffold")
    for seed in SEEDS:                                                    # one full CV run per seed -> per-seed dir
        out, s3 = paths(seed)
        out.mkdir(parents=True, exist_ok=True)
        sp = out / "suite_summary.json"
        summary = json.loads(sp.read_text()) if sp.exists() else {}
        summary.setdefault("_arm", run_name(seed)); summary.setdefault("_protocol", "scaffold-5fold-CV")
        summary.setdefault("_foundation", FOUNDATION); summary.setdefault("_seed", seed)
        # reproducibility recipe: these + the CheMeleon foundation checkpoint fully determine the models
        summary.setdefault("_recipe", {"chemprop": _chemprop_version(), "epochs": EPOCHS, "patience": 15,
                                       "split_sizes": [0.9, 0.1, 0.0], "class_balance_clf": True,
                                       "fold_scheme": "eval_v2._scaffold_kfold_indices(seed=0)",
                                       "pytorch_seed": seed, "data_seed": seed, "foundation_md5": _foundation_md5()})
        for name, tt in DATASETS:
            try:
                summary = run_dataset_seed(name, tt, seed, out, s3, summary)
            except Exception as exc:
                print(f"[molnet-e2e] {name} s{seed}: FAILED {exc}", flush=True)
        done = [d for d, _ in DATASETS if f"{d}_MEAN" in summary]
        print(f"\n[molnet-e2e] seed{seed}: {len(done)}/{len(DATASETS)} datasets done: {done}", flush=True)
        if len(done) == len(DATASETS):
            (out / "verified.json").write_text(json.dumps({"arm": run_name(seed), "protocol": "scaffold-5fold-CV",
                                                            "datasets": done, "seed": seed}))
            subprocess.run(["aws", "s3", "cp", "--recursive", str(out), s3, "--only-show-errors"], check=False)
    if all((paths(s)[0] / "verified.json").exists() for s in SEEDS):
        Path("MOLNET_CHEMELEON_E2E_DONE").write_text("done\n")


if __name__ == "__main__":
    main()
