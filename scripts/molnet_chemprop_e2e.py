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

OUT = ROOT / "figure_data" / "climb_v2_phase2" / RUN / "moleculenet_cv"
S3 = f"s3://climb-s3-bucket/experiments/climb_v2_phase2/{RUN}/moleculenet_cv"


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


def _train_predict(task_type, tr_smi, tr_Y, te_smi, cols, seed, td):
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
    pr = subprocess.run([CHEMPROP, "predict", "--test-path", str(tep), "--model-path", model,
                         "--preds-path", str(predp), "--smiles-columns", "smiles"],
                        capture_output=True, text=True)
    if not predp.exists():
        raise RuntimeError(f"predict failed (seed{seed}): {pr.stderr[-1200:]}")
    return _read_preds(predp, cols)


def run_dataset(name, task_type, summary):
    if f"{name}_MEAN" in summary:
        print(f"[molnet-e2e] SKIP {name}: already in summary", flush=True)
        return summary
    smi, Y = eval_v2._load_moleculenet_full(name)
    Y = Y if Y.ndim > 1 else Y.reshape(-1, 1)
    ncol = Y.shape[1]
    cols = ["y"] if ncol == 1 else [f"t{j}" for j in range(ncol)]
    folds = eval_v2._scaffold_kfold_indices(smi, K, FOLD_SEED, labels=Y)
    fold_m, fold_nef = [], []
    for j, test_idx in enumerate(folds):
        test_idx = np.asarray(test_idx, dtype=int)
        if len(test_idx) == 0:
            continue
        train_idx = np.asarray([i for t, f in enumerate(folds) if t != j for i in f], dtype=int)
        tr_smi = [smi[i] for i in train_idx]
        te_smi = [smi[i] for i in test_idx]
        seed_preds = []
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            for seed in SEEDS:
                seed_preds.append(_train_predict(task_type, tr_smi, Y[train_idx], te_smi, cols, seed, td))
        pred = np.mean(np.stack(seed_preds, 0), 0)  # ensemble over seeds (mirrors eval_v2 head-seed avg)
        m = compute_metric(pred, Y[test_idx], task_type)
        fold_m.append(m)
        if task_type == "classification":
            fold_nef.append(compute_nef(pred, Y[test_idx]))
        print(f"[molnet-e2e] {name} fold{j}: {'roc' if task_type=='classification' else 'rmse'}={m:.4f}"
              f"{'' if task_type=='regression' else f' nef1={fold_nef[-1]:.4f}'} (ntr={len(train_idx)}, nte={len(test_idx)})",
              flush=True)
    arr = np.array(fold_m, dtype=np.float64)
    summary[f"{name}_MEAN"] = float(np.nanmean(arr)); summary[f"{name}_STD"] = float(np.nanstd(arr))
    if fold_nef:
        na = np.array(fold_nef, dtype=np.float64)
        summary[f"{name}_nef1_MEAN"] = float(np.nanmean(na)); summary[f"{name}_nef1_STD"] = float(np.nanstd(na))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "suite_summary.json").write_text(json.dumps(summary, indent=2))
    subprocess.run(["aws", "s3", "cp", "--recursive", str(OUT), S3, "--only-show-errors"], check=False)
    print(f"[molnet-e2e] {name}: MEAN={summary[f'{name}_MEAN']:.4f} +-{summary[f'{name}_STD']:.4f} -> synced", flush=True)
    return summary


def main():
    eval_v2.set_cv_scheme("scaffold")
    OUT.mkdir(parents=True, exist_ok=True)
    sp = OUT / "suite_summary.json"
    summary = json.loads(sp.read_text()) if sp.exists() else {}
    summary.setdefault("_arm", RUN); summary.setdefault("_protocol", "scaffold-5fold-CV")
    summary.setdefault("_foundation", FOUNDATION); summary.setdefault("_seeds", SEEDS)
    for name, tt in DATASETS:
        try:
            summary = run_dataset(name, tt, summary)
        except Exception as exc:
            print(f"[molnet-e2e] {name}: FAILED {exc}", flush=True)
    done = [d for d, _ in DATASETS if f"{d}_MEAN" in summary]
    print(f"\n[molnet-e2e] {len(done)}/{len(DATASETS)} datasets done: {done}", flush=True)
    if len(done) == len(DATASETS):
        (OUT / "verified.json").write_text(json.dumps({"arm": RUN, "protocol": "scaffold-5fold-CV",
                                                        "datasets": done, "seeds": SEEDS}))
        subprocess.run(["aws", "s3", "cp", "--recursive", str(OUT), S3, "--only-show-errors"], check=False)
        Path(f"MOLNET_{RUN.upper()}_DONE").write_text("done\n")


if __name__ == "__main__":
    main()
