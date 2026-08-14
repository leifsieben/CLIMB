"""Native chemprop END-TO-END arms for the CBS rare-actives VS benchmark (Truong et al. 2026),
so the CheMeleon / chemprop D-MPNN can sit on the SAME cbs plot as our CLIMB arms + XGBoost.

Two e2e arms, each 3 seeds, on the benchmark's OWN provided 5 folds (fold col 1..5), NEF1% headline:
  * chemprop_e2e  — vanilla chemprop D-MPNN trained from scratch (no pretraining)
  * chemeleon_e2e — chemprop D-MPNN initialised from the CheMeleon foundation (--from-foundation
                    CHEMELEON), fine-tuned end-to-end. This IS the CheMeleon model, e2e.

Protocol is IDENTICAL to every other cbs arm so the numbers are comparable: provided folds, NEF1%
computed with the exact `compute_nef` formula from heads_v2 (H_a / min(n,A), n=ceil(0.01*N)), plus
ROC-AUC secondary. Per run = one seed = mean over the 5 folds (within-run STD = fold spread), so it
drops straight into scripts/build_cbs_summary.py alongside the frozen/e2e arms. Completion is judged
from ACHIEVED WORK (suite_summary.json carries cbs_nef1_MEAN), never from a file merely existing;
verified runs are skipped; every finished run syncs to S3 immediately.

Run on the box with the chemprop venv:  ~/venvs/chemeleon/bin/python scripts/cbs_chemprop_e2e.py
Env overrides: CBS_CSV (default data/cbs.csv), CBS_EPOCHS (40), CBS_SEEDS ("0 1 2").
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
os.chdir(ROOT)

CHEMPROP = os.environ.get("CHEMPROP_BIN", str(Path(sys.executable).with_name("chemprop")))
CBS_CSV = Path(os.environ.get("CBS_CSV", "data/cbs.csv"))
EPOCHS = int(os.environ.get("CBS_EPOCHS", "40"))
SEEDS = [int(s) for s in os.environ.get("CBS_SEEDS", "0 1 2").split()]
S3_OUT = "s3://climb-s3-bucket/experiments/cbs_benchmark"
FD = ROOT / "figure_data" / "cbs_benchmark"

# arm label -> foundation name for --from-foundation (None = train from scratch)
ARMS = [("chemprop_e2e", None), ("chemeleon_e2e", "CHEMELEON")]


def _chemprop_version():
    try:
        import chemprop
        return chemprop.__version__
    except Exception:
        return "unknown"


def _foundation_md5(name):
    """MD5 of the cached CheMeleon foundation checkpoint (Zenodo record 15460715), for provenance."""
    if not name:
        return None
    import hashlib
    p = Path.home() / ".chemprop" / "chemeleon_mp.pt"
    if not p.exists():
        return None
    h = hashlib.md5()
    h.update(p.read_bytes())
    return h.hexdigest()


def compute_nef(preds, labels, top_frac=0.01):
    """EXACT copy of heads_v2.compute_nef (single-column case). NEF_x% = H_a / min(n, A)."""
    y = np.asarray(labels, dtype=np.float64).ravel()
    s = np.asarray(preds, dtype=np.float64).ravel()
    mask = ~np.isnan(y)
    y, s = y[mask], s[mask]
    N = len(y)
    A = int((y > 0.5).sum())
    if N == 0 or A == 0:
        return float("nan")
    n = max(1, int(np.ceil(top_frac * N)))
    order = np.argsort(-s, kind="mergesort")
    H_a = int((y[order[:n]] > 0.5).sum())
    return float(H_a / min(n, A))


def roc_auc(preds, labels):
    from sklearn.metrics import roc_auc_score
    y = np.asarray(labels, dtype=np.float64).ravel()
    s = np.asarray(preds, dtype=np.float64).ravel()
    m = ~np.isnan(y)
    y, s = y[m], s[m]
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, s))


def load_cbs():
    rows = list(csv.DictReader(CBS_CSV.open()))
    smi = [r["smiles"] for r in rows]
    y = np.array([float(r["y"]) for r in rows], dtype=np.float64)
    fold = np.array([int(r["fold"]) for r in rows])
    return smi, y, fold


def _write_csv(path, smi, y):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["smiles", "y"])
        for s, v in zip(smi, y):
            w.writerow([s, int(v)])


def _find_model(out_dir):
    for pat in ("**/best*.pt", "**/model*.pt", "**/*.ckpt"):
        hits = sorted(Path(out_dir).glob(pat))
        if hits:
            return str(hits[0])
    return None


def _read_preds(preds_path, n):
    rows = list(csv.DictReader(open(preds_path)))
    # chemprop writes smiles + the target column ("y"); grab the non-smiles numeric column.
    keys = [k for k in rows[0].keys() if k.lower() not in ("smiles",)]
    col = "y" if "y" in keys else keys[-1]
    return np.array([float(r[col]) for r in rows], dtype=np.float64)


def _is_done(run_dir):
    p = run_dir / "moleculenet_cv" / "suite_summary.json"
    if not p.exists():
        return False
    try:
        return json.loads(p.read_text()).get("cbs_nef1_MEAN") is not None
    except Exception:
        return False


def run_arm(arm, foundation, seed):
    run = f"{arm}_s{seed}"
    run_dir = FD / run
    if _is_done(run_dir):
        print(f"[cbs-chemprop] SKIP {run}: already verified", flush=True)
        return True
    smi, y, fold = load_cbs()
    folds = sorted(set(fold.tolist()))
    fold_nef, fold_roc, per_fold = [], [], []
    for f in folds:
        tr = fold != f
        te = fold == f
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            trp, tep, outp, predp = td / "train.csv", td / "test.csv", td / "out", td / "preds.csv"
            _write_csv(trp, [s for s, m in zip(smi, tr) if m], y[tr])
            _write_csv(tep, [s for s, m in zip(smi, te) if m], y[te])
            train_cmd = [CHEMPROP, "train", "--data-path", str(trp), "--task-type", "classification",
                         "--smiles-columns", "smiles", "--target-columns", "y", "--output-dir", str(outp),
                         "--epochs", str(EPOCHS), "--split-sizes", "0.9", "0.1", "0.0",
                         "--pytorch-seed", str(seed), "--data-seed", str(seed),
                         "--num-workers", "0", "--class-balance", "--metrics", "prc", "roc"]
            if foundation:
                train_cmd += ["--from-foundation", foundation]
            r = subprocess.run(train_cmd, capture_output=True, text=True)
            model = _find_model(outp)
            if model is None:
                print(f"[cbs-chemprop] {run} fold{f}: TRAIN FAIL\nSTDERR {r.stderr[-1500:]}", flush=True)
                return False
            if os.environ.get("SAVE_MODELS"):   # persist the trained D-MPNN checkpoint before temp cleanup
                import shutil
                mdir = run_dir / "models"; mdir.mkdir(parents=True, exist_ok=True)
                shutil.copy(model, mdir / f"fold{f}_seed{seed}.pt")
            pr = subprocess.run([CHEMPROP, "predict", "--test-path", str(tep), "--model-path", model,
                                 "--preds-path", str(predp), "--smiles-columns", "smiles"],
                                capture_output=True, text=True)
            if not predp.exists():
                print(f"[cbs-chemprop] {run} fold{f}: PREDICT FAIL\nSTDERR {pr.stderr[-1500:]}", flush=True)
                return False
            p = _read_preds(predp, int(te.sum()))
        nef, roc = compute_nef(p, y[te]), roc_auc(p, y[te])
        fold_nef.append(nef); fold_roc.append(roc)
        per_fold.append({"fold": int(f), "nef1": nef, "roc_auc": roc, "n_test": int(te.sum()),
                         "n_active": int((y[te] > 0.5).sum())})
        print(f"[cbs-chemprop] {run} fold{f}: NEF1%={nef:.3f} ROC={roc:.3f} "
              f"(n={int(te.sum())}, act={int((y[te]>0.5).sum())})", flush=True)

    out = run_dir / "moleculenet_cv"
    out.mkdir(parents=True, exist_ok=True)
    summary = {
        "cbs_nef1_MEAN": float(np.nanmean(fold_nef)), "cbs_nef1_STD": float(np.nanstd(fold_nef)),
        "cbs_MEAN": float(np.nanmean(fold_roc)), "cbs_STD": float(np.nanstd(fold_roc)),
        "arm": arm, "foundation": foundation, "seed": seed, "n_folds": len(folds),
        "epochs": EPOCHS, "cv_scheme": "provided", "metric": "nef1",
        # reproducibility recipe: these + the CheMeleon foundation checkpoint fully determine the models
        "_recipe": {"chemprop": _chemprop_version(), "task_type": "classification",
                    "split_sizes": [0.9, 0.1, 0.0], "class_balance": True,
                    "foundation_md5": _foundation_md5(foundation),
                    "pytorch_seed": seed, "data_seed": seed, "fold_col": "fold (1..5, provided)"},
    }
    (out / "suite_summary.json").write_text(json.dumps(summary, indent=2))
    with (out / "per_fold.csv").open("w", newline="") as fcsv:
        w = csv.DictWriter(fcsv, fieldnames=["fold", "nef1", "roc_auc", "n_test", "n_active"])
        w.writeheader(); w.writerows(per_fold)
    (run_dir / "verified.json").write_text(json.dumps({"run": run, "metric": "nef1", "cv": "provided-5fold"}))
    subprocess.run(["aws", "s3", "cp", "--recursive", str(out),
                    f"{S3_OUT}/{run}/moleculenet_cv", "--only-show-errors"], check=False)
    print(f"[cbs-chemprop] {run}: DONE NEF1%={summary['cbs_nef1_MEAN']:.3f}"
          f"+-{summary['cbs_nef1_STD']:.3f} ROC={summary['cbs_MEAN']:.3f}", flush=True)
    return True


def main():
    if not CBS_CSV.exists():
        sys.exit(f"[cbs-chemprop] FATAL: {CBS_CSV} not found")
    ok, tot = 0, 0
    for arm, foundation in ARMS:
        for seed in SEEDS:
            tot += 1
            if run_arm(arm, foundation, seed):
                ok += 1
    print(f"\n[cbs-chemprop] {ok}/{tot} runs verified", flush=True)
    if ok == tot:
        Path("CBS_CHEMPROP_E2E_DONE").write_text("all cbs chemprop e2e runs verified\n")
        subprocess.run(["aws", "s3", "cp", "CBS_CHEMPROP_E2E_DONE",
                        f"{S3_OUT}/CBS_CHEMPROP_E2E_DONE"], check=False)


if __name__ == "__main__":
    main()
