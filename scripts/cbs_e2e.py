"""CBS end-to-end fine-tune vs frozen — the focused e2e experiment on the ONE dataset in the suite
big enough (10.4k) to give e2e a fighting chance. Best-two encoders (unsup_only, sup_only:dense),
provided 5-fold UMAP CV, NEF1% (+ ROC-AUC), 3 fine-tune seeds — directly comparable to the frozen
CBS numbers in figure_data/cbs_benchmark/<arm>/.

No fraction/downsampling dimension: CBS has only 43 actives (8-10/fold), so subsampling train
strips the positive signal (100%≈34 actives, 50%≈17, 25%≈8) and the low-fraction points are noise.
So this answers the meaningful question — does fine-tuning end-to-end beat the frozen probe on the
rare-active screen at full data — rather than a meaningless crossover.

Reuses finetune_predict per fold (respects the provided fold column); does NOT touch shared CV code
(which is MoleculeNet-only). Idempotent per arm, S3-synced, gated by a completion marker.
"""
from __future__ import annotations
import csv, json, os, subprocess, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); os.chdir(ROOT)
from rdkit import RDLogger; RDLogger.DisableLog("rdApp.*")
from finetune_e2e_v2 import finetune_predict, compute_nef, FT_HPARAMS
from sklearn.metrics import roc_auc_score

S3B = "s3://climb-s3-bucket/experiments/climb_v2_phase2"
S3OUT = "s3://climb-s3-bucket/experiments/cbs_benchmark"
TOK = "figure_data/_tokenizer"
CBS = Path("data/cbs.csv")
ENCODERS = {"unsup_8M": "unsup_8M_e2e", "skip_dense_8M": "skip_dense_8M_e2e"}  # src prefix -> output run label
SEEDS = [int(s) for s in os.environ.get("CBS_E2E_SEEDS", "0 1 2").split()]
EPOCHS = int(os.environ.get("CBS_E2E_EPOCHS", FT_HPARAMS["epochs"]))
SMOKE = os.environ.get("CBS_E2E_SMOKE") == "1"  # 1 fold, tiny train, fast path check
LOG = Path("analysis/cbs_e2e.log"); LOG.parent.mkdir(exist_ok=True)


def log(m):
    print(f"[cbs-e2e] {m}", flush=True)
    with LOG.open("a") as f:
        f.write(f"[cbs-e2e] {m}\n")


def sh(c):
    return subprocess.run(c, check=False)


def load_cbs():
    rows = list(csv.DictReader(CBS.open()))
    smi = [r["smiles"] for r in rows]
    y = np.array([[float(r["y"])] for r in rows], dtype=np.float64)
    fold = np.array([int(r["fold"]) for r in rows])
    return smi, y, fold


def stage(prefix):
    enc = ROOT / "figure_data" / "climb_v2_phase2" / prefix / "encoder"
    if not (enc / "model.safetensors").exists():
        enc.mkdir(parents=True, exist_ok=True)
        sh(["aws", "s3", "sync", f"{S3B}/{prefix}/encoder", str(enc), "--only-show-errors"])
    return str(enc)


def done(run):
    p = ROOT / "figure_data" / "cbs_benchmark" / run / "moleculenet_cv" / "suite_summary.json"
    if not p.exists():
        return False
    try:
        return json.loads(p.read_text()).get("cbs_nef1_MEAN") is not None
    except Exception:
        return False


def run_arm(prefix, run):
    from transformers import PreTrainedTokenizerFast
    tok = PreTrainedTokenizerFast.from_pretrained(TOK)
    enc = stage(prefix)
    if not Path(enc, "model.safetensors").exists():
        log(f"ERROR {prefix}: encoder missing"); return False
    smi, y, fold = load_cbs()
    folds = sorted(set(fold.tolist()))
    if SMOKE:
        folds = folds[:1]
    per_fold = []
    for f in folds:
        te = np.where(fold == f)[0]
        pool = np.where(fold != f)[0]
        rng = np.random.default_rng(0)
        perm = rng.permutation(len(pool))
        n_va = max(1, int(0.1 * len(pool)))
        va = pool[perm[:n_va]]; tr = pool[perm[n_va:]]
        if SMOKE:  # tiny train to verify the full path in ~1-2 min
            tr = tr[:400]
        seed_preds = []
        for seed in SEEDS:
            (pred,) = finetune_predict(
                enc, tok, [smi[i] for i in tr], y[tr], [smi[i] for i in va], y[va],
                [[smi[i] for i in te]], "classification", seed=seed, epochs=EPOCHS)
            seed_preds.append(np.asarray(pred, dtype=np.float64).ravel())
        pred = np.mean(np.stack(seed_preds, 0), 0)
        yte = y[te].ravel()
        nef = compute_nef(pred.reshape(-1, 1), y[te])
        roc = roc_auc_score(yte, pred) if len(set(yte.tolist())) > 1 else float("nan")
        per_fold.append({"fold": int(f), "nef1": float(nef), "roc_auc": float(roc),
                         "n_test": int(len(te)), "n_active": int(yte.sum())})
        log(f"{run} fold{f}: NEF1={nef:.3f} ROC={roc:.3f} (n_test={len(te)}, act={int(yte.sum())})")
    nefs = np.array([p["nef1"] for p in per_fold]); rocs = np.array([p["roc_auc"] for p in per_fold])
    out = ROOT / "figure_data" / "cbs_benchmark" / run / "moleculenet_cv"
    out.mkdir(parents=True, exist_ok=True)
    summary = {"cbs_nef1_MEAN": float(np.nanmean(nefs)), "cbs_nef1_STD": float(np.nanstd(nefs)),
               "cbs_MEAN": float(np.nanmean(rocs)), "cbs_STD": float(np.nanstd(rocs)),
               "arm": run, "n_folds": len(folds), "epochs": EPOCHS, "seeds": SEEDS,
               "cv_scheme": "provided", "metric": "nef1", "featurizer": "encoder_finetune"}
    (out / "suite_summary.json").write_text(json.dumps(summary, indent=2))
    with (out / "per_fold.csv").open("w", newline="") as fc:
        w = csv.DictWriter(fc, fieldnames=["fold", "nef1", "roc_auc", "n_test", "n_active"])
        w.writeheader(); w.writerows(per_fold)
    (ROOT / "figure_data" / "cbs_benchmark" / run / "verified.json").write_text(
        json.dumps({"run": run, "metric": "nef1", "cv": "provided-5fold", "arm": "e2e"}))
    sh(["aws", "s3", "cp", "--recursive", f"figure_data/cbs_benchmark/{run}",
        f"{S3OUT}/{run}", "--only-show-errors"])
    log(f"{run}: DONE NEF1%={summary['cbs_nef1_MEAN']:.3f}±{summary['cbs_nef1_STD']:.3f} "
        f"ROC={summary['cbs_MEAN']:.3f}")
    return True


def main():
    if not (ROOT / TOK / "tokenizer.json").exists():
        (ROOT / TOK).mkdir(parents=True, exist_ok=True)
        sh(["aws", "s3", "sync", "s3://climb-s3-bucket/tokenizer_10M", TOK, "--only-show-errors"])
    ok = 0
    for prefix, run in ENCODERS.items():
        if done(run):
            log(f"SKIP {run} (done)"); ok += 1; continue
        if run_arm(prefix, run):
            ok += 1
    log(f"DONE {ok}/{len(ENCODERS)}")
    if ok == len(ENCODERS):
        (ROOT / "figure_data" / "CBS_E2E_DONE").write_text("best-two CBS e2e done\n")
        log("CBS_E2E_DONE written")


if __name__ == "__main__":
    main()
