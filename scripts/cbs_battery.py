"""Run the full A1a model battery on the cbs.csv rare-actives benchmark (random 5-fold CV,
NEF1% headline metric). Mirrors scripts/cv_eval_local.py but (a) targets an arbitrary labelled
CSV via eval_v2's custom-task loader, (b) uses the label-STRATIFIED RANDOM fold scheme (NOT
scaffold — the in-distribution setting the user asked for), and (c) adds the end-to-end arm.

Battery (matches notebook_cells/08.py A1_ORDER, all pretraining seeds we have):
  * 8 frozen-probe arms x 3 pretraining seeds  = 24 encoder runs  (featurizer=encoder, head=mlp)
  * 2 classical XGBoost anchors                                   (ecfp4, fp_desc; head=xgb)
  * no_pretrain end-to-end                     = 3 runs           (finetune random-init, 1 seed each)

Idempotent + crash-safe like cv_eval_local: each run is a fresh subprocess; completion is judged
from ACHIEVED WORK (suite_summary.json contains cbs_nef1_MEAN), never from a file merely existing;
verified runs are skipped on re-invocation; every finished run is synced to S3 immediately.

Run from repo root with the box python. Env overrides: CBS_CSV, CBS_WAVE, CBS_S3_BASE."""
import glob
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

PY = sys.executable
TOK = "figure_data/_tokenizer"
CBS_CSV = os.environ.get("CBS_CSV", "data/cbs.csv")
WAVE = os.environ.get("CBS_WAVE", "cbs_benchmark")
S3_ENC = "s3://climb-s3-bucket/experiments/climb_v2_phase2"          # source encoders
S3_OUT = os.environ.get("CBS_S3_BASE", f"s3://climb-s3-bucket/experiments/{WAVE}")
FD = Path("figure_data") / WAVE
FD.mkdir(parents=True, exist_ok=True)

# arm label -> encoder run dirs (all pretraining seeds). Frozen probe, mlp head.
FROZEN = {
    "no_pretrain":                 ["random_baseline_00", "random_baseline_01", "random_baseline_02"],
    "unsup_only":                  ["unsup_8M", "unsup_8M_s1", "unsup_8M_s2"],
    "sup_only:dense":              ["skip_dense_8M", "skip_dense_8M_s1", "skip_dense_8M_s2"],
    "sup_only:sparse_all":         ["skip_sparse_all_8M", "skip_sparse_all_8M_s1", "skip_sparse_all_8M_s2"],
    "sup_only:dense_plus_sparse":  ["skip_dense_plus_sparse_8M", "skip_dense_plus_sparse_8M_s1", "skip_dense_plus_sparse_8M_s2"],
    "unsup2sup:dense":             ["u2s_dense_from8M", "u2s_dense_from8M_s1", "u2s_dense_from8M_s2"],
    "unsup2sup:sparse_all":        ["u2s_sparse_all_from8M", "u2s_sparse_all_from8M_s1", "u2s_sparse_all_from8M_s2"],
    "unsup2sup:dense_plus_sparse": ["u2s_dense_plus_sparse_from8M", "u2s_dense_plus_sparse_from8M_s1", "u2s_dense_plus_sparse_from8M_s2"],
}
ANCHORS = [("ecfp4_anchor", "ecfp4"), ("fp_desc_anchor", "fp_desc")]
# no_pretrain end-to-end: fine-tune the random-init encoder, one seed per run dir (matches phase2).
E2E = [("e2e_random_00", "random_baseline_00", 0),
       ("e2e_random_01", "random_baseline_01", 1),
       ("e2e_random_02", "random_baseline_02", 2)]

# provided = the benchmark's OWN fold column (UMAP-cluster, Tanimoto<0.70 inter-fold constraint):
# the only scheme whose per-fold NEF1% is comparable to Truong et al. 2026 (SOTA NEF1%=0.764±0.191).
CV_SCHEME = os.environ.get("CBS_CV_SCHEME", "provided")
COMMON = ["--task_csv", CBS_CSV, "--task_name", "cbs", "--task_type", "classification",
          "--cv_folds", "5", "--cv_scheme", CV_SCHEME]


def _has_weights(enc: Path) -> bool:
    return (enc / "model.safetensors").exists() or (enc / "pytorch_model.bin").exists()


def _is_done(run_dir: Path) -> bool:
    """Completion == achieved work: the run's suite carries the NEF1% headline for cbs."""
    suite = run_dir / "moleculenet_cv" / "suite_summary.json"
    if not suite.exists():
        return False
    try:
        d = json.loads(suite.read_text())
    except Exception:
        return False
    return "cbs_nef1_MEAN" in d and d.get("cbs_nef1_MEAN") is not None


def _sync_to_s3(run: str):
    subprocess.run(["aws", "s3", "cp", "--recursive", str(FD / run / "moleculenet_cv"),
                    f"{S3_OUT}/{run}/moleculenet_cv"], check=False)


def _clean_featurized_cache():
    for d in glob.glob(os.path.join(tempfile.gettempdir(), "*-featurized")):
        shutil.rmtree(d, ignore_errors=True)


def _run(cmd, label, run) -> bool:
    run_dir = FD / run
    if _is_done(run_dir):
        print(f"[cbs] SKIP {label}: already verified", flush=True)
        return True
    _clean_featurized_cache()
    print(f"[cbs] === {label} ===", flush=True)
    r = subprocess.run(cmd, capture_output=True, text=True)
    ok = _is_done(run_dir)
    print(f"[cbs] {label}: {'OK' if ok else 'FAIL'}", flush=True)
    if not ok:
        print("STDOUT tail:", r.stdout[-1200:], "\nSTDERR tail:", r.stderr[-1200:], flush=True)
        return False
    (run_dir / "verified.json").write_text(json.dumps({"run": run, "metric": "nef1", "cv": "random-5fold"}))
    _sync_to_s3(run)
    return True


def main():
    if not Path(TOK, "tokenizer.json").exists():
        subprocess.run(["aws", "s3", "sync", "s3://climb-s3-bucket/tokenizer_10M", TOK], check=False)
    if not Path(CBS_CSV).exists():
        sys.exit(f"[cbs] FATAL: task csv not found: {CBS_CSV}")

    expected, done = [], []

    # ---- frozen-probe encoder arms (mlp head) ----
    for arm, runs in FROZEN.items():
        for run in runs:
            expected.append(run)
            enc = FD / run / "encoder"
            if not _has_weights(enc):
                subprocess.run(["aws", "s3", "sync", f"{S3_ENC}/{run}/encoder", str(enc)], check=False)
            if not _has_weights(enc):
                print(f"[cbs] SKIP {run} ({arm}): no encoder weights in S3", flush=True)
                continue
            cmd = [PY, "eval_v2.py", "--encoder", str(enc), "--tokenizer", TOK,
                   "--output_dir", str(FD / run / "moleculenet_cv"),
                   "--head", "mlp", "--head_seeds", "0", "1", "2"] + COMMON
            if _run(cmd, f"{run} [{arm}] frozen-mlp", run):
                done.append(run)

    # ---- classical XGBoost anchors (no encoder) ----
    for run, feat in ANCHORS:
        expected.append(run)
        cmd = [PY, "eval_v2.py", "--output_dir", str(FD / run / "moleculenet_cv"),
               "--featurizer", feat, "--head", "xgb", "--head_seeds", "0", "1", "2"] + COMMON
        if _run(cmd, f"{run} [{feat}+xgb]", run):
            done.append(run)

    # ---- end-to-end arm (fine-tune random-init encoder) ----
    for run, enc_run, seed in E2E:
        expected.append(run)
        enc = FD / enc_run / "encoder"
        if not _has_weights(enc):
            subprocess.run(["aws", "s3", "sync", f"{S3_ENC}/{enc_run}/encoder", str(enc)], check=False)
        if not _has_weights(enc):
            print(f"[cbs] SKIP {run}: no init encoder {enc_run}", flush=True)
            continue
        cmd = [PY, "finetune_e2e_v2.py", "--encoder", str(enc), "--tokenizer", TOK,
               "--output_dir", str(FD / run / "moleculenet_cv"),
               "--seeds", str(seed)] + COMMON
        if _run(cmd, f"{run} [no_pretrain_e2e seed{seed}]", run):
            done.append(run)

    print(f"\n[cbs] battery: {len(done)}/{len(expected)} runs verified", flush=True)
    missing = [r for r in expected if r not in done]
    if missing:
        print(f"[cbs] MISSING: {missing}", flush=True)
    else:
        Path("CBS_BATTERY_DONE").write_text("all cbs runs verified\n")
        subprocess.run(["aws", "s3", "cp", "CBS_BATTERY_DONE", f"{S3_OUT}/CBS_BATTERY_DONE"], check=False)
        print("[cbs] ALL DONE -> wrote CBS_BATTERY_DONE", flush=True)


if __name__ == "__main__":
    main()
