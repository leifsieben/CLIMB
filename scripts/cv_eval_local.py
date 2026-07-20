"""Generate scaffold 5-fold CV results for every model in Fig A1, written to a SEPARATE
`moleculenet_cv/` dir (so the single-split `moleculenet/` results are preserved for the
tougher SI variant). Each run is a fresh subprocess → a crash in one can't kill the rest,
and there is no cross-run DeepChem cache state. Run from repo root with .venv_sanity python."""
import subprocess, sys, os, tempfile, shutil, glob
from pathlib import Path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); os.chdir(ROOT)

PY = sys.executable
TOK = "figure_data/_tokenizer"
S3 = "s3://climb-s3-bucket/experiments/climb_v2_phase2"
# Full v2 eval suite (matches config_v2.MOLECULENET_TASKS_V2). HIV last: it is the large
# (~41k) virtual-screening task scored primarily by NEF1% (top-1% early enrichment) — its
# CV is the slowest, so ordering it last lets the small tasks land first.
CORE = ["ESOL", "Lipophilicity", "QM7", "BBBP", "BACE", "Tox21", "HIV"]
FD = Path("figure_data/climb_v2_phase2")
ENCODER_RUNS = ["unsup_8M", "skip_dense_8M", "skip_sparse_all_8M", "skip_dense_plus_sparse_8M",
                "skip_minimol_full_8M", "skip_mixed_8M",
                "random_baseline_00", "random_baseline_01", "random_baseline_02"]

if not Path(TOK, "tokenizer.json").exists():
    subprocess.run(["aws", "s3", "sync", "s3://climb-s3-bucket/tokenizer_10M", TOK], check=False)

def has_weights(enc):
    return (enc/"model.safetensors").exists() or (enc/"pytorch_model.bin").exists()

def run_eval(args, label):
    for d in glob.glob(os.path.join(tempfile.gettempdir(), "*-featurized")):
        shutil.rmtree(d, ignore_errors=True)  # avoid DeepChem cache collisions
    print(f"[cv] === {label} ===", flush=True)
    r = subprocess.run([PY, "eval_v2.py"] + args, capture_output=True, text=True)
    ok = "[eval_v2] wrote" in r.stdout
    print(f"[cv] {label}: {'OK' if ok else 'FAIL'}", flush=True)
    if not ok:
        print("STDOUT tail:", r.stdout[-800:], "\nSTDERR tail:", r.stderr[-800:], flush=True)

for run in ENCODER_RUNS:
    enc = FD/run/"encoder"
    if not has_weights(enc):
        subprocess.run(["aws", "s3", "sync", f"{S3}/{run}/encoder", str(enc)], check=False)
    if not has_weights(enc):
        print(f"[cv] SKIP {run}: no encoder weights", flush=True); continue
    run_eval(["--encoder", str(enc), "--tokenizer", TOK, "--output_dir", str(FD/run/"moleculenet_cv"),
              "--cv_folds", "5", "--head_seeds", "0", "1", "2", "--datasets"] + CORE, f"{run} (CV)")

run_eval(["--output_dir", str(FD/"ecfp4_anchor"/"moleculenet_cv"), "--featurizer", "ecfp4", "--head", "xgb",
          "--cv_folds", "5", "--head_seeds", "0", "1", "2", "--datasets"] + CORE, "ecfp4_anchor (CV)")
# Toughest classical baseline: Morgan fingerprints ++ RDKit descriptors → XGBoost.
run_eval(["--output_dir", str(FD/"fp_desc_anchor"/"moleculenet_cv"), "--featurizer", "fp_desc", "--head", "xgb",
          "--cv_folds", "5", "--head_seeds", "0", "1", "2", "--datasets"] + CORE, "fp_desc_anchor (CV)")
print("[cv] ALL DONE", flush=True)
