"""Download the valid 8M-ladder encoders and run scaffold 5-fold CV eval locally (CPU),
writing results into figure_data/ so the notebook's A1/A2 error bars populate with real
fold spread. Encoder-only (no xgboost needed). Run from repo root with .venv_sanity python."""
import subprocess, sys
from pathlib import Path
import eval_v2

CORE = [("ESOL","regression"),("BBBP","classification"),("BACE","classification"),
        ("Tox21","classification"),("QM7","regression")]
S3 = "s3://climb-s3-bucket/experiments/climb_v2_phase2"
TOK_S3 = "s3://climb-s3-bucket/tokenizer_10M"
FD = Path("figure_data/climb_v2_phase2")
RUNS = ["unsup_8M","skip_dense_8M","skip_sparse_all_8M",
        "skip_dense_plus_sparse_8M","skip_minimol_full_8M","skip_mixed_8M"]

tok = Path("figure_data/_tokenizer"); tok.mkdir(parents=True, exist_ok=True)
if not (tok/"tokenizer.json").exists():
    subprocess.run(["aws","s3","sync",TOK_S3,str(tok)], check=False)

for run in RUNS:
    enc = FD/run/"encoder"
    if not enc.exists():
        print(f"[cv] downloading {run} encoder ...", flush=True)
        subprocess.run(["aws","s3","sync",f"{S3}/{run}/encoder",str(enc)], check=False)
    if not (enc/"model.safetensors").exists() and not (enc/"pytorch_model.bin").exists():
        print(f"[cv] SKIP {run}: no encoder weights found", flush=True); continue
    print(f"[cv] === {run}: scaffold 5-fold CV ===", flush=True)
    eval_v2.evaluate(str(enc), str(tok), str(FD/run/"moleculenet"),
                     head_seeds=[0,1,2], datasets=CORE, featurizer="encoder", cv_folds=5)
print("[cv] DONE — all encoders CV-evaluated into figure_data/", flush=True)
