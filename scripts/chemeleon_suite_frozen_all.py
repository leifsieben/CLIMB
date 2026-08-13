"""Run the FROZEN-probe benchmark for the whole model battery across both suite tracks (MoleculeACE +
Polaris), on the box. Encoder + Morgan arms use the CLIMB venv (this interpreter); the CheMeleon arm is
shelled out to the chemeleon venv (chemprop). MoleculeACE is scored locally by the runner; Polaris writes
per-seed test_predictions.csv (scored later via benchmark.evaluate() off-box).

Idempotent (skips a (model,track) whose test_predictions.csv covers all tasks), syncs each result dir to
S3. Run from repo root with ~/venvs/climb/bin/python. Env: CHEMELEON_PY (path to chemeleon venv python)."""
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
PY = sys.executable
CHEM_PY = os.environ.get("CHEMELEON_PY", str(Path.home() / "venvs" / "chemeleon" / "bin" / "python"))
TOK = "figure_data/_tokenizer"
S3_ENC = "s3://climb-s3-bucket/experiments/climb_v2_phase2"
S3_OUT = "s3://climb-s3-bucket/experiments/chemeleon_suite"
FD = ROOT / "figure_data" / "chemeleon_suite"
SEEDS = ["42", "117", "709"]
TRACKS = ["moleculeace", "polaris"]

# CLIMB 8M frozen battery (base model per arm; 3 head seeds). featurizer=encoder, head=mlp.
ENCODERS = ["unsup_8M", "skip_dense_8M", "skip_sparse_all_8M", "skip_dense_plus_sparse_8M",
            "skip_minimol_full_8M", "skip_mixed_8M",
            "u2s_dense_from8M", "u2s_sparse_all_from8M", "u2s_dense_plus_sparse_from8M",
            "u2s_minimol_full_from8M", "u2s_mixed_from8M", "random_baseline_00"]
# classical + external baselines: (model, featurizer, head, python)
BASELINES = [("ecfp4", "ecfp4", "xgb", PY), ("fp_desc", "fp_desc", "xgb", PY),
             ("chemeleon_frozen", "chemeleon", "mlp", CHEM_PY)]


def _n_tasks(track):
    name = {"moleculeace": "moleculeace_tasks.txt", "polaris": "polaris_tasks.txt"}[track]
    return len((ROOT / "chemeleon_suite" / "tasks" / name).read_text().split())


def _done(track, model):
    p = FD / track / model / "test_predictions.csv"
    if not p.exists():
        return False
    tasks = set()
    with p.open() as f:
        next(f, None)
        for line in f:
            tasks.add(line.split(",", 1)[0])
    return len(tasks) >= _n_tasks(track)


def _sync(track, model):
    subprocess.run(["aws", "s3", "cp", "--recursive", str(FD / track / model),
                    f"{S3_OUT}/{track}/{model}", "--only-show-errors"], check=False)


def _run(cmd, track, model, label):
    if _done(track, model):
        print(f"[frozen] SKIP {label} ({track}): done", flush=True); return True
    print(f"[frozen] === {label} ({track}) ===", flush=True)
    r = subprocess.run(cmd, capture_output=True, text=True)
    ok = _done(track, model)
    print(f"[frozen] {label} ({track}): {'OK' if ok else 'FAIL'}", flush=True)
    if not ok:
        print("STDOUT tail:", r.stdout[-1200:], "\nSTDERR tail:", r.stderr[-1200:], flush=True)
    else:
        _sync(track, model)
    return ok


def main():
    if not Path(TOK, "tokenizer.json").exists():
        subprocess.run(["aws", "s3", "sync", "s3://climb-s3-bucket/tokenizer_10M", TOK], check=False)
    done = []
    for track in TRACKS:
        # encoder arms
        for run in ENCODERS:
            enc = FD.parent / "climb_v2_phase2" / run / "encoder"
            if not (enc / "model.safetensors").exists():
                subprocess.run(["aws", "s3", "sync", f"{S3_ENC}/{run}/encoder", str(enc),
                                "--only-show-errors"], check=False)
            if not (enc / "model.safetensors").exists():
                print(f"[frozen] SKIP {run}: no encoder weights", flush=True); continue
            cmd = [PY, "scripts/chemeleon_suite_run.py", "--track", track, "--featurizer", "encoder",
                   "--model", run, "--encoder", str(enc), "--tokenizer", TOK, "--head", "mlp",
                   "--seeds"] + SEEDS
            done.append(_run(cmd, track, run, run))
        # baselines
        for model, feat, head, py in BASELINES:
            cmd = [py, "scripts/chemeleon_suite_run.py", "--track", track, "--featurizer", feat,
                   "--model", model, "--head", head, "--seeds"] + SEEDS
            done.append(_run(cmd, track, model, model))
    print(f"\n[frozen] battery: {sum(done)}/{len(done)} (model,track) runs OK", flush=True)
    if all(done):
        Path("CHEMELEON_FROZEN_ALL_DONE").write_text("frozen battery complete\n")
        subprocess.run(["aws", "s3", "cp", "CHEMELEON_FROZEN_ALL_DONE",
                        f"{S3_OUT}/CHEMELEON_FROZEN_ALL_DONE"], check=False)
        print("[frozen] ALL DONE", flush=True)


if __name__ == "__main__":
    main()
