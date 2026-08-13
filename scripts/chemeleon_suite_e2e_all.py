"""Drive the end-to-end fine-tune battery for the CheMeleon suite: 3 arms x 2 tracks x 3 seeds.
Arms (METHODOLOGY §3): unsup_8M, skip_dense_8M (best supervised), no_pretrain_e2e (from random_baseline_00).
Idempotent (skips a (model,track) whose test_predictions.csv covers all tasks), syncs each dir to S3,
gated self-stop marker. Run from repo root with ~/venvs/climb/bin/python (torch+transformers, GPU)."""
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
PY = sys.executable
TOK = "figure_data/_tokenizer"
S3_ENC = "s3://climb-s3-bucket/experiments/climb_v2_phase2"
S3_OUT = "s3://climb-s3-bucket/experiments/chemeleon_suite"
FD = ROOT / "figure_data" / "chemeleon_suite"
SEEDS = ["42", "117", "709"]
TRACKS = ["moleculeace", "polaris"]
# (output model label, encoder run to fine-tune FROM)
ARMS = [("unsup_8M", "unsup_8M"), ("skip_dense_8M", "skip_dense_8M"), ("no_pretrain_e2e", "random_baseline_00")]


def _n_tasks(track):
    name = {"moleculeace": "moleculeace_tasks.txt", "polaris": "polaris_tasks.txt"}[track]
    return len((ROOT / "chemeleon_suite" / "tasks" / name).read_text().split())


def _done(track, model):
    p = FD / track / f"{model}_e2e" / "test_predictions.csv"
    if not p.exists():
        return False
    tasks = set()
    with p.open() as f:
        next(f, None)
        for line in f:
            tasks.add(line.split(",", 1)[0])
    return len(tasks) >= _n_tasks(track)


def main():
    if not Path(TOK, "tokenizer.json").exists():
        subprocess.run(["aws", "s3", "sync", "s3://climb-s3-bucket/tokenizer_10M", TOK], check=False)
    done = []
    for track in TRACKS:
        for model, enc_run in ARMS:
            if _done(track, model):
                print(f"[e2e-all] SKIP {model} ({track}): done", flush=True); done.append(True); continue
            enc = FD.parent / "climb_v2_phase2" / enc_run / "encoder"
            if not (enc / "model.safetensors").exists():
                subprocess.run(["aws", "s3", "sync", f"{S3_ENC}/{enc_run}/encoder", str(enc),
                                "--only-show-errors"], check=False)
            if not (enc / "model.safetensors").exists():
                print(f"[e2e-all] SKIP {model}: no encoder {enc_run}", flush=True); done.append(False); continue
            print(f"[e2e-all] === {model} ({track}) e2e from {enc_run} ===", flush=True)
            r = subprocess.run([PY, "scripts/chemeleon_suite_e2e.py", "--track", track, "--model", model,
                                "--encoder", str(enc), "--tokenizer", TOK, "--seeds"] + SEEDS,
                               capture_output=True, text=True)
            ok = _done(track, model)
            print(f"[e2e-all] {model} ({track}): {'OK' if ok else 'FAIL'}", flush=True)
            if not ok:
                print("STDOUT tail:", r.stdout[-1000:], "\nSTDERR tail:", r.stderr[-1000:], flush=True)
            else:
                subprocess.run(["aws", "s3", "cp", "--recursive", str(FD / track / f"{model}_e2e"),
                                f"{S3_OUT}/{track}/{model}_e2e", "--only-show-errors"], check=False)
            done.append(ok)
    print(f"\n[e2e-all] battery: {sum(done)}/{len(done)} OK", flush=True)
    if all(done):
        Path("CHEMELEON_E2E_ALL_DONE").write_text("e2e battery complete\n")
        subprocess.run(["aws", "s3", "cp", "CHEMELEON_E2E_ALL_DONE", f"{S3_OUT}/CHEMELEON_E2E_ALL_DONE"], check=False)
        print("[e2e-all] ALL DONE", flush=True)


if __name__ == "__main__":
    main()
