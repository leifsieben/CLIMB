"""MoleculeACE pretraining-seed top-up (rigor fix, see notes/six-panel-migration.md).

The mainline MoleculeACE panel was scored on ONE pretraining seed per arm, while CBS/MolNet have
their _s1/_s2 replicates. That matters because the best-two headline (sup_only:dense 0.774 vs
unsup_only 0.777) rests on MoleculeACE and the two are 0.003 apart — inside likely seed noise.

This scores the 22 existing _s1/_s2 mainline encoders (11 arms x 2 replicate seeds) on MoleculeACE
via the FROZEN probe, so every arm gets 3 pretraining seeds on MoleculeACE (matching CBS). Frozen
only, no retraining. Idempotent (skip verified), S3-synced. Standard 10M-vocab tokenizer.

Results -> figure_data/chemeleon_suite/moleculeace/<prefix>/ (one home for all arms/seeds/scales).
"""
from __future__ import annotations
import os, subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
PY = os.environ.get("CLIMB_PY", str(Path.home() / "venvs" / "climb" / "bin" / "python"))
S3B = "s3://climb-s3-bucket/experiments/climb_v2_phase2"
S3OUT = "s3://climb-s3-bucket/experiments/chemeleon_suite/moleculeace"
TOK = "figure_data/_tokenizer"
STAGE = ROOT / "figure_data" / "_stage_mace_topup"
LOG = ROOT / "analysis" / "mace_seedtopup.log"
LOG.parent.mkdir(exist_ok=True)

ARMS = ["unsup_8M", "skip_dense_8M", "skip_dense_plus_sparse_8M", "skip_sparse_all_8M",
        "skip_mixed_8M", "skip_minimol_full_8M",
        "u2s_dense_from8M", "u2s_dense_plus_sparse_from8M", "u2s_sparse_all_from8M",
        "u2s_mixed_from8M", "u2s_minimol_full_from8M"]
ENCODERS = [f"{a}_{s}" for a in ARMS for s in ("s1", "s2")]  # 22 replicate encoders


def log(m):
    print(f"[mace-topup] {m}", flush=True)
    with LOG.open("a") as f:
        f.write(f"[mace-topup] {m}\n")


def done(prefix):
    return (ROOT / "figure_data" / "chemeleon_suite" / "moleculeace" / prefix / "verified.json").exists()


def main():
    if not (ROOT / TOK / "tokenizer.json").exists():
        (ROOT / TOK).mkdir(parents=True, exist_ok=True)
        subprocess.run(["aws", "s3", "sync", "s3://climb-s3-bucket/tokenizer_10M", TOK, "--only-show-errors"], check=False)
    log(f"START {len(ENCODERS)} replicate encoders x MoleculeACE")
    ok = 0
    for prefix in ENCODERS:
        if done(prefix):
            log(f"SKIP {prefix} (done)"); ok += 1; continue
        enc = STAGE / prefix / "encoder"
        if not (enc / "model.safetensors").exists():
            enc.mkdir(parents=True, exist_ok=True)
            subprocess.run(["aws", "s3", "sync", f"{S3B}/{prefix}/encoder", str(enc), "--only-show-errors"], check=False)
        if not (enc / "model.safetensors").exists():
            log(f"ERROR {prefix}: encoder missing after sync -> skip"); continue
        log(f"MoleculeACE frozen: {prefix}")
        r = subprocess.run([PY, "scripts/chemeleon_suite_run.py", "--track", "moleculeace",
                            "--featurizer", "encoder", "--model", prefix, "--encoder", str(enc),
                            "--tokenizer", TOK, "--head", "mlp", "--seeds", "42", "117", "709"])
        if r.returncode == 0 and done(prefix):
            subprocess.run(["aws", "s3", "cp", "--recursive",
                            f"figure_data/chemeleon_suite/moleculeace/{prefix}", f"{S3OUT}/{prefix}",
                            "--only-show-errors"], check=False)
            ok += 1
        subprocess.run(["rm", "-rf", str(STAGE / prefix)], check=False)
    log(f"DONE {ok}/{len(ENCODERS)}")
    if ok == len(ENCODERS):
        (ROOT / "figure_data" / "MACE_SEEDTOPUP_DONE").write_text("all 22 replicate encoders scored on MoleculeACE\n")
        log("MACE_SEEDTOPUP_DONE written")


if __name__ == "__main__":
    main()
