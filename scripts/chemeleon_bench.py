"""Benchmark the FROZEN CheMeleon fingerprint (Burns et al. 2025, descriptor-pretrained D-MPNN) as
an external featurizer across our 8 tasks, under the SAME frozen-probe protocol we use for our own
encoders: CheMeleon 2048-d embedding -> z-score -> MLP head, 3 head seeds.

  * 7 MoleculeNet tasks: scaffold 5-fold CV (A1b protocol of record)
  * CBS: provided folds + NEF1% (Truong 2026 benchmark)

Idempotent (verified.json = achieved-work check), syncs each run to S3. Run from repo root with the
box python (needs chemprop>=2.2.0 for chemeleon_fingerprint + deepchem for the MoleculeNet loaders).
Env overrides: CBS_CSV."""
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

PY = sys.executable
TOK = "figure_data/_tokenizer"   # unused by chemeleon featurizer, kept for arg parity
CBS_CSV = os.environ.get("CBS_CSV", "data/cbs.csv")
CORE = ["ESOL", "Lipophilicity", "QM7", "BBBP", "BACE", "Tox21", "HIV"]
S3 = "s3://climb-s3-bucket/experiments"

# (run_dir, wave) — MoleculeNet frozen arm lives beside the phase2 anchors; CBS beside the cbs battery.
MOLNET_OUT = Path("figure_data/climb_v2_phase2/chemeleon_frozen/moleculenet_cv")
CBS_OUT = Path("figure_data/cbs_benchmark/chemeleon_frozen/moleculenet_cv")


def _suite_ok(out: Path, keys) -> bool:
    p = out / "suite_summary.json"
    if not p.exists():
        return False
    try:
        d = json.loads(p.read_text())
    except Exception:
        return False
    return all(k in d for k in keys)


def _sync(local: Path, s3_uri: str):
    subprocess.run(["aws", "s3", "cp", "--recursive", str(local), s3_uri, "--only-show-errors"], check=False)


def _run(cmd, label, out: Path, need_keys) -> bool:
    if _suite_ok(out, need_keys):
        print(f"[chemeleon] SKIP {label}: already verified", flush=True)
        return True
    print(f"[chemeleon] === {label} ===", flush=True)
    r = subprocess.run(cmd, capture_output=True, text=True)
    ok = _suite_ok(out, need_keys)
    print(f"[chemeleon] {label}: {'OK' if ok else 'FAIL'}", flush=True)
    if not ok:
        print("STDOUT tail:", r.stdout[-1500:], "\nSTDERR tail:", r.stderr[-1500:], flush=True)
    return ok


def main():
    if not Path(CBS_CSV).exists():
        print(f"[chemeleon] WARN: {CBS_CSV} missing — CBS arm will be skipped", flush=True)
    done = []

    # --- 7 MoleculeNet, scaffold 5-fold CV ---
    molnet_keys = [f"{t}_MEAN" for t in CORE]
    cmd = [PY, "eval_v2.py", "--featurizer", "chemeleon", "--head", "mlp",
           "--standardize", "zscore", "--cv_folds", "5", "--cv_scheme", "scaffold",
           "--head_seeds", "0", "1", "2", "--output_dir", str(MOLNET_OUT), "--datasets"] + CORE
    if _run(cmd, "MoleculeNet-7 (scaffold 5fold, frozen)", MOLNET_OUT, molnet_keys):
        (MOLNET_OUT.parent / "verified.json").write_text(json.dumps({"arm": "chemeleon_frozen", "tasks": CORE}))
        _sync(MOLNET_OUT, f"{S3}/climb_v2_phase2/chemeleon_frozen/moleculenet_cv")
        done.append("molnet")

    # --- CBS, provided folds + NEF1% ---
    if Path(CBS_CSV).exists():
        cmd = [PY, "eval_v2.py", "--featurizer", "chemeleon", "--head", "mlp",
               "--standardize", "zscore", "--cv_folds", "5", "--cv_scheme", "provided",
               "--task_csv", CBS_CSV, "--task_name", "cbs", "--task_type", "classification",
               "--head_seeds", "0", "1", "2", "--output_dir", str(CBS_OUT)]
        if _run(cmd, "CBS (provided folds, NEF1%, frozen)", CBS_OUT, ["cbs_nef1_MEAN"]):
            (CBS_OUT.parent / "verified.json").write_text(json.dumps({"arm": "chemeleon_frozen", "task": "cbs"}))
            _sync(CBS_OUT, f"{S3}/cbs_benchmark/chemeleon_frozen/moleculenet_cv")
            done.append("cbs")

    print(f"\n[chemeleon] frozen probe done: {done}", flush=True)
    if "molnet" in done and (("cbs" in done) or not Path(CBS_CSV).exists()):
        Path("CHEMELEON_FROZEN_DONE").write_text("frozen probe complete\n")
        subprocess.run(["aws", "s3", "cp", "CHEMELEON_FROZEN_DONE",
                        f"{S3}/chemeleon_meta/CHEMELEON_FROZEN_DONE"], check=False)
        print("[chemeleon] ALL FROZEN DONE", flush=True)


if __name__ == "__main__":
    main()
