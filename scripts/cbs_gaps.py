"""Fill the four long-standing CBS gaps: sup_only:mixed, sup_only:minimol_full, unsup2sup:mixed,
unsup2sup:minimol_full were never run on CBS (they weren't in the A2 ladder that Wave 2 covered).
Frozen probe, provided 5-fold UMAP CV, NEF1% — identical protocol to every other CBS arm, so the
numbers drop straight into figure_data/cbs_benchmark/ and build_cbs_summary picks them up.

Runs on the box with the CLIMB venv. Idempotent, S3-synced, gated by a completion marker.
"""
from __future__ import annotations
import json, os, subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
PY = os.environ.get("CLIMB_PY", str(Path.home() / "venvs" / "climb" / "bin" / "python"))
S3B = "s3://climb-s3-bucket/experiments/climb_v2_phase2"
S3OUT = "s3://climb-s3-bucket/experiments/cbs_benchmark"
TOK = "figure_data/_tokenizer"
ENCODERS = ["skip_mixed_8M", "skip_minimol_full_8M", "u2s_mixed_from8M", "u2s_minimol_full_from8M"]
LOG = Path("analysis/cbs_gaps.log"); LOG.parent.mkdir(exist_ok=True)


def log(m):
    print(f"[cbs-gaps] {m}", flush=True)
    with LOG.open("a") as f:
        f.write(f"[cbs-gaps] {m}\n")


def sh(c):
    return subprocess.run(c, check=False)


def done(prefix):
    p = ROOT / "figure_data" / "cbs_benchmark" / prefix / "moleculenet_cv" / "suite_summary.json"
    if not p.exists():
        return False
    try:
        return json.loads(p.read_text()).get("cbs_nef1_MEAN") is not None
    except Exception:
        return False


def main():
    if not (ROOT / TOK / "tokenizer.json").exists():
        (ROOT / TOK).mkdir(parents=True, exist_ok=True)
        sh(["aws", "s3", "sync", "s3://climb-s3-bucket/tokenizer_10M", TOK, "--only-show-errors"])
    ok = 0
    for prefix in ENCODERS:
        if done(prefix):
            log(f"SKIP {prefix} (done)"); ok += 1; continue
        enc = ROOT / "figure_data" / "climb_v2_phase2" / prefix / "encoder"
        if not (enc / "model.safetensors").exists():
            enc.mkdir(parents=True, exist_ok=True)
            sh(["aws", "s3", "sync", f"{S3B}/{prefix}/encoder", str(enc), "--only-show-errors"])
        if not (enc / "model.safetensors").exists():
            log(f"ERROR {prefix}: encoder missing after sync"); continue
        out = f"figure_data/cbs_benchmark/{prefix}/moleculenet_cv"
        log(f"CBS frozen: {prefix}")
        r = sh([PY, "eval_v2.py", "--encoder", str(enc), "--tokenizer", TOK, "--output_dir", out,
                "--head", "mlp", "--head_seeds", "0", "1", "2",
                "--task_csv", "data/cbs.csv", "--task_name", "cbs", "--task_type", "classification",
                "--cv_folds", "5", "--cv_scheme", "provided"])
        if r.returncode == 0 and done(prefix):
            (ROOT / "figure_data" / "cbs_benchmark" / prefix / "verified.json").write_text(
                json.dumps({"run": prefix, "metric": "nef1", "cv": "provided-5fold", "panel": "cbs_gaps"}))
            sh(["aws", "s3", "cp", "--recursive", f"figure_data/cbs_benchmark/{prefix}",
                f"{S3OUT}/{prefix}", "--only-show-errors"])
            ok += 1
    log(f"DONE {ok}/{len(ENCODERS)}")
    if ok == len(ENCODERS):
        (ROOT / "figure_data" / "CBS_GAPS_DONE").write_text("4 CBS gap arms done\n")
        log("CBS_GAPS_DONE written")


if __name__ == "__main__":
    main()
