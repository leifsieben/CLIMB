"""Canonical-six evals for the SFT-family ablation wave (seq_*) — the fig_C_D blocker.

fig_C1/C2/fig_D are indexed by the SFT family x eval task, and were only ever scored on
MoleculeNet. This adds the three canonical panels the ablation wave never had — MoleculeACE
(macro-RMSE), CBS (NEF1%) and the Polaris track (the Ames panel) — via the FROZEN probe.
BACE/BBBP/Tox21/QM7 already exist in each run's moleculenet/, so they are NOT recomputed.

WAVE CHOICE (important): the encoders come from `climb_v2_ablation_dedup`, NOT the original
`climb_v2_ablation`. Two independent reasons, and they agree:
  1. The pre-dedup wave saved NO encoder/ at all (verified on S3: 0 files under every seq_*/
     encoder/), so it is not evaluable even in principle.
  2. figures/fig_C2.py and figures/fig_D.py already read figure_data/climb_v2_ablation_dedup,
     and scripts/reproducibility_audit.py lists climb_v2_ablation under SUPERSEDED.
So scoring the dedup wave keeps the new canonical panels on the SAME pretraining corpus as the
MoleculeNet numbers already in those figures. Scoring the other wave would silently mix corpora.

PER-MOLECULE PREDICTIONS come free: chemeleon_suite_run.py always writes test_predictions.csv
(task,seed,test_index,smiles,y_pred), which is exactly what fig_C1 a/b needs to bin test
molecules by max Tanimoto to the pretraining corpus. No extra pass, no extra cost.

Runs ON THE BOX from repo root. Idempotent per (encoder, panel); each result dir is synced to S3.
Reuses the proven runners as subprocesses so numbers are protocol-identical to the 8M battery.

Contract (the paths fig_C_D reads):
  MoleculeACE -> figure_data/chemeleon_suite/moleculeace/<arm>/
  Polaris     -> figure_data/chemeleon_suite/polaris/<arm>/
  CBS         -> figure_data/cbs_benchmark/<arm>/moleculenet_cv/
"""
from __future__ import annotations
import csv, json, os, subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
PY = os.environ.get("CLIMB_PY", str(Path.home() / "venvs" / "climb" / "bin" / "python"))
POLARIS_PY = os.environ.get("POLARIS_PY", str(ROOT / ".venv_polaris" / "bin" / "python"))
S3B = "s3://climb-s3-bucket/experiments"
WAVE = "climb_v2_ablation_dedup"
TOK_STD = "figure_data/_tokenizer"
STAGE = ROOT / "figure_data" / "_stage_ablation"
LOG = ROOT / "analysis" / "six_panel_ablation.log"
LOG.parent.mkdir(exist_ok=True)

ARMS = ["seq_mtr", "seq_dense_plus_sparse", "seq_pcba", "seq_l1000", "seq_pcqm", "seq_sparse_all"]


def log(m):
    line = f"[abl] {m}"
    print(line, flush=True)
    with LOG.open("a") as f:
        f.write(line + "\n")


def sh(c):
    return subprocess.run(c, check=False)


def _stage(arm):
    enc = STAGE / arm / "encoder"
    if not (enc / "model.safetensors").exists():
        enc.mkdir(parents=True, exist_ok=True)
        sh(["aws", "s3", "sync", f"{S3B}/{WAVE}/{arm}/encoder", str(enc), "--only-show-errors"])
    return str(enc)


def _mace_done(arm):
    return (ROOT / "figure_data" / "chemeleon_suite" / "moleculeace" / arm / "verified.json").exists()


def _cbs_done(arm):
    p = ROOT / "figure_data" / "cbs_benchmark" / arm / "moleculenet_cv" / "suite_summary.json"
    try:
        return json.loads(p.read_text()).get("cbs_nef1_MEAN") is not None
    except Exception:
        return False


def _polaris_done(arm):
    """Full-track coverage, same gate as six_panel_herg.py: count DISTINCT tasks, never file
    existence. A crashed run still leaves a partial polaris_scores.csv behind."""
    f = ROOT / "figure_data" / "chemeleon_suite" / "polaris" / arm / "polaris_scores.csv"
    try:
        return len({r["task"] for r in csv.DictReader(open(f))}) >= 20
    except Exception:
        return False


def _run_mace(arm, enc):
    log(f"MoleculeACE frozen: {arm}")
    r = sh([PY, "scripts/chemeleon_suite_run.py", "--track", "moleculeace", "--featurizer", "encoder",
            "--model", arm, "--encoder", enc, "--tokenizer", TOK_STD,
            "--head", "mlp", "--seeds", "42", "117", "709"])
    if r.returncode == 0 and _mace_done(arm):
        sh(["aws", "s3", "cp", "--recursive", f"figure_data/chemeleon_suite/moleculeace/{arm}",
            f"{S3B}/chemeleon_suite/moleculeace/{arm}", "--only-show-errors"])
        return True
    return False


def _run_cbs(arm, enc):
    log(f"CBS frozen: {arm}")
    out = f"figure_data/cbs_benchmark/{arm}/moleculenet_cv"
    r = sh([PY, "eval_v2.py", "--encoder", enc, "--tokenizer", TOK_STD, "--output_dir", out,
            "--head", "mlp", "--head_seeds", "0", "1", "2",
            "--task_csv", "data/cbs.csv", "--task_name", "cbs", "--task_type", "classification",
            "--cv_folds", "5", "--cv_scheme", "provided"])
    if r.returncode == 0 and _cbs_done(arm):
        (ROOT / "figure_data" / "cbs_benchmark" / arm / "verified.json").write_text(
            json.dumps({"run": arm, "metric": "nef1", "cv": "provided-5fold", "panel": "six_panel_ablation"}))
        sh(["aws", "s3", "cp", "--recursive", f"figure_data/cbs_benchmark/{arm}",
            f"{S3B}/cbs_benchmark/{arm}", "--only-show-errors"])
        return True
    return False


def _run_polaris(arm, enc):
    """Two-step: predict in the CLIMB venv, score in .venv_polaris (test labels are withheld).
    Full 28-task track, not one named task — a panel swap (hERG -> Ames) must not invalidate it."""
    log(f"Polaris frozen: {arm}")
    r = sh([PY, "scripts/chemeleon_suite_run.py", "--track", "polaris", "--featurizer", "encoder",
            "--model", arm, "--encoder", enc, "--tokenizer", TOK_STD,
            "--head", "mlp", "--seeds", "42", "117", "709"])
    if r.returncode != 0:
        return False
    sh([POLARIS_PY, "scripts/chemeleon_suite_score_polaris.py",
        f"figure_data/chemeleon_suite/polaris/{arm}"])
    if _polaris_done(arm):
        sh(["aws", "s3", "cp", "--recursive", f"figure_data/chemeleon_suite/polaris/{arm}",
            f"{S3B}/chemeleon_suite/polaris/{arm}", "--only-show-errors"])
        return True
    return False


def _stage_eval_data():
    """Stage the MoleculeACE CSVs and data/cbs.csv from S3 if absent.

    A Polaris-only box has neither, and without this the MoleculeACE and CBS steps fail instantly
    with FileNotFoundError while the Polaris step succeeds -- which is exactly how this driver
    first ran (mace=False cbs=False polaris=True on all six arms). Staging here means the driver
    is runnable on ANY fresh box, not just one that happened to be rsynced from the laptop.
    """
    mace = ROOT / "chemeleon_suite" / "data" / "moleculeace"
    if len(list(mace.glob("*.csv"))) < 30:
        mace.mkdir(parents=True, exist_ok=True)
        log("staging MoleculeACE CSVs from S3")
        sh(["aws", "s3", "sync", "s3://climb-s3-bucket/datasets/moleculeace/", str(mace),
            "--only-show-errors"])
    cbs = ROOT / "data" / "cbs.csv"
    if not cbs.exists():
        cbs.parent.mkdir(parents=True, exist_ok=True)
        log("staging data/cbs.csv from S3")
        sh(["aws", "s3", "cp", "s3://climb-s3-bucket/datasets/cbs.csv", str(cbs),
            "--only-show-errors"])
    n = len(list(mace.glob("*.csv")))
    if n < 30 or not cbs.exists():
        log(f"FATAL eval data incomplete (moleculeace={n}/30, cbs={cbs.exists()})")
        return False
    return True


def main():
    if not _stage_eval_data():
        return
    if not (ROOT / TOK_STD / "tokenizer.json").exists():
        (ROOT / TOK_STD).mkdir(parents=True, exist_ok=True)
        sh(["aws", "s3", "sync", "s3://climb-s3-bucket/tokenizer_10M", TOK_STD, "--only-show-errors"])
    log(f"START {len(ARMS)} ablation arms x {{MoleculeACE, CBS, Polaris}} from {WAVE}")
    done = 0
    for arm in ARMS:
        enc = _stage(arm)
        if not Path(enc, "model.safetensors").exists():
            log(f"ERROR {arm}: encoder missing after sync"); continue
        okm = _mace_done(arm) or _run_mace(arm, enc)
        okc = _cbs_done(arm) or _run_cbs(arm, enc)
        okp = _polaris_done(arm) or _run_polaris(arm, enc)
        log(f"{arm}: mace={okm} cbs={okc} polaris={okp}")
        if okm and okc and okp:
            done += 1
        sh(["rm", "-rf", str(STAGE / arm)])
    log(f"DONE {done}/{len(ARMS)}")
    if done == len(ARMS):
        (ROOT / "figure_data" / "SIX_PANEL_ABLATION_DONE").write_text(
            "canonical six panels on all 6 seq_* ablation arms\n")
        log("SIX_PANEL_ABLATION_DONE written")


if __name__ == "__main__":
    main()
