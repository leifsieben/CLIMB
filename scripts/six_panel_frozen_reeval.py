"""Wave 2 of the six-panel migration (see notes/six-panel-migration.md).

For every SCALING encoder (A2 compute ladder + H1 data-fraction + vocab), add the two panels the
scaling figures never had — MoleculeACE (macro-RMSE) and CBS (NEF1%) — via the FROZEN probe. The
other four panels (BACE/BBBP/Tox21/QM7) already exist in each scaling run's moleculenet_cv/, so
they are NOT recomputed here. No pretraining, no fine-tuning: encoder is frozen.

Runs ON THE BOX from repo root with the CLIMB venv. Idempotent: skips any (encoder, panel) whose
verified marker exists. Each result dir is synced to S3. Reuses the two PROVEN runners as
subprocesses (chemeleon_suite_run.py for MoleculeACE, eval_v2.py for CBS) so the numbers are
protocol-identical to the 8M battery.

Tight-organization contract:
  MoleculeACE frozen -> figure_data/chemeleon_suite/moleculeace/<prefix>/   (one home for ALL arms+scales)
  CBS frozen         -> figure_data/cbs_benchmark/<prefix>/moleculenet_cv/  (one home for ALL arms+scales)
Both keyed by the unique encoder prefix, so a scaling point never overwrites an arm or a MolNet result.
"""
from __future__ import annotations
import os, subprocess, sys, json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
PY = os.environ.get("CLIMB_PY", str(Path.home() / "venvs" / "climb" / "bin" / "python"))
S3B = "s3://climb-s3-bucket/experiments"
TOK_STD = "figure_data/_tokenizer"            # standard 10M-vocab tokenizer (phase2 + h1)
STAGE = ROOT / "figure_data" / "_stage_sixpanel"
LOG = ROOT / "analysis" / "six_panel_w2.log"
LOG.parent.mkdir(exist_ok=True)


def log(m):
    line = f"[six-panel-w2] {m}"
    print(line, flush=True)
    with LOG.open("a") as f:
        f.write(line + "\n")


def sh(cmd, **kw):
    return subprocess.run(cmd, **kw)


# --- encoder manifest: (axis, prefix, s3_wave, own_tokenizer) ---------------------------------
def _manifest():
    m = []
    # A2 compute ladder (phase2) — 5 regimes x their budgets; std tokenizer
    a2 = {
        "unsup": ["unsup_2M", "unsup_8M", "unsup_24M", "unsup_48M", "unsup_50M", "unsup_100M"],
        "skip_dense": ["skip_dense_2M", "skip_dense_8M", "skip_dense_24M", "skip_dense_48M", "skip_dense_96M"],
        "skip_sparse_all": ["skip_sparse_all_2M", "skip_sparse_all_8M", "skip_sparse_all_24M", "skip_sparse_all_48M"],
        "skip_dps": ["skip_dense_plus_sparse_2M", "skip_dense_plus_sparse_8M", "skip_dense_plus_sparse_24M", "skip_dense_plus_sparse_48M"],
        "u2s_dense": ["u2s_dense_from2M", "u2s_dense_from8M", "u2s_dense_from24M", "u2s_dense_from48M"],
    }
    for regime, prefs in a2.items():
        for p in prefs:
            m.append(("a2", p, "climb_v2_phase2", False))
    # H1 data-fraction (climb_v2_h1) — 30 dirs. H1 varies the pretraining DATA only; the tokenizer
    # is held constant at the standard 10M vocab (s1/s2 don't even carry a tokenizer/ dir on S3), so
    # use TOK_STD (own_tok=False), NOT a per-encoder sync.
    for mode in ("canonical", "enumerated"):
        for frac in ("0p001", "0p01", "0p1", "0p3", "full"):
            for s in ("s0", "s1", "s2"):
                m.append(("h1", f"scaling_{mode}_frac{frac}_{s}", "climb_v2_h1", False))
    # Vocab (climb_v2_vocab) — 8 dirs; MUST use their own tokenizer (different vocab size)
    for p in ("bpe_261", "bpe_1000", "bpe_3000", "bpe_12000", "unigram_261", "unigram_700", "unigram_1200", "unigram_3000"):
        m.append(("vocab", p, "climb_v2_vocab", True))
    return m


def _stage(prefix, wave, own_tok):
    """Sync encoder (+ tokenizer if own) from S3. Return (encoder_dir, tokenizer_dir)."""
    dst = STAGE / prefix
    enc = dst / "encoder"
    if not (enc / "model.safetensors").exists():
        enc.mkdir(parents=True, exist_ok=True)
        sh(["aws", "s3", "sync", f"{S3B}/{wave}/{prefix}/encoder", str(enc), "--only-show-errors"], check=False)
    tok = TOK_STD
    if own_tok:
        td = dst / "tokenizer"
        if not (td / "tokenizer.json").exists():
            td.mkdir(parents=True, exist_ok=True)
            sh(["aws", "s3", "sync", f"{S3B}/{wave}/{prefix}/tokenizer", str(td), "--only-show-errors"], check=False)
        tok = str(td)
    return str(enc), tok


def _mace_done(prefix):
    return (ROOT / "figure_data" / "chemeleon_suite" / "moleculeace" / prefix / "verified.json").exists()


def _cbs_done(prefix):
    p = ROOT / "figure_data" / "cbs_benchmark" / prefix / "moleculenet_cv" / "suite_summary.json"
    if not p.exists():
        return False
    try:
        return json.loads(p.read_text()).get("cbs_nef1_MEAN") is not None
    except Exception:
        return False


def _run_mace(prefix, enc, tok):
    log(f"MoleculeACE frozen: {prefix}")
    r = sh([PY, "scripts/chemeleon_suite_run.py", "--track", "moleculeace", "--featurizer", "encoder",
            "--model", prefix, "--encoder", enc, "--tokenizer", tok,
            "--head", "mlp", "--seeds", "42", "117", "709"])
    if r.returncode == 0:
        sh(["aws", "s3", "cp", "--recursive", f"figure_data/chemeleon_suite/moleculeace/{prefix}",
            f"{S3B}/chemeleon_suite/moleculeace/{prefix}", "--only-show-errors"], check=False)
    return r.returncode == 0


def _run_cbs(prefix, enc, tok):
    log(f"CBS frozen: {prefix}")
    out = f"figure_data/cbs_benchmark/{prefix}/moleculenet_cv"
    r = sh([PY, "eval_v2.py", "--encoder", enc, "--tokenizer", tok, "--output_dir", out,
            "--head", "mlp", "--head_seeds", "0", "1", "2",
            "--task_csv", "data/cbs.csv", "--task_name", "cbs", "--task_type", "classification",
            "--cv_folds", "5", "--cv_scheme", "provided"])
    if r.returncode == 0 and _cbs_done(prefix):
        (ROOT / "figure_data" / "cbs_benchmark" / prefix / "verified.json").write_text(
            json.dumps({"run": prefix, "metric": "nef1", "cv": "provided-5fold", "panel": "six_panel_w2"}))
        sh(["aws", "s3", "cp", "--recursive", f"figure_data/cbs_benchmark/{prefix}",
            f"{S3B}/cbs_benchmark/{prefix}", "--only-show-errors"], check=False)
    return r.returncode == 0


def main():
    # ensure standard tokenizer present
    if not (ROOT / TOK_STD / "tokenizer.json").exists():
        (ROOT / TOK_STD).mkdir(parents=True, exist_ok=True)
        sh(["aws", "s3", "sync", "s3://climb-s3-bucket/tokenizer_10M", TOK_STD, "--only-show-errors"], check=False)

    man = _manifest()
    log(f"START {len(man)} scaling encoders x {{MoleculeACE, CBS}}")
    ok = 0
    for axis, prefix, wave, own_tok in man:
        need_m = not _mace_done(prefix)
        need_c = not _cbs_done(prefix)
        if not (need_m or need_c):
            log(f"SKIP {prefix} (both done)"); ok += 1; continue
        enc, tok = _stage(prefix, wave, own_tok)
        if not Path(enc, "model.safetensors").exists():
            log(f"ERROR {prefix}: encoder weights missing after sync -> skip"); continue
        good = True
        if need_m:
            good = _run_mace(prefix, enc, tok) and good
        if need_c:
            if (ROOT / "data" / "cbs.csv").exists():
                good = _run_cbs(prefix, enc, tok) and good
            else:
                log(f"CBS SKIP {prefix}: data/cbs.csv absent (fold-provenance blocked); re-run when staged")
                need_c = False  # do not count CBS against completeness while data is unavailable
        # free staging to keep disk bounded
        sh(["rm", "-rf", str(STAGE / prefix)], check=False)
        if good:
            ok += 1
    log(f"DONE {ok}/{len(man)} encoders complete")
    if ok == len(man):
        (ROOT / "figure_data" / "SIX_PANEL_W2_DONE").write_text("all scaling encoders have MoleculeACE+CBS\n")
        log("SIX_PANEL_W2_DONE written")


if __name__ == "__main__":
    main()
