"""QM7 in NATIVE units for the one end-to-end arm the frozen re-eval cannot reach: e2e_no_pretrain.

Companion to qm7_native_reeval.py. That script fixes the 14 arms scored by eval_v2; this one fixes
e2e_no_pretrain, whose stored QM7 (0.8546) is z-scored while the other e2e arms (chemeleon_e2e
199.5, s2u_dense 197.8) are already native -- so the e2e family is internally inconsistent today.

Same root cause, same remedy: finetune_e2e_v2 ALREADY standardizes regression targets on the train
split and unscales predictions back to native units before scoring (it calls the very same
eval_v2._fit_target_scaler / _unscale_preds the frozen path uses). The stored value predates that
change, so re-running with today's code emits native kcal/mol without any conversion constant.

ENCODER CHOICE mirrors run_e2e_random.py exactly, and this matters: e2e_random_XX has no saved
encoder of its own -- it fine-tunes the SAME saved weights that random_baseline_XX was frozen at.
Seeding a fresh random encoder would make the frozen and e2e bars differ in two things instead of
one, which is precisely the comparison those two bars exist to isolate.

Non-destructive (writes moleculenet_cv_qm7native/), and the completion gate checks the UNIT.
"""
from __future__ import annotations
import csv, os, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
PY = os.environ.get("CLIMB_PY", str(Path.home() / "venvs" / "climb" / "bin" / "python"))
S3B = "s3://climb-s3-bucket/experiments"
TOK = "figure_data/_tokenizer"
OUTSUB = "moleculenet_cv_qm7native"
LOG = ROOT / "analysis" / "qm7_native_e2e.log"
LOG.parent.mkdir(exist_ok=True)

# (output run label, encoder run whose SAVED WEIGHTS get fine-tuned) -- same pairing as run_e2e_random.py
PAIRS = [("e2e_random_00", "random_baseline_00"),
         ("e2e_random_01", "random_baseline_01"),
         ("e2e_random_02", "random_baseline_02")]


def log(m):
    print(f"[qm7-e2e] {m}", flush=True)
    with LOG.open("a") as f:
        f.write(f"[qm7-e2e] {m}\n")


def sh(c):
    return subprocess.run(c, check=False)


def native_ok(run):
    """Native QM7 RMSE is ~200 kcal/mol; the stale z-scored value is ~0.85. Gate on the unit."""
    f = ROOT / "figure_data" / "climb_v2_phase2" / run / OUTSUB / "moleculenet_summary.csv"
    try:
        vals = [float(r["main_value"]) for r in csv.DictReader(f.open())
                if r["dataset"] == "QM7" and r["main_metric"] == "rmse"
                and str(r["head_seed"]).startswith("fold")]
    except Exception:
        return False
    return len(vals) >= 5 and min(vals) > 10.0


def stage_encoder(run):
    enc = ROOT / "figure_data" / "_stage_qm7e2e" / run / "encoder"
    if (enc / "model.safetensors").exists():
        return str(enc)
    enc.mkdir(parents=True, exist_ok=True)
    for wave in ("climb_v2_phase2", "climb_v2_ablation_dedup", "climb_v2_headline"):
        sh(["aws", "s3", "sync", f"{S3B}/{wave}/{run}/encoder", str(enc), "--only-show-errors"])
        if (enc / "model.safetensors").exists():
            log(f"encoder for {run} from {wave}")
            return str(enc)
    return None


def main():
    if not (ROOT / TOK / "tokenizer.json").exists():
        (ROOT / TOK).mkdir(parents=True, exist_ok=True)
        sh(["aws", "s3", "sync", "s3://climb-s3-bucket/tokenizer_10M", TOK, "--only-show-errors"])
    log(f"START {len(PAIRS)} e2e runs -> {OUTSUB}")
    ok = 0
    for out_run, enc_run in PAIRS:
        if native_ok(out_run):
            log(f"SKIP {out_run} already native"); ok += 1; continue
        enc = stage_encoder(enc_run)
        if not enc:
            log(f"ERROR {out_run}: no encoder for {enc_run} on S3"); continue
        out = f"figure_data/climb_v2_phase2/{out_run}/{OUTSUB}"
        log(f"QM7 native e2e: {out_run} (weights from {enc_run})")
        sh([PY, "finetune_e2e_v2.py", "--encoder", enc, "--tokenizer", TOK,
            "--output_dir", out, "--datasets", "QM7", "--seeds", "0", "1", "2",
            "--cv_folds", "5", "--cv_scheme", "scaffold", "--subsample_seed", "0"])
        if native_ok(out_run):
            sh(["aws", "s3", "cp", "--recursive", out,
                f"{S3B}/climb_v2_phase2/{out_run}/{OUTSUB}", "--only-show-errors"])
            log(f"OK {out_run}"); ok += 1
        else:
            log(f"FAIL {out_run}: no native QM7 rows produced")
        sh(["rm", "-rf", str(ROOT / "figure_data" / "_stage_qm7e2e" / enc_run)])
    log(f"DONE {ok}/{len(PAIRS)}")
    if ok == len(PAIRS):
        (ROOT / "figure_data" / "QM7_NATIVE_E2E_DONE").write_text("qm7 native e2e re-eval\n")
        log("marker written")


if __name__ == "__main__":
    main()
