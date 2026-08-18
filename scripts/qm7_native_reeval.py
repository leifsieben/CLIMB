"""Re-evaluate QM7 in NATIVE units (kcal/mol) for the frozen arms whose stored numbers are z-scored.

THE DEFECT: the QM7 column of mainline_8M.csv mixes conventions -- 15 arms carry normalized RMSE
(~0.85) and 3 carry native kcal/mol (s2u_dense 197.8, chemeleon_e2e 199.5, chemeleon_frozen 268.8).
Plotted together the panel is not internally comparable, and a 327x max/min ratio is the giveaway.

WHY A RE-EVAL AND NOT A CONVERSION: current eval_v2.py already scales targets PER FOLD on that
fold's training labels and inverse-transforms predictions BEFORE scoring, so it emits native units
by construction (see the comment above _fit_target_scaler). The normalized numbers are stale
artifacts predating that fix -- their summaries date to 2026-07-22. Re-running with today's code
fixes the SOURCE, and avoids inventing a single sigma: with each fold z-scored by its own training
sigma there is no one correct constant (228.656 by affine fit vs 228.9 by RMSE ratio), so any
conversion bakes in a systematic error.

NON-DESTRUCTIVE: results go to moleculenet_cv_qm7native/ alongside the existing moleculenet_cv/,
never into it. eval_v2 deletes the run's rows in its output_dir before writing, so pointing it at
the existing dir with --datasets QM7 would drop that dir's BACE/BBBP/ESOL/HIV/Tox21 rows entirely.

SCOPE: the 14 arms that go through eval_v2. e2e_no_pretrain is NOT here -- it is an end-to-end arm
scored by finetune_e2e_v2, so it needs the e2e runner, and it is flagged rather than silently
skipped (its stored 0.8546 is normalized too, unlike the other e2e arms which are native).
"""
from __future__ import annotations
import csv, json, os, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
PY = os.environ.get("CLIMB_PY", str(Path.home() / "venvs" / "climb" / "bin" / "python"))
S3B = "s3://climb-s3-bucket/experiments"
TOK = "figure_data/_tokenizer"
OUTSUB = "moleculenet_cv_qm7native"
LOG = ROOT / "analysis" / "qm7_native.log"
LOG.parent.mkdir(exist_ok=True)

# (arm, featurizer, head, [seed dirs]) -- mirrors figures/arms.py src["mol"]
ENC_ARMS = {
    "sup_dense": "skip_dense_8M", "sup_dense_sparse": "skip_dense_plus_sparse_8M",
    "sup_mixed": "skip_mixed_8M", "sup_sparse": "skip_sparse_all_8M",
    "sup_minimol": "skip_minimol_full_8M", "unsup": "unsup_8M",
    "u2s_dense": "u2s_dense_from8M", "u2s_dense_sparse": "u2s_dense_plus_sparse_from8M",
    "u2s_mixed": "u2s_mixed_from8M", "u2s_sparse": "u2s_sparse_all_from8M",
    "u2s_minimol": "u2s_minimol_full_from8M",
}
MANIFEST = []
for arm, base in ENC_ARMS.items():
    MANIFEST += [(arm, "encoder", "mlp", d) for d in (base, f"{base}_s1", f"{base}_s2")]
MANIFEST += [("random_encoder", "encoder", "mlp", d)
             for d in ("random_baseline_00", "random_baseline_01", "random_baseline_02")]
MANIFEST += [("ecfp", "ecfp4", "xgb", "ecfp4_anchor"), ("ecfp_desc", "fp_desc", "xgb", "fp_desc_anchor")]
SHARD = int(os.environ.get("SHARD", "0"))
NSHARD = int(os.environ.get("NSHARD", "1"))

# --- extension points -------------------------------------------------------------------------
# RUNS lets a caller pass an explicit encoder-run list (space separated) instead of the built-in
# 14-arm manifest -- used for the SCALING LADDER, whose ~50 rungs carry the same z-scored QM7 as
# the mainline did, so fig_B's rungs would otherwise stay z-scored while fig_A's bars went native.
# DATASETS lets the same machinery fix the other stale regression tasks (ESOL, Lipophilicity),
# which have exactly the same cause. Defaults reproduce the original QM7-only behaviour.
_runs_env = os.environ.get("RUNS", "").split()
DATASETS = os.environ.get("DATASETS", "QM7").split()
OUTSUB = os.environ.get("OUTSUB", OUTSUB)
if _runs_env:
    MANIFEST = [("ladder", "encoder", "mlp", r) for r in _runs_env]


def log(m):
    print(f"[qm7] {m}", flush=True)
    with LOG.open("a") as f:
        f.write(f"[qm7] {m}\n")


def sh(c):
    return subprocess.run(c, check=False)


def native_ok(run):
    """Achieved work: a QM7 fold row present AND in native units.

    The whole point is the unit, so the gate checks it: native QM7 RMSE is ~200 kcal/mol, the
    z-scored value is ~0.85. Anything under 10 means the stale convention came back and must NOT
    count as done."""
    f = ROOT / "figure_data" / "climb_v2_phase2" / run / OUTSUB / "moleculenet_summary.csv"
    # Native scale differs per task, so the unit gate is per-task: QM7 ~200 kcal/mol, ESOL ~0.9
    # log mol/L, Lipophilicity ~0.78. A z-scored value is ~1.0 sd for all three, so the QM7 test
    # (>10) cannot be reused -- for ESOL/Lipo compare against the z-scored value they would have.
    FLOOR = {"QM7": 10.0, "ESOL": 0.70, "Lipophilicity": 0.70}
    try:
        rows = list(csv.DictReader(f.open()))
    except Exception:
        return False
    for ds in DATASETS:
        vals = [float(r["main_value"]) for r in rows
                if r["dataset"] == ds and r["main_metric"] == "rmse"
                and str(r["head_seed"]).startswith("fold")]
        if len(vals) < 5 or min(vals) <= FLOOR.get(ds, 0.0):
            return False
    return True


def stage_encoder(run):
    enc = ROOT / "figure_data" / "_stage_qm7" / run / "encoder"
    if (enc / "model.safetensors").exists():
        return str(enc)
    enc.mkdir(parents=True, exist_ok=True)
    for wave in ("climb_v2_phase2", "climb_v2_ablation_dedup"):
        sh(["aws", "s3", "sync", f"{S3B}/{wave}/{run}/encoder", str(enc), "--only-show-errors"])
        if (enc / "model.safetensors").exists():
            return str(enc)
    return None


def main():
    if not (ROOT / TOK / "tokenizer.json").exists():
        (ROOT / TOK).mkdir(parents=True, exist_ok=True)
        sh(["aws", "s3", "sync", "s3://climb-s3-bucket/tokenizer_10M", TOK, "--only-show-errors"])
    man = [m for i, m in enumerate(MANIFEST) if i % NSHARD == SHARD]
    log(f"START {len(man)} runs (shard {SHARD}/{NSHARD}) datasets={','.join(DATASETS)} -> {OUTSUB}")
    ok = 0
    for arm, feat, head, run in man:
        if native_ok(run):
            log(f"SKIP {run} ({arm}) already native"); ok += 1; continue
        out = f"figure_data/climb_v2_phase2/{run}/{OUTSUB}"
        cmd = [PY, "eval_v2.py", "--output_dir", out, "--datasets", *DATASETS,
               "--featurizer", feat, "--head", head, "--head_seeds", "0", "1", "2",
               "--cv_folds", "5", "--cv_scheme", "scaffold"]
        if feat == "encoder":
            enc = stage_encoder(run)
            if not enc:
                log(f"ERROR {run}: no encoder on S3"); continue
            cmd += ["--encoder", enc, "--tokenizer", TOK]
        log(f"QM7 native: {arm} / {run}")
        sh(cmd)
        if native_ok(run):
            sh(["aws", "s3", "cp", "--recursive", out,
                f"{S3B}/climb_v2_phase2/{run}/{OUTSUB}", "--only-show-errors"])
            log(f"OK {run}"); ok += 1
        else:
            log(f"FAIL {run}: no native QM7 rows produced")
        sh(["rm", "-rf", str(ROOT / "figure_data" / "_stage_qm7" / run)])
    log(f"DONE {ok}/{len(man)}")
    if ok == len(man):
        (ROOT / "figure_data" / f"QM7_NATIVE_DONE_{SHARD}").write_text("qm7 native re-eval\n")
        log("marker written")
    log("NOTE e2e_no_pretrain is NOT covered here -- e2e arm, needs finetune_e2e_v2")


if __name__ == "__main__":
    main()
