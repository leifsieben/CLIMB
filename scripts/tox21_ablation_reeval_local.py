"""Re-evaluate Tox21 for the 10 climb_v2_ablation_dedup runs IN THE REFERENCE ENVIRONMENT.

Why this is a local script and not a box job. Those runs' prediction dumps are 93,876 rows --
pre-fix, missing-label cells scored as true inactives -- so nothing is recoverable from disk;
they must be re-evaluated against their checkpoints. The earlier attempt ran on EC2 and landed
~0.008 off, because deepchem 2.8.0 (which defines the 77,864-row parse) has no Python 3.12 wheel
while the box needs >=3.11. fig_C2 and fig_D are LIFT figures against a shared phase-2 floor, so
that offset would be charged to pretraining rather than to the environment. This laptop IS the
reference environment, so running here removes the offset by construction instead of correcting
for it afterwards.

Writes moleculenet_cv_tox21fixed/, never touching moleculenet_cv/, and MERGES so a run that
gains other datasets later cannot be clobbered. Gates on the reference row count.
"""
from __future__ import annotations
import shutil, subprocess, sys, tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
S3 = "s3://climb-s3-bucket/experiments/climb_v2_ablation_dedup"
WAVE = ROOT / "figure_data" / "climb_v2_ablation_dedup"
TOK = ROOT / "figure_data" / "_tokenizer"
REF_ROWS = 77864
OUTSUB = "moleculenet_cv_tox21fixed"

RUNS = ["seq_mtr", "seq_dense_plus_sparse", "seq_pcba", "seq_l1000", "seq_pcqm",
        "seq_sparse_all", "random_baseline_00", "random_baseline_01", "random_baseline_02",
        "ecfp4_anchor"]
# ecfp4_anchor has no encoder -- it is the classical anchor inside this wave.
FEATURIZER = {"ecfp4_anchor": ("ecfp4", "xgb")}


def sh(cmd):
    return subprocess.run(cmd, capture_output=True, text=True)


def has_weights(enc: Path) -> bool:
    return (enc / "model.safetensors").exists() or (enc / "pytorch_model.bin").exists()


def main(argv) -> int:
    runs = argv or RUNS
    if not (TOK / "tokenizer.json").exists():
        sh(["aws", "s3", "sync", "s3://climb-s3-bucket/tokenizer_10M", str(TOK)])
    ok, bad = [], []
    for run in runs:
        d = WAVE / run
        feat, head = FEATURIZER.get(run, (None, "mlp"))
        args = ["--cv_folds", "5", "--head_seeds", "0", "1", "2", "--datasets", "Tox21"]
        if feat:
            args += ["--featurizer", feat, "--head", head]
        else:
            enc = d / "encoder"
            if not has_weights(enc):
                sh(["aws", "s3", "sync", f"{S3}/{run}/encoder", str(enc), "--only-show-errors"])
            if not has_weights(enc):
                print(f"  {run}: NO ENCODER -> skipped", flush=True)
                bad.append(run)
                continue
            args += ["--encoder", str(enc), "--tokenizer", str(TOK)]

        tmp = Path(tempfile.mkdtemp(prefix="t21ab-"))
        print(f"=== {run} ===", flush=True)
        r = sh([sys.executable, "eval_v2.py", "--output_dir", str(tmp)] + args)
        if "[eval_v2] wrote" not in r.stdout:
            print(f"  {run}: FAIL\n{r.stdout[-800:]}\n{r.stderr[-800:]}", flush=True)
            bad.append(run)
            shutil.rmtree(tmp, ignore_errors=True)
            continue
        n = sum(1 for line in (tmp / "test_predictions.csv").open() if line.startswith("Tox21,"))
        if n != REF_ROWS:
            # Not a warning: a different row count means a different molecule set, and the whole
            # point of running here is that the set matches phase-2 exactly.
            print(f"  {run}: ROW COUNT {n} != {REF_ROWS} -> REJECTED, not written", flush=True)
            bad.append(run)
            shutil.rmtree(tmp, ignore_errors=True)
            continue
        dest = d / OUTSUB
        dest.mkdir(parents=True, exist_ok=True)
        sys.path.insert(0, str(ROOT / "scripts"))
        from merge_summary_rows import main as merge
        merge(str(tmp / "moleculenet_summary.csv"), str(dest / "moleculenet_summary.csv"), "Tox21")
        shutil.copy2(tmp / "test_predictions.csv", dest / "test_predictions.csv")
        if (tmp / "suite_summary.json").exists():
            shutil.copy2(tmp / "suite_summary.json", dest / "suite_summary.json")
        val = ""
        for line in (dest / "moleculenet_summary.csv").open():
            f = line.split(",")
            if len(f) > 9 and f[0] == "Tox21" and f[7] == "MEAN" and f[6] == "roc_auc":
                val = f"{float(f[9]):.4f}"
        print(f"  {run}: OK  {n} rows  Tox21 roc_auc={val}", flush=True)
        ok.append(run)
        shutil.rmtree(tmp, ignore_errors=True)

    print(f"\nDONE  ok={len(ok)}  failed={len(bad)}  {', '.join(bad)}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
