"""Surgical HF upload of the cbs virtual-screening battery RESULTS to the private climb-results
dataset repo, under cbs_benchmark/<run>/moleculenet_cv/ (+ experiment_cbs/ summary CSVs), WITHOUT
re-staging or overwriting any other wave. Parallel to scripts/upload_expA_results_hf.py.

These runs are eval-only (frozen probe / xgb anchor / e2e), so each has only moleculenet_cv/
(suite_summary.json, moleculenet_summary.csv, test_predictions.csv) + verified.json. NEF1% headline.

Usage:  python scripts/upload_cbs_results_hf.py --org lsieben [--execute]
"""
from __future__ import annotations

import argparse
import os
import subprocess
import tempfile
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

WAVE = "cbs_benchmark"
BUCKET = "s3://climb-s3-bucket/experiments"
RUNS = [
    "ecfp4_anchor", "fp_desc_anchor",
    "random_baseline_00", "random_baseline_01", "random_baseline_02",
    "e2e_random_00", "e2e_random_01", "e2e_random_02",
    "unsup_8M", "unsup_8M_s1", "unsup_8M_s2",
    "skip_dense_8M", "skip_dense_8M_s1", "skip_dense_8M_s2",
    "skip_sparse_all_8M", "skip_sparse_all_8M_s1", "skip_sparse_all_8M_s2",
    "skip_dense_plus_sparse_8M", "skip_dense_plus_sparse_8M_s1", "skip_dense_plus_sparse_8M_s2",
    "u2s_dense_from8M", "u2s_dense_from8M_s1", "u2s_dense_from8M_s2",
    "u2s_sparse_all_from8M", "u2s_sparse_all_from8M_s1", "u2s_sparse_all_from8M_s2",
    "u2s_dense_plus_sparse_from8M", "u2s_dense_plus_sparse_from8M_s1", "u2s_dense_plus_sparse_from8M_s2",
    # CheMeleon / chemprop comparators (2026-08-14): frozen fingerprint probe + native chemprop e2e
    # (vanilla + CheMeleon foundation), 3 seeds each. See scripts/{chemeleon_bench,cbs_chemprop_e2e}.py.
    "chemeleon_frozen",
    "chemprop_e2e_s0", "chemprop_e2e_s1", "chemprop_e2e_s2",
    "chemeleon_e2e_s0", "chemeleon_e2e_s1", "chemeleon_e2e_s2",
]
RESULT_KEEP = ["verified.json",
               "moleculenet_cv/suite_summary.json", "moleculenet_cv/moleculenet_summary.csv",
               "moleculenet_cv/test_predictions.csv",
               "moleculenet_cv/per_fold.csv"]   # chemprop e2e arms emit per_fold.csv (no test_predictions)
SUMMARY_CSVS = ["experiment_cbs/cbs_nef1_summary.csv", "experiment_cbs/cbs_per_run.csv",
                "experiment_cbs/cbs_reference_lines.csv"]


LOCAL_ROOT = Path(__file__).resolve().parent.parent / "figure_data"


def _stage(run: str, rel: str, dst: Path) -> bool:
    """Prefer the already-pulled local copy (fast); fall back to S3 if absent."""
    local = LOCAL_ROOT / WAVE / run / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    if local.exists():
        dst.write_bytes(local.read_bytes())
        return True
    uri = f"{BUCKET}/{WAVE}/{run}/{rel}"
    if not subprocess.run(["aws", "s3", "ls", uri], capture_output=True).stdout.strip():
        return False
    return subprocess.run(["aws", "s3", "cp", uri, str(dst), "--only-show-errors"]).returncode == 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--org", required=True)
    ap.add_argument("--execute", action="store_true")
    a = ap.parse_args()
    repo_id = f"{a.org}/climb-results"

    from huggingface_hub import whoami, create_repo, upload_folder
    try:
        who = whoami()["name"]
    except Exception:
        who = None
    print(f"HF auth: {'logged in as ' + who if who else 'NOT LOGGED IN'}")
    if a.execute and not who:
        print("Refusing to --execute while not logged in. Run `hf auth login` first.")
        return 2

    n = 0
    with tempfile.TemporaryDirectory(prefix="cbs_res_") as tmp:
        stage = Path(tmp)
        for run in RUNS:
            got = [rel for rel in RESULT_KEEP
                   if _stage(run, rel, stage / WAVE / run / rel)]
            n += len(got)
            print(f"  {WAVE}/{run}: {len(got)}/{len(RESULT_KEEP)} files")
        for rel in SUMMARY_CSVS:
            p = Path(rel)
            if p.exists():
                dst = stage / "experiment_cbs" / p.name
                dst.parent.mkdir(parents=True, exist_ok=True)
                dst.write_bytes(p.read_bytes()); n += 1
                print(f"  staged {rel}")
        print(f"staged {n} files -> {repo_id}")
        if not a.execute:
            print("[dry-run] pass --execute to upload.")
            return 0
        create_repo(repo_id, repo_type="dataset", private=True, exist_ok=True)
        upload_folder(folder_path=str(stage), repo_id=repo_id, repo_type="dataset",
                      commit_message=f"cbs VS benchmark results ({WAVE}): per-run NEF1% CV + summary CSVs")
        print(f"    uploaded -> https://huggingface.co/datasets/{repo_id}/tree/main/{WAVE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
