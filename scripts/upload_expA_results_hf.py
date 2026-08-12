"""Surgical HF upload of the Experiment-A raw eval RESULTS to the private climb-results dataset repo,
matching the per-run layout every other wave uses (config/metadata/metrics/verified + moleculenet{,_cv}),
WITHOUT re-staging or risking overwrite of any other wave. Parallel to upload_expA_encoders_hf.py.

Stages, under a temp dir mirroring the repo layout:
  climb_v2_expA/<arm_run>/{config.yaml,metadata.json,metrics.jsonl,verified.json,
                          moleculenet/{suite_summary.json,moleculenet_summary.csv,test_predictions.csv},
                          moleculenet_cv/{...}}                         (the 8 NEW arm runs)
  climb_v2_expA/_baselines/<run>/moleculenet_cv/{...}                   (native re-evals -> reproducible ladder)
  experiment_a/expA_ladder_{per_run,summary}.csv                       (the headline result)
then upload_folder -> lsieben/climb-results. Idempotent; ambient HF login.

Usage:  python scripts/upload_expA_results_hf.py --org lsieben [--execute]
"""
from __future__ import annotations

import argparse
import os
import subprocess
import tempfile
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

WAVE = "climb_v2_expA"
BUCKET = "s3://climb-s3-bucket/experiments"
ARM_RUNS = ["unigram_8M", "unigram_8M_s1", "unigram_8M_s2",
            "corrupt_mlm_8M_s1", "corrupt_mlm_8M_s2",
            "bigram_8M", "bigram_8M_s1", "bigram_8M_s2"]
BASELINE_RUNS = ["unsup_8M", "unsup_8M_s1", "unsup_8M_s2", "corrupt_mlm_8M",
                 "random_baseline_00", "random_baseline_01", "random_baseline_02"]
RESULT_KEEP = ["config.yaml", "metadata.json", "metrics.jsonl", "verified.json",
               "moleculenet/suite_summary.json", "moleculenet/moleculenet_summary.csv",
               "moleculenet/test_predictions.csv",
               "moleculenet_cv/suite_summary.json", "moleculenet_cv/moleculenet_summary.csv",
               "moleculenet_cv/test_predictions.csv"]
CV_ONLY = ["moleculenet_cv/suite_summary.json", "moleculenet_cv/moleculenet_summary.csv",
           "moleculenet_cv/test_predictions.csv"]
LADDER_CSVS = ["analysis/rigor/expA_ladder_per_run.csv", "analysis/rigor/expA_ladder_summary.csv"]


def _cp(uri: str, dst: Path) -> bool:
    if not subprocess.run(["aws", "s3", "ls", uri], capture_output=True).stdout.strip():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
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
    with tempfile.TemporaryDirectory(prefix="expA_res_") as tmp:
        stage = Path(tmp)
        for run in ARM_RUNS:
            got = [rel for rel in RESULT_KEEP
                   if _cp(f"{BUCKET}/{WAVE}/{run}/{rel}", stage / WAVE / run / rel)]
            n += len(got)
            print(f"  {WAVE}/{run}: {len(got)}/{len(RESULT_KEEP)} files")
        for run in BASELINE_RUNS:
            got = [rel for rel in CV_ONLY
                   if _cp(f"{BUCKET}/{WAVE}/_baselines/{run}/{rel}", stage / WAVE / "_baselines" / run / rel)]
            n += len(got)
            print(f"  {WAVE}/_baselines/{run}: {len(got)}/{len(CV_ONLY)} files")
        for rel in LADDER_CSVS:
            p = Path(rel)
            if p.exists():
                dst = stage / "experiment_a" / p.name
                dst.parent.mkdir(parents=True, exist_ok=True)
                dst.write_bytes(p.read_bytes()); n += 1
        print(f"staged {n} files -> {repo_id}")
        if not a.execute:
            print("[dry-run] pass --execute to upload.")
            return 0
        create_repo(repo_id, repo_type="dataset", private=True, exist_ok=True)
        upload_folder(folder_path=str(stage), repo_id=repo_id, repo_type="dataset",
                      commit_message=f"Experiment A results ({WAVE}): per-run eval + native baselines + ladder CSVs")
        print(f"    ✅ uploaded -> https://huggingface.co/datasets/{repo_id}/tree/main/{WAVE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
