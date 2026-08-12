"""Surgical HF upload of the Experiment-B raw eval RESULTS + analysis artifacts to the private
climb-results dataset repo, matching the per-run layout every other wave uses. Mirrors
scripts/upload_expA_results_hf.py. The real/no_pretrain comparators are reused from
climb_v2_expA/_baselines (already on HF), so only the 3 wiki_real runs are uploaded here, plus the
Exp B analysis CSVs/JSON under experiment_b/.

Usage:  python scripts/upload_expB_results_hf.py --org lsieben [--execute]
"""
from __future__ import annotations

import argparse
import os
import subprocess
import tempfile
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

WAVE = "climb_v2_expB"
BUCKET = "s3://climb-s3-bucket/experiments"
RUNS = ["wiki_real_8M", "wiki_real_8M_s1", "wiki_real_8M_s2"]
RESULT_KEEP = ["config.yaml", "metadata.json", "metrics.jsonl", "verified.json",
               "moleculenet/suite_summary.json", "moleculenet/moleculenet_summary.csv",
               "moleculenet/test_predictions.csv",
               "moleculenet_cv/suite_summary.json", "moleculenet_cv/moleculenet_summary.csv",
               "moleculenet_cv/test_predictions.csv"]
# analysis artifacts staged under experiment_b/
ANALYSIS = ["analysis/rigor/expB_wiki_summary.csv", "analysis/rigor/expB_wiki_per_run.csv",
            "analysis/rigor/wiki_coverage.json", "analysis/rigor/wiki_vs_smiles_stats.json"]


def _cp(uri, dst):
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
        print("Refusing to --execute while not logged in."); return 2

    n = 0
    with tempfile.TemporaryDirectory(prefix="expB_res_") as tmp:
        stage = Path(tmp)
        for run in RUNS:
            got = [rel for rel in RESULT_KEEP if _cp(f"{BUCKET}/{WAVE}/{run}/{rel}", stage / WAVE / run / rel)]
            n += len(got); print(f"  {WAVE}/{run}: {len(got)}/{len(RESULT_KEEP)}")
        for rel in ANALYSIS:
            p = Path(rel)
            if p.exists():
                dst = stage / "experiment_b" / p.name
                dst.parent.mkdir(parents=True, exist_ok=True)
                dst.write_bytes(p.read_bytes()); n += 1
        print(f"staged {n} files -> {repo_id}")
        if not a.execute:
            print("[dry-run] pass --execute to upload."); return 0
        create_repo(repo_id, repo_type="dataset", private=True, exist_ok=True)
        upload_folder(folder_path=str(stage), repo_id=repo_id, repo_type="dataset",
                      commit_message=f"Experiment B results ({WAVE}): wiki_real per-run eval + analysis")
        print(f"    ✅ -> https://huggingface.co/datasets/{repo_id}/tree/main/{WAVE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
