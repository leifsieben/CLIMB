"""Surgical HF upload of ONLY the 8 new Experiment-A encoders to the private climb-encoders model
repo, in the same `<wave>/<run>/` layout as scripts/publish_to_hf.py (stage_encoders) — without
re-staging the ~19 GB of all other paper waves. Idempotent (create_repo exist_ok, upload_folder
only sends changed files). Uses the ambient HF login; never handles a token.

Usage:  python scripts/upload_expA_encoders_hf.py --org lsieben [--execute]
"""
from __future__ import annotations

import argparse
import os
import subprocess
import tempfile
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")  # matches the repo's known-good upload path

WAVE = "climb_v2_expA"
BUCKET = "s3://climb-s3-bucket/experiments"
RUNS = ["unigram_8M", "unigram_8M_s1", "unigram_8M_s2",
        "corrupt_mlm_8M_s1", "corrupt_mlm_8M_s2",
        "bigram_8M", "bigram_8M_s1", "bigram_8M_s2"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--org", required=True)
    ap.add_argument("--execute", action="store_true", help="actually upload (default: dry-run)")
    a = ap.parse_args()
    repo_id = f"{a.org}/climb-encoders"

    from huggingface_hub import whoami, create_repo, upload_folder
    try:
        who = whoami()["name"]
    except Exception:
        who = None
    print(f"HF auth: {'logged in as ' + who if who else 'NOT LOGGED IN'}")
    if a.execute and not who:
        print("Refusing to --execute while not logged in. Run `hf auth login` first.")
        return 2

    with tempfile.TemporaryDirectory(prefix="expA_hf_") as tmp:
        stage = Path(tmp)
        pulled = []
        for run in RUNS:
            uri = f"{BUCKET}/{WAVE}/{run}/encoder/"
            if not subprocess.run(["aws", "s3", "ls", f"{uri}model.safetensors"],
                                  capture_output=True).stdout.strip():
                print(f"  SKIP {run}: no encoder on S3")
                continue
            dst = stage / WAVE / run
            dst.mkdir(parents=True, exist_ok=True)
            subprocess.run(["aws", "s3", "cp", uri, str(dst), "--recursive", "--only-show-errors"],
                           check=True)
            pulled.append(run)
            print(f"  staged {WAVE}/{run}/  ({', '.join(p.name for p in dst.iterdir())})")
        print(f"staged {len(pulled)} encoders -> {repo_id}")
        if not a.execute:
            print("[dry-run] pass --execute to create_repo(private=True) + upload_folder.")
            return 0
        create_repo(repo_id, repo_type="model", private=True, exist_ok=True)
        upload_folder(folder_path=str(stage), repo_id=repo_id, repo_type="model",
                      commit_message=f"Experiment A encoders ({WAVE}): unigram/bigram/shuffle arms")
        print(f"    ✅ uploaded -> https://huggingface.co/{repo_id}/tree/main/{WAVE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
