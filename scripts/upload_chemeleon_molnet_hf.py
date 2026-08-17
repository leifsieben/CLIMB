"""Surgical HF upload of the CheMeleon MoleculeNet A1 arms + the Polaris/MoleculeACE suite RESULTS to
the private climb-results dataset, WITHOUT re-staging any other wave. Mirrors upload_cbs_results_hf.py.

Uploads (idempotent — upload_folder re-syncs changed files only):
  figure_data/climb_v2_phase2/chemeleon_{frozen,e2e,e2e_s1,e2e_s2}/moleculenet_cv/  (A1.b; e2e has per-molecule OOF)
  figure_data/chemeleon_suite/<track>/<model>/                                       (Polaris + MoleculeACE, frozen+e2e)

Usage:  python scripts/upload_chemeleon_molnet_hf.py --org lsieben [--execute]
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

ROOT = Path(__file__).resolve().parent.parent
FD = ROOT / "figure_data"
# (local dir, path-in-repo) pairs. MoleculeNet CheMeleon arms live under the climb_v2_phase2 wave.
MOLNET_ARMS = ["chemeleon_frozen", "chemeleon_e2e", "chemeleon_e2e_s1", "chemeleon_e2e_s2"]
# raw per-run eval files worth publishing (no analysis on top)
ALLOW = ["*.json", "*.csv", "*.md"]


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
        print("Refusing to upload while logged out. Run: python3 -c \"from huggingface_hub import login; login()\"")
        return 1

    uploads = []  # (local_path, path_in_repo)
    for arm in MOLNET_ARMS:
        d = FD / "climb_v2_phase2" / arm / "moleculenet_cv"
        if d.exists():
            uploads.append((d, f"climb_v2_phase2/{arm}/moleculenet_cv"))
        else:
            print(f"  SKIP {arm}: {d} missing")
    suite = FD / "chemeleon_suite"
    if suite.exists():
        uploads.append((suite, "chemeleon_suite"))
    # The aggregated 6-panel tables the figures actually read (mainline_8M*, scaling_ladders,
    # labeleff_clean_*, STATUS.md). Added 2026-08-17: these were covered by NO uploader, so the
    # paper's core tables would never have reached HF at all.
    sixp = FD / "six_panel"
    if sixp.exists():
        uploads.append((sixp, "six_panel"))
    # rigor/analysis tables (bootstrap co-best, label-efficiency summaries, ...)
    rig = ROOT / "analysis" / "rigor"
    if rig.exists():
        uploads.append((rig, "analysis_rigor"))

    for local, pir in uploads:
        n = sum(1 for _ in local.rglob("*") if _.is_file())
        print(f"  {'UPLOAD' if a.execute else 'would upload'} {local.relative_to(ROOT)} -> {repo_id}/{pir}  ({n} files)")

    if not a.execute:
        print("[dry-run] pass --execute to upload.")
        return 0

    create_repo(repo_id, repo_type="dataset", private=True, exist_ok=True)
    for local, pir in uploads:
        upload_folder(repo_id=repo_id, repo_type="dataset", folder_path=str(local),
                      path_in_repo=pir, allow_patterns=ALLOW,
                      commit_message=f"CheMeleon results: {pir}")
        print(f"  uploaded {pir}")
    print(f"done -> https://huggingface.co/datasets/{repo_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
