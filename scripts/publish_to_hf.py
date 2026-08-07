"""Publish the three CLIMB artifacts to the Hugging Face Hub as PRIVATE repos (reviewer links).

Three repos, assembled from S3 + the local tree:

  <org>/climb-encoders     (model)   -- every run's encoder/ (model.safetensors + config.json) + tokenizer
  <org>/climb-results      (dataset) -- RAW eval outputs only: per-run moleculenet{,_cv}/ + metrics.jsonl
                                         (no derived analysis: _tanimoto / _eval_ceiling / figures)
  <org>/climb-pretrain-data(dataset) -- 12M filtered corpus + 217-desc targets + 5.38M supervised table
                                         + blocklist + descriptor stats + tokenizer. The 124M corpus is
                                         NOT re-hosted (linked to hheiden/PubChem-124M-... in the card).

Design goals: reviewer can reproduce; nothing published by accident. So:
  * DRY-RUN BY DEFAULT. Pass --execute to actually create repos + upload.
  * repos are created private=True; share with reviewers via HF repo settings.
  * idempotent: create_repo(exist_ok=True), upload_folder re-syncs changed files.
  * requires you to be logged in (`hf auth login`); this script never handles your token.

Usage:
  python scripts/publish_to_hf.py --org YOURORG --repo all               # dry-run plan
  python scripts/publish_to_hf.py --org YOURORG --repo cards --execute   # push the 3 READMEs only
  python scripts/publish_to_hf.py --org YOURORG --repo results --execute
  python scripts/publish_to_hf.py --org YOURORG --repo encoders --execute
  python scripts/publish_to_hf.py --org YOURORG --repo pretrain --execute
"""
from __future__ import annotations
import argparse, os, shutil, subprocess, sys
from pathlib import Path

BUCKET = "s3://climb-s3-bucket"
STAGING = Path(os.environ.get("HF_STAGING", ".hf_staging"))   # gitignored; large
CARD = {"encoders": "hf/model_card.md", "results": "hf/dataset_card_results.md",
        "pretrain": "hf/dataset_card_pretrain.md"}
REPO_TYPE = {"encoders": "model", "results": "dataset", "pretrain": "dataset"}
REPO_NAME = {"encoders": "climb-encoders", "results": "climb-results", "pretrain": "climb-pretrain-data"}

# Only these waves' encoders/results are part of the release (paper-critical + SI vocab).
# NOTE: the label-efficiency data is `climb_v2_labeleff_v2_frac_e2e` (per-task fractions, e2e raw
# per-cell) PLUS the canonical fraction CSVs staged by stage_labeleff_csvs() below — NOT the old
# absolute-budget `climb_v2_labeleff_v2` (superseded; must be deleted from the HF repo if present).
PAPER_WAVES = ["climb_v2_phase2", "climb_v2_ablation_dedup", "climb_v2_labeleff_v2_frac_e2e",
               "climb_v2_h1", "climb_v2_vocab"]
# Superseded wave paths to DELETE from the HF results repo (uploaded by an earlier release).
RESULTS_STALE_DELETE = ["climb_v2_labeleff_v2"]
# Canonical label-efficiency figure inputs (aggregated fraction CSVs) staged under label_efficiency/.
LABELEFF_CSVS = ["analysis/rigor/label_efficiency_fractions_all.csv",
                 "analysis/rigor/label_efficiency_fractions_all_summary.csv",
                 "analysis/rigor/label_efficiency_fractions.csv",
                 "analysis/rigor/label_efficiency_fractions_summary.csv",
                 "analysis/rigor/label_efficiency_fractions_e2e.csv",
                 "analysis/rigor/label_efficiency_fractions_e2e_summary.csv"]
# raw per-run eval files that belong in climb-results (NO analysis on top):
RESULT_KEEP = ["config.yaml", "metadata.json", "metrics.jsonl", "verified.json",
               "moleculenet/suite_summary.json", "moleculenet/moleculenet_summary.csv",
               "moleculenet/test_predictions.csv",
               "moleculenet_cv/suite_summary.json", "moleculenet_cv/moleculenet_summary.csv",
               "moleculenet_cv/test_predictions.csv"]


def sh(cmd, **kw):
    return subprocess.run(cmd, shell=True, text=True, capture_output=True, **kw)


def logged_in():
    try:
        from huggingface_hub import whoami
        return whoami().get("name")
    except Exception:
        return None


def render_card(kind: str, org: str, out: Path):
    """Substitute <org> into a card; leave <CITATION>/<BIBTEX> for the user to fill (flag if present)."""
    txt = Path(CARD[kind]).read_text().replace("<org>", org)
    out.write_text(txt)
    left = [t for t in ("<CITATION", "<BIBTEX", "please confirm", "confirm before release") if t in txt]
    return left


def s3_prefix_report(prefix: str) -> str:
    r = sh(f"aws s3 ls {prefix} --recursive --summarize 2>/dev/null | tail -2")
    return r.stdout.strip() or "(empty / not found)"


# ---- per-repo staging assembly -------------------------------------------------

def stage_results(stage: Path, execute: bool) -> dict:
    """Copy RAW per-run eval files from local figure_data/ into the staging tree."""
    src = Path("figure_data")
    n = 0
    for wave in PAPER_WAVES:
        wdir = src / wave
        if not wdir.exists():
            continue
        for run in sorted(p for p in wdir.iterdir() if p.is_dir() and not p.name.startswith("_")):
            for rel in RESULT_KEEP:
                f = run / rel
                if f.exists():
                    n += 1
                    if execute:
                        dst = stage / wave / run.name / rel
                        dst.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(f, dst)
    # Canonical label-efficiency figure inputs (aggregated fraction CSVs) under label_efficiency/.
    for rel in LABELEFF_CSVS:
        f = Path(rel)
        if f.exists():
            n += 1
            if execute:
                dst = stage / "label_efficiency" / f.name
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(f, dst)
    return {"files": n, "source": "local figure_data/ raw eval + analysis/rigor label-efficiency CSVs"}


def stage_encoders(stage: Path, execute: bool) -> dict:
    """Pull each run's encoder/ from S3 and the shared tokenizer."""
    plan = []
    # tokenizer (vocab-1000 main tokenizer)
    plan.append((f"{BUCKET}/tokenizer_10M/", stage / "tokenizer"))
    for wave in PAPER_WAVES:
        ls = sh(f"aws s3 ls {BUCKET}/experiments/{wave}/ 2>/dev/null")
        for line in ls.stdout.splitlines():
            if not line.strip().endswith("/"):
                continue
            run = line.split()[-1].rstrip("/")
            uri = f"{BUCKET}/experiments/{wave}/{run}/encoder/"
            if sh(f"aws s3 ls {uri}model.safetensors 2>/dev/null").stdout.strip():
                plan.append((uri, stage / wave / run))
    if execute:
        for uri, dst in plan:
            dst.mkdir(parents=True, exist_ok=True)
            subprocess.run(f"aws s3 sync {uri} {dst} --only-show-errors", shell=True, check=False)
    return {"encoders": len(plan) - 1, "source": "S3 experiments/<wave>/<run>/encoder/ + tokenizer_10M"}


def stage_pretrain(stage: Path, execute: bool) -> dict:
    """Pull the shipped pre-training data (NOT the 124M corpus)."""
    items = [
        (f"{BUCKET}/tokenized_sources/pubchem_filtered/", stage / "tokenized_sources/pubchem_filtered"),
        (f"{BUCKET}/tokenized_sources/pubchem_descriptors/", stage / "tokenized_sources/pubchem_descriptors"),
        (f"{BUCKET}/tokenized/supervised_wide_parquet/", stage / "tokenized/supervised_wide_parquet"),
        (f"{BUCKET}/tokenizer_10M/", stage / "tokenizer"),
    ]
    files = [(f"{BUCKET}/configs/eval_blocklist.json", stage / "configs/eval_blocklist.json"),
             (f"{BUCKET}/configs/descriptor_stats.json", stage / "configs/descriptor_stats.json")]
    if execute:
        for uri, dst in items:
            dst.mkdir(parents=True, exist_ok=True)
            subprocess.run(f"aws s3 sync {uri} {dst} --only-show-errors", shell=True, check=False)
        for uri, dst in files:
            dst.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(f"aws s3 cp {uri} {dst} --only-show-errors", shell=True, check=False)
    return {"prefixes": [i[0] for i in items] + [f[0] for f in files],
            "note": "124M corpus NOT included (linked to hheiden/PubChem-124M-... in the card)"}


STAGERS = {"results": stage_results, "encoders": stage_encoders, "pretrain": stage_pretrain}


def publish_one(kind: str, org: str, execute: bool):
    repo_id = f"{org}/{REPO_NAME[kind]}"
    rtype = REPO_TYPE[kind]
    stage = STAGING / kind
    print(f"\n=== {repo_id}  ({rtype}, private) ===")
    if execute:
        if stage.exists():
            shutil.rmtree(stage)
        stage.mkdir(parents=True)
    info = STAGERS[kind](stage, execute)
    for k, v in info.items():
        print(f"    {k}: {v}")
    left = render_card(kind, org, (stage / "README.md") if execute else Path("/dev/null"))
    if left:
        print(f"    ⚠️  card still has unfilled placeholders: {left}  (fill before real release)")
    if not execute:
        print("    [dry-run] would create_repo(private=True) + upload_folder. Pass --execute to do it.")
        return
    from huggingface_hub import create_repo, upload_folder, delete_folder, list_repo_files
    create_repo(repo_id, repo_type=rtype, private=True, exist_ok=True)
    # Delete superseded folders first (upload_folder only adds/updates; it never removes stale paths,
    # which is how a "competing version" survives on the Hub). Only the results dataset has any.
    if kind == "results":
        existing = set(list_repo_files(repo_id, repo_type=rtype))
        for stale in RESULTS_STALE_DELETE:
            if any(p.startswith(f"{stale}/") for p in existing):
                print(f"    🗑  deleting superseded '{stale}/' from {repo_id}")
                delete_folder(path_in_repo=stale, repo_id=repo_id, repo_type=rtype,
                              commit_message=f"remove superseded {stale} (replaced by per-task fractions)")
    upload_folder(folder_path=str(stage), repo_id=repo_id, repo_type=rtype,
                  commit_message="CLIMB release upload")
    print(f"    ✅ uploaded {stage} -> https://huggingface.co/{'datasets/' if rtype=='dataset' else ''}{repo_id}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--org", required=True, help="HF namespace (user or org) the repos live under")
    ap.add_argument("--repo", choices=["cards", "results", "encoders", "pretrain", "all"], default="all")
    ap.add_argument("--execute", action="store_true", help="actually create+upload (default: dry-run)")
    a = ap.parse_args()

    who = logged_in()
    print(f"HF auth: {'logged in as ' + who if who else 'NOT LOGGED IN'}")
    if a.execute and not who:
        print("Refusing to --execute while not logged in. Run `hf auth login` first.", file=sys.stderr)
        return 2

    kinds = ["encoders", "results", "pretrain"] if a.repo in ("all", "cards") else [a.repo]
    if a.repo == "cards":
        # cards-only: just push each README, skip the heavy data staging
        for kind in kinds:
            repo_id = f"{a.org}/{REPO_NAME[kind]}"
            print(f"\n=== card -> {repo_id} ===")
            tmp = STAGING / f"{kind}_card"; tmp.mkdir(parents=True, exist_ok=True)
            left = render_card(kind, a.org, tmp / "README.md")
            if left:
                print(f"    ⚠️  unfilled placeholders: {left}")
            if not a.execute:
                print("    [dry-run] would upload README.md only.")
                continue
            from huggingface_hub import create_repo, upload_file
            create_repo(repo_id, repo_type=REPO_TYPE[kind], private=True, exist_ok=True)
            upload_file(path_or_fileobj=str(tmp / "README.md"), path_in_repo="README.md",
                        repo_id=repo_id, repo_type=REPO_TYPE[kind], commit_message="card")
            print(f"    ✅ card pushed")
        return 0

    print("\nS3 size probes (informational):")
    if a.repo in ("all", "encoders"):
        print("  encoders:", s3_prefix_report(f"{BUCKET}/experiments/climb_v2_phase2/unsup_8M/encoder/"))
    for kind in kinds:
        publish_one(kind, a.org, a.execute)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
