"""Push runs that are on S3/local but missing from HuggingFace.

The standing rule is results + checkpoints + training data live locally, on S3 AND on HF. HF is
the copy a reader of the paper can actually fetch, so a run that never reached it is not
reproducible even though two backups exist.

Uploads RESULT artifacts only (summaries, suite json, prediction dumps, per-fold csv, configs,
metrics) -- never optimizer state or raw features, matching what is already there. Encoders go
to the model repo separately, weights only.

Idempotent: it re-lists the repo and skips runs already present, so it can be re-run after an
interruption without duplicating uploads.
"""
from __future__ import annotations
import sys
from collections import defaultdict
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parent.parent
FD = ROOT / "figure_data"
RESULTS_REPO = "lsieben/climb-results"
ENCODER_REPO = "lsieben/climb-encoders"
WAVES = ["climb_v2_phase2", "climb_v2_h1", "climb_v2_ablation_dedup", "climb_v2_vocab",
         "climb_v2_expA", "climb_v2_expB", "climb_v2_lrsweep", "cbs_benchmark"]
RESULT_NAMES = {"moleculenet_summary.csv", "suite_summary.json", "test_predictions.csv",
                "per_fold.csv", "metrics.jsonl", "config.yaml", "metadata.json",
                "verified.json", "reference_scoring.json"}
ENCODER_NAMES = {"model.safetensors", "config.json", "tokenizer.json", "tokenizer_config.json",
                 "special_tokens_map.json"}
# Quarantined copies and smoke-test stubs must never reach the public mirror: the first
# are known-wrong numbers, the second are 3-molecule placeholders that look like runs.
SKIP_DIR_MARKERS = (".INVALID", ".REJECTED", ".DRIFT")
SKIP_RUN_MARKERS = ("SMOKE", "STUB", "_TEST")


def repo_runs(api, repo, typ):
    out = defaultdict(set)
    for f in api.list_repo_files(repo, repo_type=typ):
        b = f.split("/")
        if len(b) >= 3 and b[0] in WAVES:
            out[b[0]].add(b[1])
    return out


def files_for(run_dir: Path, names: set) -> list:
    out = []
    for p in run_dir.rglob("*"):
        if not p.is_file() or p.name not in names:
            continue
        if any(m in str(p) for m in SKIP_DIR_MARKERS):
            continue
        out.append(p)
    return out


def push(api, repo, typ, wave, run, names, dry) -> int:
    d = FD / wave / run
    fs = files_for(d, names)
    if not fs:
        print(f"    {wave}/{run}: no eligible files, skipped")
        return 0
    size = sum(f.stat().st_size for f in fs)
    print(f"    {wave}/{run}: {len(fs)} files, {size/1e6:.1f} MB", flush=True)
    if dry:
        return 0
    api.upload_folder(repo_id=repo, repo_type=typ, folder_path=str(d),
                      path_in_repo=f"{wave}/{run}",
                      allow_patterns=[f"**/{n}" for n in names] + list(names),
                      ignore_patterns=[f"*{m}*" for m in SKIP_DIR_MARKERS],
                      commit_message=f"backup {wave}/{run}")
    return 1


def main(argv) -> int:
    dry = "--dry-run" in argv
    only = [a for a in argv if not a.startswith("--")]
    api = HfApi()
    have_res = repo_runs(api, RESULTS_REPO, "dataset")
    have_enc = repo_runs(api, ENCODER_REPO, "model")

    print("=== RESULTS ===")
    n = 0
    for w in WAVES:
        d = FD / w
        if not d.is_dir() or (only and w not in only):
            continue
        miss = sorted({p.name for p in d.iterdir()
                       if p.is_dir() and not p.name.startswith("_")
                       and not any(m in p.name for m in SKIP_RUN_MARKERS)}
                      - have_res.get(w, set()))
        if not miss:
            continue
        print(f"  {w}: {len(miss)} to upload")
        for r in miss:
            n += push(api, RESULTS_REPO, "dataset", w, r, RESULT_NAMES, dry)

    print("\n=== ENCODERS ===")
    m = 0
    for w in WAVES:
        d = FD / w
        if not d.is_dir() or (only and w not in only):
            continue
        for p in sorted(d.iterdir()):
            e = p / "encoder"
            if not (e / "model.safetensors").exists():
                continue
            if p.name in have_enc.get(w, set()):
                continue
            fs = files_for(e, ENCODER_NAMES)
            size = sum(f.stat().st_size for f in fs)
            print(f"    {w}/{p.name}: {len(fs)} files, {size/1e6:.1f} MB", flush=True)
            if dry:
                continue
            api.upload_folder(repo_id=ENCODER_REPO, repo_type="model", folder_path=str(e),
                              path_in_repo=f"{w}/{p.name}",
                              allow_patterns=list(ENCODER_NAMES),
                              commit_message=f"encoder {w}/{p.name}")
            m += 1

    print(f"\n{'DRY RUN -- nothing uploaded' if dry else f'uploaded {n} result runs, {m} encoders'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
