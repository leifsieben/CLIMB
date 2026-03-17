#!/usr/bin/env python3
"""Validate run manifests before launching training."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from supervised_streaming import resolve_family_specs
from storage_utils import list_data_files, path_exists


def _load_manifest(path: str) -> Dict:
    with open(path) as f:
        return json.load(f)


def _selected_runs(manifest: Dict, run_ids: List[str]) -> List[Dict]:
    if not run_ids:
        return manifest["runs"]
    wanted = set(run_ids)
    return [run for run in manifest["runs"] if run["run_id"] in wanted]


def _validate_imports() -> List[str]:
    errors = []
    for module_name in ("deepchem", "torch", "transformers", "yaml", "fsspec"):
        try:
            __import__(module_name)
        except Exception as exc:
            errors.append(f"Missing import {module_name}: {exc}")
    try:
        import pyarrow  # noqa: F401
    except Exception as exc:
        errors.append(f"Missing import pyarrow: {exc}")
    return errors


def _validate_run(run: Dict) -> List[str]:
    errors = []
    cfg = run["pretrain_config"]

    tokenizer_path = cfg["tokenizer_path"]
    if tokenizer_path.startswith("s3://"):
        if not path_exists(f"{tokenizer_path.rstrip('/')}/tokenizer.json"):
            errors.append(f"{run['run_id']}: tokenizer.json not found at {tokenizer_path}")
    else:
        tokenizer_file = Path(tokenizer_path) / "tokenizer.json"
        if not tokenizer_file.exists():
            errors.append(f"{run['run_id']}: missing tokenizer {tokenizer_file}")

    if cfg.get("unsupervised_data"):
        unsup_files = list_data_files(cfg["unsupervised_data"])
        if not unsup_files:
            errors.append(f"{run['run_id']}: no unsupervised shards resolved")

    parquet_path = cfg.get("supervised_tokenized_parquet_path") or cfg.get("supervised_parquet_path")
    if parquet_path:
        if not path_exists(parquet_path):
            errors.append(f"{run['run_id']}: supervised parquet not found: {parquet_path}")
        else:
            try:
                _, resolved = resolve_family_specs(parquet_path, cfg.get("supervised_families"))
                missing = [item["name"] for item in resolved if not item["columns"]]
                if missing:
                    errors.append(f"{run['run_id']}: supervised families missing columns: {missing}")
            except Exception as exc:
                errors.append(f"{run['run_id']}: could not resolve supervised families: {exc}")

    output_dir = Path(run["output_dir"])
    try:
        output_dir.parent.mkdir(parents=True, exist_ok=True)
    except Exception as exc:
        errors.append(f"{run['run_id']}: cannot create output parent {output_dir.parent}: {exc}")

    backup_s3_uri = run.get("backup_s3_uri")
    if backup_s3_uri and not backup_s3_uri.startswith("s3://"):
        errors.append(f"{run['run_id']}: invalid backup S3 URI: {backup_s3_uri}")

    if cfg["compute_budget"].get("total_tokens") and not (
        cfg["mlm_training"].get("tokens_per_step_estimate")
        or cfg["supervised_training"].get("tokens_per_step_estimate")
    ):
        # This is informational only; the pipeline can estimate lengths dynamically.
        pass

    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description="Preflight CLIMB experiment runs")
    parser.add_argument("--manifest", required=True, help="Resolved manifest JSON")
    parser.add_argument("--run_id", action="append", default=[], help="Specific run_id to validate")
    args = parser.parse_args()

    manifest = _load_manifest(args.manifest)
    runs = _selected_runs(manifest, args.run_id)
    errors = _validate_imports()
    for run in runs:
        errors.extend(_validate_run(run))

    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        raise SystemExit(1)

    print(f"Validated {len(runs)} run(s) successfully.")


if __name__ == "__main__":
    main()
