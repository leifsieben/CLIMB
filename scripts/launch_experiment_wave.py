#!/usr/bin/env python3
"""Launch resolved manifest runs sequentially."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiment_manifest import dump_yaml

PYTHON = sys.executable


def _load_manifest(path: str) -> Dict:
    with open(path) as f:
        return json.load(f)


def _select_runs(manifest: Dict, stage: str, run_type: str, run_ids: List[str], max_runs: int) -> List[Dict]:
    runs = manifest["runs"]
    if stage:
        runs = [run for run in runs if run["stage"] == stage]
    if run_type:
        runs = [run for run in runs if run["run_type"] == run_type]
    if run_ids:
        wanted = set(run_ids)
        runs = [run for run in runs if run["run_id"] in wanted]
    if max_runs:
        runs = runs[:max_runs]
    return runs


def _background_sync(run_dir: Path, s3_dest: str, interval_seconds: int) -> subprocess.Popen:
    return subprocess.Popen(
        [
            PYTHON,
            "scripts/sync_run_to_s3.py",
            "--run_dir",
            str(run_dir),
            "--s3_dest",
            s3_dest,
            "--interval_seconds",
            str(interval_seconds),
        ]
    )


def _terminate(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    proc.send_signal(signal.SIGTERM)
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()


def _s3_object_exists(s3_uri: str) -> bool:
    result = subprocess.run(
        ["aws", "s3", "ls", s3_uri],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.returncode == 0


def _should_skip_existing(run: Dict) -> bool:
    run_dir = Path(run["output_dir"])
    if (run_dir / "metadata.json").exists():
        return True
    if Path(run["evaluation_output_dir"], "suite_summary.json").exists():
        return True

    backup_s3_uri = run.get("backup_s3_uri")
    if not backup_s3_uri:
        return False
    return _s3_object_exists(f"{backup_s3_uri}/moleculenet/suite_summary.json")


def _write_run_context(run: Dict, worker_name: str) -> None:
    run_dir = Path(run["output_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "run_id": run["run_id"],
        "run_type": run["run_type"],
        "stage": run["stage"],
        "worker_name": worker_name,
        "output_dir": run["output_dir"],
        "backup_s3_uri": run.get("backup_s3_uri"),
        "evaluation_output_dir": run["evaluation_output_dir"],
    }
    with (run_dir / "run_context.json").open("w") as f:
        json.dump(payload, f, indent=2)


def _run_pretrain(run: Dict, resume: bool) -> None:
    run_dir = Path(run["output_dir"])
    config_path = run_dir / "config.yaml"
    dump_yaml(str(config_path), run["pretrain_config"])
    cmd = [
        PYTHON,
        "pretrain_pipeline.py",
        "--config",
        str(config_path),
    ]
    if resume:
        cmd.append("--resume")
    subprocess.run(cmd, check=True)


def _run_eval(run: Dict, evaluation_cfg: Dict) -> None:
    cfg = {
        "freeze_encoder": True,
        "num_epochs": 50,
        "batch_size": 32,
        "learning_rate": 2e-5,
        **evaluation_cfg,
    }
    tokenizer_path = run["pretrain_config"]["tokenizer_path"]
    if run["pretrain_config"]["compute_budget"]["supervised_fraction"] > 0:
        pretrained_model = str(Path(run["output_dir"]) / "supervised" / "encoder")
    else:
        pretrained_model = str(Path(run["output_dir"]) / "unsupervised" / "encoder")

    cmd = [
        PYTHON,
        "scripts/run_moleculenet_suite.py",
        "--pretrained_model",
        pretrained_model,
        "--output",
        run["evaluation_output_dir"],
        "--tokenizer",
        tokenizer_path,
        "--num_epochs",
        str(cfg["num_epochs"]),
        "--batch_size",
        str(cfg["batch_size"]),
        "--learning_rate",
        str(cfg["learning_rate"]),
        "--skip_existing",
    ]
    if cfg.get("freeze_encoder", True):
        cmd.append("--freeze_encoder")
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch a wave of CLIMB manifest runs")
    parser.add_argument("--manifest", required=True, help="Resolved manifest JSON")
    parser.add_argument("--stage", help="Optional stage filter (smoke/main)")
    parser.add_argument("--run_type", help="Optional run type filter")
    parser.add_argument("--run_id", action="append", default=[], help="Specific run_id to launch")
    parser.add_argument("--max_runs", type=int, default=0, help="Launch at most this many runs")
    parser.add_argument("--preflight", action="store_true", help="Run preflight validation before launching")
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoints")
    parser.add_argument("--skip_eval", action="store_true", help="Skip MoleculeNet evaluation")
    parser.add_argument("--worker_name", default=os.environ.get("HOSTNAME", "local"), help="Label for this launcher host")
    parser.add_argument("--backup_interval_seconds", type=int, default=900, help="Background S3 sync interval")
    parser.add_argument("--skip_existing", action="store_true", help="Skip runs with existing metadata.json")
    parser.add_argument("--dry_run", action="store_true", help="Print selected runs and exit")
    args = parser.parse_args()

    manifest = _load_manifest(args.manifest)
    runs = _select_runs(manifest, args.stage, args.run_type, args.run_id, args.max_runs)
    if not runs:
        raise SystemExit("No runs selected.")

    if args.preflight:
        cmd = [PYTHON, "scripts/preflight_experiment.py", "--manifest", args.manifest]
        for run in runs:
            cmd.extend(["--run_id", run["run_id"]])
        subprocess.run(cmd, check=True)

    if args.dry_run:
        for run in runs:
            print(f"{run['run_id']} -> {run['output_dir']}")
        return

    evaluation_cfg = manifest.get("evaluation", {})
    for run in runs:
        if args.skip_existing and _should_skip_existing(run):
            print(f"Skipping existing run {run['run_id']}")
            continue
        run_dir = Path(run["output_dir"])
        _write_run_context(run, args.worker_name)
        backup_proc = None
        try:
            if run.get("backup_s3_uri"):
                backup_proc = _background_sync(run_dir, run["backup_s3_uri"], args.backup_interval_seconds)
            _run_pretrain(run, resume=args.resume)
            if not args.skip_eval:
                _run_eval(run, evaluation_cfg)
        finally:
            if backup_proc is not None:
                _terminate(backup_proc)
                subprocess.run(
                    [
                        PYTHON,
                        "scripts/sync_run_to_s3.py",
                        "--run_dir",
                        str(run_dir),
                        "--s3_dest",
                        run["backup_s3_uri"],
                    ],
                    check=False,
                )


if __name__ == "__main__":
    main()
