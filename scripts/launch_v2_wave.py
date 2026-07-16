"""v2 launch script. Replaces v1's launch_experiment_wave.py.

For each requested run_id from the manifest:
  1. Materialise the run dir + write its config.yaml.
  2. Decide path: random_baseline → random_baseline_v2.py; else pretrain_v2.py + eval_v2.py.
  3. Spawn the watchdog as a separate subprocess.
  4. Run the trainer; on success run eval. On stall, the watchdog kills it and
     we record STALLED status.
  5. Backup outputs to S3 (best-effort).

Skip rules:
  - If <run_dir>/moleculenet/suite_summary.json exists locally OR on S3, skip
    (configurable via --no_skip_existing).

Status tracking:
  - <run_dir>/run_status.json with {status: "ok"|"stalled"|"failed", ...}
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import yaml


def _path_exists_local(p: str) -> bool:
    return Path(p).exists()


def _path_exists_s3(uri: str) -> bool:
    try:
        out = subprocess.run(
            ["aws", "s3", "ls", uri], check=False, capture_output=True, text=True, timeout=20,
        )
        return out.returncode == 0 and bool(out.stdout.strip())
    except Exception:
        return False


def _should_skip(run: dict, no_skip: bool) -> bool:
    if no_skip:
        return False
    eval_dir = Path(run["evaluation_output_dir"])
    if (eval_dir / "suite_summary.json").exists():
        return True
    s3_summary = run["backup_s3_uri"].rstrip("/") + "/moleculenet/suite_summary.json"
    return _path_exists_s3(s3_summary)


def _write_config_yaml(run: dict) -> Path:
    run_dir = Path(run["output_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = run_dir / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(run["pretrain_config"], sort_keys=False))
    return cfg_path


def _run_status(run_dir: Path, status: str, **extra) -> None:
    payload = {"status": status, "updated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **extra}
    (run_dir / "run_status.json").write_text(json.dumps(payload, indent=2))


def _spawn_watchdog(pid: int, metrics_path: Path, stall_seconds: int, max_seconds: int) -> subprocess.Popen:
    cmd = [
        sys.executable, str(Path(__file__).parent / "run_watchdog.py"),
        "--pid", str(pid),
        "--metrics", str(metrics_path),
        "--stall_seconds", str(stall_seconds),
        "--max_seconds", str(max_seconds),
    ]
    return subprocess.Popen(cmd)


def _stall_seconds_for(run: dict) -> int:
    sel = run.get("selection", {})
    fps = int(sel.get("total_forward_passes") or run["pretrain_config"]["selection"].get("total_forward_passes", 250_000_000))
    if fps >= 1_000_000_000:
        return 60 * 60     # 60 min
    if fps >= 500_000_000:
        return 45 * 60     # 45 min
    return 30 * 60         # 30 min default


def _max_seconds_for(run: dict) -> int:
    sel = run.get("selection", {})
    fps = int(sel.get("total_forward_passes") or run["pretrain_config"]["selection"].get("total_forward_passes", 250_000_000))
    # Generous cap: ~12s per 1k FPs ≈ 50M FPs/h → 250M FPs ≈ 5 h, +eval. Pad heavily.
    if fps >= 1_000_000_000:
        return 36 * 3600
    if fps >= 500_000_000:
        return 24 * 3600
    return 12 * 3600


def _run_pretrain(run: dict) -> str:
    """Returns one of: 'ok', 'stalled', 'failed'."""
    run_dir = Path(run["output_dir"])
    cfg_path = _write_config_yaml(run)
    metrics_path = run_dir / "metrics.jsonl"

    cmd = [sys.executable, "pretrain_v2.py", "--run_dir", str(run_dir), "--config", str(cfg_path)]

    print(f"[launch_v2] running: {' '.join(cmd)}", flush=True)
    proc = subprocess.Popen(cmd)
    watchdog = _spawn_watchdog(
        proc.pid, metrics_path,
        stall_seconds=_stall_seconds_for(run),
        max_seconds=_max_seconds_for(run),
    )
    try:
        proc.wait()
    except KeyboardInterrupt:
        proc.send_signal(signal.SIGTERM)
        proc.wait()
        watchdog.terminate()
        raise

    watchdog.terminate()
    try:
        watchdog.wait(timeout=10)
    except subprocess.TimeoutExpired:
        watchdog.kill()

    if proc.returncode == 0:
        return "ok"
    if proc.returncode in (-signal.SIGTERM, signal.SIGTERM, -signal.SIGKILL, signal.SIGKILL):
        return "stalled"
    return "failed"


def _run_eval(run: dict) -> str:
    run_dir = Path(run["output_dir"])
    cfg_path = run_dir / "config.yaml"
    encoder_path = run_dir / "encoder"
    tokenizer_path = run_dir / "tokenizer"
    eval_dir = run_dir / "moleculenet"
    eval_dir.mkdir(exist_ok=True)

    if not encoder_path.exists():
        return "failed_no_encoder"
    if not tokenizer_path.exists():
        return "failed_no_tokenizer"

    ev = (run.get("pretrain_config", {}) or {}).get("evaluation", {}) or {}
    cmd = [
        sys.executable, "eval_v2.py",
        "--encoder", str(encoder_path),
        "--tokenizer", str(tokenizer_path),
        "--output_dir", str(eval_dir),
        "--pool", str(ev.get("pool", "mean")),
        "--standardize", str(ev.get("standardize", "zscore")),
        "--head", str(ev.get("head", "mlp")),
        "--max_length", str(ev.get("max_length", 256)),
    ]
    if ev.get("head_seeds"):
        cmd += ["--head_seeds", *[str(s) for s in ev["head_seeds"]]]
    print(f"[launch_v2] eval: {' '.join(cmd)}", flush=True)
    rc = subprocess.run(cmd).returncode
    return "ok" if rc == 0 else "failed"


def _run_ecfp_anchor(run: dict) -> str:
    """Classical ECFP4 + XGBoost baseline — no encoder, pure eval."""
    run_dir = Path(run["output_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    eval_dir = run_dir / "moleculenet"
    eval_dir.mkdir(exist_ok=True)
    ov = run.get("eval_override", {}) or {}
    ev = (run.get("pretrain_config", {}) or {}).get("evaluation", {}) or {}
    cmd = [
        sys.executable, "eval_v2.py",
        "--output_dir", str(eval_dir),
        "--featurizer", str(ov.get("featurizer", "ecfp4")),
        "--head", str(ov.get("head", "xgb")),
    ]
    if ev.get("head_seeds"):
        cmd += ["--head_seeds", *[str(s) for s in ev["head_seeds"]]]
    print(f"[launch_v2] ecfp4_anchor: {' '.join(cmd)}", flush=True)
    rc = subprocess.run(cmd).returncode
    return "ok" if rc == 0 else "failed"


def _run_random_baseline(run: dict) -> str:
    run_dir = Path(run["output_dir"])
    cfg_path = _write_config_yaml(run)
    cmd = [sys.executable, "random_baseline_v2.py", "--run_dir", str(run_dir), "--config", str(cfg_path)]
    print(f"[launch_v2] random_baseline: {' '.join(cmd)}", flush=True)
    rc = subprocess.run(cmd).returncode
    return "ok" if rc == 0 else "failed"


def _backup_to_s3(run: dict) -> None:
    run_dir = Path(run["output_dir"])
    s3_uri = run["backup_s3_uri"]
    # Sync only essential files (skip large model dirs to save bandwidth).
    essentials = ["metrics.jsonl", "metadata.json", "config.yaml", "run_status.json", "heartbeat.json"]
    for f in essentials:
        src = run_dir / f
        if src.exists():
            try:
                subprocess.run(["aws", "s3", "cp", str(src), f"{s3_uri}/{f}"], check=False, timeout=120)
            except Exception:
                pass
    eval_dir = run_dir / "moleculenet"
    if eval_dir.exists():
        try:
            subprocess.run(
                ["aws", "s3", "sync", str(eval_dir), f"{s3_uri}/moleculenet"],
                check=False, timeout=600,
            )
        except Exception:
            pass


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--run_id", action="append", default=[])
    p.add_argument("--run_type", action="append", default=[])
    p.add_argument("--worker_name", default="local")
    p.add_argument("--no_skip_existing", action="store_true")
    args = p.parse_args()

    with open(args.manifest) as f:
        manifest = json.load(f)

    selected = manifest["runs"]
    if args.run_id:
        selected = [r for r in selected if r["run_id"] in set(args.run_id)]
    if args.run_type:
        selected = [r for r in selected if r["run_type"] in set(args.run_type)]

    if not selected:
        print("[launch_v2] no runs match selection")
        return 0

    print(f"[launch_v2] {args.worker_name}: {len(selected)} runs queued")

    for run in selected:
        run_id = run["run_id"]
        run_dir = Path(run["output_dir"])
        run_dir.mkdir(parents=True, exist_ok=True)

        if _should_skip(run, args.no_skip_existing):
            print(f"[launch_v2] SKIP {run_id} (already evaluated)")
            continue

        print(f"\n========== {run_id} ({run['run_type']}) ==========")
        run_started = time.time()

        if run["run_type"] == "random_baseline":
            status = _run_random_baseline(run)
            _run_status(run_dir, status, elapsed_seconds=time.time() - run_started)
        elif run["run_type"] == "ecfp4_anchor":
            status = _run_ecfp_anchor(run)
            _run_status(run_dir, status, elapsed_seconds=time.time() - run_started)
        else:
            pre_status = _run_pretrain(run)
            if pre_status != "ok":
                _run_status(run_dir, pre_status, phase="pretrain", elapsed_seconds=time.time() - run_started)
                _backup_to_s3(run)
                continue
            ev_status = _run_eval(run)
            _run_status(run_dir, ev_status, phase="eval" if ev_status != "ok" else "complete",
                        elapsed_seconds=time.time() - run_started)

        _backup_to_s3(run)

    print("[launch_v2] done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
