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


NOTIFY = str(Path(__file__).parent / "notify.sh")


def _notify(level: str, subject: str, message: str) -> None:
    """Best-effort direct SNS notification (never fatal)."""
    try:
        subprocess.run(["bash", NOTIFY, level, subject, message], timeout=30, check=False)
    except Exception:
        pass


def _budget_fp(run: dict) -> int:
    """Forward-pass budget, or 0 for runs that legitimately have none.

    Eval-only runs (ecfp4/fp_desc anchors, random baselines) carry a `selection` without
    `total_forward_passes` AND a `pretrain_config` without a `selection` block at all -- so the
    old fall-through raised KeyError on the FIRST such run and killed the whole worker before it
    trained anything. That stayed hidden while every anchor in flight happened to be already
    complete (and so skipped before this call); it surfaced the moment a wave containing a
    not-yet-complete anchor was launched, which reported as 5 runs 'missing' with no training and
    no obvious cause.

    Returning 0 is correct rather than defensive: these runs have no training to budget. Callers
    must treat 0 as 'no budget' -- _reached_budget already does, via _is_complete's anchor branch.
    """
    sel = run.get("selection") or {}
    fp = sel.get("total_forward_passes")
    if not fp:
        pc_sel = (run.get("pretrain_config") or {}).get("selection") or {}
        fp = pc_sel.get("total_forward_passes")
    if not fp:
        return 0 if not run.get("requires_pretrain") else 250_000_000
    return int(fp)


def _final_fp_from_jsonl_text(text: str) -> int:
    last = None
    for line in text.splitlines():
        if line.strip():
            last = line
    if not last:
        return 0
    try:
        return int(json.loads(last).get("forward_passes_seen", 0))
    except Exception:
        return 0


def _final_fp_local(run_dir: Path) -> int:
    mp = run_dir / "metrics.jsonl"
    if not mp.exists():
        return 0
    return _final_fp_from_jsonl_text(mp.read_text())


def _final_fp_s3(run: dict) -> int:
    uri = run["backup_s3_uri"].rstrip("/") + "/metrics.jsonl"
    try:
        out = subprocess.run(["aws", "s3", "cp", uri, "-"], capture_output=True,
                             text=True, timeout=90, check=False)
        if out.returncode != 0:
            return 0
        return _final_fp_from_jsonl_text(out.stdout)
    except Exception:
        return 0


def _reached_budget(run: dict, *, prefer_s3: bool = False) -> bool:
    """Ground-truth completion test: did training actually reach >=98% of its
    forward-pass budget? This replaces the old 'a suite_summary.json exists' test,
    which a TRUNCATED run also satisfies and which silently poisoned re-runs."""
    budget = _budget_fp(run)
    fp = _final_fp_local(Path(run["output_dir"]))
    if prefer_s3 and fp < 0.98 * budget:
        fp = max(fp, _final_fp_s3(run))
    return fp >= 0.98 * budget


def _is_complete(run: dict) -> bool:
    """THE completion test. Precedence:
      1) A `verified.json` marker (local or S3) — written only on genuine completion.
      2) For non-pretraining anchors (no FP budget), fall back to suite existence.
      3) Otherwise: training must have reached >=98% of its FP budget.

    Every consumer of "is this run done?" MUST call this — skip logic, the worker's
    shutdown gate, and any downstream artifact gate. Having two different notions of
    completion is its own bug class: the shutdown gate previously accepted ONLY a local
    verified.json, so runs that this function legitimately considers done (anchors, and
    complete-but-unmarked runs verified via S3 or FP) were reported "missing" forever and
    the box never self-terminated — billing indefinitely while looking like a real alarm."""
    run_dir = Path(run["output_dir"])
    # (1) explicit verified marker is the single source of truth
    if (run_dir / "verified.json").exists():
        return True
    if _path_exists_s3(run["backup_s3_uri"].rstrip("/") + "/verified.json"):
        return True
    # (2) anchors / random baselines have no training budget
    if run.get("run_type") in ("ecfp4_anchor", "random_baseline"):
        eval_dir = Path(run["evaluation_output_dir"])
        return ((eval_dir / "suite_summary.json").exists()
                or _path_exists_s3(run["backup_s3_uri"].rstrip("/") + "/moleculenet/suite_summary.json"))
    # (3) pretraining runs: require the FP budget to have actually been reached
    return _reached_budget(run, prefer_s3=True)


def _should_skip(run: dict, no_skip: bool) -> bool:
    """Skip only a run that is VERIFIED complete (see `_is_complete`).
    A stale/truncated suite_summary.json NO LONGER causes a skip."""
    if no_skip:
        return False
    return _is_complete(run)


def _write_verified_marker(run: dict, run_dir: Path, final_fp: int) -> None:
    """Only genuine, budget-reaching completion earns this marker (the thing
    `_should_skip` trusts). Downstream / future waves treat it as authoritative."""
    (run_dir / "verified.json").write_text(json.dumps({
        "run_id": run["run_id"],
        "budget_fp": _budget_fp(run),
        "final_fp": final_fp,
        "fraction": round(final_fp / max(1, _budget_fp(run)), 4),
        "verified_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }, indent=2))


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
    # Third copy of the budget fall-through; same KeyError on eval-only runs. Delegated too.
    fps = _budget_fp(run)
    if fps >= 1_000_000_000:
        return 60 * 60     # 60 min
    if fps >= 500_000_000:
        return 45 * 60     # 45 min
    return 30 * 60         # 30 min default


def _max_seconds_for(run: dict) -> int:
    # Delegates to _budget_fp rather than repeating its fall-through: this function carried its
    # own copy, which raised the same KeyError on eval-only runs and would drift from the real
    # budget definition. One definition of the budget, one place to fix it.
    fps = _budget_fp(run)
    if not fps:
        return 4 * 3600      # eval-only run: no training to size against
    # Hard ceiling only — real stalls are caught separately by stall_seconds (no-progress).
    # Measured throughput is ~700-760 seq/s; budget at a CONSERVATIVE 400 seq/s + 4h for
    # eval/warmup so genuinely long runs are never killed prematurely (the old flat 12h cap
    # silently truncated every 48M≈18h and 96M≈35h run at ~33M FP).
    #
    # 700-760 seq/s is the PRECOMPUTED-descriptor rate. An MTR run without a precompute directory
    # calls rdkit in the dataloader and measures ~300 seq/s on a 16-vCPU box, so 400 is no longer
    # conservative for it -- it is optimistic by a third, and this ceiling would have killed the
    # 50M c124 rung at ~42M forward passes and the 100M at ~80M. That is the same silent
    # truncation the flat 12h cap used to cause, from the other direction: a ceiling is only
    # conservative relative to a throughput it actually applies to.
    pc = run.get("pretrain_config") or {}
    live_descriptors = "mtr" in (pc.get("selection", {}) or {}).get("objectives", run.get(
        "selection", {}).get("objectives", {})) and not pc.get("descriptor_precompute_dir")
    floor = 200 if live_descriptors else 400
    return int(fps / floor) + 4 * 3600


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
    """Back up EVERYTHING irreplaceable: metrics, eval outputs, AND the trained encoder.

    The encoder used to be excluded here "to save bandwidth". That is the one file that cannot be
    regenerated without repeating the whole pretraining run, and it nearly cost us the three
    s2u_dense_from8M checkpoints on 2026-08-17: the box self-stopped on completion, the results had
    synced, and the encoders existed on that instance's disk and nowhere else — one terminate away
    from being gone. 165 MB per run is nothing against re-pretraining. Never exclude it again.
    """
    run_dir = Path(run["output_dir"])
    s3_uri = run["backup_s3_uri"]
    enc = run_dir / "encoder"
    if enc.exists():
        try:
            subprocess.run(["aws", "s3", "sync", str(enc), f"{s3_uri}/encoder", "--only-show-errors"],
                           check=False, timeout=1800)
        except Exception:
            pass
    tok = run_dir / "tokenizer"
    if tok.exists():
        try:
            subprocess.run(["aws", "s3", "sync", str(tok), f"{s3_uri}/tokenizer", "--only-show-errors"],
                           check=False, timeout=600)
        except Exception:
            pass
    essentials = ["metrics.jsonl", "metadata.json", "config.yaml", "run_status.json", "heartbeat.json", "verified.json"]
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
    p.add_argument("--check_complete", action="store_true",
                   help="Do not run anything: report {done, missing} for the manifest using the "
                        "SAME completion test as the skip logic, and exit 0 iff all runs are "
                        "complete. This is what the worker's shutdown gate must call — a gate "
                        "with its own separate notion of 'done' will strand boxes forever.")
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

    if args.check_complete:
        done, missing = [], []
        for run in selected:
            (done if _is_complete(run) else missing).append(run["run_id"])
        print(json.dumps({"done": done, "missing": missing}))
        return 0 if not missing else 1

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
        budget = _budget_fp(run)
        _notify("INFO", f"{args.worker_name}: START {run_id}",
                f"run_type={run['run_type']} budget_fp={budget:,}")

        if run["run_type"] == "random_baseline":
            status = _run_random_baseline(run)
            _run_status(run_dir, status, elapsed_seconds=time.time() - run_started)
            _notify("DONE" if status == "ok" else "ALERT",
                    f"{args.worker_name}: {status.upper()} {run_id}", "random baseline")
        elif run["run_type"] == "ecfp4_anchor":
            status = _run_ecfp_anchor(run)
            _run_status(run_dir, status, elapsed_seconds=time.time() - run_started)
            _notify("DONE" if status == "ok" else "ALERT",
                    f"{args.worker_name}: {status.upper()} {run_id}", "ecfp4 anchor")
        else:
            pre_status = _run_pretrain(run)
            final_fp = _final_fp_local(run_dir)
            reached = final_fp >= 0.98 * budget
            if pre_status != "ok" or not reached:
                # TRUNCATION / STALL: never silent. Do NOT write a verified marker, so
                # a later wave will re-run this instead of treating it as complete.
                pct = 100.0 * final_fp / max(1, budget)
                _run_status(run_dir, "truncated", phase="pretrain",
                            final_fp=final_fp, budget_fp=budget, fraction=round(final_fp / max(1, budget), 4),
                            pretrain_status=pre_status, elapsed_seconds=time.time() - run_started)
                _notify("ALERT", f"{args.worker_name}: TRUNCATED {run_id}",
                        f"pretrain_status={pre_status} reached {final_fp:,}/{budget:,} FP ({pct:.0f}%). "
                        f"NOT marked complete — will be re-run. Investigate before assuming this datapoint is valid.")
                _backup_to_s3(run)
                continue
            ev_status = _run_eval(run)
            if ev_status == "ok":
                _write_verified_marker(run, run_dir, final_fp)
                _run_status(run_dir, "ok", phase="complete", final_fp=final_fp, budget_fp=budget,
                            elapsed_seconds=time.time() - run_started)
                _notify("DONE", f"{args.worker_name}: COMPLETE {run_id}",
                        f"reached {final_fp:,}/{budget:,} FP (100%), eval OK, verified.")
            else:
                _run_status(run_dir, ev_status, phase="eval", final_fp=final_fp, budget_fp=budget,
                            elapsed_seconds=time.time() - run_started)
                _notify("ALERT", f"{args.worker_name}: EVAL FAILED {run_id}",
                        f"training reached budget ({final_fp:,} FP) but eval status={ev_status}.")

        _backup_to_s3(run)

    print("[launch_v2] done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
