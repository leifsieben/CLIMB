"""Standalone watchdog for a v2 training run.

Args:
    --pid <int>           PID of the training process to watch
    --metrics <path>      Path to the run's metrics.jsonl
    --heartbeat <path>    Path to the run's heartbeat.json (optional)
    --stall_seconds <N>   Default 1800 (30 min). SIGTERM the PID if metrics.jsonl
                          mtime hasn't advanced in N seconds.
    --max_seconds <N>     Hard cap; SIGKILL the PID if still alive after N seconds total.

Behaviour:
- Polls every 30 s.
- Pauses the SIGTERM check while a `*.tmp` file exists in the run dir (legitimate
  checkpoint save in progress).
- Exits 0 when the watched PID exits. Exits 1 if it had to kill.

The watchdog is a separate process so it survives even if the launcher hangs.
"""

from __future__ import annotations

import argparse
import os
import signal
import sys
import time
from pathlib import Path


POLL_SECONDS = 30


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _has_tmp_file(run_dir: Path) -> bool:
    if not run_dir.exists():
        return False
    for p in run_dir.rglob("*.tmp"):
        try:
            if p.is_file():
                return True
        except Exception:
            continue
    return False


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pid", type=int, required=True)
    p.add_argument("--metrics", required=True)
    p.add_argument("--heartbeat", default=None)
    p.add_argument("--stall_seconds", type=int, default=1800)
    p.add_argument("--max_seconds", type=int, default=24 * 3600)
    p.add_argument("--poll_seconds", type=int, default=POLL_SECONDS)
    p.add_argument("--sigterm_grace_seconds", type=int, default=60)
    args = p.parse_args()

    metrics_path = Path(args.metrics)
    run_dir = metrics_path.parent
    started = time.time()

    print(f"[watchdog] watching pid={args.pid} metrics={metrics_path} "
          f"stall={args.stall_seconds}s max={args.max_seconds}s")

    while _pid_alive(args.pid):
        time.sleep(args.poll_seconds)
        now = time.time()
        elapsed = now - started

        # Hard cap
        if elapsed > args.max_seconds:
            print(f"[watchdog] hard cap exceeded ({elapsed:.0f}s); SIGKILL pid={args.pid}")
            try:
                os.kill(args.pid, signal.SIGKILL)
            except OSError:
                pass
            return 1

        # Stall check (skip if checkpoint write in progress)
        if _has_tmp_file(run_dir):
            print(f"[watchdog] tmp file present; pausing stall check")
            continue

        if metrics_path.exists():
            metrics_age = now - metrics_path.stat().st_mtime
            if metrics_age > args.stall_seconds:
                print(f"[watchdog] metrics stale ({metrics_age:.0f}s > {args.stall_seconds}s); "
                      f"SIGTERM pid={args.pid}")
                try:
                    os.kill(args.pid, signal.SIGTERM)
                    deadline = time.time() + args.sigterm_grace_seconds
                    while _pid_alive(args.pid) and time.time() < deadline:
                        time.sleep(min(2, max(0.1, args.sigterm_grace_seconds / 5)))
                    if _pid_alive(args.pid):
                        os.kill(args.pid, signal.SIGKILL)
                except OSError:
                    pass
                return 1
        else:
            # metrics.jsonl never appeared — something failed at startup
            startup_grace = 600  # 10 min for first metric write
            if elapsed > startup_grace:
                print(f"[watchdog] no metrics.jsonl after {elapsed:.0f}s; SIGTERM pid={args.pid}")
                try:
                    os.kill(args.pid, signal.SIGTERM)
                except OSError:
                    pass
                return 1

    print(f"[watchdog] pid={args.pid} exited cleanly")
    return 0


if __name__ == "__main__":
    sys.exit(main())
