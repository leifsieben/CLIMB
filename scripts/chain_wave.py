"""Wait for one wave to finish, then start the boxes back up and launch the next wave on them.

Needed because a worker self-stops the instant its manifest is fully verified (the completion
gate). So "launch B when A frees up" cannot be done by holding the boxes idle -- they will be
gone. This polls for A's completion, restarts the instances, redeploys, stages manifests
OUTSIDE the S3-synced tree, launches detached, and verifies the launch actually took.

Every step is verified rather than assumed, because each has failed before in this project:
  * `ssh host 'nohup ... &'` can hang while the job runs fine -- so launch is confirmed by
    polling the box afterwards (PPID 1 + the worker's own log), never by the ssh exit code.
  * us-east-1d intermittently returns InsufficientInstanceCapacity on start -- retried, with
    the batch call avoided in favour of per-instance starts, which have succeeded when the
    batched form failed.
  * A manifest placed inside experiments/ gets clobbered by the startup sync -- staged in
    ~/CLIMB/ instead.
  * A dead warm-start base kills runs at startup with no metrics at all (README 13.8) -- the
    deployed worker now aborts loudly on that, and this script surfaces the log either way.

Usage:
  python scripts/chain_wave.py --wait-wave climb_v2_lrsweep --wait-runs a,b,c \\
      --targets i-abc:manifests/w0.json:ab0,i-def:manifests/w1.json:ab1 [--max-hours 8]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

BUCKET = "s3://climb-s3-bucket/experiments"
KEY = "climb-gpu-key.pem"
SNS = "arn:aws:sns:us-east-1:075120018132:climb-experiments"
SSH = ["ssh", "-i", KEY, "-o", "ConnectTimeout=10", "-o", "StrictHostKeyChecking=no"]


def log(m: str) -> None:
    print(f"[chain {time.strftime('%H:%M:%S')}] {m}", flush=True)


def notify(subject: str, body: str) -> None:
    subprocess.run(["aws", "sns", "publish", "--topic-arn", SNS,
                    "--subject", subject[:99], "--message", body],
                   capture_output=True, check=False)


def sh(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, check=False, **kw)


def verified(wave: str, run: str) -> bool:
    r = sh(["aws", "s3", "ls", f"{BUCKET}/{wave}/{run}/verified.json"])
    return r.returncode == 0 and bool(r.stdout.strip())


def state_and_ip(iid: str):
    r = sh(["aws", "ec2", "describe-instances", "--instance-ids", iid,
            "--query", "Reservations[].Instances[].[State.Name,PublicIpAddress]",
            "--output", "text"])
    parts = r.stdout.split()
    return (parts[0], parts[1] if len(parts) > 1 else None) if parts else (None, None)


def ensure_running(iid: str, tries: int = 20) -> str | None:
    for attempt in range(tries):
        st, ip = state_and_ip(iid)
        if st == "running" and ip:
            return ip
        if st == "stopped":
            r = sh(["aws", "ec2", "start-instances", "--instance-ids", iid])
            if r.returncode != 0:
                # capacity is per-AZ and transient; keep retrying rather than giving up
                log(f"  {iid}: start failed ({r.stderr.strip()[:90]}) — retry {attempt+1}/{tries}")
        time.sleep(20)
    return None


def wait_ssh(ip: str, tries: int = 20) -> bool:
    for _ in range(tries):
        if sh(SSH + [f"ec2-user@{ip}", "echo ok"]).returncode == 0:
            return True
        time.sleep(15)
    return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wait-wave", required=True)
    ap.add_argument("--wait-runs", required=True)
    ap.add_argument("--targets", required=True,
                    help="comma-separated instance:manifest_path:worker_name")
    ap.add_argument("--poll-seconds", type=int, default=300)
    ap.add_argument("--max-hours", type=float, default=8.0)
    ap.add_argument("--branch", default="v2-redux")
    a = ap.parse_args()

    pending = [r.strip() for r in a.wait_runs.split(",") if r.strip()]
    targets = []
    for t in a.targets.split(","):
        iid, mpath, wname = t.split(":")
        targets.append((iid.strip(), Path(mpath.strip()), wname.strip()))

    deadline = time.time() + a.max_hours * 3600
    log(f"waiting for {len(pending)} runs in {a.wait_wave} to verify")
    while time.time() < deadline:
        left = [r for r in pending if not verified(a.wait_wave, r)]
        if not left:
            log("prerequisite wave COMPLETE")
            break
        log(f"  {len(left)} not yet verified: {', '.join(left[:4])}{' …' if len(left) > 4 else ''}")
        time.sleep(a.poll_seconds)
    else:
        msg = f"gave up after {a.max_hours}h; still unverified: {', '.join(left)}"
        log(msg); notify("chain_wave ABORTED", msg)
        return 1

    launched, failed = [], []
    for iid, mpath, wname in targets:
        log(f"{wname}: bringing up {iid}")
        ip = ensure_running(iid)
        if not ip or not wait_ssh(ip):
            failed.append(f"{wname} ({iid}): unreachable"); log(f"  {wname}: UNREACHABLE"); continue
        log(f"  {wname}: {ip} up; deploying {a.branch}")

        dep = sh(SSH + [f"ec2-user@{ip}",
                        f"cd ~/CLIMB && git fetch -q origin && git reset -q --hard origin/{a.branch} "
                        f"&& chmod +x scripts/*.sh && git log --oneline -1"])
        if dep.returncode != 0:
            failed.append(f"{wname}: deploy failed"); log(f"  {wname}: DEPLOY FAILED"); continue
        log(f"  {wname}: HEAD {dep.stdout.strip()}")

        remote_manifest = f"/home/ec2-user/CLIMB/{mpath.name}"   # OUTSIDE experiments/
        if sh(["scp", "-i", KEY, "-o", "StrictHostKeyChecking=no",
               str(mpath), f"ec2-user@{ip}:{remote_manifest}"]).returncode != 0:
            failed.append(f"{wname}: manifest copy failed"); log(f"  {wname}: SCP FAILED"); continue

        sh(SSH + [f"ec2-user@{ip}",
                  f"cd ~/CLIMB && setsid nohup bash scripts/phase2_worker.sh {remote_manifest} "
                  f"{wname} > phase2_{wname}.log 2>&1 < /dev/null & echo started"], timeout=60)
        time.sleep(45)
        chk = sh(SSH + [f"ec2-user@{ip}",
                        "pgrep -f phase2_worker.sh >/dev/null && echo ALIVE || echo DEAD; "
                        f"tail -5 ~/CLIMB/phase2_{wname}.log"])
        alive = "ALIVE" in chk.stdout
        log(f"  {wname}: {'LAUNCHED' if alive else 'NOT RUNNING'}")
        for line in chk.stdout.strip().splitlines()[-4:]:
            log(f"      {line}")
        (launched if alive else failed).append(f"{wname} ({ip})")

    summary = f"launched={len(launched)} failed={len(failed)}\n" + \
              "\n".join(["  OK " + x for x in launched] + ["  FAIL " + x for x in failed])
    log(summary)
    notify(f"chain_wave: {a.wait_wave} done -> next wave launched ({len(launched)}/{len(targets)})",
           summary)
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
