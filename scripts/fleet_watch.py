"""Watch every figB training box and ALARM on the states that have actually cost us time today.

Three dead runs sat unnoticed for 1.5h, 11h and 5.5h. In each case a watcher existed and reported
nothing wrong, because it was written to detect SUCCESS and treated everything else as "not yet":
absence of a step count looks identical to a run that has not started. So this checks for the
failure states explicitly, and a box is UNHEALTHY unless it proves otherwise:

  ABORTED  -- the current run slice contains ABORT / OOM / Traceback
  STALLED  -- forward passes did not advance between polls
  IDLE     -- no training process, and the run is not verifiably complete
  DONE     -- encoder on S3 AND verified.json (the same test launch_v2_wave uses)

Exits non-zero the moment any box is unhealthy, so the caller is interrupted rather than having to
read a log it will not read.
"""
from __future__ import annotations
import json, subprocess, sys, time

KEY = "climb-gpu-key.pem"
S3 = "s3://climb-s3-bucket/experiments/climb_v2_phase2"


def sh(cmd: list[str], timeout: int = 60) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout).stdout.strip()
    except subprocess.TimeoutExpired:
        return ""


def boxes() -> list[tuple[str, str, str]]:
    out = sh(["aws", "ec2", "describe-instances",
              "--filters", "Name=instance-state-name,Values=running",
              "Name=tag:Name,Values=climb-figB-*",
              "--query", "Reservations[].Instances[].[Tags[?Key=='Name']|[0].Value,"
                         "InstanceId,PublicIpAddress]", "--output", "text"])
    rows = []
    for line in out.splitlines():
        p = line.split()
        if len(p) == 3 and p[2] != "None":
            rows.append((p[0], p[1], p[2]))
    return rows


def remote(ip: str, cmd: str) -> str:
    return sh(["ssh", "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=10",
               "-i", KEY, f"ec2-user@{ip}", cmd], timeout=45)


def run_id_of(ip: str) -> str:
    # The run the box is ACTUALLY executing, from its live process -- not from its Name tag. The
    # tag said 100M while the GPU was running a 50M, and trusting the tag is what hid that.
    out = remote(ip, "pgrep -af 'pretrain_v[2].py' | head -1")
    for tok in out.split():
        if tok.startswith("experiments/"):
            return tok.rstrip("/").split("/")[-1]
    return ""


def s3_done(run: str) -> bool:
    enc = sh(["aws", "s3", "ls", f"{S3}/{run}/encoder/model.safetensors"])
    ver = sh(["aws", "s3", "ls", f"{S3}/{run}/verified.json"])
    return bool(enc) and bool(ver)


def poll(prev: dict) -> tuple[list[str], bool]:
    lines, bad = [], False
    for name, iid, ip in boxes():
        run = run_id_of(ip)
        tag_run = name.replace("climb-figB-", "").replace("-fast", "")
        log = f"/home/ec2-user/CLIMB/analysis/figB_{run or tag_run}.log"
        # Slice from the last launch marker: these logs are append-only across relaunches and the
        # previous attempt's ABORT is still sitting in them.
        cur = remote(ip, f"awk '/start on/{{buf=\"\"}} {{buf=buf $0 \"\\n\"}} END{{printf \"%s\", buf}}' {log} 2>/dev/null")
        fp = ""
        for tok in reversed(cur.split()):
            if tok.startswith("fp="):
                fp = tok; break
        procs = remote(ip, "pgrep -cf 'pretrain_v[2].py'") or "0"
        state = "RUNNING"
        if any(k in cur for k in ("ABORT", "OutOfMemoryError", "Traceback")):
            state, bad = "ABORTED", True
        elif procs.strip() in ("", "0"):
            state = "DONE" if (run or tag_run) and s3_done(run or tag_run) else "IDLE"
            if state == "IDLE":
                bad = True
        elif fp and prev.get(ip) == fp:
            state, bad = "STALLED", True
        prev[ip] = fp
        lines.append(f"{state:8s} {name:42s} run={run or '(none)':24s} {fp or 'fp=?'}")
    return lines, bad


def main() -> int:
    prev: dict[str, str] = {}
    for i in range(int(sys.argv[1]) if len(sys.argv) > 1 else 480):
        lines, bad = poll(prev)
        stamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        print(f"--- {stamp}", flush=True)
        for l in lines:
            print(l, flush=True)
        if bad and i > 0:      # first pass seeds the stall baseline
            print("UNHEALTHY -- see above", flush=True)
            return 1
        time.sleep(300)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
