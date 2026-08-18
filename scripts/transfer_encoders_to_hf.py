"""Robust S3 -> HF transfer of every paper-critical encoder checkpoint.

Replaces the publish_to_hf.py --repo encoders path for this job, which had a SILENT-FAILURE bug:
it decided whether to include a run via `if aws s3 ls <uri>model.safetensors` returning output, so
a transient S3 connection error returned empty and the run was silently SKIPPED. With 34 such
errors it would have reported success while uploading almost nothing.

This version instead:
  * enumerates every encoder in ONE recursive listing (1 S3 call, not 2 per run),
  * retries each download and VERIFIES the byte size against the S3 listing,
  * uploads per-wave so one failure cannot lose the rest,
  * FAILS LOUDLY: exits non-zero and names every encoder it could not transfer.

Run: python3 scripts/transfer_encoders_to_hf.py [--execute] [--wave climb_v2_phase2]
"""
from __future__ import annotations
import argparse, os, subprocess, sys, tempfile, shutil
from pathlib import Path

BUCKET = "s3://climb-s3-bucket"
REPO = "lsieben/climb-encoders"
WAVES = ["climb_v2_phase2", "climb_v2_h1", "climb_v2_vocab",
         "climb_v2_ablation_dedup", "climb_v2_expA", "climb_v2_expB"]


def sh(cmd, timeout=1800):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)


def enumerate_encoders():
    """One recursive listing -> {(wave, run): size}. Retries; raises if it cannot enumerate."""
    for attempt in range(4):
        r = sh(f"aws s3 ls {BUCKET}/experiments/ --recursive")
        if r.returncode == 0 and r.stdout.strip():
            out = {}
            for line in r.stdout.splitlines():
                p = line.split()
                if len(p) < 4 or not p[3].endswith("encoder/model.safetensors"):
                    continue
                parts = p[3].split("/")
                if len(parts) < 4 or parts[0] != "experiments":
                    continue
                wave, run = parts[1], parts[2]
                if wave in WAVES:
                    out[(wave, run)] = int(p[2])
            return out
        print(f"  enumeration attempt {attempt+1} failed, retrying", flush=True)
    raise RuntimeError("could not enumerate S3 encoders after 4 attempts")


def fetch(wave, run, size, dst):
    """Sync one encoder dir with retries; verify model.safetensors byte size."""
    for attempt in range(4):
        sh(f"aws s3 sync {BUCKET}/experiments/{wave}/{run}/encoder {dst} --only-show-errors")
        f = dst / "model.safetensors"
        if f.exists() and f.stat().st_size == size:
            return True
        print(f"    retry {attempt+1} {wave}/{run} (got {f.stat().st_size if f.exists() else 0}/{size})", flush=True)
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--execute", action="store_true")
    ap.add_argument("--wave", default=None)
    a = ap.parse_args()

    enc = enumerate_encoders()
    waves = [a.wave] if a.wave else WAVES
    enc = {k: v for k, v in enc.items() if k[0] in waves}
    print(f"encoders to transfer: {len(enc)}  (~{sum(enc.values())/1e9:.1f} GB)")
    for w in waves:
        print(f"  {w}: {sum(1 for k in enc if k[0]==w)}")
    if not a.execute:
        print("[dry-run] pass --execute")
        return 0

    from huggingface_hub import upload_folder
    failed = []
    for w in waves:
        runs = sorted(r for (ww, r) in enc if ww == w)
        if not runs:
            continue
        stage = Path(tempfile.mkdtemp(prefix=f"climbenc_{w}_"))
        try:
            for run in runs:
                d = stage / run
                d.mkdir(parents=True, exist_ok=True)
                if not fetch(w, run, enc[(w, run)], d):
                    failed.append(f"{w}/{run}")
                    shutil.rmtree(d, ignore_errors=True)
            got = [p for p in stage.iterdir() if (p / "model.safetensors").exists()]
            print(f"  {w}: staged {len(got)}/{len(runs)} -> uploading", flush=True)
            if got:
                for attempt in range(3):
                    try:
                        upload_folder(repo_id=REPO, repo_type="model", folder_path=str(stage),
                                      path_in_repo=w, commit_message=f"encoders: {w} ({len(got)} runs)")
                        print(f"  {w}: uploaded", flush=True)
                        break
                    except Exception as e:
                        print(f"  {w}: upload attempt {attempt+1} failed: {str(e)[:100]}", flush=True)
                else:
                    failed.append(f"{w}:UPLOAD")
        finally:
            shutil.rmtree(stage, ignore_errors=True)

    if failed:
        print(f"\nFAILED ({len(failed)}): {failed}")
        return 1
    print("\nAll encoders transferred.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
