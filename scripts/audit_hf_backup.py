"""Three-way audit: what exists locally, on S3, and on HuggingFace.

The standing rule is results + checkpoints + training data live in all three. This checks it
rather than assuming it, because the failure is silent: a run that never reached HF looks
identical on disk to one that did.

Compares at RUN level (<wave>/<run>), not file level -- S3 and HF hold different file subsets by
design (HF omits raw encoder optimizer state, S3 keeps working files).
"""
from __future__ import annotations
import subprocess, sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
S3B = "s3://climb-s3-bucket/experiments"
# A HAND-WRITTEN WAVE LIST IS AN ABSENCE CLAIM, AND IT WENT STALE SILENTLY. This listed 8 trees
# while S3 held ~30, so the audit printed "complete -- nothing missing" while saying NOTHING about
# chemeleon_suite -- the tree holding MoleculeACE and Polaris, i.e. 58 of the 65 ranking datasets.
# All 595 of its files were absent from HF and no check anywhere reported it. Same for six_panel_e2e,
# climb_v2_labeleff*, climb_v2_headline and climb_v2.
#
# Trees are now DISCOVERED from S3 and only explicitly-excluded ones are skipped, so a new wave is
# audited the day it appears rather than the day someone remembers to add it here.
SKIP_TREES = {
    "_logs", "_scripts", "_quarantine", "_pending_verdict", "_bench",   # infrastructure
    "climb_v1", "climb_v2_ablation",                                    # superseded, kept for history
    "robust_matrix", "robust_matrix_v2",                                # 350k files, DEEP_ARCHIVE
}


def waves() -> list:
    """Every result tree on S3 except the explicitly excluded ones."""
    out = subprocess.run(["aws", "s3", "ls", S3B + "/"], capture_output=True, text=True).stdout
    found = [l.split()[-1].rstrip("/") for l in out.splitlines()
             if l.strip().startswith("PRE")]
    return sorted(w for w in found if w not in SKIP_TREES)


WAVES = waves()


def local_runs() -> dict:
    out = defaultdict(set)
    for w in WAVES:
        d = ROOT / "figure_data" / w
        if d.is_dir():
            out[w] = {p.name for p in d.iterdir() if p.is_dir()}
    return out


def local_encoders() -> dict:
    out = defaultdict(set)
    for w in WAVES:
        d = ROOT / "figure_data" / w
        if not d.is_dir():
            continue
        for p in d.iterdir():
            e = p / "encoder"
            if (e / "model.safetensors").exists() or (e / "pytorch_model.bin").exists():
                out[w].add(p.name)
    return out


def s3_encoders() -> dict:
    """Encoders present on S3, by wave.

    THE AUDIT USED TO COMPARE local_encoders() AGAINST HF, AND THAT HAS A HOLE: an encoder that
    lives on S3 but was never pulled down is in NEITHER side of that comparison, so it reads as
    "complete" while sitting in exactly one place. corrupt_mtr_8M_s1 and _s2 sat like that until
    2026-08-20 -- safe on S3 the whole time, never lost, but one copy short of the three-copy rule
    and invisible to the check that exists to enforce it.

    S3 is the authority for what SHOULD exist, because it is the store nothing is deleted from, so
    the comparison has to be S3 -> HF rather than local -> HF. One recursive listing, not a call
    per run.
    """
    out = defaultdict(set)
    r = subprocess.run(["aws", "s3", "ls", f"{S3B}/", "--recursive"],
                       capture_output=True, text=True)
    for line in r.stdout.splitlines():
        parts = line.split()
        if len(parts) < 4:
            continue
        bits = parts[3].split("/")
        if bits and bits[0] == "experiments":
            bits = bits[1:]
        if (len(bits) >= 4 and bits[0] in WAVES and bits[2] == "encoder"
                and bits[3] in ("model.safetensors", "pytorch_model.bin")):
            out[bits[0]].add(bits[1])
    return out


def s3_runs() -> dict:
    out = defaultdict(set)
    r = subprocess.run(["aws", "s3", "ls", f"{S3B}/", "--recursive"],
                       capture_output=True, text=True)
    for line in r.stdout.splitlines():
        parts = line.split()
        if len(parts) < 4:
            continue
        # `aws s3 ls --recursive` prints keys relative to the BUCKET, so they still carry the
        # "experiments/" prefix -- splitting without stripping it puts every run under a wave
        # named "experiments" and reports the whole of S3 as missing.
        bits = parts[3].split("/")
        if bits and bits[0] == "experiments":
            bits = bits[1:]
        if len(bits) >= 3 and bits[0] in WAVES:
            out[bits[0]].add(bits[1])
    return out


def hf_runs():
    from huggingface_hub import HfApi
    api = HfApi()
    enc, res = defaultdict(set), defaultdict(set)
    for f in api.list_repo_files("lsieben/climb-encoders", repo_type="model"):
        b = f.split("/")
        if len(b) >= 3 and b[0] in WAVES:
            enc[b[0]].add(b[1])
    for f in api.list_repo_files("lsieben/climb-results", repo_type="dataset"):
        b = f.split("/")
        if len(b) >= 3 and b[0] in WAVES:
            res[b[0]].add(b[1])
    return enc, res


def report(title, have: dict, want: dict):
    print(f"\n===== {title} =====")
    total_missing = 0
    for w in WAVES:
        miss = sorted(want.get(w, set()) - have.get(w, set()))
        if miss:
            total_missing += len(miss)
            print(f"  {w}: {len(miss)} missing of {len(want.get(w, set()))}")
            for m in miss[:12]:
                print(f"      {m}")
            if len(miss) > 12:
                print(f"      ... {len(miss) - 12} more")
    if not total_missing:
        print("  complete -- nothing missing")
    return total_missing


def main() -> int:
    loc, enc_loc = local_runs(), local_encoders()
    s3 = s3_runs()
    hf_enc, hf_res = hf_runs()
    for w in WAVES:
        print(f"{w:28} local={len(loc.get(w,set())):4}  s3={len(s3.get(w,set())):4}  "
              f"hf_results={len(hf_res.get(w,set())):4}  "
              f"local_enc={len(enc_loc.get(w,set())):4}  hf_enc={len(hf_enc.get(w,set())):4}")
    n = 0
    n += report("RESULTS on S3 but NOT on HF", hf_res, s3)
    n += report("RESULTS local but NOT on S3", s3, loc)
    # S3 -> HF is the load-bearing check; local -> HF is kept because a local copy is the one you
    # can actually work from, but it can never be the completeness test.
    n += report("ENCODERS on S3 but NOT on HF  (three-copy rule)", hf_enc, s3_encoders())
    n += report("ENCODERS on S3 but NOT local  (informational)", enc_loc, s3_encoders())
    print(f"\nTOTAL gaps: {n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
