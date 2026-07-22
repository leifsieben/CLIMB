"""Peer-review reproducibility audit: every model, how it was trained, how it was evaluated.

Answers three questions in one pass, from S3 and the local tree:

  1. What models exist?      -- pretraining type x budget x seed, per wave
  2. What do we hold?        -- encoder checkpoint, training metrics, completion proof
  3. How was each evaluated? -- which eval schemes, which tasks, train/test metrics present

and then lists what is MISSING for full reproducibility, which is the part that matters: a run
whose encoder is gone cannot be re-evaluated by a reviewer, and a figure built on it cannot be
regenerated. The `climb_v2` round-1 scaling sweep is exactly that case -- its evals survive but
its encoders were never backed up, so H1 could not be re-scored on HIV without retraining.

Usage:
    aws s3 ls s3://climb-s3-bucket/experiments/ --recursive > listing.txt     # or --refresh
    python scripts/reproducibility_audit.py --listing listing.txt --out audit/
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
from collections import defaultdict
from pathlib import Path

BUCKET = "s3://climb-s3-bucket"
LOCAL = Path("figure_data")

# wave -> what it is for in the paper
WAVES = {
    "climb_v2_phase2":         "main wave: unsup_only / sup_only / unsup->sup ladders + anchors (A1, A2, B2, E1)",
    "climb_v2_ablation_dedup": "SFT-family ablation, eval-leakage deduped (C1J1)",
    "climb_v2_ablation":       "SFT-family ablation, PRE-dedup (kept only for the leakage before/after)",
    "climb_v2_labeleff_v2":    "label-efficiency sweep, frozen probe (B1p1)",
    "climb_v2_labeleff":       "label-efficiency, superseded by _v2",
    "climb_v2_lrsweep":        "SFT learning-rate sweep",
    "climb_v2_headline":       "round-1 headline",
    "climb_v2":                "round-1: canonical-vs-enumerated scaling sweep (H1) + round-1 baselines",
}

# The five evaluation surfaces a reviewer would expect to reproduce.
EVAL_ARTIFACTS = {
    "moleculenet/suite_summary.json":       "single-split summary",
    "moleculenet/moleculenet_summary.csv":  "single-split per-seed metrics",
    "moleculenet_cv/suite_summary.json":    "5-fold CV summary",
    "moleculenet_cv/moleculenet_summary.csv": "5-fold CV per-seed metrics",
    "moleculenet_cv/test_predictions.csv":  "per-molecule CV predictions (I1)",
}


def parse_run(name: str):
    """-> (pretraining type, budget label, seed) or None for non-model dirs."""
    m = re.match(r"(.+)_s(\d+)$", name)
    seed = 0
    if m:
        base, seed = m.group(1), int(m.group(2))
    else:
        base = name
    if base.startswith("random_baseline"):
        return ("no_pretrain (random init, frozen)", "-", int(base[-2:]) if base[-2:].isdigit() else 0)
    if base == "ecfp4_anchor":
        return ("classical: Morgan+XGBoost", "-", seed)
    if base == "fp_desc_anchor":
        return ("classical: Morgan+desc+XGBoost", "-", seed)
    m = re.match(r"corrupt_(mlm|mtr)_(\d+M)$", base)
    if m:
        return (f"corrupted control ({m[1]}: content destroyed)", m[2], seed)
    m = re.match(r"unsup_(\d+M)$", base)
    if m:
        return ("unsup_only (MLM)", m[1], seed)
    m = re.match(r"skip_(.+)_(\d+M)$", base)
    if m:
        return (f"sup_only: {m[1]}", m[2], seed)
    m = re.match(r"u2s_(.+)_from(\d+M)$", base)
    if m:
        return (f"unsup->sup: {m[1]}", f"{m[2]}+2M", seed)
    m = re.match(r"seq_(.+)$", base)
    if m:
        return (f"unsup->sup (ablation): {m[1]}", "2M+2M", seed)
    m = re.match(r"scaling_(canonical|enumerated)_(frac\S+)$", base)
    if m:
        return (f"unsup_only, {m[1]} SMILES", f"2M @ {m[2]}", seed)
    m = re.match(r"(random|sup|unsup|unsup2sup)_n(\d+)$", base)
    if m:
        return (f"label-efficiency probe: {m[1]}", f"n={m[2]}", seed)
    if base in ("unsup_only_seed0", "sup_only_seed0", "mixed_seed0"):
        return (f"round-1 {base.replace('_seed0','')}", "?", 0)
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--listing", required=True, help="output of `aws s3 ls <bucket>/ --recursive`")
    ap.add_argument("--out", default="audit")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    # ---- fold the S3 listing into per-run facts ----
    s3 = defaultdict(dict)                    # (wave, run) -> {relpath: bytes}
    for line in Path(a.listing).read_text().splitlines():
        parts = line.split()
        if len(parts) < 4 or not parts[0][:2].isdigit():
            continue
        size, key = int(parts[2]), parts[3]
        seg = key.split("/")
        if len(seg) < 3 or seg[0] != "experiments":
            continue
        wave, run, rel = seg[1], seg[2], "/".join(seg[3:])
        s3[(wave, run)][rel] = size

    rows = []
    for (wave, run), files in sorted(s3.items()):
        meta = parse_run(run)
        if meta is None:
            continue
        ptype, budget, seed = meta
        enc = files.get("encoder/model.safetensors", 0)
        loc = LOCAL / wave / run
        rows.append(dict(
            wave=wave, run=run, pretraining=ptype, budget=budget, seed=seed,
            encoder_s3_gb=round(enc / 1e9, 3) if enc else 0.0,
            encoder_local=int((loc / "encoder" / "model.safetensors").exists()),
            metrics_jsonl=int("metrics.jsonl" in files),
            config=int("config.yaml" in files),
            verified=int("verified.json" in files),
            **{f"eval::{lbl}": int(any(k.startswith(p) for k in files))
               for p, lbl in EVAL_ARTIFACTS.items()},
            eval_local_single=int((loc / "moleculenet" / "suite_summary.json").exists()),
            eval_local_cv=int((loc / "moleculenet_cv" / "suite_summary.json").exists()),
        ))

    p = out / "model_inventory.csv"
    with open(p, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

    # ---- gaps that block reproducibility ----
    gaps = defaultdict(list)
    for r in rows:
        if r["wave"] not in WAVES:
            continue
        trained = r["pretraining"] not in (
            "classical: Morgan+XGBoost", "classical: Morgan+desc+XGBoost")
        if trained and not r["encoder_s3_gb"]:
            gaps["NO ENCODER IN S3 - cannot be re-evaluated or released"].append(f'{r["wave"]}/{r["run"]}')
        if trained and not r["metrics_jsonl"]:
            gaps["no training curve (metrics.jsonl)"].append(f'{r["wave"]}/{r["run"]}')
        if trained and not r["verified"]:
            gaps["no completion proof (verified.json)"].append(f'{r["wave"]}/{r["run"]}')
        if not r["eval::single-split summary"] and not r["eval::5-fold CV summary"]:
            gaps["never evaluated"].append(f'{r["wave"]}/{r["run"]}')

    (out / "gaps.json").write_text(json.dumps({k: sorted(v) for k, v in gaps.items()}, indent=2))

    # ---- console report ----
    print(f"{len(rows)} runs across {len({r['wave'] for r in rows})} waves -> {p}\n")
    by_wave = defaultdict(list)
    for r in rows:
        by_wave[r["wave"]].append(r)
    print(f"{'wave':<26} {'runs':>5} {'w/enc':>6} {'w/ver':>6} {'single':>7} {'cv':>4}  purpose")
    for wv in sorted(by_wave, key=lambda w: (w not in WAVES, w)):
        rs = by_wave[wv]
        print(f"{wv:<26} {len(rs):>5} {sum(1 for r in rs if r['encoder_s3_gb']):>6} "
              f"{sum(r['verified'] for r in rs):>6} "
              f"{sum(r['eval::single-split summary'] for r in rs):>7} "
              f"{sum(r['eval::5-fold CV summary'] for r in rs):>4}  {WAVES.get(wv,'(not used in the paper)')[:52]}")

    tot = sum(r["encoder_s3_gb"] for r in rows if r["wave"] in WAVES)
    print(f"\npaper-critical encoders in S3: {tot:.1f} GB "
          f"({sum(1 for r in rows if r['wave'] in WAVES and r['encoder_s3_gb'])} checkpoints)")

    if gaps:
        print("\nGAPS:")
        for k, v in sorted(gaps.items(), key=lambda kv: -len(kv[1])):
            print(f"  {k}: {len(v)}")
            for x in v[:8]:
                print(f"     - {x}")
            if len(v) > 8:
                print(f"     ... and {len(v)-8} more")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
