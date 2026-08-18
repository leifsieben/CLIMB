#!/usr/bin/env python3
"""Build the manifest for the 124M-corpus extension of the unsupervised ladder.

Three runs:
  unsup_50M      50M forward passes  } the two requested ladder rungs, the first that are
  unsup_100M    100M forward passes  } genuinely single-epoch over UNIQUE molecules
  unsup_8M_c124   8M forward passes  - NOTATION BRIDGE CONTROL, see below

WHY THE BRIDGE RUN EXISTS (do not drop it):
The existing ladder (unsup_2M/8M/24M/48M) trained on pubchem_filtered, whose SMILES are the
raw upstream PubChem strings -- `--recanonicalize` was a silent no-op there because RDKit was
not installed on the box that built it (prepare_pubchem_124m.py returns the input unchanged
when `Chem is None`). Measured: 0.0% of those SMILES use lowercase aromatic notation.
The 124M corpus WAS recanonicalized (73.0% aromatic). The eval sets are 39.5% aromatic.
So the new runs differ from the old ladder in TWO ways at once: unique-molecule count AND
SMILES notation. unsup_8M_c124 holds the budget fixed at a rung that already exists and is
single-epoch on BOTH corpora (8M FP < 12M corpus, so 8M unique molecules either way), which
makes it differ from unsup_8M in notation ONLY. The 8M-vs-8M_c124 gap is the notation offset;
without it the 50M/100M points are uninterpretable.

The pretrain_config is CLONED from a generated unsup_48M rather than hand-written: a
hand-written config containing only {run_id, selection} is accepted by the manifest loader and
then dies inside pretrain_v2 on cfg["tokenizer_path"].
"""
import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

TEMPLATE = "unsup_48M"
CORPUS = "s3://climb-s3-bucket/tokenized_sources/pubchem_124m_full_tokenized_pkl/"
RAW = "s3://climb-s3-bucket/tokenized_sources/pubchem_124m_full/"
RESULTS_ROOT = "experiments/climb_v2_phase2"
S3_ROOT = "s3://climb-s3-bucket/experiments/climb_v2_phase2"

# (run_id, forward passes, worker index)  -- worker1 gets the long pole alone.
RUNS = [
    ("unsup_8M_c124", 8_000_000, 0),
    ("unsup_50M", 50_000_000, 0),
    ("unsup_100M", 100_000_000, 1),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="experiments/climb_v2_phase2/manifests/unsup124")
    ap.add_argument("--spec", default="configs/v2_phase2.yaml")
    args = ap.parse_args()

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
        tmp = tf.name
    rc = subprocess.run(
        [sys.executable, "experiment_v2.py", "--spec", args.spec, "--output", tmp],
        capture_output=True, text=True,
    )
    if rc.returncode != 0:
        print("FATAL: experiment_v2.py failed\n", rc.stdout, rc.stderr)
        return 1
    gen = json.load(open(tmp))
    by_id = {r["run_id"]: r for r in gen["runs"]}
    if TEMPLATE not in by_id:
        print(f"FATAL: template {TEMPLATE} not in generated manifest")
        return 1
    tmpl = by_id[TEMPLATE]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    workers = {}

    for run_id, fp, w in RUNS:
        r = json.loads(json.dumps(tmpl))          # deep copy of a known-good entry
        r["run_id"] = run_id
        r["output_dir"] = f"{RESULTS_ROOT}/{run_id}"
        r["backup_s3_uri"] = f"{S3_ROOT}/{run_id}"
        r["evaluation_output_dir"] = f"{RESULTS_ROOT}/{run_id}/moleculenet"
        pc = r["pretrain_config"]
        pc["run_id"] = run_id
        # THE corpus swap -- this is the only substantive difference from the old ladder.
        pc["unsupervised_data_paths"] = [CORPUS]
        pc["unsupervised_raw_smiles_paths"] = [RAW]
        for sel in (r["selection"], pc["selection"]):
            sel["total_forward_passes"] = fp
        workers.setdefault(w, []).append(r)

        # the clone must be complete or pretrain_v2 dies at runtime, not here
        need = ["tokenizer_path", "unsupervised_data_paths", "model", "training", "evaluation"]
        miss = [k for k in need if k not in pc]
        if miss:
            print(f"FATAL: cloned pretrain_config for {run_id} missing {miss}")
            return 1
        for sel in (r["selection"], pc["selection"]):
            assert sel["objectives"] == {"mlm": 1.0}, "must stay MLM-only"
            assert sel["augmentation"] == "canonical", "must use the pre-tokenized path"
            assert sel["init_encoder_path"] is None, "must train from scratch"
            assert sel["pretraining_seed"] == 0, "ladder uses pretraining_seed 0"

    for w, runs in sorted(workers.items()):
        # short-first: fast feedback, and the 8M bridge acts as a canary for the new corpus
        runs.sort(key=lambda r: r["selection"]["total_forward_passes"])
        man = {
            "name": "climb_v2_phase2",
            "results_root": RESULTS_ROOT,
            "s3_backup_root": S3_ROOT,
            "tokenizer_path": gen["tokenizer_path"],
            "runs": runs,
        }
        p = out_dir / f"worker{w}.json"
        p.write_text(json.dumps(man, indent=2))
        fp_tot = sum(r["selection"]["total_forward_passes"] for r in runs)
        print(f"wrote {p}: {[r['run_id'] for r in runs]}  "
              f"{fp_tot:,} FP = {fp_tot/749/3600:.1f} GPU-h @749 seq/s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
