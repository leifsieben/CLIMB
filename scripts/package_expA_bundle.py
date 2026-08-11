"""Package Experiment A (synthetic-statistics ladder) into a single self-describing zip to share
with a collaborator for further analysis.

Contents (analysis-ready; NOT the multi-hundred-MB weights — those are indexed with S3/HF URIs):
  README.md                     what the experiment is, the arms, the result, how to reproduce
  methodology.md                the arm table (what each corruption preserves/destroys)
  ladder/expA_ladder_summary.csv, expA_ladder_per_run.csv     the headline result
  runs/<arm>/moleculenet_cv/{moleculenet_summary.csv,test_predictions.csv}   per-molecule CV outputs
  runs/<arm>/{config.yaml,metadata.json,metrics.jsonl}        exact config + training curve
  corpora/{unigram,bigram}_diagnostics.json                  KL / entropy corruption certificates
  checkpoints.csv               S3 + HF URIs and sizes for every encoder (fetch on demand)

Pulls per-run eval outputs from S3 (that is where the wave backs them up), reads the ladder CSVs
locally. Run AFTER the bigram wave + finalize (encoders on S3). Idempotent; writes to --out.

Usage:
    python scripts/package_expA_bundle.py --out dist/experiment_a_bundle.zip [--hf-org lsieben]
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import subprocess
import zipfile
from pathlib import Path

BUCKET = "s3://climb-s3-bucket/experiments/climb_v2_expA"
CORPUS = "s3://climb-s3-bucket/tokenized_sources"

# arm label -> (run_id, s3_subpath). Frozen comparators come from the native _baselines re-eval.
ARMS = [
    ("real__unsup_8M", "_baselines/unsup_8M"),
    ("real__unsup_8M_s1", "_baselines/unsup_8M_s1"),
    ("real__unsup_8M_s2", "_baselines/unsup_8M_s2"),
    ("shuffle__corrupt_mlm_8M", "_baselines/corrupt_mlm_8M"),
    ("shuffle__corrupt_mlm_8M_s1", "corrupt_mlm_8M_s1"),
    ("shuffle__corrupt_mlm_8M_s2", "corrupt_mlm_8M_s2"),
    ("unigram__unigram_8M", "unigram_8M"),
    ("unigram__unigram_8M_s1", "unigram_8M_s1"),
    ("unigram__unigram_8M_s2", "unigram_8M_s2"),
    ("bigram__bigram_8M", "bigram_8M"),
    ("bigram__bigram_8M_s1", "bigram_8M_s1"),
    ("bigram__bigram_8M_s2", "bigram_8M_s2"),
    ("no_pretrain__random_baseline_00", "_baselines/random_baseline_00"),
    ("no_pretrain__random_baseline_01", "_baselines/random_baseline_01"),
    ("no_pretrain__random_baseline_02", "_baselines/random_baseline_02"),
]
# per-run files to include (small; predictions are the useful bit for per-molecule analysis)
RUN_FILES = ["config.yaml", "metadata.json", "metrics.jsonl",
             "moleculenet_cv/moleculenet_summary.csv", "moleculenet_cv/test_predictions.csv"]
# encoders that are NEW checkpoints (baselines re-use phase2 encoders already published)
ENC_RUNS = ["unigram_8M", "unigram_8M_s1", "unigram_8M_s2", "corrupt_mlm_8M_s1", "corrupt_mlm_8M_s2",
            "bigram_8M", "bigram_8M_s1", "bigram_8M_s2"]

METHODOLOGY = """# Experiment A — synthetic-statistics ladder: what each arm preserves

| Arm | What survives | What's destroyed |
|---|---|---|
| real (unsup_only) | everything | — |
| shuffle_tokens | each molecule's exact token multiset (which tokens, how many) | token order → SMILES grammar, ring/branch matching, adjacency |
| bigram_resample | corpus token frequencies + local pairwise adjacency (a 1st-order Markov chain fit on the corpus) | the per-molecule multiset (composition), all long-range structure |
| unigram_resample | only the corpus-wide token frequencies (the marginal) | per-molecule composition, order, and adjacency |
| no_pretrain | nothing (random encoder) | — |

bigram and shuffle_tokens probe complementary axes (adjacency vs composition), not a strict nesting.
All arms: 8M forward passes, frozen-probe 5-fold scaffold CV, 3 pretraining seeds, otherwise
bit-identical to the real `unsup_8M` run. Corruption certified by token-frequency KL(real‖synth)≈0
(see corpora/*_diagnostics.json) and, for unigram, the MLM loss plateauing at the unigram entropy.
"""


def sh(cmd: str) -> str:
    return subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout


def s3_bytes(uri: str) -> bytes | None:
    r = subprocess.run(["aws", "s3", "cp", uri, "-"], capture_output=True)
    return r.stdout if r.returncode == 0 else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="dist/experiment_a_bundle.zip")
    ap.add_argument("--hf-org", default="lsieben")
    args = ap.parse_args()
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)

    n_run, n_missing = 0, []
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        # ladder CSVs (local)
        for rel in ("analysis/rigor/expA_ladder_summary.csv", "analysis/rigor/expA_ladder_per_run.csv"):
            p = Path(rel)
            if p.exists():
                z.write(p, f"ladder/{p.name}")
            else:
                n_missing.append(rel)
        # methodology + note
        z.writestr("methodology.md", METHODOLOGY)
        note = Path("notes/note-c-dedup-reanalysis-2026-08-10.md")  # not expA, skip if absent
        # per-run eval outputs from S3
        for label, sub in ARMS:
            for rel in RUN_FILES:
                data = s3_bytes(f"{BUCKET}/{sub}/{rel}")
                if data is not None:
                    z.writestr(f"runs/{label}/{rel}", data)
                    n_run += 1
                else:
                    n_missing.append(f"{sub}/{rel}")
        # corpus corruption diagnostics
        for mode in ("unigram", "bigram"):
            d = s3_bytes(f"{CORPUS}/pubchem_filtered_{mode}_pkl/_diagnostics.json")
            if d is not None:
                z.writestr(f"corpora/{mode}_diagnostics.json", d)
        # checkpoint index (URIs + sizes; NOT the weights)
        ck = io.StringIO(); w = csv.writer(ck)
        w.writerow(["run_id", "size", "s3_uri", "hf_uri"])
        for rid in ENC_RUNS:
            ls = sh(f"aws s3 ls {BUCKET}/{rid}/encoder/model.safetensors 2>/dev/null").split()
            size = ls[2] if len(ls) >= 3 else "?"
            w.writerow([rid, size, f"{BUCKET}/{rid}/encoder/",
                        f"https://huggingface.co/{args.hf_org}/climb-encoders/tree/main/climb_v2_expA/{rid}"])
        z.writestr("checkpoints.csv", ck.getvalue())
        # README
        z.writestr("README.md", f"""# Experiment A — synthetic-statistics ladder (CLIMB)

Which statistic of the SMILES corpus does a masked-LM actually need? We pretrain on corpora that
preserve progressively less structure and measure downstream (frozen-probe, 5-fold scaffold CV).

**Result:** shuffle_tokens ≈ real and unigram_resample ≈ no_pretrain across all 7 MoleculeNet tasks —
token order barely matters and the token marginal buys ~nothing over random init, so the benefit
lives in the per-molecule token **composition**. (bigram_resample tests whether local adjacency can
substitute; see ladder/.)

## Layout
- `ladder/` — the headline: per-arm mean±std (summary) and per-run (per_run) CV metrics. NATIVE units,
  same eval version across every arm.
- `runs/<arm>/` — exact config, training curve (metrics.jsonl), and per-molecule CV predictions
  (moleculenet_cv/test_predictions.csv) for your own analysis.
- `corpora/` — corruption certificates (token-freq KL, entropy).
- `checkpoints.csv` — S3 + HF URIs for each encoder ({len(ENC_RUNS)} new checkpoints). Weights are not
  in this zip; fetch from HF (`{args.hf_org}/climb-encoders`, private — request access) or S3.
- `methodology.md` — the arm table.

Arms: real=unsup_8M · shuffle=corrupt_mlm_8M(+seeds) · unigram=unigram_8M · bigram=bigram_8M ·
no_pretrain=random_baseline. 3 seeds each.
""")

    print(f"wrote {out}  ({n_run} per-run files from S3, {out.stat().st_size/1e6:.1f} MB)")
    if n_missing:
        print(f"  NOTE {len(n_missing)} expected files missing (run before bigram/finalize done?):")
        for m in n_missing[:12]:
            print(f"    - {m}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
