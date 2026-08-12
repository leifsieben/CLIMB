"""Experiment B — token-coverage confound guard. A wiki-pretrained encoder only trains the embedding
rows for tokens that appear in Wikipedia; SMILES-specific tokens may stay near-random, which would make
a NULL transfer result ambiguous ("no transfer" vs "those embeddings were never trained"). This reports
what fraction of the EVAL molecules' token mass the Wikipedia pretraining corpus actually covered.

Coverage = sum of eval-token frequency over tokens with wiki_count >= threshold, / total eval-token mass.
Reported at several thresholds. Also the per-task breakdown. A high coverage means a null is clean; a low
coverage means the embedding-unfreeze control (noted follow-up) would be needed to interpret a null.

Reads the wiki token unigram from the corpus builder's diagnostics (local or S3) and tokenizes the eval
molecules with the SAME frozen SMILES tokenizer. Run locally.

Usage:  python scripts/wiki_coverage_report.py --tokenizer figure_data/_tokenizer \
            --out analysis/rigor/wiki_coverage.json
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

WIKI_DIAG_S3 = "s3://climb-s3-bucket/tokenized_sources/pubchem_filtered_wiki_pkl/_diagnostics.json"
TASKS = ["ESOL", "QM7", "BBBP", "BACE", "Tox21", "HIV"]
THRESHOLDS = [1, 10, 100, 1000]


def _wiki_unigram(local_diag: str | None) -> np.ndarray:
    if local_diag and Path(local_diag).exists():
        d = json.loads(Path(local_diag).read_text())
    else:
        raw = subprocess.run(["aws", "s3", "cp", WIKI_DIAG_S3, "-"], capture_output=True)
        d = json.loads(raw.stdout.decode())
    return np.asarray(d["wiki_token_unigram"], dtype=np.int64)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--local_diag", default=None, help="wiki_pkl/_diagnostics.json if present locally")
    ap.add_argument("--out", default="analysis/rigor/wiki_coverage.json")
    a = ap.parse_args()

    from transformers import PreTrainedTokenizerFast
    from eval_v2 import _load_moleculenet
    tok = PreTrainedTokenizerFast.from_pretrained(a.tokenizer)
    wiki = _wiki_unigram(a.local_diag)

    per_task = {}
    total = Counter()
    for t in TASKS:
        try:
            tr_s, _, va_s, _, te_s, _ = _load_moleculenet(t)
        except Exception as e:
            print(f"  skip {t}: {e}"); continue
        c = Counter()
        for s in list(tr_s) + list(va_s) + list(te_s):
            for tid in tok(str(s), add_special_tokens=False)["input_ids"]:
                c[tid] += 1
        mass = sum(c.values())
        cov = {str(th): round(sum(f for tid, f in c.items() if wiki[tid] >= th) / max(mass, 1), 4)
               for th in THRESHOLDS}
        per_task[t] = {"token_mass": mass, "distinct_tokens": len(c), "coverage_at_threshold": cov}
        total.update(c)
        print(f"  {t:6} mass={mass:>9,}  coverage(≥1/≥10/≥100/≥1000)="
              f"{cov['1']:.3f}/{cov['10']:.3f}/{cov['100']:.3f}/{cov['1000']:.3f}")

    mass = sum(total.values())
    overall = {str(th): round(sum(f for tid, f in total.items() if wiki[tid] >= th) / max(mass, 1), 4)
               for th in THRESHOLDS}
    out = {
        "what": "fraction of eval-molecule token mass covered by the Wikipedia pretraining corpus",
        "thresholds": THRESHOLDS,
        "overall_coverage_at_threshold": overall,
        "per_task": per_task,
        "note": ("High coverage => a null transfer result is clean. Low coverage => SMILES-specific "
                 "tokens were undertrained on Wikipedia, so the embedding-unfreeze control (follow-up) "
                 "is needed to interpret a null."),
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2))
    print(f"\noverall coverage ≥1/≥10/≥100/≥1000 = "
          f"{overall['1']}/{overall['10']}/{overall['100']}/{overall['1000']}  -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
