"""Package Experiment B (Wikipedia-transfer) into a self-describing zip for a collaborator.
Mirrors scripts/package_expA_bundle.py. Analysis-ready; weights indexed by URI, not inlined.

Usage:  python scripts/package_expB_bundle.py --out dist/experiment_b_bundle.zip [--hf-org lsieben]
"""
from __future__ import annotations

import argparse
import csv
import io
import subprocess
import zipfile
from pathlib import Path

BUCKET = "s3://climb-s3-bucket/experiments"

# arm label -> s3 subpath (wiki from climb_v2_expB; comparators reuse expA native _baselines)
ARMS = [
    ("wiki_real__wiki_real_8M", "climb_v2_expB/wiki_real_8M"),
    ("wiki_real__wiki_real_8M_s1", "climb_v2_expB/wiki_real_8M_s1"),
    ("wiki_real__wiki_real_8M_s2", "climb_v2_expB/wiki_real_8M_s2"),
    ("real__unsup_8M", "climb_v2_expA/_baselines/unsup_8M"),
    ("real__unsup_8M_s1", "climb_v2_expA/_baselines/unsup_8M_s1"),
    ("real__unsup_8M_s2", "climb_v2_expA/_baselines/unsup_8M_s2"),
    ("no_pretrain__random_baseline_00", "climb_v2_expA/_baselines/random_baseline_00"),
    ("no_pretrain__random_baseline_01", "climb_v2_expA/_baselines/random_baseline_01"),
    ("no_pretrain__random_baseline_02", "climb_v2_expA/_baselines/random_baseline_02"),
]
RUN_FILES = ["config.yaml", "metadata.json", "metrics.jsonl",
             "moleculenet_cv/moleculenet_summary.csv", "moleculenet_cv/test_predictions.csv"]
ENC_RUNS = ["wiki_real_8M", "wiki_real_8M_s1", "wiki_real_8M_s2"]
LOCAL = ["analysis/rigor/expB_wiki_summary.csv", "analysis/rigor/expB_wiki_per_run.csv",
         "analysis/rigor/wiki_coverage.json", "analysis/rigor/wiki_vs_smiles_stats.json"]

METHOD = """# Experiment B — does non-chemical pretraining transfer?

Pretrain a ModernBERT on **English Wikipedia tokenized with the frozen SMILES byte-level BPE**,
chunked to MATCH the SMILES token-length distribution (sampled from the real corpus, so length is a
controlled variable, not a confound). Frozen-probe 5-fold scaffold CV, 3 seeds, otherwise identical
to the real `unsup_8M` run. Comparators: `real` = unsup_8M, `no_pretrain` = random_baseline (both
native-unit re-evals).

**Result:** wiki_real beats no_pretrain on 6/7 tasks and MATCHES real SMILES on QM7 — English (zero
chemistry) transfers a task-dependent share of the benefit. Guards: (1) coverage — Wikipedia trained
96.9% of eval-token MASS (see wiki_coverage.json), so not an undertrained-embedding artifact; (2)
same tokenizer + matched lengths but token MARGINALS near-maximally divergent (JS=0.93 bits;
wiki_vs_smiles_stats.json), with 435 chemistry tokens (incl. stereo) unfilled. So transfer happens
despite an orthogonal token marginal.
"""


def sh(c): return subprocess.run(c, shell=True, capture_output=True, text=True).stdout
def s3b(uri):
    r = subprocess.run(["aws", "s3", "cp", uri, "-"], capture_output=True)
    return r.stdout if r.returncode == 0 else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="dist/experiment_b_bundle.zip")
    ap.add_argument("--hf-org", default="lsieben")
    a = ap.parse_args()
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    miss = []
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for rel in LOCAL:
            p = Path(rel)
            if p.exists(): z.write(p, f"result/{p.name}")
            else: miss.append(rel)
        z.writestr("methodology.md", METHOD)
        for label, sub in ARMS:
            for rel in RUN_FILES:
                d = s3b(f"{BUCKET}/{sub}/{rel}")
                if d is not None: z.writestr(f"runs/{label}/{rel}", d)
                else: miss.append(f"{sub}/{rel}")
        d = s3b("s3://climb-s3-bucket/tokenized_sources/pubchem_filtered_wiki_pkl/_diagnostics.json")
        if d is not None: z.writestr("corpus/wiki_corpus_diagnostics.json", d)
        ck = io.StringIO(); w = csv.writer(ck); w.writerow(["run_id", "size", "s3_uri", "hf_uri"])
        for rid in ENC_RUNS:
            ls = sh(f"aws s3 ls {BUCKET}/climb_v2_expB/{rid}/encoder/model.safetensors 2>/dev/null").split()
            w.writerow([rid, ls[2] if len(ls) >= 3 else "?", f"{BUCKET}/climb_v2_expB/{rid}/encoder/",
                        f"https://huggingface.co/{a.hf_org}/climb-encoders/tree/main/climb_v2_expB/{rid}"])
        z.writestr("checkpoints.csv", ck.getvalue())
        z.writestr("README.md", f"""# Experiment B — Wikipedia-through-SMILES-tokenizer transfer (CLIMB)

Does pretraining on a NON-chemical corpus (English Wikipedia, frozen SMILES tokenizer) help the
chemistry suite? See methodology.md. `result/` has the headline (wiki_real vs real vs no_pretrain,
native 5-fold CV), plus the coverage guard and the SMILES-vs-Wiki corpus comparison. `runs/<arm>/`
has per-molecule CV predictions + configs for your own analysis. `checkpoints.csv` indexes the 3
wiki_real encoders (weights on HF `{a.hf_org}/climb-encoders`, private — request access).
""")
    print(f"wrote {out} ({out.stat().st_size/1e6:.1f} MB)")
    if miss: print(f"  NOTE {len(miss)} expected files missing (e.g. baselines have no training artifacts)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
