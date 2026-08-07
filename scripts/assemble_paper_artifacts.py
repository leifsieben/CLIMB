"""Assemble one tidy, self-describing bundle of everything the paper and its review need.

Layout (identical locally and in the backup bucket):

    paper_artifacts/
      README.md              what this is, and how to regenerate any part of it
      INVENTORY.md           the human table: every model x how trained x how evaluated
      model_inventory.csv    the same, machine-readable
      checkpoints.csv        S3 URI + size for every encoder (see the note on downloading)
      fetch_checkpoint.sh    pull any single checkpoint on demand
      figures/               the eight paper figures, png + pdf
      results/<wave>/<run>/  that run's config, training curve, completion proof, evaluations
      derived/               Tanimoto similarity, eval-ceiling, tokenizer

Checkpoints are deliberately NOT copied into the local bundle: the paper-critical encoders total
~13.4 GB, above the 10 GB the user allowed on their laptop. They live in S3 and in the backup
bucket, and checkpoints.csv + fetch_checkpoint.sh make any one of them a single command away.

Usage:
    python scripts/assemble_paper_artifacts.py --inventory audit/model_inventory.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import defaultdict
from pathlib import Path

OUT = Path("paper_artifacts")
FIGS = Path("figures_out")
DATA = Path("figure_data")

# per-run files worth keeping: everything except the multi-hundred-MB weights
KEEP = ["config.yaml", "metadata.json", "metrics.jsonl", "verified.json", "run_status.json",
        "moleculenet/suite_summary.json", "moleculenet/moleculenet_summary.csv",
        "moleculenet_cv/suite_summary.json", "moleculenet_cv/moleculenet_summary.csv",
        "moleculenet_cv/test_predictions.csv"]

FIGURE_DOC = [
    ("A1a", "figA1a_best_model_headline_holdout", "Which model performs best (headline scaffold hold-out)?",
     "climb_v2_phase2 @ 8M, DeepChem scaffold hold-out"),
    ("A1b", "figA1b_best_model_headline_cv", "Which model performs best (5-fold scaffold CV)?",
     "climb_v2_phase2 @ 8M, 5-fold scaffold CV"),
    ("A2a", "figA2a_scaling_tokens", "How does pretraining scale in tokens seen?",
     "climb_v2_phase2, 2M/8M/24M/48M/96M (+50M/100M)"),
    ("A2b", "figA2b_scaling_unique_molecules", "How does pretraining scale in unique molecules?",
     "climb_v2_phase2 scaling rungs, 5-fold CV"),
    ("B1p1", "figB1p1_label_efficiency_train_test", "Does pretraining help small datasets more, and how?",
     "analysis/rigor/label_efficiency_fractions_all_summary.csv — per-task fractions 5/10/25/50/100%, 5 arms, native-unit regression"),
    ("E1", "figE1_H5_eval_ceiling", "Is the sup/unsup ordering a frozen-probe artifact?",
     "climb_v2_phase2 + derived/_eval_ceiling{,_sup}"),
    ("B2", "figB2_corrupted_control", "Does content-free pretraining help just as much?",
     "climb_v2_phase2 corrupt_mlm_8M / corrupt_mtr_8M"),
    ("C1J1", "figC1J1_sft_family_transfer", "Which SFT data helps, how much, does it track chemistry?",
     "climb_v2_ablation_dedup + derived/_tanimoto"),
    ("I1", "figI1_memorization_vs_representation", "Do corpus-similar or novel molecules benefit?",
     "climb_v2_phase2 CV predictions + derived/_tanimoto"),
    ("C1J1+I1", "figC1J1_I1_combined", "Combined SFT-transfer + memorization panel (paper layout)",
     "reuses climb_v2_ablation_dedup + climb_v2_phase2 + derived/_tanimoto"),
    ("H1", "figH1_canonical_vs_enumerated", "Does enumeration beat canonical repetition?",
     "climb_v2_h1 (retrained, 3 seeds)"),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inventory", default="audit/model_inventory.csv")
    ap.add_argument("--bucket", default="s3://climb-s3-bucket")
    a = ap.parse_args()

    rows = list(csv.DictReader(open(a.inventory)))
    OUT.mkdir(exist_ok=True)

    # ---- figures ----
    (OUT / "figures").mkdir(exist_ok=True)
    n_fig = 0
    for _, name, _, _ in FIGURE_DOC:
        for ext in ("png", "pdf"):
            for cand in (FIGS / f"{name}.{ext}", FIGS / f"{name}_PLACEHOLDER.{ext}"):
                if cand.exists():
                    shutil.copy2(cand, OUT / "figures" / cand.name); n_fig += 1
                    break

    # ---- per-run results (no weights) ----
    n_files = 0
    for r in rows:
        src = DATA / r["wave"] / r["run"]
        if not src.exists():
            continue
        for rel in KEEP:
            f = src / rel
            if f.exists():
                dst = OUT / "results" / r["wave"] / r["run"] / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(f, dst); n_files += 1

    # ---- derived analysis products ----
    for d in ("_tanimoto", "_eval_ceiling", "_eval_ceiling_sup", "_tokenizer"):
        if (DATA / d).exists():
            shutil.copytree(DATA / d, OUT / "derived" / d.lstrip("_"), dirs_exist_ok=True)

    # ---- checkpoint index (URIs, not bytes) ----
    ck = [r for r in rows if float(r["encoder_s3_gb"] or 0) > 0]
    with open(OUT / "checkpoints.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["wave", "run", "pretraining", "budget", "seed", "size_gb", "s3_uri"])
        for r in sorted(ck, key=lambda x: (x["wave"], x["run"])):
            w.writerow([r["wave"], r["run"], r["pretraining"], r["budget"], r["seed"],
                        r["encoder_s3_gb"],
                        f'{a.bucket}/experiments/{r["wave"]}/{r["run"]}/encoder/'])
    total_gb = sum(float(r["encoder_s3_gb"]) for r in ck)

    (OUT / "fetch_checkpoint.sh").write_text(
        "#!/usr/bin/env bash\n"
        "# Pull one encoder checkpoint by run name.  Usage: ./fetch_checkpoint.sh unsup_8M [dest]\n"
        "# Checkpoints are not shipped in this bundle: they total "
        f"{total_gb:.1f} GB, which is more than\n"
        "# belongs on a laptop. Every one is in S3 and in the backup bucket; this fetches on demand.\n"
        "set -euo pipefail\n"
        'RUN=${1:?run name, e.g. unsup_8M}; DEST=${2:-./checkpoints/$RUN}\n'
        'URI=$(awk -F, -v r="$RUN" \'$2==r{print $7}\' "$(dirname "$0")/checkpoints.csv")\n'
        '[ -n "$URI" ] || { echo "no checkpoint called $RUN in checkpoints.csv"; exit 1; }\n'
        'mkdir -p "$DEST" && aws s3 sync "$URI" "$DEST" && echo "-> $DEST"\n')
    (OUT / "fetch_checkpoint.sh").chmod(0o755)

    # ---- INVENTORY.md: the human-facing table ----
    by_type = defaultdict(list)
    for r in rows:
        by_type[r["pretraining"]].append(r)

    L = ["# Model inventory\n",
         "Every model trained for this paper, grouped by pretraining type, then by the budget it ",
         "was trained at, then by how it was evaluated.\n",
         "\n**Legend** — `ckpt`: encoder weights in S3 · `curve`: metrics.jsonl training curve · ",
         "`proof`: verified.json (achieved forward passes ≥98% of budget) · `single`: DeepChem ",
         "scaffold hold-out eval · `cv`: 5-fold scaffold CV eval · `preds`: per-molecule CV ",
         "predictions.\n\n"]
    for ptype in sorted(by_type):
        rs = sorted(by_type[ptype], key=lambda x: (x["budget"], x["seed"]))
        L.append(f"## {ptype}\n\n")
        L.append("| wave | run | budget | seed | ckpt | curve | proof | single | cv | preds |\n")
        L.append("|---|---|---|---|:--:|:--:|:--:|:--:|:--:|:--:|\n")
        tick = lambda v: "✅" if str(v) == "1" else "—"
        for r in rs:
            L.append(f'| {r["wave"]} | `{r["run"]}` | {r["budget"]} | {r["seed"]} | '
                     f'{"✅" if float(r["encoder_s3_gb"] or 0) else "—"} | '
                     f'{tick(r["metrics_jsonl"])} | {tick(r["verified"])} | '
                     f'{tick(r["eval::single-split summary"])} | {tick(r["eval::5-fold CV summary"])} | '
                     f'{tick(r["eval::per-molecule CV predictions (I1)"])} |\n')
        L.append("\n")
    (OUT / "INVENTORY.md").write_text("".join(L))

    # ---- README ----
    fig_rows = "\n".join(f"| **{fid}** | {q} | `figures/{name}.png` / `.pdf` | {src} |"
                         for fid, name, q, src in FIGURE_DOC)
    (OUT / "README.md").write_text(f"""# CLIMB — paper & peer-review artifacts

Everything needed to check, re-run or extend the paper. The same tree exists in the backup
bucket, so this directory and `s3://climb-paper-backup-075120018132/paper_artifacts/` are
interchangeable.

## What is here

| path | contents |
|---|---|
| `INVENTORY.md` | every model: pretraining type x budget x seed, and what we hold for it |
| `model_inventory.csv` | the same, machine-readable |
| `checkpoints.csv` | S3 URI and size for all {len(ck)} encoder checkpoints ({total_gb:.1f} GB) |
| `fetch_checkpoint.sh` | pull any one checkpoint: `./fetch_checkpoint.sh unsup_8M` |
| `figures/` | the eight paper figures, PNG + PDF |
| `results/<wave>/<run>/` | that run's config, training curve, completion proof, evaluations |
| `derived/` | Tanimoto similarity tables, eval-ceiling results, the tokenizer |

**Checkpoints are not in this bundle.** They total {total_gb:.1f} GB, more than belongs on a
laptop, and they exist in two places already (the working bucket and the backup bucket). Fetch
them individually rather than mirroring them.

## The figures

| ID | Question | File | Built from |
|---|---|---|---|
{fig_rows}

## Reproducing

- **A figure**: `python scripts/build_figure_notebook.py && jupyter nbconvert --execute --inplace
  climb_figures.ipynb`. The notebook's last cell reports, per figure, whether it is FINAL or
  still PROVISIONAL and exactly what is blocking it.
- **An evaluation**: `python eval_v2.py --encoder <encoder dir> --tokenizer derived/tokenizer
  --output_dir <out> --datasets ESOL QM7 BBBP BACE Tox21 HIV --head_seeds 0 1 2`
  (add `--cv_folds 5` for the CV scheme). Lipophilicity is excluded everywhere — the eval
  blocklist predates it.
- **A model**: every run's `config.yaml` is the exact config it was trained with; feed it back to
  `pretrain_v2.py --run_dir <dir> --config <config.yaml>`.

## Completion is proven, not assumed

`verified.json` is written only after achieved forward passes reach ≥98% of the budget. Any
"is this done?" decision reads that marker. Runs without it are not finished, regardless of which
output files exist — a truncated run still writes a summary, which is how half-trained models
reached figures earlier in this project.
""")

    shutil.copy2(a.inventory, OUT / "model_inventory.csv")
    print(f"paper_artifacts/: {n_fig} figure files, {n_files} result files, "
          f"{len(ck)} checkpoints indexed ({total_gb:.1f} GB, not copied)")
    print(f"  -> {OUT}/README.md, INVENTORY.md, checkpoints.csv, fetch_checkpoint.sh")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
