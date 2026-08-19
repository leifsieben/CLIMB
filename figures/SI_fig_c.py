"""SI Fig c — featurization cost: classical descriptors vs the transformer encoder.

ONE script, ONE TABLE: figures_v2/SI_fig_c.csv + SI_fig_c.tex

The paper argues about compute cost at virtual-screening scale, so the number has to be measured,
not asserted. `scripts/bench_featurization.py` times the three featurizers we actually ship on the
SAME 1000 MoleculeNet molecules with the SAME settings production uses, 5 repeats after a warm-up
(RDKit lazy imports, torch kernel autotune / MPS shader compile).

  ECFP4 (stereo)       r=2, bits, chirality on, 2048-d   (featurize_v2.ecfp4_features)
  RDKit descriptors    217 descriptors                  (descriptors_v2.rdkit_descriptors)
  ECFP4 + descriptors  the ECFP4+desc anchor's features
  CLIMB encoder        ModernBERT ~41M, tokenize + forward + mean pool (eval_v2._encoder_features)

THE RESULT: the transformer is not the expensive part -- RDKit descriptors are. On one CPU core the
descriptors cost 4.71 s/1000 molecules against ECFP4's 0.094 s, a 50x gap. Concatenating the
fingerprint on top costs nothing measurable: ECFP4+desc comes in at 4.65 s against the descriptors'
own 4.71 s, a difference inside the run-to-run spread, so the anchor's featurization cost IS its
descriptor block. The encoder on one A10G runs 1000 molecules in
0.587 s -- 6.3x the cost of single-core ECFP4, but 7.9x FASTER than the ECFP4+desc anchor that
outranks it in Fig A1. At 1B molecules that is 163 GPU-hours for the encoder against
1292 single-core CPU-hours for ECFP4+desc.

So "transformers are too slow to screen with" is not supported by our own measurements: the
classical baseline that beats CLIMB on accuracy is the slower of the two to featurize, unless the
descriptors are parallelised across cores (12 processes brings ECFP+desc to 0.76 s).

THE r3-COUNTS ROW AND WHY IT COMES WITH A CONTROL. The third XGBoost anchor uses Morgan radius 3
COUNTS, which is strictly more work than radius-2 bits, and it costs 1.15x -- 0.088 s against
0.077 s per 1000 molecules, both timed interleaved in one process (scripts/bench_fp_variants.py).
It was first measured in its own bench run and came out 7% FASTER than ECFP4, which cannot be
true; the tell was the RDKit descriptor row, identical work timed at 4.709 s in one run and 4.119 s
in the other, so that machine was simply 14% quicker that afternoon. The paired control row is kept
in the table precisely so a reader cannot re-derive the impossible comparison from the published
ECFP4 row, and the "x ECFP4" column normalises each row against the reference measured under ITS
OWN machine state (the `ratio_vs` column of the CSV says which). Either way the conclusion is
unchanged: both fingerprints are ~50x cheaper than the descriptor block they are concatenated to.

HARDWARE. Two machines, and the table says which per row because they are not comparable:
`cpu`/`mps` rows are the Apple M4 Pro; `cuda` rows are an AWS g5.2xlarge (NVIDIA A10G). RDKit has
no GPU path, so the classical rows exist only on CPU. Full per-repeat timings, molecule/token
statistics and library versions are in figure_data/_bench/featurization_timing.json.

Run:  python3 -m figures.SI_fig_c
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "figure_data" / "_bench" / "featurization_timing.json"
OUTDIR = ROOT / "figures_v2"

METHOD_LABEL = {"ecfp4": "ECFP4 (stereo)", "rdkit_desc": "RDKit descriptors",
                "fp_desc": "ECFP4 + descriptors", "encoder": "CLIMB encoder",
                "ecfp4_r3c": "Morgan r3-counts"}
# Label override where the METHOD alone is ambiguous: the paired-A/B rows are the same featurizer
# as the published one, timed again as a control, and must not read as a second measurement of it.
LABEL_BY_KEY = {("ecfp4", "single core (paired A/B)"): "ECFP4 (stereo), paired control"}
# Rows whose x-ECFP4 ratio must be taken against the PAIRED control rather than the published
# ECFP4 row, because they were measured in a different bench run from it.
PAIRED = {("ecfp4", "single core (paired A/B)"): True, ("ecfp4_r3c", "cpu"): True,
          ("ecfp4_r3c", "single core"): True}

# the rows worth printing in the paper; the rest stay in the JSON
#
# THE TWO PAIRED ROWS SIT TOGETHER AND ARE THE ONLY VALID r3-COUNTS COMPARISON. Read against the
# published ECFP4 row above them, r3-counts appears FASTER (0.088 vs 0.094 s) -- which is
# impossible, since radius-3 counts is strictly more work than radius-2 bits. Those two numbers
# come from different bench runs, and the give-away is the RDKit descriptor row: identical work,
# 4.709 s in one run and 4.119 s in the other, i.e. the machine itself was 14% quicker that
# afternoon. scripts/bench_fp_variants.py re-times both fingerprints interleaved in ONE process,
# and the honest answer is 1.15x. Keeping the control row visible is what stops a reader
# re-deriving the impossible one.
KEEP = [("ecfp4", "cpu", "single core"), ("ecfp4", "cpu", "12 processes"),
        ("ecfp4", "cpu", "single core (paired A/B)"),
        ("ecfp4_r3c", "cpu", "single core"),
        ("rdkit_desc", "cpu", "single core"), ("rdkit_desc", "cpu", "12 processes"),
        ("fp_desc", "cpu", "single core"), ("fp_desc", "cpu", "12 processes"),
        ("encoder", "cpu", "8 threads, bs256, dynamic pad"),
        ("encoder", "mps", "bs256, dynamic pad"),
        ("encoder", "cuda", "bs256, dynamic pad")]
# device -> which machine it ran on (results rows carry no hardware label of their own)
MACHINE = {"cpu": "Apple M4 Pro", "mps": "Apple M4 Pro (GPU)", "cuda": "AWS g5.2xlarge (A10G)"}
PRECISION_PICK = {"cuda": "bf16", "mps": "fp16"}      # the fastest measured precision per device


def main():
    j = json.load(open(SRC))
    n = j["config"]["n_molecules"]
    rows, by_key = [], {}
    for r in j["results"]:
        by_key.setdefault((r["method"], r["device"], r["notes"]), []).append(r)

    base = paired_base = None
    for method, device, notes in KEEP:
        cands = by_key.get((method, device, notes), [])
        if device in PRECISION_PICK:
            cands = [c for c in cands if c.get("precision") == PRECISION_PICK[device]] or cands
        if not cands:
            continue
        r = cands[0]
        # A CARRIED-FORWARD row has the summary statistics but not the raw repeats -- it was
        # recovered from the published CSV of a run on hardware this machine does not have (the
        # A10G encoder row). Its mean and sd are the real measured ones; synthesising a `times_s`
        # list that happens to reproduce them would be fabricating measurements to satisfy a
        # schema, so the reader takes the summaries directly and the row stays labelled.
        if "times_s" in r:
            t = np.array(r["times_s"], dtype=float)
            mean_s, sd_s = float(t.mean()), float(t.std(ddof=1))
        else:
            mean_s, sd_s = float(r["s_per_1k"]), float(r.get("sd_s", float("nan")))
        row = dict(method=LABEL_BY_KEY.get((method, notes), METHOD_LABEL[method]),
                   machine=MACHINE[device], device=device,
                   precision=r.get("precision") if r.get("precision") != "n/a" else "",
                   config=notes, n_molecules=n,
                   s_per_1k=round(mean_s, 4), sd_s=round(sd_s, 4),
                   mol_per_s=round(float(r["mol_per_s"]), 1),
                   hours_per_1M=round(float(r.get("hours_1M", r.get("hours_per_1M"))), 3),
                   hours_per_1B=round(float(r.get("hours_1B", r.get("hours_per_1B"))), 1),
                   source=r.get("carried_from", "measured in this run"))
        if base is None:
            base = row["s_per_1k"]                      # ECFP4, single core = the reference
        if (method, notes) == ("ecfp4", "single core (paired A/B)"):
            paired_base = row["s_per_1k"]
        # The r3-counts row is normalised against ITS OWN paired control, not against the published
        # ECFP4 row. Dividing by `base` here is what produced "r3-counts = 0.94x ECFP4" -- a
        # cross-run ratio that says the more expensive fingerprint is cheaper. Same rule, one line:
        # a ratio is only meaningful between two numbers measured under one machine state.
        ref = paired_base if PAIRED.get((method, notes)) else base
        row["vs_ecfp4_1core"] = round(row["s_per_1k"] / ref, 2)
        row["ratio_vs"] = "paired control" if PAIRED.get((method, notes)) else "published ECFP4"
        rows.append(row)

    OUTDIR.mkdir(exist_ok=True)
    cols = ["method", "machine", "device", "precision", "config", "n_molecules", "s_per_1k",
            "sd_s", "mol_per_s", "hours_per_1M", "hours_per_1B", "vs_ecfp4_1core", "ratio_vs", "source"]
    with open(OUTDIR / "SI_fig_c.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)

    hdr = ["Featurizer", "Hardware", "Config", "s / 1k mol", "mol / s", "GPU- or CPU-h / 1B",
           r"$\times$ ECFP4"]
    with open(OUTDIR / "SI_fig_c.tex", "w") as fh:
        fh.write("% SI Fig c — featurization cost. Generated by figures/SI_fig_c.py; do not hand-edit.\n")
        fh.write("\\begin{tabular}{lllrrrr}\n\\hline\n")
        fh.write(" & ".join(hdr) + " \\\\\n\\hline\n")
        for r in rows:
            prec = f" {r['precision']}" if r["precision"] else ""
            fh.write(f"{r['method']} & {r['machine']}{prec} & {r['config']} & "
                     f"{r['s_per_1k']:.3f} $\\pm$ {r['sd_s']:.3f} & {r['mol_per_s']:.0f} & "
                     f"{r['hours_per_1B']:.1f} & {r['vs_ecfp4_1core']:.2f} \\\\\n")
        fh.write("\\hline\n\\end{tabular}\n")

    w_ = [len(h) for h in hdr]
    print("\nSI Fig c — featurization cost (1000 molecules, 5 repeats after warm-up):\n")
    print(f"  {'Featurizer':<31}{'Hardware':<27}{'Config':<31}{'s/1k':>9}{'mol/s':>9}{'h/1B':>9}{'xECFP4':>9}")
    for r in rows:
        prec = f" {r['precision']}" if r["precision"] else ""
        print(f"  {r['method']:<31}{r['machine']+prec:<27}{r['config']:<31}"
              f"{r['s_per_1k']:>9.3f}{r['mol_per_s']:>9.0f}{r['hours_per_1B']:>9.1f}"
              f"{r['vs_ecfp4_1core']:>9.2f}")
    print("\n  wrote figures_v2/SI_fig_c.csv + SI_fig_c.tex")


if __name__ == "__main__":
    main()
