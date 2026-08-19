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
                "fp_desc": "ECFP4 + descriptors", "encoder": "CLIMB encoder"}
# the rows worth printing in the paper; the rest stay in the JSON
KEEP = [("ecfp4", "cpu", "single core"), ("ecfp4", "cpu", "12 processes"),
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

    base = None
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
        row = dict(method=METHOD_LABEL[method], machine=MACHINE[device], device=device,
                   precision=r.get("precision") if r.get("precision") != "n/a" else "",
                   config=notes, n_molecules=n,
                   s_per_1k=round(mean_s, 4), sd_s=round(sd_s, 4),
                   mol_per_s=round(float(r["mol_per_s"]), 1),
                   hours_per_1M=round(float(r.get("hours_1M", r.get("hours_per_1M"))), 3),
                   hours_per_1B=round(float(r.get("hours_1B", r.get("hours_per_1B"))), 1),
                   source=r.get("carried_from", "measured in this run"))
        if base is None:
            base = row["s_per_1k"]                      # Morgan counts, single core = the reference
        row["vs_morgan_1core"] = round(row["s_per_1k"] / base, 2)
        rows.append(row)

    OUTDIR.mkdir(exist_ok=True)
    cols = ["method", "machine", "device", "precision", "config", "n_molecules", "s_per_1k",
            "sd_s", "mol_per_s", "hours_per_1M", "hours_per_1B", "vs_morgan_1core", "source"]
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
                     f"{r['hours_per_1B']:.1f} & {r['vs_morgan_1core']:.2f} \\\\\n")
        fh.write("\\hline\n\\end{tabular}\n")

    w_ = [len(h) for h in hdr]
    print("\nSI Fig c — featurization cost (1000 molecules, 5 repeats after warm-up):\n")
    print(f"  {'Featurizer':<29}{'Hardware':<27}{'Config':<31}{'s/1k':>9}{'mol/s':>9}{'h/1B':>9}{'xMorgan':>9}")
    for r in rows:
        prec = f" {r['precision']}" if r["precision"] else ""
        print(f"  {r['method']:<29}{r['machine']+prec:<27}{r['config']:<31}"
              f"{r['s_per_1k']:>9.3f}{r['mol_per_s']:>9.0f}{r['hours_per_1B']:>9.1f}"
              f"{r['vs_morgan_1core']:>9.2f}")
    print("\n  wrote figures_v2/SI_fig_c.csv + SI_fig_c.tex")


if __name__ == "__main__":
    main()
