"""SI Fig c — featurization cost: classical descriptors vs the transformer encoder.

ONE script, ONE TABLE: figures_v2/SI_Fig_c.csv + SI_Fig_c.tex

The paper argues about compute cost at virtual-screening scale, so the number has to be measured,
not asserted. `scripts/bench_featurization.py` times the three featurizers we actually ship on the
SAME 1000 MoleculeNet molecules with the SAME settings production uses, 5 repeats after a warm-up
(RDKit lazy imports, torch kernel autotune / MPS shader compile).

  ECFP4                Morgan r=2, 2048 bits            (featurize_v2.ecfp4_features)
  RDKit descriptors    217 descriptors                  (descriptors_v2.rdkit_descriptors)
  ECFP4 + descriptors  the ECFP+desc anchor's features
  CLIMB encoder        ModernBERT ~41M, tokenize + forward + mean pool (eval_v2._encoder_features)

THE RESULT: the transformer is not the expensive part — RDKit descriptors are. On one CPU core the
descriptors cost 4.37 s/1000 molecules against ECFP4's 0.078 s, a 56x gap, and they dominate the
ECFP+desc anchor almost entirely (4.46 s, i.e. 98% descriptors). The encoder on one A10G runs
1000 molecules in 0.59 s — 7.5x the cost of single-core ECFP4, but 7.4x FASTER than the ECFP+desc
anchor that outranks it in Fig A1. At 1B molecules that is 163 GPU-hours for the encoder against
1238 single-core CPU-hours for ECFP+desc.

So "transformers are too slow to screen with" is not supported by our own measurements: the
classical baseline that beats CLIMB on accuracy is the slower of the two to featurize, unless the
descriptors are parallelised across cores (12 processes brings ECFP+desc to 0.76 s).

HARDWARE. Two machines, and the table says which per row because they are not comparable:
`cpu`/`mps` rows are the Apple M4 Pro; `cuda` rows are an AWS g5.2xlarge (NVIDIA A10G). RDKit has
no GPU path, so the classical rows exist only on CPU. Full per-repeat timings, molecule/token
statistics and library versions are in figure_data/_bench/featurization_timing.json.

Run:  python3 -m figures.fig_SI_c
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "figure_data" / "_bench" / "featurization_timing.json"
OUTDIR = ROOT / "figures_v2"

METHOD_LABEL = {"ecfp4": "ECFP4", "rdkit_desc": "RDKit descriptors",
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
        t = np.array(r["times_s"], dtype=float)
        row = dict(method=METHOD_LABEL[method], machine=MACHINE[device], device=device,
                   precision=r.get("precision") if r.get("precision") != "n/a" else "",
                   config=notes, n_molecules=n,
                   s_per_1k=round(float(t.mean()), 4), sd_s=round(float(t.std(ddof=1)), 4),
                   mol_per_s=round(float(r["mol_per_s"]), 1),
                   hours_per_1M=round(float(r["hours_1M"]), 3),
                   hours_per_1B=round(float(r["hours_1B"]), 1))
        if base is None:
            base = row["s_per_1k"]                      # ECFP4, single core = the reference cost
        row["vs_ECFP4_1core"] = round(row["s_per_1k"] / base, 2)
        rows.append(row)

    OUTDIR.mkdir(exist_ok=True)
    cols = ["method", "machine", "device", "precision", "config", "n_molecules", "s_per_1k",
            "sd_s", "mol_per_s", "hours_per_1M", "hours_per_1B", "vs_ECFP4_1core"]
    with open(OUTDIR / "SI_Fig_c.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)

    hdr = ["Featurizer", "Hardware", "Config", "s / 1k mol", "mol / s", "GPU- or CPU-h / 1B",
           r"$\times$ ECFP4"]
    with open(OUTDIR / "SI_Fig_c.tex", "w") as fh:
        fh.write("% SI Fig c — featurization cost. Generated by figures/fig_SI_c.py; do not hand-edit.\n")
        fh.write("\\begin{tabular}{lllrrrr}\n\\hline\n")
        fh.write(" & ".join(hdr) + " \\\\\n\\hline\n")
        for r in rows:
            prec = f" {r['precision']}" if r["precision"] else ""
            fh.write(f"{r['method']} & {r['machine']}{prec} & {r['config']} & "
                     f"{r['s_per_1k']:.3f} $\\pm$ {r['sd_s']:.3f} & {r['mol_per_s']:.0f} & "
                     f"{r['hours_per_1B']:.1f} & {r['vs_ECFP4_1core']:.2f} \\\\\n")
        fh.write("\\hline\n\\end{tabular}\n")

    w_ = [len(h) for h in hdr]
    print("\nSI Fig c — featurization cost (1000 molecules, 5 repeats after warm-up):\n")
    print(f"  {'Featurizer':<21}{'Hardware':<27}{'Config':<31}{'s/1k':>9}{'mol/s':>9}{'h/1B':>9}{'xECFP4':>9}")
    for r in rows:
        prec = f" {r['precision']}" if r["precision"] else ""
        print(f"  {r['method']:<21}{r['machine']+prec:<27}{r['config']:<31}"
              f"{r['s_per_1k']:>9.3f}{r['mol_per_s']:>9.0f}{r['hours_per_1B']:>9.1f}"
              f"{r['vs_ECFP4_1core']:>9.2f}")
    print("\n  wrote figures_v2/SI_Fig_c.csv + SI_Fig_c.tex")


if __name__ == "__main__":
    main()
