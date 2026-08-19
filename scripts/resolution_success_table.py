"""Turn pairwise distances into a SUCCESS RATE per (embedding, failure mode).

RESOLVED IS BINARY: the two embedding vectors are DIFFERENT, or they are not.

That is the whole definition. No chosen threshold, because there is nothing to choose -- the only
guard is float32 epsilon (cosine > 1e-6), and it is there for arithmetic, not for chemistry.
scripts/resolution_noise_floor.py measures the need for it: re-embedding the same molecules in a
different batch order reproduces every representation BIT-FOR-BIT (128/128), and the largest
cosine seen across that probe is 1.8e-07, i.e. float32 rounding in the cosine itself. So 1e-6 sits
two orders above measured noise and far below any real difference.

An earlier version scored success against a normalised-distance threshold of 0.01. It was
answering a different and worse question -- "is the difference BIG" rather than "is the difference
THERE" -- and it made four of thirteen modes knife-edge: CLIMB unsup scored 94% on stereo_flip at
eps=0.001 and 4% at eps=0.01, so the headline number was a property of the threshold. The magnitude
question is still worth asking, so median_separation is still reported beside the binary answer;
it is just no longer what decides success.

Success needs a threshold, because for a continuous embedding "not bit-identical" is satisfied by
floating-point noise -- CheMeleon reads as 0% identical on Class B purely from summation order,
while its actual separation is exactly zero. So success is defined on the SCALE-FREE separation
ratio (cosine(A,B) divided by the median cosine from A to 1,000 random molecules):

  class A (different molecules)  RESOLVED      if separation >= EPS
  class B (same molecule)        CORRECT       if separation <= EPS

EPS = 0.01 means "at least 1% of the distance to a random molecule". It is reported at three
values so the reader can see the ranking does not hinge on the choice.
"""
from __future__ import annotations
import csv, json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "figure_data/embedding_resolution"
# THE DEFINITION: two molecules are RESOLVED if they sit farther apart than that representation
# could put a molecule from ITSELF -- above the 95th percentile of its own noise distribution, so
# under a 5% chance the difference is noise. The threshold is MEASURED per representation by
# resolution_null_calibration.py (200 anchors x 10 random SMILES re-writings, which denote exactly
# the same compound), never chosen. It comes out at 1.5e-07 for a fingerprint, which is invariant,
# and 3.5e-01 for CLIMB sup, which is not -- and that gap IS the experiment.
#
# CALIBRATION CAVEAT, stated rather than hidden: `smiles_enumeration` is the same transformation
# the null is built from, so scoring it against the null is circular by construction. It is
# reported as the calibration itself -- the noise magnitude -- and excluded from the success bars.
# `kekule` and `symmetry_equivalent` are different transformations and are scored normally.
CALIBRATION_MODE = "smiles_enumeration"
EPS_LIST = [0.001, 0.01, 0.05]     # legacy magnitude columns, retained, NOT used for success


def main() -> int:
    rows = list(csv.DictReader((OUT / "distances.csv").open()))
    # Floor the measured threshold at the arithmetic noise of the distance itself. For an exactly
    # invariant fingerprint the null is degenerate at 0 and its 95th percentile lands at 1.5e-07,
    # which is BELOW the float32 rounding of the cosine -- so ~10% of genuinely identical pairs
    # would score as "resolved" on rounding alone, and the stereo-blind control read 10% instead
    # of the 0% that validates the harness.
    FLOAT_GUARD = 1e-6
    thr = {k: max(v["p95"], FLOAT_GUARD)
           for k, v in json.loads((OUT / "null_threshold.json").read_text()).items()}
    by = defaultdict(list)
    order_mode, order_emb = [], []
    for r in rows:
        by[(r["embedding"], r["mode"])].append(r)
        if (r["klass"], r["mode"]) not in order_mode:
            order_mode.append((r["klass"], r["mode"]))
        if r["embedding"] not in order_emb:
            order_emb.append(r["embedding"])

    out = []
    for (klass, mode) in order_mode:
        for emb in order_emb:
            rs = by[(emb, mode)]
            if not rs:
                continue
            sep = np.array([float(r["separation"]) for r in rs])
            t = thr.get(emb)
            if t is None:
                continue
            different = sep > t
            rec = dict(klass=klass, mode=mode, embedding=emb, n=len(rs),
                       median_separation=float(np.median(sep)),
                       null_p95=float(t),
                       identical=int(sum(1 for r in rs if r["identical"] == "True")),
                       success=(None if mode == CALIBRATION_MODE else
                                round(100.0 * (different if klass == "A" else ~different).mean(), 1)))
            for e in EPS_LIST:
                ok = (sep >= e) if klass == "A" else (sep <= e)
                rec[f"magnitude_eps{e}"] = round(100.0 * ok.mean(), 1)
            out.append(rec)

    fields = list(out[0].keys())
    with (OUT / "success_rates.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(out)

    show = [e for e in order_emb if e != "ECFP4 stereo-blind"]
    hdr = f"{'mode':26}" + "".join(f"{e[:12]:>14}" for e in show)
    print("RESOLVED = farther apart than the representation's own noise (p<0.05)")
    print("  A: resolved is correct   B: NOT resolved is correct\n")
    print("  measured noise threshold (95th pct of self-vs-self separation):")
    for k, v in thr.items():
        print(f"      {k:22} {v:.2e}")
    print()
    print(hdr); print("-" * len(hdr))
    for (klass, mode) in order_mode:
        line = f"{('A ' if klass=='A' else 'B ')+mode:26}"
        for emb in show:
            rec = next((r for r in out if r["mode"] == mode and r["embedding"] == emb), None)
            line += (f"{rec['success']:>13.0f}%" if rec and rec["success"] is not None
                     else f"{'calib':>14}" if rec else f"{'--':>14}")
        print(line)
    ctrl = [r for r in out if r["embedding"] == "ECFP4 stereo-blind"
            and r["mode"] in ("stereo_flip", "ez_flip")]
    print("\ncontrol, ECFP4 stereo-blind: " +
          ", ".join(f"{r['mode']} {r['success']:.0f}%" for r in ctrl))
    print(f"\nwrote {OUT/'success_rates.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
