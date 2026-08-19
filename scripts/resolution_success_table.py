"""Turn pairwise distances into a SUCCESS RATE per (embedding, failure mode).

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
EPS_LIST = [0.001, 0.01, 0.05]
EPS_MAIN = 0.01


def main() -> int:
    rows = list(csv.DictReader((OUT / "distances.csv").open()))
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
            rec = dict(klass=klass, mode=mode, embedding=emb, n=len(rs),
                       median_separation=float(np.median(sep)),
                       identical=int(sum(1 for r in rs if r["identical"] == "True")))
            for e in EPS_LIST:
                ok = (sep >= e) if klass == "A" else (sep <= e)
                rec[f"success_eps{e}"] = round(100.0 * ok.mean(), 1)
            rec["success"] = rec[f"success_eps{EPS_MAIN}"]
            out.append(rec)

    fields = list(out[0].keys())
    with (OUT / "success_rates.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(out)

    show = [e for e in order_emb if e != "ECFP4 stereo-blind"]
    hdr = f"{'mode':26}" + "".join(f"{e[:12]:>14}" for e in show)
    print(f"SUCCESS RATE %  (eps={EPS_MAIN}; A: resolved, B: correctly NOT resolved)\n")
    print(hdr); print("-" * len(hdr))
    for (klass, mode) in order_mode:
        line = f"{('A ' if klass=='A' else 'B ')+mode:26}"
        for emb in show:
            rec = next((r for r in out if r["mode"] == mode and r["embedding"] == emb), None)
            line += f"{rec['success']:>13.0f}%" if rec else f"{'--':>14}"
        print(line)
    ctrl = [r for r in out if r["embedding"] == "ECFP4 stereo-blind"
            and r["mode"] in ("stereo_flip", "ez_flip")]
    print("\ncontrol, ECFP4 stereo-blind: " +
          ", ".join(f"{r['mode']} {r['success']:.0f}%" for r in ctrl))
    print(f"\nwrote {OUT/'success_rates.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
