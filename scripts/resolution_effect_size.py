"""Magnitude of a chemical change, in units of the embedding's own per-dimension spread.

Why this replaces the noise-calibrated test. That test used SMILES re-writing as the null, and it
is not a fair null for a sequence model: non-invariance to how a SMILES is written is a known
property of CLMs, and in the actual pipeline every molecule is embedded from its CANONICAL SMILES,
so that variation never occurs at inference. Using it as the null charged the CLMs for something
they are never asked to do, and drove every class-A cell to 0%.

Under canonical input every representation here is DETERMINISTIC -- re-embedding reproduces the
vectors bit-for-bit, 128/128 (resolution_noise_floor.py). There is no noise. So "resolved" is
simply whether the vectors differ, and the interesting question is not significance but SIZE.

effect = RMS over dimensions of (e(A) - e(B)) / sigma_d,  sigma_d = that dimension's SD over BACKGROUND_N
background molecules (10,000). effect = 1.0 means the two molecules differ, per dimension, by as much as
two molecules typically do. 0.01 means the change is a hundredth of the space's natural scale.

SMILES non-invariance stays in the figure, as the class B RESULT it always was, rather than being
promoted to the yardstick everything else is judged against.
"""
from __future__ import annotations
import csv, json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "figure_data/embedding_resolution"
EMB_DIR = OUT / "embeddings"
import os
CANON = os.environ.get("RESOLUTION_INPUT") == "canonical"
SUFFIX = "_canonical" if CANON else ""
# CheMeleon is RETIRED (figures.arms.RETIRED) and is not listed: its stored npz covers only the
# superseded 100-pair molecule set, and re-embedding needs a chemprop>=2.2 host. The npz files are
# kept on disk rather than deleted.
NAMES = {"ECFP4+stereo": "ECFP4_stereo", "ECFP4+desc": "ECFP4_desc",
         "Morgan r3-counts": "Morgan_r3-counts", "Morgan r3-cnt+desc": "Morgan_r3-cnt_desc",
         "CLIMB sup": "CLIMB_sup", "CLIMB unsup": "CLIMB_unsup",
         "CLIMB unsup (enum-aug)": "CLIMB_unsup_(enum-aug)",
         "CLIMB unsup (canon-ctrl)": "CLIMB_unsup_(canon-ctrl)",
         "random encoder": "random_encoder", "ECFP4 stereo-blind": "ECFP4_stereo-blind"}


def main() -> int:
    pairs = list(csv.DictReader((OUT / ("pairs" + SUFFIX + ".csv")).open()))
    bg = set((OUT / ("molecules" + SUFFIX + ".txt")).read_text().split("\n"))
    paired = {r["smiles_a"] for r in pairs} | {r["smiles_b"] for r in pairs}
    bg = sorted(bg - paired)          # background = molecules not in any reported pair
    rows = []
    for label, fname in NAMES.items():
        p = EMB_DIR / f"{fname}{SUFFIX}.npz"
        if not p.exists():
            continue
        z = np.load(p, allow_pickle=True)
        idx = {str(s): i for i, s in enumerate(z["smiles"])}
        X = z["X"].astype(np.float64)
        # A stale npz from an earlier, smaller pair set covers some molecules and not others, and
        # the per-pair lookup below would raise on the first miss halfway through. Check coverage
        # up front and skip the whole arm, so a stale file is a named skip rather than a crash.
        uncovered = [r for r in pairs if r["smiles_a"] not in idx or r["smiles_b"] not in idx]
        if uncovered:
            print(f"{label:22} SKIPPED: {len(uncovered)}/{len(pairs)} pairs absent from "
                  f"{p.name} -- stale embedding file, re-run embed_resolution_pairs.py", flush=True)
            continue
        sigma = X[[idx[s] for s in bg if s in idx]].std(axis=0)
        sigma[sigma == 0] = np.nan            # dead dimensions carry no information; exclude them
        live = np.isfinite(sigma)
        for r in pairs:
            d = (X[idx[r["smiles_a"]]] - X[idx[r["smiles_b"]]])[live] / sigma[live]
            rows.append(dict(embedding=label, mode=r["mode"], klass=r["klass"],
                             pair_id=r["pair_id"], effect=float(np.sqrt(np.mean(d ** 2))),
                             n_live_dims=int(live.sum())))
        print(f"{label:22} {int(live.sum())}/{len(sigma)} live dimensions", flush=True)
    with (OUT / f"effect_sizes{SUFFIX}.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["embedding", "mode", "klass", "pair_id", "effect",
                                          "n_live_dims"])
        w.writeheader(); w.writerows(rows)
    print(f"\nwrote {OUT}/effect_sizes{SUFFIX}.csv: {len(rows)} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
