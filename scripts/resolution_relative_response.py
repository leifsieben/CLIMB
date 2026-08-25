"""Aggregate per-pair effect sizes into the table fig_G draws, WITH dispersion.

This file did not exist. `relative_response_figure.csv` was assembled by hand, which meant the
figure's only input had no generator in the repo and could not be regenerated or diffed. It is the
same provenance gap `pairs_canonical.csv` had, and both are now closed.

THE UNIT. `effect` (scripts/resolution_effect_size.py) is the RMS over live dimensions of
(e(A) - e(B)) / sigma_j, sigma_j estimated once per representation on BACKGROUND_N unedited
molecules that appear in no reported pair. That number is not comparable across architectures on
its own, so every cell is divided by the SAME MODEL'S response to a matched-molecular-weight
substitution -- a completely different compound of near-identical mass and Tanimoto < 0.30:

    relative_response(pair) = effect(pair) / median_over_matched_mw_pairs(effect)

1.00 therefore reads "this edit moves the representation as far as swapping in a different
molecule", and a 512-d transformer and a 2048-bit fingerprint sit on one axis honestly. There is no
threshold anywhere. A count of dimensions displaced past a threshold was considered and rejected:
measured on this data it tracks how DENSE a representation is rather than how well it resolves the
edit (ECFP4 moves 28 of 2048 bits for a completely different compound, CLIMB 418 of 512), and it is
threshold-critical exactly where the CLM arms live (median count for one added methyl: 140 at
sigma/3, 56 at sigma/2, 1 at sigma). See notes/figG-resolution-metric.md.

THE REFERENCE IS ONE SCALAR PER MODEL, taken on CANONICAL input for both classes, because it is a
property of the representation's scale rather than of the edit. Class B numerators are measured as
written -- that IS the notation question -- but they are expressed against the same yardstick as
class A so the panels can be read side by side.

WHAT DISPERSION MEANS HERE. Two different intervals, and they answer different questions:

  q1/q3   -- the interquartile range of relative_response ACROSS PAIRS. This is chemistry, not
             noise: an inverted stereocentre on a rigid ring is not the same edit as one on a
             flexible chain. It is what the figure's whiskers draw.
  ci_lo/ci_hi -- a percentile bootstrap of the MEDIAN, resampling the numerator pairs and the
             matched-MW reference pairs independently so the reference's own uncertainty is
             carried rather than assumed away. This is what a caption should quote.

ARM-vs-ARM CLAIMS USE THE PAIRED FORM, in resolution_contrasts.csv. Every arm sees the identical
pair list, so the difference between two arms is a per-pair quantity and pair difficulty cancels.
The marginal IQRs overlap for contrasts that the paired test separates cleanly, so reading the
whiskers as an arm-vs-arm test would understate what the data supports.

Run:  python3 scripts/resolution_relative_response.py
"""
from __future__ import annotations
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "figure_data/embedding_resolution"

# embedding name -> the `short` key figures/fig_G.py selects on. Anything not listed is still
# written to the CSV with short = "", so adding an arm here is a labelling decision, not a
# filtering one -- a table that silently dropped arms would look complete.
SHORT = {"ECFP4+stereo": "ECFP", "ECFP4+desc": "ECFP+d",
         "Morgan r3-counts": "r3fp", "Morgan r3-cnt+desc": "r3fp+d",
         "CLIMB unsup (enum-aug)": "uns-ENUM", "CLIMB unsup (canon-ctrl)": "uns-CANON",
         "CLIMB sup": "sup", "CLIMB unsup": "uns-MAIN",
         "random encoder": "rand", "ECFP4 stereo-blind": "ECFP-blind"}

REFERENCE_MODE = "matched_mw"
N_BOOT = 2000
SEED = 0

# Contrasts the paper actually asserts. Each is (arm_a, arm_b): reported as a - b, per pair.
CONTRASTS = [("CLIMB unsup (enum-aug)", "CLIMB unsup (canon-ctrl)"),
             ("Morgan r3-counts", "ECFP4+stereo"),
             ("ECFP4+desc", "ECFP4+stereo"),
             ("Morgan r3-cnt+desc", "Morgan r3-counts")]


def load(suffix):
    """{(embedding, mode): {pair_id: effect}} for one input convention."""
    p = OUT / f"effect_sizes{suffix}.csv"
    assert p.exists(), f"{p} missing -- run scripts/resolution_effect_size.py first"
    d = defaultdict(dict)
    klass = {}
    for r in csv.DictReader(p.open()):
        d[(r["embedding"], r["mode"])][r["pair_id"]] = float(r["effect"])
        klass[r["mode"]] = r["klass"]
    return d, klass


def main() -> int:
    canon, klass_c = load("_canonical")
    aswr, klass_a = load("")
    klass = {**klass_a, **klass_c}

    embeddings = sorted({e for e, _ in canon} | {e for e, _ in aswr})
    # The reference is taken on CANONICAL input for every arm: matched_mw is a class-A mode, and a
    # yardstick that changed with the input convention would make the two classes incomparable.
    ref = {}
    for e in embeddings:
        v = list(canon.get((e, REFERENCE_MODE), {}).values())
        assert v, f"{e}: no {REFERENCE_MODE} pairs on canonical input -- nothing to scale by"
        ref[e] = float(np.median(v))
        assert ref[e] > 0, f"{e}: {REFERENCE_MODE} reference is {ref[e]}, cannot divide by it"

    rng = np.random.default_rng(SEED)
    rows = []
    for e in embeddings:
        refs = np.array(list(canon[(e, REFERENCE_MODE)].values()))
        for src, tag in ((canon, "canonical"), (aswr, "as_written")):
            for (emb, mode), pairs in sorted(src.items()):
                if emb != e:
                    continue
                v = np.array(list(pairs.values()))
                rel = v / ref[e]
                # joint bootstrap: the reference is a median over a finite pair set too
                bs = np.array([np.median(rng.choice(v, v.size)) /
                               np.median(rng.choice(refs, refs.size)) for _ in range(N_BOOT)])
                q1, q3 = np.percentile(rel, [25, 75])
                lo, hi = np.percentile(bs, [2.5, 97.5])
                rows.append(dict(input=tag, klass=klass[mode], mode=mode, embedding=e,
                                 short=SHORT.get(e, ""),
                                 relative_response=round(float(np.median(rel)), 4),
                                 q1=round(float(q1), 4), q3=round(float(q3), 4),
                                 ci_lo=round(float(lo), 4), ci_hi=round(float(hi), 4),
                                 n=int(v.size), reference_matched_mw=round(ref[e], 4)))

    fields = ["input", "klass", "mode", "embedding", "short", "relative_response",
              "q1", "q3", "ci_lo", "ci_hi", "n", "reference_matched_mw"]
    with (OUT / "relative_response_figure.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
    print(f"wrote {OUT/'relative_response_figure.csv'}: {len(rows)} rows "
          f"({len(embeddings)} arms x {len({r['mode'] for r in rows})} modes x 2 inputs)")

    # ---- paired arm-vs-arm contrasts -----------------------------------------------------------
    crows = []
    for a, b in CONTRASTS:
        for src, tag in ((canon, "canonical"), (aswr, "as_written")):
            for mode in sorted({m for (e, m) in src if e == a} & {m for (e, m) in src if e == b}):
                pa, pb = src[(a, mode)], src[(b, mode)]
                ids = sorted(set(pa) & set(pb))
                assert len(ids) == len(pa) == len(pb), (
                    f"{a} vs {b} on {mode}: {len(pa)} and {len(pb)} pairs, {len(ids)} shared. The "
                    f"paired test requires the identical pair list; unequal sets mean one arm was "
                    f"scored on a different pair file.")
                d = np.array([pa[i] / ref[a] - pb[i] / ref[b] for i in ids])
                se = float(d.std(ddof=1) / np.sqrt(d.size))
                crows.append(dict(input=tag, klass=klass[mode], mode=mode, arm_a=a, arm_b=b,
                                  mean_diff=round(float(d.mean()), 4), se=round(se, 4),
                                  t=round(float(d.mean() / se), 2) if se > 0 else float("inf"),
                                  n=int(d.size)))
    with (OUT / "resolution_contrasts.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["input", "klass", "mode", "arm_a", "arm_b",
                                          "mean_diff", "se", "t", "n"])
        w.writeheader(); w.writerows(crows)
    print(f"wrote {OUT/'resolution_contrasts.csv'}: {len(crows)} paired contrasts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
