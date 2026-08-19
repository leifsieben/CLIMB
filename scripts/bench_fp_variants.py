"""Time the fingerprint variants AGAINST EACH OTHER, in one process, interleaved.

WHY THIS EXISTS RATHER THAN A SECOND bench_featurization.py RUN. The r3-counts row was first
measured in its own run and came out 7% FASTER than ECFP4 -- impossible, since radius-3 counts is
strictly more work than radius-2 bits. The tell was the RDKit descriptor row: identical work, timed
4.709 s in one run and 4.119 s in the other, so that machine was simply 14% quicker that afternoon.
Two bench runs on one laptop are not one experiment.

So both variants are timed HERE, in a single process, with the repeats INTERLEAVED (A B A B ...)
rather than blocked, which spreads any thermal or scheduler drift evenly across both arms instead
of loading it onto whichever ran second. The reported delta is then a property of the fingerprint,
not of the hour it was measured in.

Writes the two rows into figure_data/_bench/featurization_timing.json, carrying every other row
through untouched.

Run:  python3 scripts/bench_fp_variants.py [--repeats 7]
"""
from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
OUT = ROOT / "figure_data" / "_bench" / "featurization_timing.json"

# radius/counts passed EXPLICITLY, so one process can time both without touching FP_VARIANT --
# _fp_settings() resolves explicit kwargs over the env var.
VARIANTS = [("ecfp4",     dict(radius=2, counts=False, include_chirality=True)),
            ("ecfp4_r3c", dict(radius=3, counts=True,  include_chirality=True))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=7)
    ap.add_argument("--n", type=int, default=1000)
    a = ap.parse_args()

    import logging
    logging.disable(logging.INFO)          # the per-call variant banner would be 14 x n lines
    from featurize_v2 import ecfp4_features

    j = json.load(open(OUT))
    # THE SAME 1000 MOLECULES the published table was timed on -- same loader, same seed. A
    # different molecule sample would put a second uncontrolled variable next to the one being
    # measured, which is the mistake this script exists to correct.
    sys.path.insert(0, str(ROOT / "scripts"))
    from bench_featurization import load_smiles
    smiles = load_smiles(j["config"]["source"], a.n, j["config"]["seed"])
    assert len(smiles) == a.n, f"need {a.n} molecules, got {len(smiles)}"

    for name, kw in VARIANTS:                                   # warm-up, both arms
        ecfp4_features(smiles[:64], **kw)

    times = {name: [] for name, _ in VARIANTS}
    for i in range(a.repeats):
        for name, kw in VARIANTS:                               # INTERLEAVED, not blocked
            t0 = time.perf_counter()
            X = ecfp4_features(smiles, **kw)
            times[name].append(time.perf_counter() - t0)
            assert X.shape[0] == a.n
        print(f"  repeat {i+1}/{a.repeats}  " +
              "  ".join(f"{n}={times[n][-1]:.4f}s" for n, _ in VARIANTS))

    keep = [r for r in j["results"]
            if not (r["method"] == "ecfp4_r3c" and r["device"] == "cpu"
                    and r["notes"] == "single core")]
    rows = {}
    for name, kw in VARIANTS:
        t = np.array(times[name])
        rows[name] = dict(method=name, device="cpu", precision="n/a", notes="single core",
                          n_molecules=a.n, repeats=a.repeats, times_s=[round(x, 6) for x in t],
                          total_s_mean=round(float(t.mean()), 6),
                          total_s_sd=round(float(t.std(ddof=1)), 6),
                          ms_per_mol=round(float(t.mean()) / a.n * 1000, 6),
                          mol_per_s=round(a.n / float(t.mean()), 3),
                          hours_1M=round(float(t.mean()) * 1e6 / a.n / 3600, 6),
                          hours_1B=round(float(t.mean()) * 1e9 / a.n / 3600, 3),
                          fp_settings=kw,
                          paired_with="ecfp4" if name == "ecfp4_r3c" else "ecfp4_r3c",
                          note="interleaved same-process A/B (scripts/bench_fp_variants.py)")
    # The existing ecfp4 row stays the published one; this run's ecfp4 is the PAIRED control and is
    # kept only so the delta can be read from two rows measured under one machine state.
    keep = [r for r in keep
            if not (r["method"] == "ecfp4" and r["notes"] == "single core (paired A/B)")]
    rows["ecfp4"]["notes"] = "single core (paired A/B)"
    j["results"] = keep + [rows["ecfp4"], rows["ecfp4_r3c"]]
    json.dump(j, open(OUT, "w"), indent=2)

    e2, e3 = np.array(times["ecfp4"]).mean(), np.array(times["ecfp4_r3c"]).mean()
    print(f"\n  ECFP4 (r=2, bits)      {e2:.4f} s / {a.n}")
    print(f"  R3FP (r=3, counts)     {e3:.4f} s / {a.n}   ({e3/e2:.2f}x)")
    print(f"\n  wrote {OUT}")


if __name__ == "__main__":
    main()
