"""How much of an arm's MolNet number is the PRETRAINING SEED?

WHY THIS EXISTS. Every CLIMB arm in fig A1 rests on 3 pretraining seeds. The two end-to-end arms
(unsup_8M_e2e, skip_dense_8M_e2e) rest on 3 FINE-TUNE seeds against ONE pretrained encoder on the
MoleculeACE and Polaris tracks -- a different axis, and on the frozen arm the pretraining axis
carries 2.31x the head-seed spread (30 MoleculeACE targets, median SD 0.0204 vs 0.0085). So a
single-pretraining-seed suite cell is the larger unknown, and the question is whether it is small
enough to declare or has to be closed with ~700 more fine-tunes.

WHAT MAKES THIS LIKE-FOR-LIKE. The obvious comparison -- frozen SD on MoleculeACE against e2e SD on
MolNet -- is not one: different suites, metrics and task counts. This script instead computes the
pretraining-seed SD for BOTH probes on the SAME MolNet datasets with the SAME metric, so the
frozen-to-e2e RATIO is measured rather than assumed. That ratio applied to the frozen arm's
MoleculeACE SD is the estimate for the e2e arm there, and the transfer assumption is stated rather
than hidden: it assumes the probe changes seed sensitivity by a similar factor on both suites.

Each seed dir's cell is the MEAN row over its own folds, so the SD reported is across pretraining
seeds with fold noise already averaged down inside each -- the same construction as the 0.0204.

METRIC MUST MATCH ON BOTH SIDES, and this is not a formality. The first datapoint (HIV, from the
peer 2026-08-19) was read as a ratio of ~1.55 -- above 1, the side that costs ~700 fine-tunes --
by comparing the e2e arm's nef1 spread against the frozen arm's roc_auc spread. Matched on metric
the same runs give 0.56 (unsup, roc_auc), 0.67 (unsup, nef1), 0.92 and 0.97 (sup:dense), i.e. every
one below 1 and the opposite conclusion. nef1 relative SD runs ~2.3x roc_auc's on the same
predictions -- it is a rank statistic on a handful of actives and it is quantised -- so a
cross-metric ratio measures the metric, not the probe. This script reads main_metric from the row
it uses and pairs like with like.

Run:  python3 scripts/pretrain_seed_variance.py
"""
from __future__ import annotations
import csv, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from figures.sixpanel import NATIVE_SUBDIRS

FD = ROOT / "figure_data" / "climb_v2_phase2"
# DENOMINATOR FLOOR, on the peer's suggestion (2026-08-19) and set as a rule rather than case by
# case after seeing the numbers. Below this relative SD the frozen denominator is comparable to
# fold-level rounding, and a ratio like 1.64%/0.07% = 23x is finite, meaningless, and looks like a
# result. Datasets under the floor are REPORTED AS SKIPPED with their value, never silently
# dropped -- a quietly shorter dataset list is how a ratio gets computed on whichever panels
# happened to be noisy.
#
# Honest note on when this was chosen: HIV's e2e numerators were already visible (peer's message)
# when the floor was set. It cannot have been tuned to HIV -- both HIV denominators (1.14%, 0.49%)
# clear any floor in this range -- but the other five datasets' numerators were not yet known.
MIN_DENOM_REL = 0.0025   # 0.25% of the mean

ARMS = {"unsup frozen":   ["unsup_8M", "unsup_8M_s1", "unsup_8M_s2"],
        "unsup e2e":      ["unsup_8M_e2e", "unsup_8M_e2e_s1", "unsup_8M_e2e_s2"],
        "sup:dense froz": ["skip_dense_8M", "skip_dense_8M_s1", "skip_dense_8M_s2"],
        "sup:dense e2e":  ["skip_dense_8M_e2e", "skip_dense_8M_e2e_s1", "skip_dense_8M_e2e_s2"]}


def _cells(run, ds):
    """{metric: value} for `ds` in `run` -- EVERY MEAN row, not the first one.

    HIV and the other classification sets carry a roc_auc MEAN and a nef1 MEAN in the same file.
    Returning "the MEAN row" takes whichever is written first, which is a positional metric read
    and the exact defect that produced the 1.55-vs-0.56 disagreement this script exists to settle.
    Keying by metric makes the matching structural instead of a convention to remember.
    """
    for sub in NATIVE_SUBDIRS.get(ds, ("moleculenet_cv",)):
        f = FD / run / sub / "moleculenet_summary.csv"
        if not f.exists():
            continue
        got = {r["main_metric"]: float(r["main_value"]) for r in csv.DictReader(f.open())
               if r["dataset"] == ds and r["head_seed"] == "MEAN"}
        if got:
            return got
    return {}


def main():
    datasets = sorted({r["dataset"] for run in sum(ARMS.values(), [])
                       for sub in ("moleculenet_cv",)
                       if (FD / run / sub / "moleculenet_summary.csv").exists()
                       for r in csv.DictReader((FD / run / sub / "moleculenet_summary.csv").open())})
    out = {}
    for arm, runs in ARMS.items():
        print(f"\n=== {arm} ===")
        present = [r for r in runs if (FD / r).exists()]
        if len(present) < 2:
            print(f"  only {len(present)} of 3 pretraining-seed dirs on disk "
                  f"({', '.join(present) or 'none'}) -- SD not computable")
            continue
        if len(present) < len(runs):
            print(f"  WARNING {len(present)} of 3 seed dirs; SD is on fewer seeds than the others")
        for ds in datasets:
            per_run = [_cells(r, ds) for r in present]
            metrics = sorted(set.intersection(*[set(c) for c in per_run]) if all(per_run) else [])
            for m in metrics:
                vals = np.array([c[m] for c in per_run], dtype=float)
                sd = float(vals.std(ddof=1))
                out.setdefault(arm, {})[(ds, m)] = (sd, sd / abs(vals.mean()))
                print(f"  {ds:<15} {m:<8} n={len(vals)}  mean {vals.mean():9.4f}  "
                      f"SD {sd:8.4f}  ({100 * sd / abs(vals.mean()):.2f}% of mean)")

    for probe_f, probe_e in (("unsup frozen", "unsup e2e"), ("sup:dense froz", "sup:dense e2e")):
        if probe_f in out and probe_e in out:
            shared = sorted(set(out[probe_f]) & set(out[probe_e]))
            if shared:
                # ratio of RELATIVE SDs, so datasets on different metrics can be pooled
                usable = [d for d in shared if out[probe_f][d][1] >= MIN_DENOM_REL]
                for d in sorted(set(shared) - set(usable)):
                    print(f"  SKIPPED {d[0]}/{d[1]}: frozen SD {100 * out[probe_f][d][1]:.2f}% is below the "
                          f"{100 * MIN_DENOM_REL:.2f}% floor -- ratio not well conditioned")
                if not usable:
                    print(f"\n{probe_e} / {probe_f}: every dataset below the denominator floor; "
                          f"no ratio reported.")
                    continue
                shared = usable
                rat = [out[probe_e][d][1] / out[probe_f][d][1] for d in shared]
                print(f"\n{probe_e} / {probe_f} pretraining-seed SD ratio over {len(shared)} "
                      f"MolNet dataset x metric cells: median {np.median(rat):.2f}  (per-dataset "
                      f"{', '.join(f'{d[0]}/{d[1]} {r:.2f}' for d, r in zip(shared, rat))})")
                print("  >1 means end-to-end AMPLIFIES pretraining-seed spread; <1 means it damps it.")
                print(f"  Frozen arm's MoleculeACE SD is 0.0204 (30 targets); x{np.median(rat):.2f} "
                      f"=> {0.0204 * np.median(rat):.4f} estimated for the e2e arm there.")


if __name__ == "__main__":
    main()
