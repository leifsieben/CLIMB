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
ARMS = {"unsup frozen":   ["unsup_8M", "unsup_8M_s1", "unsup_8M_s2"],
        "unsup e2e":      ["unsup_8M_e2e", "unsup_8M_e2e_s1", "unsup_8M_e2e_s2"],
        "sup:dense froz": ["skip_dense_8M", "skip_dense_8M_s1", "skip_dense_8M_s2"],
        "sup:dense e2e":  ["skip_dense_8M_e2e", "skip_dense_8M_e2e_s1", "skip_dense_8M_e2e_s2"]}


def _cell(run, ds):
    """That run's value for `ds`: the MEAN row of the first subdir that has one."""
    for sub in NATIVE_SUBDIRS.get(ds, ("moleculenet_cv",)):
        f = FD / run / sub / "moleculenet_summary.csv"
        if not f.exists():
            continue
        for r in csv.DictReader(f.open()):
            if r["dataset"] == ds and r["head_seed"] == "MEAN":
                return float(r["main_value"]), r["main_metric"], sub
    return None


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
            cells = [c for c in (_cell(r, ds) for r in present) if c is not None]
            if len(cells) < 2:
                continue
            vals = np.array([c[0] for c in cells])
            sd = float(vals.std(ddof=1))
            out.setdefault(arm, {})[ds] = (sd, sd / abs(vals.mean()))
            print(f"  {ds:<15} {cells[0][1]:<8} n={len(vals)}  mean {vals.mean():9.4f}  "
                  f"SD {sd:8.4f}  ({100 * sd / abs(vals.mean()):.2f}% of mean)")

    for probe_f, probe_e in (("unsup frozen", "unsup e2e"), ("sup:dense froz", "sup:dense e2e")):
        if probe_f in out and probe_e in out:
            shared = sorted(set(out[probe_f]) & set(out[probe_e]))
            if shared:
                # ratio of RELATIVE SDs, so datasets on different metrics can be pooled
                rat = [out[probe_e][d][1] / out[probe_f][d][1] for d in shared
                       if out[probe_f][d][1] > 0]
                print(f"\n{probe_e} / {probe_f} pretraining-seed SD ratio over {len(shared)} "
                      f"MolNet datasets: median {np.median(rat):.2f}  (per-dataset "
                      f"{', '.join(f'{d} {r:.2f}' for d, r in zip(shared, rat))})")
                print("  >1 means end-to-end AMPLIFIES pretraining-seed spread; <1 means it damps it.")
                print(f"  Frozen arm's MoleculeACE SD is 0.0204 (30 targets); x{np.median(rat):.2f} "
                      f"=> {0.0204 * np.median(rat):.4f} estimated for the e2e arm there.")


if __name__ == "__main__":
    main()
