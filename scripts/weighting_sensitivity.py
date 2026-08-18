"""Is the all-datasets ranking (Fig A1-wide) dominated by MoleculeACE?

Fig A1-wide averages a model's rank over 66 datasets, of which 30 are MoleculeACE targets and 28
are Polaris tasks. That is only a problem if (a) those datasets vote as a bloc, and (b) the final
ordering actually depends on the weighting. This script measures both, plus the price paid for the
per-dataset scheme: correlated datasets inflate the effective sample size and shrink the SE.

  1. within-suite agreement   mean pairwise Spearman between the model rankings of two datasets
                              from the same suite -> effective number of independent datasets
                              n_eff = n / (1 + (n-1) * rho_bar)
  2. weighting sensitivity    per-dataset vs per-suite vs leave-one-suite-out orderings, compared
                              by Kendall tau and by the largest single-model rank shift
  3. honest SE                SE recomputed on n_eff rather than n

Run:  python3 scripts/weighting_sensitivity.py
"""
from __future__ import annotations
import sys, itertools
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, kendalltau

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from figures.allsuites import wide_ranks, wide_table, SUITES   # noqa: E402
from figures.arms import ARMS, ARM_ORDER                       # noqa: E402

# same rule as the figure scripts: ignore arms registered ahead of their results
_S0, _ = wide_table(ARM_ORDER)
ARMS_USED = [a for a in ARM_ORDER if _S0.loc[a].notna().any()]


def mean_pairwise_spearman(R, cols, max_pairs=4000):
    """Mean Spearman correlation between the model-rankings of every pair of datasets."""
    cols = [c for c in cols if R[c].notna().sum() >= 8]
    pairs = list(itertools.combinations(cols, 2))
    if len(pairs) > max_pairs:
        pairs = [pairs[i] for i in np.linspace(0, len(pairs) - 1, max_pairs).astype(int)]
    vals = []
    for a, b in pairs:
        s = R[[a, b]].dropna()
        if len(s) >= 8:
            r = spearmanr(s[a], s[b]).statistic
            if np.isfinite(r):
                vals.append(r)
    return float(np.mean(vals)), len(cols), len(vals)


def main():
    out, R, M = wide_ranks(ARMS_USED, per_suite_equal=False)
    arms = list(out.index)

    print("=" * 78)
    print("1. Do the datasets inside a suite vote as a bloc?")
    print("=" * 78)
    print(f"{'suite':14s} {'n':>3s} {'mean pairwise rho':>18s} {'n_eff':>7s}  interpretation")
    neff = {}
    for s in SUITES:
        cols = [c for c in R.columns if M.loc[c, "suite"] == s]
        if len(cols) < 2:
            neff[s] = len(cols)
            print(f"{s:14s} {len(cols):3d} {'—':>18s} {len(cols):7.1f}  single dataset")
            continue
        rho, n, _ = mean_pairwise_spearman(R, cols)
        ne = n / (1 + (n - 1) * max(rho, 0))
        neff[s] = ne
        print(f"{s:14s} {n:3d} {rho:18.3f} {ne:7.1f}  "
              f"{n} datasets behave like ~{ne:.1f} independent ones")

    tot_n = sum(len([c for c in R.columns if M.loc[c, 'suite'] == s]) for s in SUITES)
    tot_eff = sum(neff.values())
    print(f"\nnominal datasets {tot_n}  ->  effective independent datasets ~{tot_eff:.1f}")
    print("share of the vote, nominal vs effective:")
    for s in SUITES:
        n = len([c for c in R.columns if M.loc[c, "suite"] == s])
        print(f"   {s:14s} {n/tot_n:6.1%}  ->  {neff[s]/tot_eff:6.1%}")

    print()
    print("=" * 78)
    print("2. Does the ordering actually change with the weighting?")
    print("=" * 78)
    schemes = {"per-dataset": out["mean_rank"],
               "per-suite": wide_ranks(ARMS_USED, per_suite_equal=True)[0]["mean_rank"]}
    for drop in SUITES:                       # leave-one-suite-out, per-dataset weighting
        cols = [c for c in R.columns if M.loc[c, "suite"] != drop]
        schemes[f"drop {drop}"] = R[cols].mean(axis=1)
    T = pd.DataFrame(schemes).loc[arms]
    print(f"{'model':38s} " + " ".join(f"{k[:11]:>11s}" for k in T.columns))
    for a in arms:
        print(f"{ARMS[a]['label']:38s} " + " ".join(f"{T.loc[a,k]:11.2f}" for k in T.columns))

    print(f"\nKendall tau of each scheme's ORDERING vs the per-dataset ordering:")
    base = T["per-dataset"].rank()
    for k in T.columns:
        tau = kendalltau(base, T[k].rank()).statistic
        shift = int((base - T[k].rank()).abs().max())
        print(f"   {k:22s} tau={tau:5.3f}   largest single-model position shift = {shift}")

    print()
    print("=" * 78)
    print("3. The SE the per-dataset scheme reports vs the honest one")
    print("=" * 78)
    print(f"{'model':38s} {'mean':>6s} {'SE (n=66)':>10s} {'SE (n_eff)':>11s}")
    infl = np.sqrt(tot_n / tot_eff)
    for a in arms[:6]:
        r = out.loc[a]
        print(f"{ARMS[a]['label']:38s} {r.mean_rank:6.2f} {r.se_rank:10.2f} "
              f"{r.se_rank * infl:11.2f}")
    print(f"\nSEs are understated by a factor of ~{infl:.1f} because correlated datasets are "
          f"counted as independent.")


if __name__ == "__main__":
    main()
