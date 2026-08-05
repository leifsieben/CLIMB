"""Variance-components decomposition of the 8M principal arms (pretraining-seed / fold / head-seed).

Both reviewers asked for this: with 3 pretraining seeds x 5 scaffold folds x 3 head seeds = 45
per-cell metric values per (arm, task) — all already in the CV `_cell` rows — we can partition the
metric variance and, crucially, compare pretraining-seed spread to the between-regime gap. That
converts "we did not find a difference" into "the difference is smaller than the pretraining-seed
noise a proponent would invoke."

Writes analysis/rigor/variance_decomposition.csv (per arm x task) and a per-task decisive summary.
"""
from __future__ import annotations
import re, os
from pathlib import Path
import numpy as np, pandas as pd

OUT = Path("analysis/rigor"); OUT.mkdir(parents=True, exist_ok=True)
ARMS = {"unsup_only": "unsup_8M", "sup_only:dense": "skip_dense_8M",
        "sup_only:mixed": "skip_mixed_8M", "unsup->sup": "u2s_dense_from8M"}
TASKS = [("ESOL", "rmse"), ("QM7", "rmse"), ("BBBP", "roc_auc"),
         ("BACE", "roc_auc"), ("Tox21", "roc_auc"), ("HIV", "roc_auc")]


def cells(base, ds, metric):
    out = {}
    for p, run in enumerate([base, f"{base}_s1", f"{base}_s2"]):
        f = f"figure_data/climb_v2_phase2/{run}/moleculenet_cv/moleculenet_summary.csv"
        if not os.path.exists(f):
            continue
        d = pd.read_csv(f)
        d = d[(d.dataset == ds) & (d.main_metric == f"{metric}_cell")]
        for _, r in d.iterrows():
            m = re.match(r"s(\d+)_fold(\d+)", str(r.head_seed))
            if m:
                out[(p, int(m.group(2)), int(m.group(1)))] = float(r.main_value)
    return out


def decomp(c):
    ps = sorted({k[0] for k in c}); fs = sorted({k[1] for k in c}); hs = sorted({k[2] for k in c})
    Y = np.full((len(ps), len(fs), len(hs)), np.nan)
    for (p, f, h), v in c.items():
        Y[ps.index(p), fs.index(f), hs.index(h)] = v
    if np.isnan(Y).any() or Y.size == 0:
        return None
    g = Y.mean(); nP, nF, nH = Y.shape
    SSt = ((Y - g) ** 2).sum()
    SSp = nF * nH * ((Y.mean((1, 2)) - g) ** 2).sum()
    SSf = nP * nH * ((Y.mean((0, 2)) - g) ** 2).sum()
    SSh = nP * nF * ((Y.mean((0, 1)) - g) ** 2).sum()
    SSr = SSt - SSp - SSf - SSh
    return dict(mean=g, sd_pretrain=Y.mean((1, 2)).std(), sd_fold=Y.mean((0, 2)).std(),
                sd_head=Y.mean((0, 1)).std(), pct_pretrain=100 * SSp / SSt, pct_fold=100 * SSf / SSt,
                pct_head=100 * SSh / SSt, pct_resid=100 * SSr / SSt)


def main():
    rows, store = [], {}
    for arm, base in ARMS.items():
        for ds, met in TASKS:
            r = decomp(cells(base, ds, met))
            if not r:
                continue
            store[(arm, ds)] = r
            rows.append(dict(arm=arm, task=ds, metric=met, **{k: round(v, 5) for k, v in r.items()}))
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "variance_decomposition.csv", index=False)

    # per-task decisive comparison: pretrain-seed SD vs |unsup - dense| gap
    dec = []
    for ds, met in TASKS:
        sds = [store[(a, ds)]["sd_pretrain"] for a in ARMS if (a, ds) in store]
        if ("unsup_only", ds) in store and ("sup_only:dense", ds) in store:
            gap = abs(store[("unsup_only", ds)]["mean"] - store[("sup_only:dense", ds)]["mean"])
            msp = float(np.median(sds))
            dec.append(dict(task=ds, sigma_pretrain=round(msp, 5), unsup_minus_dense_gap=round(gap, 5),
                            reliable_on_one_seed=bool(gap >= 2 * msp)))
    pd.DataFrame(dec).to_csv(OUT / "variance_decision.csv", index=False)
    print(df.to_string(index=False))
    print("\nwrote analysis/rigor/variance_decomposition.csv + variance_decision.csv")


if __name__ == "__main__":
    main()
