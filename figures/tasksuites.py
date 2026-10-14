from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd

from figures import allsuites as A

ROOT = Path(__file__).resolve().parent.parent

CATEGORIES = ["Activity cliffs", "Virtual screening", "Classification", "Regression"]
CAT_SHORT = {"Activity cliffs": "cliffs", "Virtual screening": "screen",
             "Classification": "class.", "Regression": "regr."}
CAT_MARKER = {"Activity cliffs": "^", "Virtual screening": "D",
              "Classification": "s", "Regression": "o"}


PENDING_DATASETS = {}


VS_BY_NAME = {"MolNet:HIV"}


def category_of(key, meta):
    """Task category for one dataset key, derived from the manifest and the metric."""
    if key.startswith("MolACE:"):
        return "Activity cliffs"
    if key.split(":")[0] in ("CBS", "Wong") or key in VS_BY_NAME:
        return "Virtual screening"
    if key.startswith("Polaris:"):
        man = _polaris_types()
        name = key.split(":", 1)[1]
        hit = [v for k, v in man.items() if k.split("/")[-1] == name]
        assert len(hit) == 1, (
            f"{key}: {len(hit)} manifest entries match. The Polaris category comes from the "
            f"manifest's own `type` field; an unmatched key would otherwise be silently binned.")
        return "Regression" if hit[0] == "regression" else "Classification"
    if key.startswith("FartDB:"):
        return "Classification"
    if key.startswith("MolNet:"):
        return "Classification" if meta.loc[key, "higher_better"] else "Regression"
    raise KeyError(f"{key}: no task category rule. Add one rather than defaulting -- a dataset "
                   f"binned by accident changes a category mean without changing any score.")


def _polaris_types():
    man = json.load(open(ROOT / "benchmarks/data/polaris/polaris_manifest.json"))
    return {k: v["type"] for k, v in man.items()}


EXCLUDED_DATASETS = {"MolNet:BBBP",
                     "Polaris:pkis2-ret-wt-cls-v2", "Polaris:pkis2-kit-wt-cls-v2"}


def wide_ranks(arms=None, summary="mean"):

    S, M = A.wide_table(arms)


    hit = sorted(set(S.columns) & EXCLUDED_DATASETS)
    if hit:
        S = S.drop(columns=hit)
        M = M.drop(index=hit)
    cat = pd.Series({c: category_of(c, M) for c in S.columns})
    unknown = sorted(set(cat) - set(CATEGORIES))
    assert not unknown, f"category_of returned {unknown}, not in CATEGORIES"

    N = len(S)
    R = pd.DataFrame(index=S.index, columns=S.columns, dtype=float)
    for c in S.columns:
        col = S[c].dropna()
        if len(col) < 2:
            continue
        r = col.rank(ascending=not M.loc[c, "higher_better"])
        R.loc[r.index, c] = 1 + (N - 1) * (r - 1) / (len(col) - 1)

    out = pd.DataFrame(index=S.index)
    for k in CATEGORIES:
        cols = [c for c in S.columns if cat[c] == k]
        out[k] = (R[cols].median(axis=1) if summary == "median" else R[cols].mean(axis=1)) \
                 if cols else np.nan
        out[k + "_n"] = R[cols].notna().sum(axis=1) if cols else 0


    out["mean_rank"] = out[CATEGORIES].mean(axis=1)
    n_cat = out[CATEGORIES].notna().sum(axis=1)
    out["se_rank"] = out[CATEGORIES].std(axis=1, ddof=1) / np.sqrt(n_cat.clip(lower=1))
    out["n_units"] = n_cat
    out["n_datasets"] = R.notna().sum(axis=1)


    Mc = M.copy(); Mc["suite"] = [cat[c] for c in M.index]
    _saved, A.SUITES = A.SUITES, CATEGORIES
    try:
        ne = A.effective_n(R, Mc)
    finally:
        A.SUITES = _saved
    deff = out["n_units"] / max(sum(ne.values()), 1e-9)
    out["se_rank_naive"] = out["se_rank"]
    out["se_rank"] = out["se_rank"] * np.sqrt(deff.clip(lower=1.0))


    n_eff_ds = max(sum(ne.values()), 1e-9)
    out["mean_rank_pooled"] = R.mean(axis=1)
    out["se_rank_pooled"] = R.std(axis=1, ddof=1) / np.sqrt(n_eff_ds)
    out.attrs["n_eff_datasets"] = n_eff_ds

    out.attrs["effective_n"] = ne
    out.attrs["n_datasets_total"] = len(S.columns)
    out.attrs["categories"] = {k: int((cat == k).sum()) for k in CATEGORIES}
    return out, cat, R


def coverage(arms=None):
    """{arm: {category: (n_scored, n_available)}} -- what is measured and what is missing."""
    out, cat, R = wide_ranks(arms)
    avail = {k: int((cat == k).sum()) for k in CATEGORIES}
    return {a: {k: (int(out.loc[a, k + "_n"]), avail[k]) for k in CATEGORIES} for a in out.index}
