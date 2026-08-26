"""The FOUR TASK-TYPE categories fig_A ranks over, and the ranking computed across them.

figures/allsuites.py groups the same 65 datasets by BENCHMARK (MoleculeNet / MoleculeACE /
Polaris / CBS). fig_A groups them by WHAT THE TASK IS -- activity cliffs, virtual screening,
classification, regression -- because "which benchmark did it come from" is a fact about
provenance and "what kind of problem is it" is the axis the paper's claim is about. Both
groupings run off the same score matrix (allsuites.wide_table), so a number cannot differ
between them; only the pooling differs.

allsuites is NOT modified: it feeds fig_A1, fig_A2 and both audit scripts, and its four-benchmark
SUITES list is load-bearing there.

EQUAL WEIGHT PER CATEGORY, NOT PER DATASET. Activity cliffs holds 30 datasets and virtual
screening 3, so each VS dataset carries ~8.3% of the headline against ~0.83% for each MoleculeACE
target. That is deliberate -- the axis is task type -- but it is not self-evident and the caption
has to say it.

Category assignment is DERIVED, never hand-listed per dataset:
  MolACE:*    -> Activity cliffs   (that is what the benchmark is: matched pairs across a cliff)
  CBS:*, Wong:*, MolNet:HIV -> Virtual screening. The suite is CBS + HIV + Wong
                (scripts/export_figA_smiles.py); HIV is a rare-active screen, 1.35% positive, and
                a metric-direction rule bins it as ordinary classification because it is scored by
                ROC-AUC. Named explicitly for that reason.
  Polaris:*   -> the manifest's own `type` field, regression or classification
  MolNet:*    -> by metric direction: an error metric is regression, an AUC is classification
Deriving it means a dataset added to any tree lands in a category without an edit here, and a
dataset whose metric changes cannot keep a stale category label.
"""
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

# Datasets commissioned for this figure that have not landed locally yet. Named here so the
# coverage report can say "3 of 4 categories, and the missing one is WAITING rather than absent",
# which is a different statement from "this arm scored nothing".
# EMPTY as of 2026-08-26: Wong and FartDB landed, so virtual screening is CBS + HIV + Wong (3) and
# classification is 15. The mechanism stays -- it is how a commissioned dataset shows as owed in
# the legend rather than being invisible until it arrives.
PENDING_DATASETS = {}

# Datasets that live in a benchmark tree whose default rule would bin them elsewhere. HIV is
# scored by ROC-AUC like every other MoleculeNet classification set, so the metric rule calls it
# classification -- but it is a 1.35%-positive rare-active screen and it is the third member of
# the virtual-screening suite alongside CBS and Wong.
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
    man = json.load(open(ROOT / "chemeleon_suite/data/polaris/polaris_manifest.json"))
    return {k: v["type"] for k, v in man.items()}


# DATASETS EXCLUDED FROM THE RANKING (Leif 2026-08-26: "drop Molnet BBBP").
#
# MolNet:BBBP is not merely noisy, it is mis-curated, and we verified that on our own copy rather
# than taking it on authority. Of its 2,039 molecules, 60 canonical structures appear more than
# once and TEN of those pairs carry CONTRADICTORY labels -- aspirin, levodopa, miconazole and
# methylphenidate are each labelled both brain-penetrant and non-penetrant. That matches the count
# Pat Walters reports, independently arrived at.
#
# THE STRONGER ARGUMENT IS THAT IT DOES NOT DISCRIMINATE. A randomly initialised encoder places
# 8th of 14 on it, ahead of ChemBERTa-2, SELFIES-TED and both fingerprint anchors; across the
# classification datasets, an arm's rank here correlates with the RANDOM encoder's rank at
# rho = -0.65 (p = 0.008). The field packs into 0.0737 ROC-AUC against a between-fold SD of
# 0.0354, so the ordering is largely resolved by noise and then charged as full ranks.
#
# NOTHING IS LOST BY DROPPING IT. Polaris:bbb-martins is the same endpoint from the same Martins
# source in a better curation -- every one of its 394 test molecules is also in MolNet:BBBP -- and
# ECFP4+desc places FIRST there and last here. Keeping both counted one endpoint twice.
#
# Measured cost to the headline: nothing reorders, and the largest change to any arm is 0.15.
#
# ---------------------------------------------------------------------------------------------
# THE THRESHOLDED PKIS2 TASKS: the SAME MOLECULES counted twice, in two categories.
#
# Leif 2026-08-26: "let's drop the pkis2-ret because it shouldn't be in regression and
# classification." Verified before acting -- pkis2-ret-wt-cls-v2 and pkis2-ret-wt-reg-v2 hold the
# IDENTICAL 106 test molecules, and pkis2-kit the identical 116. One is the other thresholded. So
# each target was contributing a Classification rank and a Regression rank from one experiment.
#
# THE RULE IS APPLIED TO KIT AS WELL AS RET, and that matters for a reason beyond consistency.
# ECFP4+desc places 11th on ret-cls and 2nd on kit-cls, so dropping only the one Leif named would
# have removed the dataset where our own anchor looks worst and kept the one where it looks good.
# The rule as stated -- drop the thresholded variant wherever the continuous variant exists on the
# same molecules -- removes one of each and cannot be read as choosing the outcome. pkis2-egfr has
# only a regression variant and is untouched.
#
# THE CONTINUOUS VARIANT IS THE ONE KEPT, on the evidence rather than by preference: the two agree
# at rho +0.73 (ret) and +0.54 (kit) about which arm is better, well above the ~0.45 typical of
# two unrelated classification sets, so the pair is internally consistent and the thresholded copy
# adds no information its parent lacks -- it only adds PR-AUC's small-sample fragility on 106 and
# 116 molecules.
EXCLUDED_DATASETS = {"MolNet:BBBP",
                     "Polaris:pkis2-ret-wt-cls-v2", "Polaris:pkis2-kit-wt-cls-v2"}


def wide_ranks(arms=None, summary="mean"):
    """Per-arm mean rank within each task category, and the mean of the four.

    Ranks are computed PER DATASET over the arms present on it and rescaled to the full field
    (1 .. N), so a dataset scored on only k of N arms cannot flatter the k. Category summaries are
    then averaged with EQUAL WEIGHT, and the interval is the spread across those four numbers
    inflated by the design effect -- 30 near-duplicate MoleculeACE targets do not buy sqrt(30)
    worth of precision.
    """
    S, M = A.wide_table(arms)
    # Report the exclusion rather than performing it silently: a filter that matches nothing looks
    # exactly like one that matches everything, and both read as success.
    hit = sorted(set(S.columns) & EXCLUDED_DATASETS)
    miss = sorted(EXCLUDED_DATASETS - set(S.columns))
    assert not miss, f"EXCLUDED_DATASETS names datasets not in the table: {miss}"
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
    # `summary` sets how a CATEGORY is summarised from its datasets; the four categories are
    # always averaged, because they are four numbers and equally weighted by construction.
    #
    # WHY THE CHOICE MATTERS. Mean rank is not robust on a dataset where the whole field sits
    # inside the test-set noise: BBBP packs 13 arms into 0.0737 ROC-AUC with a between-fold SD of
    # 0.0354, so a tie is resolved by noise and then charged as a full rank. ECFP4+desc scores
    # 0.9056 there against bare ECFP4's 0.8792 -- descriptors HELP -- and still takes rank 12,
    # which drags its classification mean from 2.09 to 3.93. The median is unmoved by that.
    out["mean_rank"] = out[CATEGORIES].mean(axis=1)
    n_cat = out[CATEGORIES].notna().sum(axis=1)
    out["se_rank"] = out[CATEGORIES].std(axis=1, ddof=1) / np.sqrt(n_cat.clip(lower=1))
    out["n_units"] = n_cat
    out["n_datasets"] = R.notna().sum(axis=1)

    # Design effect, computed on the CATEGORY grouping rather than the benchmark one. The SE is
    # already a spread across four category means rather than across 65 datasets, so this only
    # bites when the four categories themselves are fewer than four independent observations --
    # which is the honest thing for it to do, and it is the same correction allsuites applies.
    Mc = M.copy(); Mc["suite"] = [cat[c] for c in M.index]
    _saved, A.SUITES = A.SUITES, CATEGORIES
    try:
        ne = A.effective_n(R, Mc)
    finally:
        A.SUITES = _saved
    deff = out["n_units"] / max(sum(ne.values()), 1e-9)
    out["se_rank_naive"] = out["se_rank"]
    out["se_rank"] = out["se_rank"] * np.sqrt(deff.clip(lower=1.0))
    # POOLED ALTERNATIVE: every dataset equal weight, category structure ignored entirely.
    # Carried alongside rather than as a separate function so both summaries come off ONE rank
    # matrix -- a second code path computing "the same ranks a different way" is how two numbers
    # that should agree start disagreeing. Its interval is the spread across the 67 per-dataset
    # ranks divided by the EFFECTIVE dataset count (sum of the per-category effective n), which is
    # the matching estimand for a dataset-weighted mean rather than the four-category one.
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
