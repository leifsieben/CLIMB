"""Fig C1 -- does molecular similarity (pretrain corpus <-> eval molecule) explain the benefit of
UNSUPERVISED pretraining?

RETRACTED HEADLINE (2026-08-19). This file used to open with "THERE IS NO BENEFIT TO EXPLAIN,
AND THAT IS THE RESULT", quoting MoleculeACE -0.29% and QM7 -2.68% over all molecules. That null
was an artefact of two separate defects, and neither was visible in the figure, which looked
entirely reasonable throughout.

  1. WRONG LABEL COLUMN on MoleculeACE. chemeleon_suite/data/moleculeace/*.csv carry both `y` and
     `y [pEC50/pKi]`, differing by EXACTLY 9.0 -- `y` is -log10(exp_mean in nM), the models are
     trained and scored on pKi. The truth join took `y`, so every squared error was (residual - 9)
     and read ~81 instead of ~0.87. Lift is (floor SE - arm SE)/floor SE and the constant does NOT
     cancel: it inflates the denominator ~90x and crushes every lift toward zero. See _mace_truth.
  2. PROTOCOL-MIXED FLOOR. A FROZEN pretrained arm was lifted against a FINE-TUNED random-init
     arm, so the number mixed "did pretraining help" with "frozen vs fine-tuned".

WHAT IT READS NOW. Floor is the frozen random encoder ("no pretrain, random"), matched to the
frozen arm under test (user decision 2026-08-19); labels are the pKi column. Over ALL molecules,
before any binning:

    MoleculeACE   +18.98%  [+17.06, +20.98]        QM7   +2.79%  [+1.04, +4.52]

The two defects pulled in opposite directions and it is worth keeping them apart, because "we
changed the floor and the result grew" is not what happened:

    floor        labels      MoleculeACE      QM7
    e2e          `y` (bug)        -0.29%    -2.68%     <- the retracted headline
    e2e          pKi             +28.59%    -3.39%     <- the units fix alone
    frozen       pKi             +18.98%    +2.79%     <- current

So on MoleculeACE the null was ENTIRELY the units bug, and moving to the matched frozen floor
actually LOWERS the lift (28.6 -> 19.0). On QM7 the floor is what flips the sign. That asymmetry
is itself informative and not noise: MoleculeACE targets are a few hundred molecules each, where a
fine-tuned random init overfits and makes a WEAKER floor than the frozen one; QM7 has 6.8k, enough
for the fine-tuned floor to be the harder of the two. Which no-pretraining floor is more demanding
depends on the label budget, so neither is "the honest" one in general -- what matters is that it
is protocol-matched to the arm, which the frozen floor is and the old one was not.

ON THE PRE-REGISTRATION. This file pre-registered, before any of these numbers existed, that
"whatever the matched comparison gives is what gets reported", naming a LARGER matched lift as the
costly direction because the headline was a null. That is what happened, and the null is withdrawn
rather than defended. The pre-registered swap named e2e-vs-e2e as the matched version; the swap
actually made is frozen-vs-frozen, which is matched on the same principle and is the one this
figure's arm supports today. The e2e-vs-e2e number remains worth having as the practical claim and
is one QM7 run away (peer session).

WHAT THE FIGURE NOW SHOWS, which is a result rather than an absence of one: the benefit is real
and it does NOT concentrate on corpus-similar molecules. Among non-identical molecules the most-
similar and most-novel quantiles are indistinguishable (MoleculeACE +9.2% [6.8,11.4] vs +9.4%
[6.8,12.1]; QM7 +2.0% vs +0.8%, CIs overlapping). The corpus-identical group sits modestly higher
(+11.3% MoleculeACE, +2.6% QM7) but its interval overlaps the most-similar quantile, so it is not
a clean memorization signal either. Generalisation, not recall -- which is the claim fig_I1 makes
from the other direction.

Do not read this against fig_E's +6.4% (MoleculeACE) as though it were the same quantity: fig E
lifts a suite-level metric per task, this figure lifts POOLED PER-MOLECULE squared error, and the
two aggregate differently even with the same arm and the same floor.

ONE script, ONE figure: figures_v2/figC1.png / .pdf

What it shows
-------------
Per-molecule squared errors of the unsupervised (MLM) arm vs "no pretrain, random" -- the
random-init encoder FROZEN and probed exactly as the arm is (mean of 3 replicate runs) -- binned
by the molecule's TRUE
max ECFP4 Tanimoto similarity to the full 12M pretraining corpus (analysis/dedup_i1/
full_corpus_similarity_i1.csv, all 12 shards, NOT a subsample lower bound).

  (a) RMSE lift (%) over the floor for three molecule groups, averaged over the two regression
      tasks (MoleculeACE and QM7 -- this line said ESOL until 2026-08-20, naming a task the
      canonical migration removed): corpus-IDENTICAL molecules (Tanimoto = 1.0 or literal match -- excluded
      from the trend, reported apart), the most corpus-similar quartile, and the most novel
      quartile. This separates memorization (identity) from interpolation (similarity).
  (b) The same lift as a continuous trend: 5 quantile bins per task over NON-identical molecules,
      bootstrap 95% CIs. Near-duplicates (0.95 <= T < 1.0) are distinct inputs and stay in the
      trend as the genuine interpolation regime.

Regression tasks only, because this needs per-molecule SQUARED ERRORS -- and the canonical six has
exactly two: MoleculeACE (13.2% ECFP4-identical to corpus, median max-Tanimoto 0.76) and QM7
(15.7% identical, median 0.63). Same predictions on both sides of every comparison; MoleculeACE
labels are joined from the benchmark's own split files because its prediction dumps carry no
y_true.

THREE THINGS THE CAPTION MUST CARRY about the x-axis (compute session, 2026-08-19):
  - The similarity distribution is BIMODAL, not a tail. MoleculeACE's p75 is 0.844 but its p90 is
    already 1.0000: molecules are either clearly novel or fingerprint-identical, with little in
    between. Quoting a median alone would read as "moderate overlap" and hide that structure,
    which is why panel (a) reports the identical group SEPARATELY rather than as a percentile.
  - The two panels differ in KIND, not just degree. QM7's near-duplicate count (1,081) barely
    exceeds its exact-match count (1,075) -- essentially every QM7 near-neighbour IS an exact
    fingerprint match, as expected for small saturated molecules where ECFP4 saturates. MoleculeACE
    has a genuine 0.95-1.0 band (1,405 vs 1,206, ~200 real near-dups). So "13% overlap" and "16%
    overlap" do not mean the same thing on the two panels.
  - "Exact match" means ECFP4-IDENTICAL at 2048 bits, not necessarily the same molecule. Same
    definition as compute_tanimoto_novelty.py, so it is consistent with fig_I1.

Run:  python3 -m figures.fig_C1

PANEL SET — MIGRATED to the canonical six on 2026-08-19, when the full-corpus similarity for
MoleculeACE landed (ASK 4). The x-axis is a TRUE max Tanimoto to all 12 shards of the 12M corpus
(analysis/dedup_i1/full_corpus_similarity_i1.csv, 15,973 molecules), not the subsample lower bound
in figure_data/_tanimoto/corpus_similarity.csv -- a lower bound cannot establish Tanimoto = 1.0
identity, which is the whole point of panel (a).
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from figures.style import STYLE, FS, save, title, check_font
from figures.arms import ARMS, SHADES, LIFT_YLABEL

check_font()

ROOT = Path(__file__).resolve().parent.parent
DEDUP = ROOT / "analysis" / "dedup_i1"
TANI = DEDUP / "full_corpus_similarity_i1.csv"     # true max Tanimoto to the full 12M corpus
EXACT = DEDUP / "exact_match_per_molecule.csv"     # literal isomeric-canonical corpus match
IDENTICAL_THR = 0.99999                            # ECFP4 Tanimoto = 1.0 => corpus-identical

MODEL = "unsup_8M"                                 # the unsupervised arm (matched 8M budget)
BASE_RUNS = ["random_baseline_00", "random_baseline_01", "random_baseline_02"]  # frozen floor
# MoleculeACE lives in its own tree with its own run names for the same two arms, and its
# predictions carry no y_true -- the labels come from the benchmark's own split files.
MACE_DIR = ROOT / "figure_data" / "chemeleon_suite" / "moleculeace"
MACE_DATA = ROOT / "chemeleon_suite" / "data" / "moleculeace"
MACE_MODEL = ["unsup_8M", "unsup_8M_s1", "unsup_8M_s2"]
MACE_BASE = ["random_baseline_00", "random_baseline_01", "random_baseline_02"]
# The canonical six has exactly TWO regression panels, and this figure needs per-molecule squared
# errors, so these are the two it can use. (Pre-canonical it was ESOL + QM7.)
TASKS = ["MoleculeACE", "QM7"]
QM7_SUB = "moleculenet_cv_qm7native"               # both arm and floor have it -> one convention
# panels scored by RMSE, i.e. the ones _merged clamps. MoleculeACE is macro RMSE over 30 targets
# with per-target scales, so it is excluded: one band over pooled targets would be meaningless.
_REGRESSION_TASKS = {"QM7"}

# colours: the arm under test is `unsup`, so similarity groups/tasks run dark (similar) -> light
# (novel) along the unsupervised shade ladder; the excluded identity group is grey. No hard-coded
# colours -- everything comes from arms.py.
C_MEM = SHADES["e2e"][1]
C_SIM = SHADES["unsup"][0]
C_NOV = SHADES["unsup"][2]
TASK_COL = {"MoleculeACE": SHADES["unsup"][0], "QM7": SHADES["unsup"][2]}
FLOOR_LABEL = ARMS["random_encoder"]["label"]      # "no pretrain, random" (frozen, matched)

NBOOT = 400


# ------------------------------------------------------------------------------------------------
# data
# ------------------------------------------------------------------------------------------------
def _mace_truth():
    """{smiles: y} for every MoleculeACE TEST molecule, from the benchmark's own split files.

    The MoleculeACE prediction dumps carry (task, seed, test_index, smiles, y_pred) and no y_true,
    so the labels are joined from chemeleon_suite/data/moleculeace/<task>.csv. Verified 2026-08-19
    that `test_index` indexes the split=="test" rows of that file in order (smiles match exactly),
    so the join is by SMILES and the index is only a cross-check.

    THE LABEL COLUMN IS `y [pEC50/pKi]`, NOT `y`, AND THE DIFFERENCE IS NOT COSMETIC. Those files
    carry both, and they differ by EXACTLY 9.0 everywhere -- `y` is -log10(exp_mean in nM) while
    the models are trained and scored on pKi = 9 - log10(nM). Joining `y` therefore subtracts a
    constant 9 from every label, and squared errors go from ~0.87 to ~81: the true signal survives
    only as a rounding term on a large constant.

    That is exactly what this figure measures, so it mattered. Lift is (floor SE - arm SE)/floor
    SE, and adding the same constant to both numerator terms does NOT cancel -- it inflates the
    denominator by two orders of magnitude and crushes every lift toward zero. Read with `y`,
    MoleculeACE lift was -0.29% and the figure's headline was "there is no benefit to explain".
    That headline was measuring the unit error.

    Caught 2026-08-19 by a scale sanity check, not by the figure looking wrong -- it did not. The
    assertion below is the fix that generalises: any future truth join whose residuals are absurd
    for the metric fails here instead of rendering.
    """
    out = {}
    for f in sorted(MACE_DATA.glob("*.csv")):
        d = pd.read_csv(f)
        if "split" not in d.columns:
            continue
        col = "y [pEC50/pKi]"
        assert col in d.columns, f"{f.name}: no {col!r} column; do NOT silently fall back to 'y'"
        te = d[d.split == "test"]
        out.update(dict(zip(te.smiles, te[col])))
    assert 3.0 < min(out.values()) and max(out.values()) < 14.0, (
        f"MoleculeACE labels out of pKi range [{min(out.values()):.2f}, {max(out.values()):.2f}] "
        f"-- the 'y' column (pKi - 9) is the likely join")
    return out


def _mace_preds(runs):
    """Per-molecule mean prediction over eval seeds AND pretraining-seed dirs, MoleculeACE."""
    truth = _mace_truth()
    frames = []
    for r in runs:
        f = MACE_DIR / r / "test_predictions.csv"
        if f.exists():
            frames.append(pd.read_csv(f))
    if not frames:
        return None
    d = pd.concat(frames, ignore_index=True)
    d = d.rename(columns={"smiles": "raw_smiles"})
    d["y_true"] = d.raw_smiles.map(truth)
    d = d.dropna(subset=["y_true"])
    d["dataset"] = "MoleculeACE"
    return (d.groupby(["dataset", "raw_smiles"], as_index=False)
             .agg(y_true=("y_true", "first"), y_pred=("y_pred", "mean")))


def _preds(run):
    """Per-molecule mean prediction (over folds x head seeds) for one run, CV scheme."""
    sub = QM7_SUB if (ROOT / "figure_data" / "climb_v2_phase2" / run / QM7_SUB /
                      "test_predictions.csv").exists() else "moleculenet_cv"
    p = ROOT / "figure_data" / "climb_v2_phase2" / run / sub / "test_predictions.csv"
    if not p.exists():
        return None
    d = pd.read_csv(p)
    return (d.groupby(["dataset", "raw_smiles"], as_index=False)
             .agg(y_true=("y_true", "first"), y_pred=("y_pred", "mean")))


def _floor_preds():
    """Mean prediction across the floor's replicates, both trees."""
    bs = [b for b in (_preds(r) for r in BASE_RUNS) if b is not None]
    m = _mace_preds(MACE_BASE)
    if m is not None:
        bs.append(m)
    if not bs:
        raise FileNotFoundError("no random-encoder floor predictions found")
    return (pd.concat(bs).groupby(["dataset", "raw_smiles"], as_index=False)
             .agg(y_true=("y_true", "first"), y_pred=("y_pred", "mean")))


def _model_preds():
    """The unsupervised arm, both trees."""
    parts = [p for p in [_preds(MODEL), _mace_preds(MACE_MODEL)] if p is not None]
    return pd.concat(parts, ignore_index=True)


def _simframe():
    s = pd.read_csv(TANI).rename(columns={"max_tanimoto_to_corpus_full": "max_tanimoto_to_corpus"})
    e = pd.read_csv(EXACT)[["raw_smiles", "dataset", "exact_nosalt"]]
    s = s.merge(e, on=["raw_smiles", "dataset"], how="left")
    s["exact_nosalt"] = s.exact_nosalt.fillna(0).astype(int)
    # corpus-IDENTICAL (Tanimoto = 1.0, or a literal match); near-dups (0.95<=T<1.0) deliberately
    # NOT flagged -- they are distinct inputs and stay in the trend.
    s["memorized"] = (s.exact_nosalt == 1) | (s.max_tanimoto_to_corpus >= IDENTICAL_THR)
    return s


def _merged(task, sim, mod, base):
    m = mod[mod.dataset == task].merge(base[base.dataset == task],
                                       on=["dataset", "raw_smiles"], suffixes=("_m", "_b"))
    m = m.merge(sim[sim.dataset == task][["raw_smiles", "max_tanimoto_to_corpus", "memorized"]],
                on="raw_smiles", how="inner")
    # CLAMP REGRESSION PREDICTIONS, both arms, before squaring (user 2026-08-19). QM7's `[H]` --
    # a lone hydrogen atom, and by construction the least corpus-similar "molecule" in the set --
    # was predicted at +1,199.8 kcal/mol by the unsupervised arm against a true -1,739.2 and a
    # dataset that is negative everywhere. That single point carried 15% of the lowest-similarity
    # bin's squared error and set panel (b)'s first marker at -8.0%; without it the bin reads
    # -1.8%. The bound is eval_v2._bound_ood, the same one the suite tracks have always applied
    # and that the aggregator now uses -- so this panel and the bars in fig_A describe the same
    # predictions. Fit on the OTHER molecules of the task, not on the test molecule.
    if task in _REGRESSION_TASKS:
        import eval_v2
        for pred, true in (("y_pred_m", "y_true_m"), ("y_pred_b", "y_true_b")):
            m[pred] = eval_v2._bound_ood(m[pred].to_numpy(float), m[true].to_numpy(float),
                                         "regression")
    m["se_m"] = (m.y_pred_m - m.y_true_m) ** 2
    m["se_b"] = (m.y_pred_b - m.y_true_b) ** 2
    return m


# ------------------------------------------------------------------------------------------------
# statistics
# ------------------------------------------------------------------------------------------------
def _lift_rmse(se_m, se_b):
    rm, rb = np.sqrt(np.mean(se_m)), np.sqrt(np.mean(se_b))
    return np.nan if (not np.isfinite(rb) or rb == 0) else 100 * (rb - rm) / rb


def _boot_ci(se_m, se_b, seed):
    pt = _lift_rmse(se_m, se_b)
    if not np.isfinite(pt):
        return None
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(se_m), size=(NBOOT, len(se_m)))
    boot = np.array([_lift_rmse(se_m[i], se_b[i]) for i in idx])
    boot = boot[np.isfinite(boot)]
    lo, hi = np.percentile(boot, [2.5, 97.5]) if len(boot) else (np.nan, np.nan)
    return pt, lo, hi


def binned_lift(m, nbins=5, seed=0):
    """(centres, lift%, lo, hi, n) over the NON-memorized molecules, quantile-binned."""
    m = m[~m.memorized]
    edges = np.unique(m.max_tanimoto_to_corpus.quantile(np.linspace(0, 1, nbins + 1)).values)
    if len(edges) < 3:
        return (None,) * 5
    edges[0] -= 1e-9
    m = m.assign(bin=pd.cut(m.max_tanimoto_to_corpus, bins=edges, labels=False,
                            include_lowest=True))
    xs, ys, los, his, ns = [], [], [], [], []
    for b in range(len(edges) - 1):
        s = m[m.bin == b]
        if len(s) < 15:
            continue
        ci = _boot_ci(s.se_m.values, s.se_b.values, seed)
        if ci is None:
            continue
        xs.append(float(s.max_tanimoto_to_corpus.mean()))
        ys.append(ci[0]); los.append(ci[1]); his.append(ci[2]); ns.append(int(len(s)))
    return xs, ys, los, his, ns


def quartile_lift(m):
    """(most-similar, most-novel) lift% with CIs, from the outer quartiles (non-memorized)."""
    xs, ys, lo, hi, _ = binned_lift(m, nbins=4)
    if not xs or len(ys) < 4:
        return None
    return (ys[-1], lo[-1], hi[-1]), (ys[0], lo[0], hi[0])


def memorized_lift(m, seed=1):
    """Lift% on the EXCLUDED corpus-identical group, for the side-by-side bar."""
    s = m[m.memorized]
    if len(s) < 15:
        return None
    ci = _boot_ci(s.se_m.values, s.se_b.values, seed)
    return None if ci is None else (ci[0], ci[1], ci[2], int(len(s)))


# ------------------------------------------------------------------------------------------------
# figure
# ------------------------------------------------------------------------------------------------
def compute():
    """All C1 numbers, once. Shared by the standalone figure and the assembled fig_C."""
    sim = _simframe()
    mod = _model_preds()
    base = _floor_preds()
    merged = {t: _merged(t, sim, mod, base) for t in TASKS}
    pairs = [(t, v) for t, v in ((t, quartile_lift(m)) for t, m in merged.items()) if v]
    mpairs = [(t, v) for t, v in ((t, memorized_lift(m)) for t, m in merged.items()) if v]
    return dict(merged=merged, pairs=pairs, mpairs=mpairs)


def _agg(items):  # mean over tasks + combined SE from the per-task bootstrap CIs
    trips = [v[0] if isinstance(v[0], tuple) else v for _, v in items]
    trips = [t if isinstance(t[0], float) else t for t in trips]
    val = float(np.mean([t[0] for t in trips]))
    se = float(np.sqrt(np.sum([((t[2] - t[1]) / 2 / len(trips)) ** 2 for t in trips])))
    return val, se


def draw(ax0, ax1, data, tags=("a", "b"), compact=False):
    """Draw the two C1 panels onto existing axes (standalone or assembled context).
    compact=True shortens the bar tick labels and xlabel for the narrow assembled column."""
    merged, pairs, mpairs = data["merged"], data["pairs"], data["mpairs"]

    # --- group bars: corpus-identical / most-similar quartile / most-novel quartile ------------
    bars = []
    if mpairs:
        mem, se_m = _agg([(t, v) for t, v in mpairs])
        # NOT "(excluded)". That label made the bar read as zero by definition -- excluded from
        # the analysis, so of course it shows nothing -- when it is the opposite: this group is
        # MEASURED, and its coming out at -0.0% is the memorization control and one of the
        # figure's results. "Excluded" refers only to panel (b), where these molecules are kept
        # out of the similarity quantiles because a Tanimoto of exactly 1.0 is not a percentile.
        bars.append(("corpus-\nidentical" if compact else
                     "corpus-identical\n(Tanimoto = 1.0,\nseen in pretraining)", mem, se_m, C_MEM))
    if pairs:
        simv, se_s = _agg([(t, v[0]) for t, v in pairs])
        nov, se_n = _agg([(t, v[1]) for t, v in pairs])
        bars += [("top\nquartile" if compact else
                  "most corpus-similar\n(top quartile,\nnot identical)", simv, se_s, C_SIM),
                 ("bottom\nquartile" if compact else
                  "most novel\n(bottom quartile)", nov, se_n, C_NOV)]
    xpos = list(range(len(bars)))
    ax0.bar(xpos, [b[1] for b in bars], color=[b[3] for b in bars], width=0.6,
            yerr=[b[2] for b in bars], capsize=STYLE["cap_size"],
            error_kw=dict(lw=STYLE["lw_thin"]))
    for xi, b in zip(xpos, bars):
        ax0.text(xi, b[1] + b[2] + 0.6, f"{b[1]:+.1f}%", ha="center", fontsize=FS["annot"])
    ax0.axhline(0, color=SHADES["random"][0], lw=0.6)
    vals = [b[1] for b in bars]; errs = [b[2] for b in bars]
    lo_, hi_ = min(0, *(v - e for v, e in zip(vals, errs))), max(0, *(v + e for v, e in zip(vals, errs)))
    # Pad proportionally to the DRAWN RANGE, not to |lo_|. The old form scaled the bottom pad with
    # the magnitude of the lowest bar, so a -7.5 whisker pushed the floor past -10 and left a third
    # of the panel empty (user 2026-08-19).
    span_ = max(hi_ - lo_, 1e-9)
    ax0.set_ylim(lo_ - 0.12 * span_, hi_ + 0.18 * span_)
    ax0.set_xticks(xpos)
    ax0.set_xticklabels([b[0] for b in bars], fontsize=FS["annot"])
    ax0.set_ylabel(LIFT_YLABEL)
    # NOT "Lift by similarity group" (user 2026-08-19: "is that result real? identical SMILES have
    # no change?"). It is real, and the old title made it read as an apportionment of a benefit
    # that does not exist -- the overall lift is -0.29% / -2.68%. The title now states the finding.
    # The compact title shortens again at the A4 re-lay (2026-08-19): at 6.69in the panels are
    # ~1.75in and "No lift, at any corpus similarity" ran under panel (b)'s tag. "similarity" is
    # already carried by (b)'s own title and the x-axis below it, so dropping "corpus" here costs
    # the reader nothing.
    ax0.set_title("No lift at any similarity" if compact else
                  "No lift over fine-tuned no-pretraining, at any corpus similarity",
                  loc="left" if compact else "center",
                  fontsize=FS["title"], fontweight="bold", pad=9 if compact else 4)
    if tags and tags[0]:
        ax0.text(*((-0.15, 1.05) if compact else (-0.22, 1.02)), tags[0], transform=ax0.transAxes,
                 fontsize=FS["panel_tag"], fontweight="bold", va="bottom", ha="left")

    # --- binned trend per task ------------------------------------------------------------------
    for t, m in merged.items():
        xs, ys, lo, hi, nn = binned_lift(m)
        if not xs:
            continue
        err = np.vstack([np.array(ys) - np.array(lo), np.array(hi) - np.array(ys)])
        ax1.errorbar(xs, ys, yerr=err, color=TASK_COL[t], marker="o", mec="white", mew=0.6,
                     lw=STYLE["lw"], capsize=STYLE["cap_size"],
                     label=f"{t} (n\u2248{int(np.median(nn))})")
    ax1.axhline(0, color=SHADES["random"][0], lw=0.6)
    # LOWER RIGHT. Both curves rise to the right and flatten near zero, so the upper-left corner
    # is where the MoleculeACE line begins.
    #
    # The labels lost their "/bin" ("n/bin\u22481585" -> "n\u22481585") and the box lost a point of
    # size, because at the A4 re-lay this panel is ~1.75in wide and the long form made the legend
    # nearly panel-wide: anchored lower-RIGHT it still reached back under the QM7 error bar at
    # x=0.40, which runs down to -5.5 (user 2026-08-19). The floor is also dropped to open a clean
    # band beneath the data rather than letting the box sit on it. "per bin" is what the x-axis
    # already says, so the shortened label loses nothing.
    ax1.legend(loc="lower right", fontsize=FS["legend"] - 0.5, frameon=False,
               handletextpad=0.35, labelspacing=0.22, borderaxespad=0.3)
    lo1, hi1 = ax1.get_ylim()
    ax1.set_ylim(lo1 - 0.20 * (hi1 - lo1), hi1)
    # "ECFP4" here is NOT the anchor's fingerprint and must not be "fixed" to match it.
    # The MODEL featurizer became Morgan r=3 counts with chirality on 2026-08-19; this
    # SIMILARITY axis deliberately stays stereo-blind binary r=2 (scripts/
    # compute_tanimoto_novelty.py), because a stereo-blind match is the LOOSER definition
    # of "the model already read this molecule" and therefore over-counts memorization --
    # the conservative direction for a null result. Two questions, two answers (user
    # 2026-08-19: "the tanimoto I'm ok with. but otherwise let's definitely use the stereo").
    ax1.set_xlabel("max Tanimoto to corpus (bin mean)" if compact else
                   "max ECFP4 Tanimoto to corpus (bin mean)")
    ax1.set_ylabel(LIFT_YLABEL)
    ax1.set_title("Lift vs corpus similarity", loc="left" if compact else "center",
                  fontsize=FS["title"], fontweight="bold", pad=9 if compact else 4)
    if tags and len(tags) > 1 and tags[1]:
        ax1.text(*((-0.13, 1.05) if compact else (-0.26, 1.02)), tags[1], transform=ax1.transAxes,
                 fontsize=FS["panel_tag"], fontweight="bold", va="bottom", ha="left")


def main():
    data = compute()
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(STYLE["col2"], 3.1))
    draw(ax0, ax1, data)
    title(fig, "Fig C1 \u2014 Unsupervised pretraining: memorization or representation?",
          y=1.04)
    # Margins are set explicitly so the AXES fill the canvas: with matplotlib's defaults the
    # right 10% (0.67in) went unused, savefig's tight bbox trimmed it, and this figure came out
    # 5.75in against the set's 6.69in page width -- LaTeX then upscaled it and its fonts printed
    # larger than every other figure's. save() warns if that returns.
    fig.subplots_adjust(top=0.84, bottom=0.16, left=0.085, right=0.985, wspace=0.32)
    save(fig, "fig_C1", subdir="panels")
    plt.close(fig)

    # --- printed verdict (derived, not asserted) ------------------------------------------------
    pairs, mpairs = data["pairs"], data["mpairs"]
    for t, v in pairs:
        print(f"C1 {t}: most-similar {v[0][0]:+.1f}% [{v[0][1]:+.1f},{v[0][2]:+.1f}]   "
              f"most-novel {v[1][0]:+.1f}% [{v[1][1]:+.1f},{v[1][2]:+.1f}]  (95% bootstrap CI)")
    for t, v in mpairs:
        print(f"   {t} corpus-identical group (measured; kept out of the (b) quantiles only): {v[0]:+.1f}% [{v[1]:+.1f},{v[2]:+.1f}]"
              f"  (n={v[3]})")
    if pairs:
        sep = [t for t, v in pairs if v[0][1] > v[1][2] or v[1][1] > v[0][2]]
        if sep:
            print(f"Non-overlapping CIs on {', '.join(sep)} => the gain DOES depend on corpus "
                  f"similarity there even among non-identical molecules.")
        else:
            # The identical group's values are DERIVED here. They were hardcoded into this
            # sentence as "MoleculeACE -0.1%, QM7 +0.1%" and survived a floor change and a label
            # -units fix that moved MoleculeACE to +15%, still printing the old pair as though it
            # were a fresh reading. A narrative line that quotes numbers must compute them.
            mem = ", ".join(f"{t} {v[0]:+.1f}%" for t, v in mpairs)
            flat = all(abs(v[0]) < 2.0 for _, v in mpairs)
            print("Once corpus-identical molecules are removed, no task shows the lift "
                  "concentrating on corpus-similar molecules (CIs overlap).")
            print(f"   corpus-identical group: {mem} -- "
                  + ("also flat, so there is no memorization advantage to explain away on this "
                     "task set." if flat else
                     "NOT flat, so the identical group carries a gain that the non-identical "
                     "quantiles do not explain; report it as its own result."))


if __name__ == "__main__":
    main()
