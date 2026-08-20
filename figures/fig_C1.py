"""Fig C1 -- does molecular similarity (pretrain corpus <-> eval molecule) explain the benefit of
UNSUPERVISED pretraining?

READ THIS FIRST: THERE IS NO BENEFIT TO EXPLAIN, AND THAT IS THE RESULT.
Against the honest floor -- a random-init encoder FINE-TUNED end-to-end -- unsupervised pretraining
is worth, over ALL molecules and before any binning:

    MoleculeACE   -0.29%  [-0.40, -0.18]        QM7   -2.68%  [-5.25, -0.94]

i.e. nothing, or slightly negative, on both tasks. So this figure is not apportioning a benefit
between similar and novel molecules; it is showing that no apportionment is available, INCLUDING
on the molecules the model has literally read. That is the stronger claim, and it is what panel (a)
means: the corpus-identical group sits at -0.0%, so there is no memorization advantage to explain
away in the first place.

Do not read this against fig_E, which reports the same arm at +6.4% (MoleculeACE) and +2.9% (QM7).
Fig E's floor is no pretraining FROZEN; this figure's floor is no pretraining FINE-TUNED. Both are
true and together they are the point: pretraining beats an untrained encoder held fixed, and does
not beat the same untrained encoder allowed to learn. Beating a frozen random encoder is close to
automatic, which is exactly why it is not the comparator here.

ONE script, ONE figure: figures_v2/figC1.png / .pdf

What it shows
-------------
Per-molecule squared errors of the unsupervised (MLM) arm vs the honest floor -- the random-init
encoder FINE-TUNED end-to-end ("no pretrain, end2end", mean of 3 replicate runs; beating a frozen
random encoder is close to automatic, so it is not the comparator) -- binned by the molecule's TRUE
max ECFP4 Tanimoto similarity to the full 12M pretraining corpus (analysis/dedup_i1/
full_corpus_similarity_i1.csv, all 12 shards, NOT a subsample lower bound).

  (a) RMSE lift (%) over the floor for three molecule groups, averaged over the two regression
      tasks (ESOL, QM7): corpus-IDENTICAL molecules (Tanimoto = 1.0 or literal match -- excluded
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
BASE_RUNS = ["e2e_random_00", "e2e_random_01", "e2e_random_02"]   # no pretrain, end2end (MolNet)
# MoleculeACE lives in its own tree with its own run names for the same two arms, and its
# predictions carry no y_true -- the labels come from the benchmark's own split files.
MACE_DIR = ROOT / "figure_data" / "chemeleon_suite" / "moleculeace"
MACE_DATA = ROOT / "chemeleon_suite" / "data" / "moleculeace"
MACE_MODEL = ["unsup_8M", "unsup_8M_s1", "unsup_8M_s2"]
MACE_BASE = ["no_pretrain_e2e_e2e", "no_pretrain_e2e_e2e_s1", "no_pretrain_e2e_e2e_s2"]
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
FLOOR_LABEL = ARMS["e2e_no_pretrain"]["label"]     # "no pretrain, end2end"

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
    """
    out = {}
    for f in sorted(MACE_DATA.glob("*.csv")):
        d = pd.read_csv(f)
        if "split" not in d.columns:
            continue
        te = d[d.split == "test"]
        out.update(dict(zip(te.smiles, te.y)))
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
        raise FileNotFoundError("no e2e floor predictions found")
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
        bars.append(("identical\n(excluded)" if compact else
                     "corpus-identical\n(Tanimoto=1.0,\nexcluded)", mem, se_m, C_MEM))
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
        print(f"   {t} corpus-identical group (excluded): {v[0]:+.1f}% [{v[1]:+.1f},{v[2]:+.1f}]"
              f"  (n={v[3]})")
    if pairs:
        sep = [t for t, v in pairs if v[0][1] > v[1][2] or v[1][1] > v[0][2]]
        if sep:
            print(f"Non-overlapping CIs on {', '.join(sep)} => the gain DOES depend on corpus "
                  f"similarity there even among non-identical molecules.")
        else:
            print("Once corpus-identical molecules are removed, no task shows the lift "
                  "concentrating on corpus-similar molecules (CIs overlap). NOTE the corpus-"
                  "identical group is ALSO flat on the canonical panels (MoleculeACE -0.1%, "
                  "QM7 +0.1%), so on this task set there is no memorization advantage to explain "
                  "away -- the unsupervised arm gains nothing on molecules it has effectively "
                  "seen. That is a stronger statement than the pre-canonical ESOL/QM7 version, "
                  "where the top bin WAS carried by the identical group.")


if __name__ == "__main__":
    main()
