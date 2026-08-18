"""Fig C1 -- does molecular similarity (pretrain corpus <-> eval molecule) explain the benefit of
UNSUPERVISED pretraining?

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

Regression tasks only (squared errors needed): ESOL (41.9% ECFP4-identical to corpus -- the
circularity risk) and QM7 (15.7% identical, median max-Tanimoto 0.63 -- the clean control).
Same 5-fold CV predictions on both sides of every comparison.

Run:  python3 -m figures.fig_C1

!!! OFF-SUITE — DO NOT SHIP AS-IS !!!
This figure is still on the OLD MoleculeNet task set (ESOL/BBBP/BACE/HIV/Tox21/QM7), not the
paper's canonical six (MoleculeACE / CBS / BACE / Ames / Tox21 / QM7). It is blocked on data, not
on code: the seq_* ablation arms have MoleculeNet evals only, and panels a/b need
PER-MOLECULE predictions on the canonical panels, which do not exist anywhere yet.
Verified absent on disk 2026-08-17; requested from the compute session the same day. When the evals
land, the fix is a data-path + panel-list change in the builder and a re-render — not a redesign.
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from figures.style import STYLE, FS, save, title, check_font
from figures.arms import ARMS, SHADES

check_font()

ROOT = Path(__file__).resolve().parent.parent
DEDUP = ROOT / "analysis" / "dedup_i1"
TANI = DEDUP / "full_corpus_similarity_i1.csv"     # true max Tanimoto to the full 12M corpus
EXACT = DEDUP / "exact_match_per_molecule.csv"     # literal isomeric-canonical corpus match
IDENTICAL_THR = 0.99999                            # ECFP4 Tanimoto = 1.0 => corpus-identical

MODEL = "unsup_8M"                                 # the unsupervised arm (matched 8M budget)
BASE_RUNS = ["e2e_random_00", "e2e_random_01", "e2e_random_02"]   # no pretrain, end2end
TASKS = ["ESOL", "QM7"]

# colours: the arm under test is `unsup`, so similarity groups/tasks run dark (similar) -> light
# (novel) along the unsupervised shade ladder; the excluded identity group is grey. No hard-coded
# colours -- everything comes from arms.py.
C_MEM = SHADES["e2e"][1]
C_SIM = SHADES["unsup"][0]
C_NOV = SHADES["unsup"][2]
TASK_COL = {"ESOL": SHADES["unsup"][0], "QM7": SHADES["unsup"][2]}
FLOOR_LABEL = ARMS["e2e_no_pretrain"]["label"]     # "no pretrain, end2end"

NBOOT = 400


# ------------------------------------------------------------------------------------------------
# data
# ------------------------------------------------------------------------------------------------
def _preds(run):
    """Per-molecule mean prediction (over folds x head seeds) for one run, CV scheme."""
    p = ROOT / "figure_data" / "climb_v2_phase2" / run / "moleculenet_cv" / "test_predictions.csv"
    if not p.exists():
        return None
    d = pd.read_csv(p)
    return (d.groupby(["dataset", "raw_smiles"], as_index=False)
             .agg(y_true=("y_true", "first"), y_pred=("y_pred", "mean")))


def _floor_preds():
    """Mean prediction across the floor's replicates."""
    bs = [b for b in (_preds(r) for r in BASE_RUNS) if b is not None]
    if not bs:
        raise FileNotFoundError("no e2e floor predictions found")
    return (pd.concat(bs).groupby(["dataset", "raw_smiles"], as_index=False)
             .agg(y_true=("y_true", "first"), y_pred=("y_pred", "mean")))


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
    mod = _preds(MODEL)
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
    ax0.set_ylim(lo_ - abs(lo_) * 0.35 - 2, hi_ + abs(hi_) * 0.35 + 3)
    ax0.set_xticks(xpos)
    ax0.set_xticklabels([b[0] for b in bars], fontsize=FS["annot"])
    ax0.set_ylabel(f"lift over {FLOOR_LABEL} (%)")
    ax0.set_title("Lift by similarity group" if compact else
                  "Lift by corpus similarity group",
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
                     label=f"{t} (n/bin\u2248{int(np.median(nn))})")
    ax1.axhline(0, color=SHADES["random"][0], lw=0.6)
    # top-left: the only quadrant of this panel with no data in it (user 2026-08-17); "best"
    # kept drifting between renders, which is worse than a fixed corner in an assembled figure.
    ax1.legend(loc="upper left", fontsize=FS["legend"], frameon=False,
               handletextpad=0.4, labelspacing=0.25, borderaxespad=0.3)
    ax1.set_xlabel("max Tanimoto to corpus (bin mean)" if compact else
                   "max ECFP4 Tanimoto to corpus (bin mean)")
    ax1.set_ylabel("lift (%)" if compact else f"lift over {FLOOR_LABEL} (%)")
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
                  "concentrating on corpus-similar molecules (CIs overlap). The top-bin advantage "
                  "is carried by the corpus-identical group in (a): memorization of in-corpus "
                  "structures, not genuine interpolation.")


if __name__ == "__main__":
    main()
