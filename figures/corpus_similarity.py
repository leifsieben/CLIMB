from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd

from figures.style import STYLE, FS, check_font, LEGEND_BOX
from figures.arms import ARMS, SHADES, LIFT_YLABEL

check_font()

ROOT = Path(__file__).resolve().parent.parent
DEDUP = ROOT / "analysis" / "dedup_i1"
TANI = DEDUP / "full_corpus_similarity_i1.csv"
EXACT = DEDUP / "exact_match_per_molecule.csv"
IDENTICAL_THR = 0.99999

MODEL = "unsup_8M"
BASE_RUNS = ["random_baseline_00", "random_baseline_01", "random_baseline_02"]


MACE_DIR = ROOT / "figure_data" / "chemeleon_suite" / "moleculeace"
MACE_DATA = ROOT / "benchmarks" / "data" / "moleculeace"
MACE_MODEL = ["unsup_8M", "unsup_8M_s1", "unsup_8M_s2"]
MACE_BASE = ["random_baseline_00", "random_baseline_01", "random_baseline_02"]


TASKS = ["MoleculeACE", "QM7"]
QM7_SUB = "moleculenet_cv_qm7native"


_REGRESSION_TASKS = {"QM7"}


C_MEM = SHADES["e2e"][1]
C_SIM = SHADES["unsup"][0]
C_NOV = SHADES["unsup"][2]
TASK_COL = {"MoleculeACE": SHADES["unsup"][0], "QM7": SHADES["unsup"][2]}
FLOOR_LABEL = ARMS["random_encoder"]["label"]

NBOOT = 400


def _mace_truth():

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


    s["memorized"] = (s.exact_nosalt == 1) | (s.max_tanimoto_to_corpus >= IDENTICAL_THR)
    return s


def _merged(task, sim, mod, base):
    m = mod[mod.dataset == task].merge(base[base.dataset == task],
                                       on=["dataset", "raw_smiles"], suffixes=("_m", "_b"))
    m = m.merge(sim[sim.dataset == task][["raw_smiles", "max_tanimoto_to_corpus", "memorized"]],
                on="raw_smiles", how="inner")


    if task in _REGRESSION_TASKS:
        import evaluate
        for pred, true in (("y_pred_m", "y_true_m"), ("y_pred_b", "y_true_b")):
            m[pred] = evaluate._bound_ood(m[pred].to_numpy(float), m[true].to_numpy(float),
                                         "regression")
    m["se_m"] = (m.y_pred_m - m.y_true_m) ** 2
    m["se_b"] = (m.y_pred_b - m.y_true_b) ** 2
    return m


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


def compute():
    """All C1 numbers, once. Used by Figure 3."""
    sim = _simframe()
    mod = _model_preds()
    base = _floor_preds()
    merged = {t: _merged(t, sim, mod, base) for t in TASKS}
    pairs = [(t, v) for t, v in ((t, quartile_lift(m)) for t, m in merged.items()) if v]
    mpairs = [(t, v) for t, v in ((t, memorized_lift(m)) for t, m in merged.items()) if v]
    return dict(merged=merged, pairs=pairs, mpairs=mpairs)


def _agg(items):
    trips = [v[0] if isinstance(v[0], tuple) else v for _, v in items]
    trips = [t if isinstance(t[0], float) else t for t in trips]
    val = float(np.mean([t[0] for t in trips]))
    se = float(np.sqrt(np.sum([((t[2] - t[1]) / 2 / len(trips)) ** 2 for t in trips])))
    return val, se


def draw(ax0, ax1, data, tags=("a", "b"), compact=False):

    merged, pairs, mpairs = data["merged"], data["pairs"], data["mpairs"]


    bars = []
    if mpairs:
        mem, se_m = _agg([(t, v) for t, v in mpairs])


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


    ax0.bar(xpos, [b[1] for b in bars], color=[b[3] for b in bars], width=0.45,
            edgecolor=STYLE["ink"], linewidth=0.8,
            yerr=[b[2] for b in bars], zorder=3,
            error_kw=dict(elinewidth=1.0, capsize=2.2, capthick=1.1,
                          ecolor=STYLE["ink"], zorder=6))
    for xi, b in zip(xpos, bars):
        ax0.text(xi, b[1] + b[2] + 0.6, f"{b[1]:+.1f}%", ha="center", fontsize=FS["annot"])
    ax0.axhline(0, color=SHADES["random"][0], lw=0.6)
    vals = [b[1] for b in bars]; errs = [b[2] for b in bars]
    lo_, hi_ = min(0, *(v - e for v, e in zip(vals, errs))), max(0, *(v + e for v, e in zip(vals, errs)))


    span_ = max(hi_ - lo_, 1e-9)
    ax0.set_ylim(lo_ - 0.12 * span_, hi_ + 0.18 * span_)
    ax0.set_xticks(xpos)
    ax0.set_xticklabels([b[0] for b in bars], fontsize=FS["annot"])
    ax0.set_ylabel(LIFT_YLABEL)


    ax0.set_title("Unsupervised pretrain" if compact else
                  "The lift does not depend on corpus similarity, including on identical molecules",
                  loc="left" if compact else "center",
                  fontsize=FS["title"], fontweight="bold", pad=9 if compact else 4)
    if tags and tags[0]:
        ax0.text(*((-0.15, 1.05) if compact else (-0.22, 1.02)), tags[0], transform=ax0.transAxes,
                 fontsize=FS["panel_tag"], fontweight="bold", va="bottom", ha="left")


    for t, m in merged.items():
        xs, ys, lo, hi, nn = binned_lift(m)
        if not xs:
            continue
        err = np.vstack([np.array(ys) - np.array(lo), np.array(hi) - np.array(ys)])
        ax1.errorbar(xs, ys, yerr=err, color=TASK_COL[t], marker="o", mec="white", mew=0.6,
                     lw=STYLE["lw"], capsize=STYLE["cap_size"],
                     label=f"{t} (n\u2248{int(np.median(nn))})")
    ax1.axhline(0, color=SHADES["random"][0], lw=0.6)


    ax1.legend(loc="lower right", fontsize=FS["legend"] - 0.5, **LEGEND_BOX,
               handletextpad=0.35, labelspacing=0.22, borderaxespad=0.3)
    lo1, hi1 = ax1.get_ylim()
    ax1.set_ylim(lo1 - 0.20 * (hi1 - lo1), hi1)


    ax1.set_xlabel("max Tanimoto to corpus (bin mean)" if compact else
                   "max ECFP4 Tanimoto to corpus (bin mean)")
    ax1.set_ylabel(LIFT_YLABEL)
    ax1.set_title("Lift vs corpus similarity", loc="left" if compact else "center",
                  fontsize=FS["title"], fontweight="bold", pad=9 if compact else 4)
    if tags and len(tags) > 1 and tags[1]:
        ax1.text(*((-0.13, 1.05) if compact else (-0.26, 1.02)), tags[1], transform=ax1.transAxes,
                 fontsize=FS["panel_tag"], fontweight="bold", va="bottom", ha="left")
