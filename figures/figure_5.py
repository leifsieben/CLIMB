from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.patches import Patch

from figures.style import STYLE, FS, save, check_font, mark_empty, row_ncol, LEGEND_BOX
from figures.arms import PANELS, PANEL_ORDER, SHADES

check_font()
INK = "#000000"

ROOT = Path(__file__).resolve().parent.parent


RIGOR = ROOT / "analysis" / "rigor"

N_POS, N_NEG = 777, 680


def hanley_mcneil_se(auc: float, n1: int = N_POS, n0: int = N_NEG) -> float:
    q1 = auc / (2 - auc)
    q2 = 2 * auc * auc / (1 + auc)
    var = (auc * (1 - auc) + (n1 - 1) * (q1 - auc * auc) + (n0 - 1) * (q2 - auc * auc)) / (n1 * n0)
    return math.sqrt(var)

PANEL_TASK = {"MoleculeACE": "MoleculeACE", "HIV": "HIV", "BACE": "BACE",
              "Ames": "Ames", "Tox21": "Tox21", "QM7": "QM7"}
PRIMARY = {"MoleculeACE": "macro_rmse", "HIV": "nef1", "BACE": "roc_auc",
           "Ames": "roc_auc", "Tox21": "roc_auc", "QM7": "rmse"}
LOWER_BETTER_METRIC = {"rmse", "macro_rmse"}

FOLD_METRIC = {"macro_rmse": "rmse"}

NO_EMB = "no embedding"
EMB_CLM_U, EMB_CLM_S = "+ CLIMB unsup.", "+ CLIMB sup., desc"


EMB_CLM_R = "+ CLIMB random"
ROLE_COLOR = {NO_EMB: SHADES["anchor"][0], EMB_CLM_U: SHADES["unsup"][0],
              EMB_CLM_S: SHADES["sup"][0],
              EMB_CLM_R: SHADES["random"][0]}


ROLE_SUFFIX = {NO_EMB: None, EMB_CLM_U: "CLM", EMB_CLM_S: "CLMsup",
               EMB_CLM_R: "CLMrand"}


ROLE_ORDER = [NO_EMB, EMB_CLM_U, EMB_CLM_S, EMB_CLM_R]


REF_ORDER = [("ECFP4", "fp"), ("RDKit desc", "desc"), ("ECFP4+desc", "fp+desc")]
CLASSICAL_ADD = {
    "fp":      [("fp+desc", "+ RDKit desc", SHADES["anchor"][0])],
    "desc":    [("fp+desc", "+ ECFP4", SHADES["anchor"][1])],
    "fp+desc": [],
}
EMB_ROLES = [r for r in ROLE_ORDER if ROLE_SUFFIX[r]]


def additions(ref_key):
    """[(feature_key, label, colour)] for one reference -- classical first, then the embeddings."""
    out = list(CLASSICAL_ADD.get(ref_key, []))
    out += [(f"{ref_key}+{ROLE_SUFFIX[r]}", r, ROLE_COLOR[r]) for r in EMB_ROLES]
    return out


GROUPS = [(tick, ref, additions(ref)) for tick, ref in REF_ORDER]

CONCAT_DIR = RIGOR / "concatenation"
EMBEDDING_TAGS = {EMB_CLM_U: "CLMunsup", EMB_CLM_S: "CLMsup", EMB_CLM_R: "CLMrand"}


def _source_paths(stem):
    return ([CONCAT_DIR / f"{pre}_rdkit_sameenv_{stem}.csv" for pre in ("concat", "concat_panels")],
            [CONCAT_DIR / f"{pre}_rdkit_sameenv_{stem}_folds.csv"
             for pre in ("concat", "concat_panels")])


_REQUIRED_STEMS = ["SHARED"] + [EMBEDDING_TAGS[r] for r in EMB_ROLES]
SOURCES, FOLD_SOURCES = [], []
for _stem in _REQUIRED_STEMS:
    _a, _b = _source_paths(_stem)
    SOURCES += _a
    FOLD_SOURCES += _b


TABLES_READY = all(f.exists() for f in SOURCES + FOLD_SOURCES)


LIFT_YLABEL = "Lift over reference (%)"


def compute():


    parts = []
    if not TABLES_READY:
        missing = [f.name for f in SOURCES + FOLD_SOURCES if not f.exists()]
        raise FileNotFoundError(
            f"figure_5: missing required concatenation tables in {CONCAT_DIR}: "
            + ", ".join(missing))
    for f in SOURCES:
        parts.append(_align_tags(pd.read_csv(f)))
    d = pd.concat(parts, ignore_index=True)


    key = ["task", "features", "metric"]
    dup = d[d.duplicated(key, keep=False)]
    if len(dup):
        w = dup.groupby(key)["mean"].agg(["min", "max", "count"])
        w["rel"] = 100 * (w["max"] - w["min"]) / w["min"].abs().replace(0, float("nan"))
        worst = w.sort_values("rel", ascending=False).head(3)
        print(f"  figure_5: {len(w)} embedding-free cell(s) present in both tables; "
              f"max disagreement {w['rel'].max():.2f}% "
              f"({', '.join(f'{i[0]}/{i[1]} {r.rel:.2f}%' for i, r in worst.iterrows())})")
    return _fill_ames_se(d.drop_duplicates(key, keep="first").reset_index(drop=True))


def _align_tags(df):

    df = df.copy()
    df["features"] = df["features"].str.replace("CLMunsup", "CLM", regex=False)
    return df


def _fill_ames_se(d):
    ames = (d.task == "Ames") & (d.metric == "roc_auc") & (~np.isfinite(d["std"]))
    n = int(ames.sum())
    if n:
        d.loc[ames, "std"] = d.loc[ames, "mean"].map(hanley_mcneil_se)
        print(f"  figure_5: Ames has no fold spread by construction (labels withheld); "
              f"{n} cell(s) given the Hanley-McNeil analytic AUC SE "
              f"(n1={N_POS}, n0={N_NEG})")
    return d


def fold_table():

    import collections
    out = collections.defaultdict(dict)


    for f in FOLD_SOURCES:
        for r in _align_tags(pd.read_csv(f)).itertuples():
            out[(r.task, r.features, r.metric)][str(r.fold)] = float(r.value)
    return out


def draw_panel(ax, d, p, compact=False, tag=None, fig=None, ylims=None, xrot=None, bw=None,
               ylabel=True):

    meta = PANELS[p]
    task = PANEL_TASK[p]
    arrow = "↑" if meta["higher_better"] else "↓"
    if tag is None:
        ax.set_title(f"{meta['label']} {arrow}", fontsize=FS["title"], fontweight="bold",
                     color=INK, pad=3)
    else:
        from matplotlib.transforms import ScaledTranslation
        ax.text(0.0, 1.04, tag, transform=ax.transAxes, fontsize=FS["panel_tag"],
                fontweight="bold", va="bottom", ha="left", color=INK)


        short = {"Ames": "Ames", "MoleculeACE": "MolACE"}.get(p, meta["label"])
        ax.text(0.0, 1.04, f"{short} {arrow}", fontsize=FS["title"] - (1 if compact else 0),
                fontweight="bold",
                va="bottom", ha="left", color=INK,
                transform=ax.transAxes + ScaledTranslation(11 / 72, 0, fig.dpi_scale_trans))


    if ylabel:
        ax.set_ylabel(LIFT_YLABEL, fontsize=FS["annot"] - (1 if compact else 0), color=INK)
    ax.grid(axis="y", ls=":", lw=0.6, color=STYLE["grid"])
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    if task is None:
        ax.text(0.5, 0.5, "concatenation test\nnot run", transform=ax.transAxes,
                ha="center", va="center", fontsize=FS["annot"], color=INK)
        ax.set_xticks([]); ax.set_yticks([])


        mark_empty(ax, f"{p}: concatenation test not run on this panel")
        return
    metric = PRIMARY[task]
    FOLDS = fold_table()
    g = d[(d.task == task) & (d.metric == metric)].set_index("features")

    def cell(k):
        if k not in g.index:
            return None, None
        v = float(g.loc[k, "mean"])
        e = float(g.loc[k, "std"])


        return v, (float("nan") if not np.isfinite(e) else e)


    def lift_pct(ref_key, add_key):
        v0, _ = cell(ref_key)
        v1, _ = cell(add_key)
        if v0 is None or v1 is None or not np.isfinite(v0) or v0 == 0:
            return None
        gain = (v0 - v1) if metric in LOWER_BETTER_METRIC else (v1 - v0)
        return 100.0 * gain / abs(v0)

    GAP = 0.8
    xs, ys, cs, present, labels = [], [], [], [], []
    xi = 0.0
    group_centres = []
    for _, ref_key, adds in GROUPS:
        start = xi
        for k, lab, colour in adds:
            xs.append(xi)
            ys.append(lift_pct(ref_key, k))
            cs.append(colour)
            labels.append(lab)
            present.append(ys[-1] is not None)
            xi += 1.0
        if xi == start:
            xi += 1.0
        group_centres.append((start + xi - 1.0) / 2.0)
        xi += GAP
    x = np.array(xs)

    def lift_sd(ref_key, add_key):
        fmetric = FOLD_METRIC.get(metric, metric)
        r0 = FOLDS.get((task, ref_key, fmetric), {})
        r1 = FOLDS.get((task, add_key, fmetric), {})
        shared = sorted(set(r0) & set(r1))
        if len(shared) < 2:


            v0, _ = cell(ref_key)
            v1, _ = cell(add_key)
            g0 = g.loc[ref_key, "std"] if ref_key in g.index else float("nan")
            g1 = g.loc[add_key, "std"] if add_key in g.index else float("nan")
            if None in (v0, v1) or not all(np.isfinite(x) for x in (v0, v1, g0, g1)) or v0 == 0:
                return float("nan")

            var = (100.0 / v0) ** 2 * float(g1) ** 2 + (100.0 * v1 / v0 ** 2) ** 2 * float(g0) ** 2
            return float(np.sqrt(var))
        per = []
        for f in shared:
            v0, v1 = r0[f], r1[f]
            if not np.isfinite(v0) or v0 == 0:
                continue
            gain = (v0 - v1) if fmetric in LOWER_BETTER_METRIC else (v1 - v0)
            per.append(100.0 * gain / abs(v0))
        return float(np.std(per, ddof=1)) if len(per) > 1 else float("nan")

    es = [lift_sd(rk, k) for _, rk, adds in GROUPS for k, _, _ in adds]

    base_v = 0.0
    drawn = [v for v in ys if v is not None]
    if not drawn:
        ax.text(0.5, 0.5, "concatenation test\nnot run", transform=ax.transAxes,
                ha="center", va="center", fontsize=FS["annot"], color=INK)
        ax.set_xticks([]); ax.set_yticks([])
        mark_empty(ax, f"{p}: concatenation test not run on this panel")
        return


    span = [v for v in drawn] + [0.0]
    span += [v + e for v, e in zip(ys, es) if v is not None and np.isfinite(e)]
    span += [v - e for v, e in zip(ys, es) if v is not None and np.isfinite(e)]
    lo, hi = min(span), max(span)
    pad = 0.28 * max(hi - lo, 1e-9)
    y0, y1 = lo - pad, hi + pad
    ax.set_ylim(y0, y1)


    bw = bw if bw is not None else (0.62 if compact else 0.74)
    for xi_, v, e, c, ok in zip(x, ys, es, cs, present):
        if ok:


            ax.bar([xi_], [v], width=bw, color=c, edgecolor=INK, linewidth=0.7,
                   yerr=([e] if np.isfinite(e) else None),
                   error_kw=dict(elinewidth=0.9, capsize=1.8, capthick=0.9,
                                 ecolor=INK, zorder=6), zorder=3)
        else:


            ax.bar([xi_], [y1 - y0], bottom=y0, width=bw, facecolor="none",
                   edgecolor=STYLE["grid"], linewidth=0.7, linestyle=(0, (2, 2)), zorder=2)
            ax.text(xi_, y0 + 0.5 * (y1 - y0), "not run", rotation=90, ha="center", va="center",
                    fontsize=FS["annot"] - 2, color=INK, style="italic", zorder=4)


    ax.set_xticks(group_centres)


    ax.set_xticklabels([g for g, _, _ in GROUPS], fontsize=FS["annot"] - 1)
    ax.tick_params(axis="x", length=0)
    ax.xaxis.set_minor_locator(ticker.NullLocator())
    ax.tick_params(axis="x", which="minor", bottom=False)
    if compact:
        ax.tick_params(axis="y", labelsize=FS["annot"] - 1)
        ax.yaxis.set_major_locator(ticker.MaxNLocator(4))


def legend_handles():


    seen, out = set(), []
    for _, _, adds in GROUPS:
        for _, lab, colour in adds:
            key = (str(lab), colour)
            if key in seen:
                continue
            seen.add(key)
            out.append(Patch(facecolor=colour, edgecolor=INK, lw=0.8, label=str(lab)))
    return out


def main():
    d = compute()


    fig, axes = plt.subplots(2, 3, figsize=(STYLE["col2"], 3.45))
    for i, (ax, p) in enumerate(zip(axes.ravel(), PANEL_ORDER)):
        draw_panel(ax, d, p, ylabel=(i % 3 == 0))

    handles = legend_handles()
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.052), ncol=row_ncol(handles),
               fontsize=FS["legend"], handletextpad=0.5, labelspacing=0.3, columnspacing=1.2,
               borderpad=0.30, **LEGEND_BOX, labelcolor=INK)


    fig.tight_layout(rect=(0, 0.055, 1, 1), h_pad=0.9, w_pad=1.0)


    save(fig, "figure_5")
    plt.close(fig)


if __name__ == "__main__":
    main()
