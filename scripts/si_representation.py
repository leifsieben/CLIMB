"""What does the SUPERVISED (descriptor-regression) objective actually learn?

Three analyses on frozen embeddings dumped by scripts/embed_probe_dump.py, over the paper's usual
3 pretraining seeds per arm plus ECFP4 as the classical reference.

THE QUESTION, and why it is not the obvious one. fig_E already settles whether the supervised
objective needs the molecule->label CORRESPONDENCE: permuting the descriptor targets across the
batch preserves p(y) exactly and lands BELOW the untrained floor on all six panels. So the map is
real. The open question is whether it is anything MORE than a descriptor calculator -- and fig_F
already hints not, since concatenating CLIMB onto ECFP+desc helps on no canonical panel.

(1) PROPERTY RECOVERABILITY (the number behind the PCA picture). For each property we report the
    out-of-fold R^2 of a ridge probe from (a) the 2 PCs the figure actually draws, (b) the top 10
    PCs, and (c) the FULL embedding. The three columns separate two very different statements:
      2-PC R^2 low but full R^2 high  -> the property IS encoded, just not along the directions of
                                         greatest variance. The PCA panel is then uninformative
                                         about that property rather than evidence against it.
      full R^2 also low               -> the property genuinely is not linearly present.
    This is what a coloured scatter cannot tell you, and it is why the PCA panels are supporting
    material here rather than the claim.

    Honest framing on which properties count: the MTR objective regressed RDKit's FULL
    `Descriptors.descList`, so MolWt / MolLogP / TPSA are IN the training set -- recovering them
    tests that the training loss converged, not chemistry. An out-of-set property is only
    informative to the extent it is independent of the trained ones, so each one reports its
    variance shared with MolWt (RadiusOfGyration shares 68% on BACE and is mostly mass in
    disguise; Asphericity shares 1% and is the clean probe).

(2) DESCRIPTOR-RESIDUAL PROBING -- the decisive test. Split the embedding into the part linearly
    predictable from the 217 trained descriptors and the part that is not:
        Z = Zhat (descriptor-explained) + R (residual)
    then run the SAME probe on Z, Zhat, R and D alone.
      probe(Zhat) ~ probe(Z) and probe(R) ~ chance  =>  a learned descriptor calculator
      probe(R) meaningfully above chance            =>  it carries something descriptors do not
    The D->Z map and the probe are BOTH fit on training folds only and applied to the held-out
    fold, so no part of the decomposition sees test molecules. Folds are the paper's own scaffold
    split (eval_v2._scaffold_kfold_indices via a2_bootstrap_errorbars.fold_ids). descriptor_R2 is
    likewise computed OUT OF FOLD -- the in-sample version flatters the map.

(3) LINEAR *AND* MLP PROBES, because the choice changes what the result can claim.
    The decomposition is linear, so R is "what a LINEAR descriptor map cannot reach". Probing it
    linearly therefore tests only for linearly-decodable extra information, and a reader is right
    to ask whether an MLP would find more. Two things follow, and they point in opposite
    directions, so both probes are reported:
      - An MLP on R could recover descriptor content that is a NONLINEAR function of D -- i.e. it
        can make R look informative without any information beyond descriptors existing. So a HIGH
        MLP(R) is weak evidence for "beyond descriptors".
      - A LOW MLP(R) is strong evidence for the null: not even a nonlinear probe finds anything
        outside the linear descriptor span.
    The MLP also matches the paper's own probe head in kind, so its Z column is the one comparable
    to fig_A (not in value -- different folds, no head-seed ensembling).

Writes: figure_data/SI_repr/{property_recoverability,residual_probing}_<dataset>.csv
        figures_v2/SI_repr_pca_<dataset>.png/pdf, figures_v2/SI_repr_table.{csv,tex}
Run:    python3 scripts/embed_probe_dump.py --dataset BACE && python3 scripts/si_representation.py BACE
"""
from __future__ import annotations
import sys, warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
warnings.filterwarnings("ignore")

from sklearn.linear_model import Ridge, LogisticRegression          # noqa: E402
from sklearn.neural_network import MLPRegressor, MLPClassifier      # noqa: E402
from sklearn.decomposition import PCA                                # noqa: E402
from sklearn.metrics import roc_auc_score                            # noqa: E402
from sklearn.preprocessing import StandardScaler                     # noqa: E402
from a2_bootstrap_errorbars import fold_ids                          # noqa: E402

FD = ROOT / "figure_data"
REPR = FD / "_repr"
OUTD = FD / "SI_repr"
ARMS = ["unsup", "sup_desc", "ecfp"]
ARM_LABEL = {"unsup": "unsupervised", "sup_desc": "supervised, desc", "ecfp": "ECFP4"}
PROPS = [("MolWt", "D", True), ("MolLogP", "D", True), ("TPSA", "D", True),
         ("Asphericity", "D3", False), ("RadiusOfGyration", "D3", False), ("label", "y", False)]
RIDGE_ALPHA = 10.0
PCA_PANEL_ARMS = ["unsup", "sup_desc"]        # ECFP is a bit vector; its PCA is not comparable


def seeds_of(data, arm):
    return sorted(k for k in data.files if k.startswith(f"Z_{arm}_s"))


def clean(M, winsorize=False):
    """NaN -> column median, drop zero-variance columns, optionally winsorise.

    WINSORISE THE DESCRIPTORS. RDKit's `Ipc` (information content) overflows on larger molecules:
    on BACE its maximum is 3.9e23 against 3.6e3 for the next-largest descriptor. Standardising does
    not tame that -- one molecule dominates the column -- and it destroyed the ridge D->Z map,
    which showed up as an out-of-fold descriptor R^2 of -7e12. QM7 was unaffected because its
    molecules are small enough that Ipc stays finite, which is exactly why this had to be caught on
    BACE rather than assumed away. Clipping each column to its 0.5-99.5 percentile keeps every
    descriptor while removing the overflow.
    """
    M = np.array(M, dtype=float, copy=True)
    if M.shape[1] == 0:
        return M
    med = np.nanmedian(M, axis=0)
    med[~np.isfinite(med)] = 0.0
    idx = np.where(~np.isfinite(M))
    M[idx] = np.take(med, idx[1])
    if winsorize:
        lo, hi = np.nanpercentile(M, [0.5, 99.5], axis=0)
        M = np.clip(M, lo, hi)
    return M[:, M.std(axis=0) > 1e-8]


def prop_vector(data, name, source):
    if source == "y":
        return data["y"].astype(float)
    key, nkey = ("D", "dnames") if source == "D" else ("D3", "d3names")
    names = list(data[nkey])
    if name not in names:
        return None
    v = data[key][:, names.index(name)].astype(float)
    return v if np.isfinite(v).sum() > 10 else None


def make_probe(kind, classification):
    if kind == "mlp":
        common = dict(hidden_layer_sizes=(256,), max_iter=400, early_stopping=True,
                      n_iter_no_change=15, random_state=0)
        return MLPClassifier(**common) if classification else MLPRegressor(**common)
    return LogisticRegression(max_iter=2000, C=1.0) if classification else Ridge(alpha=RIDGE_ALPHA)


def probe_score(kind, Xtr, ytr, Xte, yte, classification):
    if classification and (len(np.unique(ytr)) < 2 or len(np.unique(yte)) < 2):
        return np.nan
    m = make_probe(kind, classification)
    if classification:
        m.fit(Xtr, ytr)
        return roc_auc_score(yte, m.predict_proba(Xte)[:, 1])
    # STANDARDISE THE REGRESSION TARGET for the MLP, then invert. QM7's labels are ~-1500 kcal/mol;
    # an MLP with default init cannot fit that scale and came out WORSE than ridge on every single
    # column (225 vs 206 RMSE, residual 358), which is a property of the optimiser, not of the
    # representation. Ridge is scale-equivariant so it needs no such treatment -- but it is applied
    # to both probes so the two columns differ only in the probe, which is the whole comparison.
    mu, sg = float(np.mean(ytr)), float(np.std(ytr)) or 1.0
    m.fit(Xtr, (ytr - mu) / sg)
    pred = m.predict(Xte) * sg + mu
    return float(np.sqrt(np.mean((pred - yte) ** 2)))


# ------------------------------------------------------------------ (2)+(3) residual probing
def residual_probing(data, dataset, probes=("linear", "mlp")):
    smiles = list(data["smiles"])
    y = data["y"].astype(float)
    classification = len(np.unique(y[np.isfinite(y)])) <= 2
    D = clean(data["D"], winsorize=True)
    folds = fold_ids("climb_v2_phase2", smiles, y)
    rows = []
    for arm in ARMS:
        for zi, zkey in enumerate(seeds_of(data, arm)):
            Z = clean(data[zkey])
            per = {(p, k): [] for p in probes for k in ("Z", "Zhat", "R", "D")}
            expl = []
            for f in sorted(set(folds[folds >= 0])):
                te, tr = folds == f, (folds >= 0) & (folds != f)
                sd, sz = StandardScaler().fit(D[tr]), StandardScaler().fit(Z[tr])
                Dtr, Dte = sd.transform(D[tr]), sd.transform(D[te])
                Ztr, Zte = sz.transform(Z[tr]), sz.transform(Z[te])
                mapper = Ridge(alpha=RIDGE_ALPHA).fit(Dtr, Ztr)
                Htr, Hte = mapper.predict(Dtr), mapper.predict(Dte)
                # POOLED R^2 over the whole matrix, not the mean of per-dimension R^2. A dimension
                # that is near-constant within a test fold has ss_tot ~ 0, so its per-dimension R^2
                # explodes; ECFP's sparse bit columns make that catastrophic (it produced -2e27).
                # Pooling weights each dimension by the variance it actually has.
                ss_res = float(((Zte - Hte) ** 2).sum())
                ss_tot = float(((Zte - Zte.mean(0)) ** 2).sum())
                expl.append(1.0 - ss_res / max(ss_tot, 1e-12))
                mats = dict(Z=(Ztr, Zte), Zhat=(Htr, Hte), R=(Ztr - Htr, Zte - Hte), D=(Dtr, Dte))
                for p in probes:
                    for k, (a, b) in mats.items():
                        per[(p, k)].append(probe_score(p, a, y[tr], b, y[te], classification))
            row = dict(dataset=dataset, arm=arm, seed=zi,
                       metric="roc_auc" if classification else "rmse",
                       descriptor_R2=round(float(np.mean(expl)), 4))
            for (p, k), v in per.items():
                row[f"{p}_{k}"] = round(float(np.nanmean(v)), 4)
            rows.append(row)
            print(f"    {arm:9s} seed{zi}  descR2={row['descriptor_R2']:.3f}  "
                  + "  ".join(f"{p}:Z={row[f'{p}_Z']:.3f} R={row[f'{p}_R']:.3f}"
                              for p in probes), flush=True)
    return pd.DataFrame(rows), classification


# ------------------------------------------------------------------ (1) property recoverability
def property_recoverability(data, dataset):
    """Out-of-fold R^2 for each property from 2 PCs / 10 PCs / the full embedding.

    Answers 'is the PCA panel evidence?' -- a property invisible in 2 PCs but recoverable from the
    full embedding is present, just not along the leading variance directions.
    """
    smiles = list(data["smiles"])
    y = data["y"].astype(float)
    folds = fold_ids("climb_v2_phase2", smiles, y)
    props = [(n, s, ins) for n, s, ins in PROPS if prop_vector(data, n, s) is not None]
    mw = prop_vector(data, "MolWt", "D")
    rows = []
    for arm in ARMS:
        for zi, zkey in enumerate(seeds_of(data, arm)):
            Z = clean(data[zkey])
            for name, source, in_set in props:
                v = prop_vector(data, name, source)
                ok = np.isfinite(v)
                share = np.nan
                if not in_set and mw is not None:
                    m = ok & np.isfinite(mw)
                    share = float(np.corrcoef(v[m], mw[m])[0, 1] ** 2)
                out = {}
                for tag, ncomp in (("pc2", 2), ("pc10", 10), ("full", None)):
                    preds, obs = [], []
                    for f in sorted(set(folds[folds >= 0])):
                        te = (folds == f) & ok
                        tr = (folds >= 0) & (folds != f) & ok
                        if te.sum() < 5 or tr.sum() < 20:
                            continue
                        sc = StandardScaler().fit(Z[tr])
                        A, B = sc.transform(Z[tr]), sc.transform(Z[te])
                        if ncomp:
                            pc = PCA(n_components=min(ncomp, A.shape[1]), random_state=0).fit(A)
                            A, B = pc.transform(A), pc.transform(B)
                        preds.append(Ridge(alpha=RIDGE_ALPHA).fit(A, v[tr]).predict(B))
                        obs.append(v[te])
                    if preds:
                        pr, ob = np.concatenate(preds), np.concatenate(obs)
                        out[tag] = 1 - ((ob - pr) ** 2).sum() / max(((ob - ob.mean()) ** 2).sum(), 1e-12)
                    else:
                        out[tag] = np.nan
                rows.append(dict(dataset=dataset, arm=arm, seed=zi, property=name,
                                 in_training_set=int(in_set),
                                 var_shared_with_MolWt=round(share, 4) if np.isfinite(share) else "",
                                 **{f"r2_{k}": round(float(x), 4) for k, x in out.items()}))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ PCA figure (supporting)
def pca_panels(data, dataset, rec: pd.DataFrame):
    import matplotlib.pyplot as plt
    from figures.style import STYLE, FS, save, check_font
    check_font()
    props = [(n, s, ins) for n, s, ins in PROPS if prop_vector(data, n, s) is not None]
    mw = prop_vector(data, "MolWt", "D")
    share = {}
    for n, src, ins in props:
        if ins:
            continue
        v = prop_vector(data, n, src)
        ok = np.isfinite(v) & np.isfinite(mw)
        share[n] = float(np.corrcoef(v[ok], mw[ok])[0, 1] ** 2) if ok.sum() > 10 else 0.0

    fig, axes = plt.subplots(len(PCA_PANEL_ARMS), len(props),
                             figsize=(STYLE["col2"], 2.55 * len(PCA_PANEL_ARMS)), squeeze=False)
    for i, arm in enumerate(PCA_PANEL_ARMS):
        zkeys = seeds_of(data, arm)
        if not zkeys:
            continue
        Z = clean(data[zkeys[0]])                       # seed 0 is drawn; all seeds are in the CSV
        Zs = StandardScaler().fit_transform(Z)
        P = PCA(n_components=2, random_state=0).fit(Zs)
        XY = P.transform(Zs)
        ev = 100 * P.explained_variance_ratio_.sum()
        for j, (name, source, in_set) in enumerate(props):
            ax = axes[i][j]
            v = prop_vector(data, name, source)
            ok = np.isfinite(v)
            r = rec[(rec.arm == arm) & (rec.seed == 0) & (rec.property == name)]
            r2_2, r2_f = (float(r.r2_pc2.iloc[0]), float(r.r2_full.iloc[0])) if len(r) else (np.nan, np.nan)
            lo, hi = np.nanpercentile(v[ok], [2, 98])
            ax.scatter(XY[ok, 0], XY[ok, 1], c=np.clip(v[ok], lo, hi), cmap="Purples",
                       s=4, lw=0.15, edgecolor="#00000022", rasterized=True)
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_color(STYLE["grid"])
            tag = "trained" if in_set else f"NOT trained ({100 * share.get(name, 0):.0f}% w/ MolWt)"
            # BOTH numbers on the panel: what the picture can show, and what the embedding holds
            ax.set_title(f"{name}\n{tag}\n2PC R²={r2_2:.2f} · full R²={r2_f:.2f}",
                         fontsize=FS["annot"] - 1, color=STYLE["ink"], linespacing=1.25)
            if j == 0:
                ax.set_ylabel(f"{ARM_LABEL[arm]}\n(PC1+PC2 = {ev:.0f}% of variance)",
                              fontsize=FS["annot"], fontweight="bold", color=STYLE["ink"])
    fig.suptitle(f"PC1–PC2 of the frozen embedding, {dataset} molecules "
                 f"(seed 0; R² is out-of-fold ridge, not the picture)",
                 fontsize=FS["title"], fontweight="bold", color=STYLE["ink"])
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save(fig, f"SI_repr_pca_{dataset}")
    plt.close(fig)


def write_tex(allr: pd.DataFrame, path: Path):
    """Compact LaTeX table: one row per (dataset, arm), mean +- SD over pretraining seeds."""
    lines = [r"\begin{tabular}{llrrrrrrr}", r"\toprule",
             r"Dataset & Representation & desc.\ $R^2$ & \multicolumn{3}{c}{linear probe} "
             r"& \multicolumn{3}{c}{MLP probe} \\",
             r"\cmidrule(lr){4-6}\cmidrule(lr){7-9}",
             r" & & & $Z$ & $\hat{Z}$ & $R$ & $Z$ & $\hat{Z}$ & $R$ \\", r"\midrule"]
    for (ds, arm), g in allr.groupby(["dataset", "arm"], sort=False):
        def cell(c):
            m, s = g[c].mean(), g[c].std(ddof=1)
            return f"{m:.3f}" if not np.isfinite(s) or len(g) < 2 else rf"{m:.3f}\,$\pm$\,{s:.3f}"
        lines.append(" & ".join([ds, ARM_LABEL.get(arm, arm), cell("descriptor_R2"),
                                 cell("linear_Z"), cell("linear_Zhat"), cell("linear_R"),
                                 cell("mlp_Z"), cell("mlp_Zhat"), cell("mlp_R")]) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    path.write_text("\n".join(lines))


def main():
    datasets = sys.argv[1:] or ["BACE"]
    OUTD.mkdir(parents=True, exist_ok=True)
    all_res = []
    for dataset in datasets:
        f = REPR / f"{dataset}_embeddings.npz"
        if not f.exists():
            sys.exit(f"missing {f} -- run scripts/embed_probe_dump.py --dataset {dataset}")
        data = np.load(f, allow_pickle=True)
        print(f"\n=== {dataset} ===", flush=True)
        rec = property_recoverability(data, dataset)
        rec.to_csv(OUTD / f"property_recoverability_{dataset}.csv", index=False)
        pca_panels(data, dataset, rec)
        res, classification = residual_probing(data, dataset)
        res.to_csv(OUTD / f"residual_probing_{dataset}.csv", index=False)
        all_res.append(res)

        print(f"\n--- {dataset}: property recoverable from 2 PCs vs 10 PCs vs the FULL embedding "
              f"(out-of-fold R², mean over seeds) ---")
        r = rec.groupby(["arm", "property"], sort=False)[
            ["r2_pc2", "r2_pc10", "r2_full"]].mean().round(3).reset_index()
        ins = rec.drop_duplicates("property").set_index("property").in_training_set
        hdr = f"  {'property':<18}{'trained':>8}" + "".join(
            f"{a[:9]:>11}" for a in ARMS for _ in range(1))
        print(f"  {'property':<18}{'trained':>8}" + "".join(
            f"{ARM_LABEL[a][:10]:>26}" for a in ARMS))
        print(f"  {'':<18}{'':>8}" + "".join(f"{'2PC':>8}{'10PC':>9}{'full':>9}" for _ in ARMS))
        for prop in rec.property.unique():
            line = f"  {prop:<18}{('yes' if ins[prop] else 'NO'):>8}"
            for a in ARMS:
                g = r[(r.arm == a) & (r["property"] == prop)]
                if g.empty:
                    line += f"{'—':>8}{'—':>9}{'—':>9}"
                else:
                    line += (f"{g.r2_pc2.iloc[0]:>8.2f}{g.r2_pc10.iloc[0]:>9.2f}"
                             f"{g.r2_full.iloc[0]:>9.2f}")
            print(line)

        better = "higher" if classification else "lower"
        print(f"\n--- {dataset}: descriptor-residual probing ({res.metric.iloc[0]}, "
              f"{better}=better; mean over {res.seed.nunique()} seed(s)) ---")
        for arm in ARMS:
            g = res[res.arm == arm]
            if g.empty:
                continue
            print(f"  {ARM_LABEL[arm]:<20} descR²={g.descriptor_R2.mean():.3f}")
            for p in ("linear", "mlp"):
                print(f"      {p:<7} " + "  ".join(
                    f"{k}={g[f'{p}_{k}'].mean():.3f}" for k in ("Z", "Zhat", "R", "D")))

    if all_res:
        allr = pd.concat(all_res, ignore_index=True)
        allr.to_csv(ROOT / "figures_v2" / "SI_repr_table.csv", index=False)
        write_tex(allr, ROOT / "figures_v2" / "SI_repr_table.tex")
        print(f"\nwrote figures_v2/SI_repr_table.csv + .tex ({len(allr)} rows)")


if __name__ == "__main__":
    main()
