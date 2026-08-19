"""What does the SUPERVISED (descriptor-regression) objective actually learn?

Two analyses on frozen embeddings dumped by scripts/embed_probe_dump.py.

THE QUESTION, and why it is not the obvious one. fig_E already settles whether the supervised
objective needs the molecule->label CORRESPONDENCE: permuting the descriptor targets across the
batch preserves p(y) exactly and lands BELOW the untrained floor on all six panels. So the map is
real. The open question is whether it is anything MORE than a descriptor calculator -- and fig_F
already hints not, since concatenating CLIMB onto ECFP+desc helps on no canonical panel.

(1) PCA, coloured by property. Honest framing matters here: the MTR objective regressed RDKit's
    full `Descriptors.descList`, so MolWt / MolLogP / TPSA are IN the training set. A supervised
    embedding that organises by them confirms its training loss converged; it is not evidence about
    chemistry. The informative comparisons are therefore
      - the SAME panels for `unsup`, which was never given those targets, and
      - properties genuinely OUTSIDE the training set: 3D shape descriptors (ETKDG conformer, so
        not a deterministic function of any 2D descriptor) and the benchmark's own label.
    Each panel carries the R^2 of a linear probe for that property from the 2 PCs actually drawn,
    so the reader gets a number rather than an impression of a coloured cloud.

(2) DESCRIPTOR-RESIDUAL PROBING -- the decisive test. Split the embedding into the part linearly
    predictable from the 217 trained descriptors and the part that is not:
        Z = Zhat (descriptor-explained) + R (residual)
    then run the SAME probe on Z, Zhat, R and D alone.
      probe(Zhat) ~ probe(Z) and probe(R) ~ chance  =>  a learned descriptor calculator
      probe(R) meaningfully above chance            =>  it carries something descriptors do not
    The D->Z map and the probe are BOTH fit on training folds only and applied to the held-out
    fold, so no part of the decomposition sees test molecules. Folds are the paper's own scaffold
    split (eval_v2._scaffold_kfold_indices via a2_bootstrap_errorbars.fold_ids).

    The probe is a LINEAR one (logistic / ridge), not the paper's MLP head. That is deliberate: the
    comparison here is Z vs Zhat vs R under one probe, and a linear probe measures what is linearly
    decodable without the head inventing structure of its own. Absolute values are therefore NOT
    comparable to fig_A's -- only the four columns to each other.

Run:  python3 scripts/embed_probe_dump.py --dataset BACE && python3 scripts/si_representation.py
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
from sklearn.decomposition import PCA                                # noqa: E402
from sklearn.metrics import roc_auc_score                            # noqa: E402
from sklearn.preprocessing import StandardScaler                     # noqa: E402
from a2_bootstrap_errorbars import fold_ids                          # noqa: E402

FD = ROOT / "figure_data"
REPR = FD / "_repr"
OUTD = FD / "SI_repr"
ARMS = ["unsup", "sup_dense"]
ARM_LABEL = {"unsup": "unsupervised", "sup_dense": "supervised, dense"}
# (name, source, in_training_set?)
PROPS = [("MolWt", "D", True), ("MolLogP", "D", True), ("TPSA", "D", True),
         ("Asphericity", "D3", False), ("RadiusOfGyration", "D3", False), ("label", "y", False)]
RIDGE_ALPHA = 10.0


def clean(M):
    """NaN -> column median, drop zero-variance columns. Returns (matrix, kept mask)."""
    M = np.array(M, dtype=float, copy=True)
    if M.shape[1] == 0:
        return M, np.zeros(0, bool)
    med = np.nanmedian(M, axis=0)
    med[~np.isfinite(med)] = 0.0
    idx = np.where(~np.isfinite(M))
    M[idx] = np.take(med, idx[1])
    keep = M.std(axis=0) > 1e-8
    return M[:, keep], keep


def prop_vector(data, name, source):
    if source == "y":
        return data["y"].astype(float)
    key, nkey = ("D", "dnames") if source == "D" else ("D3", "d3names")
    names = list(data[nkey])
    if name not in names:
        return None
    v = data[key][:, names.index(name)].astype(float)
    return v if np.isfinite(v).sum() > 10 else None


# ------------------------------------------------------------------ analysis 2: residual probing
def probe_score(Xtr, ytr, Xte, yte, classification):
    if classification:
        if len(np.unique(ytr)) < 2 or len(np.unique(yte)) < 2:
            return np.nan
        m = LogisticRegression(max_iter=2000, C=1.0)
        m.fit(Xtr, ytr)
        return roc_auc_score(yte, m.predict_proba(Xte)[:, 1])
    m = Ridge(alpha=RIDGE_ALPHA)
    m.fit(Xtr, ytr)
    return float(np.sqrt(np.mean((m.predict(Xte) - yte) ** 2)))


def residual_probing(data, dataset):
    smiles = list(data["smiles"])
    y = data["y"].astype(float)
    classification = len(np.unique(y[np.isfinite(y)])) <= 2
    D, _ = clean(data["D"])
    folds = fold_ids("climb_v2_phase2", smiles, y)

    rows = []
    for arm in ARMS:
        Z, _ = clean(data[f"Z_{arm}"])
        per = {k: [] for k in ("Z", "Zhat", "R", "D")}
        for f in sorted(set(folds[folds >= 0])):
            te, tr = folds == f, (folds >= 0) & (folds != f)
            # everything is fit on TRAIN ONLY -- the decomposition must not see the test fold
            sd, sz = StandardScaler().fit(D[tr]), StandardScaler().fit(Z[tr])
            Dtr, Dte = sd.transform(D[tr]), sd.transform(D[te])
            Ztr, Zte = sz.transform(Z[tr]), sz.transform(Z[te])
            mapper = Ridge(alpha=RIDGE_ALPHA).fit(Dtr, Ztr)      # D -> Z
            Htr, Hte = mapper.predict(Dtr), mapper.predict(Dte)  # descriptor-explained part
            Rtr, Rte = Ztr - Htr, Zte - Hte                      # residual
            for k, (a, b) in dict(Z=(Ztr, Zte), Zhat=(Htr, Hte), R=(Rtr, Rte),
                                  D=(Dtr, Dte)).items():
                per[k].append(probe_score(a, y[tr], b, y[te], classification))
        ev = float(np.mean([np.corrcoef(Z[:, j],
                   Ridge(alpha=RIDGE_ALPHA).fit(D, Z).predict(D)[:, j])[0, 1] ** 2
                   for j in range(Z.shape[1])]))
        rows.append(dict(dataset=dataset, arm=arm,
                         metric="roc_auc" if classification else "rmse",
                         descriptor_R2=round(ev, 4),
                         **{k: round(float(np.nanmean(v)), 4) for k, v in per.items()}))
    return pd.DataFrame(rows), classification


# ------------------------------------------------------------------ analysis 1: PCA
def pca_panels(data, dataset):
    import matplotlib.pyplot as plt
    from figures.style import STYLE, FS, save, check_font
    check_font()

    props = [(n, s, ins) for n, s, ins in PROPS if prop_vector(data, n, s) is not None]
    mw = prop_vector(data, "MolWt", "D")
    mw_share = {}
    for n, src, ins in props:
        if ins or mw is None:
            continue
        v = prop_vector(data, n, src)
        ok = np.isfinite(v) & np.isfinite(mw)
        mw_share[n] = float(np.corrcoef(v[ok], mw[ok])[0, 1] ** 2) if ok.sum() > 10 else 0.0
    ncol = len(props)
    fig, axes = plt.subplots(len(ARMS), ncol, figsize=(STYLE["col2"], 2.35 * len(ARMS)),
                             squeeze=False)
    r2_rows = []
    for i, arm in enumerate(ARMS):
        Z, _ = clean(data[f"Z_{arm}"])
        Zs = StandardScaler().fit_transform(Z)
        P = PCA(n_components=2, random_state=0).fit(Zs)
        XY = P.transform(Zs)
        for j, (name, source, in_set) in enumerate(props):
            ax = axes[i][j]
            v = prop_vector(data, name, source)
            ok = np.isfinite(v)
            # R^2 of a LINEAR probe for the property from the 2 PCs actually drawn -- the number
            # behind the colour, so the panel is not read as a rorschach test
            r2 = float(np.corrcoef(Ridge(alpha=1.0).fit(XY[ok], v[ok]).predict(XY[ok]),
                                   v[ok])[0, 1] ** 2)
            r2_rows.append(dict(dataset=dataset, arm=arm, property=name,
                                in_training_set=int(in_set), pc_r2=round(r2, 4)))
            lo, hi = np.nanpercentile(v[ok], [2, 98])
            ax.scatter(XY[ok, 0], XY[ok, 1], c=np.clip(v[ok], lo, hi), cmap="Purples",
                       s=4, lw=0.15, edgecolor="#00000022", rasterized=True)
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_color(STYLE["grid"])
            # A property outside the training set is only informative to the extent it is NOT a
            # restatement of one inside it. RadiusOfGyration shares 68% of its variance with MolWt
            # on BACE, so its apparent recovery is largely mass leaking through; Asphericity shares
            # 1.5% and is the clean probe. Report the overlap rather than let the reader assume 0.
            tag = ("trained" if in_set else
                   f"NOT trained ({100 * mw_share.get(name, 0):.0f}% w/ MolWt)")
            # three lines, not two: the confound note makes a single tag line collide with its
            # neighbours at this column count
            ax.set_title(f"{name}\n{tag}\nR²={r2:.2f}", fontsize=FS["annot"] - 1,
                         color=STYLE["ink"], linespacing=1.25)
            if j == 0:
                ax.set_ylabel(ARM_LABEL[arm], fontsize=FS["title"], fontweight="bold",
                              color=STYLE["ink"])
    ev = ""
    fig.suptitle(f"PC1–PC2 of the frozen embedding, {dataset} molecules{ev}",
                 fontsize=FS["title"], fontweight="bold", color=STYLE["ink"])
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    OUTD.mkdir(parents=True, exist_ok=True)
    save(fig, f"SI_repr_pca_{dataset}")
    plt.close(fig)
    return pd.DataFrame(r2_rows)


def main():
    dataset = sys.argv[1] if len(sys.argv) > 1 else "BACE"
    f = REPR / f"{dataset}_embeddings.npz"
    if not f.exists():
        sys.exit(f"missing {f} -- run scripts/embed_probe_dump.py --dataset {dataset} first")
    data = np.load(f, allow_pickle=True)

    r2 = pca_panels(data, dataset)
    res, classification = residual_probing(data, dataset)
    OUTD.mkdir(parents=True, exist_ok=True)
    r2.to_csv(OUTD / f"pca_property_r2_{dataset}.csv", index=False)
    res.to_csv(OUTD / f"residual_probing_{dataset}.csv", index=False)

    print(f"\n=== {dataset}: property recoverable from the 2 drawn PCs (linear R²) ===")
    piv = r2.pivot(index="property", columns="arm", values="pc_r2")
    ins = r2.drop_duplicates("property").set_index("property").in_training_set
    piv.insert(0, "trained_on", ins.map({1: "yes", 0: "NO"}))
    print(piv.to_string())
    print("  NOTE: an out-of-set property only tests generalisation insofar as it is independent of "
          "the trained ones; the figure prints each one's variance shared with MolWt.")

    better = "higher" if classification else "lower"
    print(f"\n=== {dataset}: descriptor-residual probing ({res.metric.iloc[0]}, {better}=better) ===")
    print("  descriptor_R2 = fraction of embedding variance linearly explained by the 217 "
          "trained descriptors")
    print(res[["arm", "descriptor_R2", "Z", "Zhat", "R", "D"]].to_string(index=False))
    print("\n  Z    full embedding          Zhat  descriptor-explained part")
    print("  R    residual (what descriptors CANNOT explain)     D  the 217 descriptors alone")


if __name__ == "__main__":
    main()
