from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
STATS = ROOT / "configs" / "descriptor_stats.json"
TOKENIZER = ROOT / "figure_data" / "_tokenizer"
DEFAULT_POOL = (ROOT / "figure_data" / "climb_v2_phase2" / "skip_dense_8M" /
                "moleculenet_cv" / "test_predictions.csv")


ENCODERS = {
    "CLIMB unsup":      "unsup_8M",
    "CLIMB sup, desc":  "skip_dense_8M",
    "CLIMB random":     "random_baseline_00",
}


GROUPS = [
    ("Chi (connectivity)",   lambda n: n.startswith("Chi")),
    ("Kappa (shape)",        lambda n: n.startswith("Kappa") or n == "HallKierAlpha"),
    ("BCUT (eigenvalue)",    lambda n: n.startswith("BCUT")),
    ("EState",               lambda n: "EState" in n),
    ("PEOE/SMR/SlogP VSA",   lambda n: "VSA" in n),
    ("Fragment counts (fr_)", lambda n: n.startswith("fr_")),
    ("Information (Ipc/SPS)", lambda n: "Ipc" in n or n in ("SPS", "Phi")),
    ("Simple counts / bulk", lambda n: True),
]


def group_of(name):
    for label, pred in GROUPS:
        if pred(name):
            return label
    return "other"


def descriptors(smiles, names):

    from rdkit import Chem, RDLogger
    from rdkit.Chem import Descriptors
    RDLogger.DisableLog("rdApp.*")
    fn = dict(Descriptors.descList)
    missing = [n for n in names if n not in fn]
    assert not missing, (
        f"this RDKit exposes {len(fn)} descriptors and is missing {len(missing)} the pretraining "
        f"statistics were fit on, e.g. {missing[:5]}. Scoring against a different descriptor set "
        f"is not a weaker version of this experiment, it is a different one.")
    out = np.full((len(smiles), len(names)), np.nan, dtype=np.float64)
    for i, s in enumerate(smiles):
        m = Chem.MolFromSmiles(s)
        if m is None:
            continue
        for j, n in enumerate(names):
            try:
                out[i, j] = fn[n](m)
            except Exception:
                pass
    return out


def ecfp4(smiles, n_bits=2048):
    from rdkit import Chem, RDLogger
    from rdkit.Chem import rdFingerprintGenerator
    RDLogger.DisableLog("rdApp.*")
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=n_bits)
    X = np.zeros((len(smiles), n_bits), dtype=np.float32)
    for i, s in enumerate(smiles):
        m = Chem.MolFromSmiles(s)
        if m is not None:
            X[i] = np.asarray(gen.GetFingerprintAsNumPy(m), dtype=np.float32)
    return X


def embed(smiles, enc_dir, batch=512):
    import torch
    from transformers import AutoTokenizer, AutoModel
    tok = AutoTokenizer.from_pretrained(str(TOKENIZER))
    model = AutoModel.from_pretrained(str(enc_dir)).eval()
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    model = model.to(dev)
    out = []
    with torch.no_grad():
        for i in range(0, len(smiles), batch):
            chunk = smiles[i:i + batch]
            e = tok(chunk, padding=True, truncation=True, max_length=128, return_tensors="pt")


            e = {k: v.to(dev) for k, v in e.items() if k in ("input_ids", "attention_mask")}
            h = model(**e).last_hidden_state
            mask = e["attention_mask"].unsqueeze(-1).float()
            out.append(((h * mask).sum(1) / mask.sum(1)).float().cpu().numpy())
    return np.vstack(out)


def probe(X, Y, n_train, alphas=(1.0, 10.0, 100.0, 1000.0)):

    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    Xtr, Xte = X[:n_train], X[n_train:]
    Ytr, Yte = Y[:n_train], Y[n_train:]
    sc = StandardScaler().fit(Xtr)
    Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)

    cut = int(0.8 * n_train)
    best, best_score = None, -np.inf
    for a in alphas:
        r = Ridge(alpha=a).fit(Xtr[:cut], Ytr[:cut])
        p = r.predict(Xtr[cut:])
        s = float(np.nanmean(1 - ((Ytr[cut:] - p) ** 2).mean(0) / Ytr[cut:].var(0).clip(1e-9)))
        if s > best_score:
            best, best_score = a, s
    model = Ridge(alpha=best).fit(Xtr, Ytr)
    pred = model.predict(Xte)

    ss_res = ((Yte - pred) ** 2).mean(0)
    ss_tot = Yte.var(0)
    r2 = 1 - ss_res / np.where(ss_tot < 1e-9, np.nan, ss_tot)
    return r2, best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smiles", type=str, default=None,
                    help="CSV with a `smiles` column, ideally drawn from OUTSIDE the pretraining "
                         "corpus. Without it the default pool is the eval molecules, which is "
                         "stated in the output rather than assumed away.")
    ap.add_argument("--out", type=str, default=str(ROOT / "analysis" / "descriptor_probe.csv"))
    args = ap.parse_args()

    stats = json.loads(STATS.read_text())
    names = stats["names"]
    mu = np.asarray(stats["mean"], float)
    sd = np.asarray(stats["std"], float)
    assert len(names) == 217, f"expected 217 fitted descriptors, found {len(names)}"

    rng = np.random.default_rng(args.seed)
    if args.smiles:
        pool = pd.read_csv(args.smiles)["smiles"].dropna().unique().tolist()
        pool_name = f"{args.smiles} (caller-supplied)"
    else:
        d = pd.read_csv(DEFAULT_POOL, usecols=["canonical_key"]).drop_duplicates()
        pool = d["canonical_key"].dropna().unique().tolist()
        pool_name = ("downstream eval molecules -- NOT verified absent from the pretraining "
                     "corpus")
    idx = rng.permutation(len(pool))[:args.n]
    smiles = [pool[i] for i in idx]
    print(f"pool: {pool_name}\n{len(smiles)} molecules, seed {args.seed}")

    t0 = time.time()
    Y_raw = descriptors(smiles, names)
    print(f"  descriptors {time.time() - t0:.0f}s")


    ok_mol = np.isfinite(Y_raw).all(1)
    Y = (Y_raw[ok_mol] - mu) / np.where(sd < 1e-9, 1.0, sd)


    Y = np.clip(Y, -10.0, 10.0)
    keep = np.isfinite(Y).all(0) & (Y.std(0) > 1e-6)
    dropped = [n for n, k in zip(names, keep) if not k]
    Y = Y[:, keep]
    kept_names = [n for n, k in zip(names, keep) if k]
    smiles = [s for s, k in zip(smiles, ok_mol) if k]
    print(f"  {len(smiles)} molecules parsed, {Y.shape[1]} of {len(names)} descriptors usable"
          + (f" (dropped: {', '.join(dropped[:6])}{'...' if len(dropped) > 6 else ''})"
             if dropped else ""))

    n_train = int(0.8 * len(smiles))
    reps = {}
    t0 = time.time()
    reps["ECFP4 (2048 bit)"] = ecfp4(smiles)
    print(f"  ECFP4 {time.time() - t0:.0f}s")
    for label, run in ENCODERS.items():
        enc = ROOT / "figure_data" / "climb_v2_phase2" / run / "encoder"
        if not enc.exists():
            print(f"  SKIP {label}: no encoder at {enc}")
            continue
        t0 = time.time()
        reps[label] = embed(smiles, enc)
        print(f"  {label} {time.time() - t0:.0f}s")

    rows = []
    for label, X in reps.items():
        r2, alpha = probe(X, Y, n_train)
        for n, v in zip(kept_names, r2):
            rows.append({"arm": label, "descriptor": n, "group": group_of(n),
                         "r2": float(v), "alpha": alpha})
    df = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)

    print(f"\nmean R^2 over {Y.shape[1]} descriptors, {len(smiles) - n_train} held-out molecules\n")
    order = [c for c in ("ECFP4 (2048 bit)", "CLIMB unsup", "CLIMB sup, desc", "CLIMB random")
             if c in df.arm.unique()]
    for how in ("mean", "median"):
        piv = df.pivot_table(index="group", columns="arm", values="r2", aggfunc=how)
        piv.loc["ALL"] = df.pivot_table(index="arm", values="r2", aggfunc=how)["r2"]
        print(f"\n{how} R^2 by descriptor group\n")
        print(piv[order].round(3).to_string())


    worst = df.groupby("descriptor").r2.mean().nsmallest(5)
    print("\nworst-decoded descriptors (mean over arms) -- these drive the mean:")
    print("   " + ", ".join(f"{n} {v:.2f}" for n, v in worst.items()))

    tex = Path(args.out).with_suffix(".tex")
    tex.write_text(latex_table(df, order, len(smiles) - n_train, pool_name))
    print(f"wrote {tex}")


TEX_COLS = [
    ("Chi (connectivity)",    "Chi"),
    ("Kappa (shape)",         "Kappa"),
    ("BCUT (eigenvalue)",     "BCUT"),
    ("Information (Ipc/SPS)", "Ipc/SPS"),
    ("EState",                "EState"),
    ("PEOE/SMR/SlogP VSA",    "VSA"),
    ("Fragment counts (fr_)", r"fr\_*"),
    ("Simple counts / bulk",  "Bulk"),
]


def latex_table(df, arm_order, n_test, pool_name):

    med = df.pivot_table(index="arm", columns="group", values="r2", aggfunc="median")
    allc = df.pivot_table(index="arm", values="r2", aggfunc="median")["r2"]
    n_desc = int(df.descriptor.nunique())

    head = " & ".join(r"\textbf{%s}" % short for _, short in TEX_COLS)
    rows = []
    for a in arm_order:
        cells = [("%.2f" % med.loc[a, g]) if g in med.columns else "---" for g, _ in TEX_COLS]
        rows.append("%s & %s & %.2f \\\\" % (a, " & ".join(cells), allc[a]))
    body = "\n".join(rows)

    tmpl = r"""\begin{table}[h]
\centering
\footnotesize
\setlength{\tabcolsep}{4.5pt}
\caption{\textbf{How much of the descriptor space each representation encodes.} Median $R^2$ of a
ridge head fit from the frozen representation to each of __NDESC__ RDKit descriptors, on __NTEST__
held-out molecules. A head is trained from scratch for every row; the representation is the only
thing that varies. Targets are $z$-scored with the pretraining corpus statistics and clipped at
$\pm10$, matching the descriptor objective. Median rather than mean because two descriptors
(Kappa3, Ipc) are exponential in molecular size and reach large negative $R^2$ after clipping.
\emph{CLIMB sup, desc was trained on these exact targets} and is included as a manipulation check,
not as a comparison.}
\label{tab:descriptor_probe}
\begin{tabular}{@{}l cccc cccc @{\hspace{6pt}}||@{\hspace{6pt}} c@{}}
\toprule
& \multicolumn{4}{c}{\emph{topology and shape}}
& \multicolumn{4}{c}{\emph{substructure and electronic}} & \\
\cmidrule(lr){2-5} \cmidrule(lr){6-9}
\textbf{Representation} & __HEAD__ & \textbf{ALL} \\
\midrule
__BODY__
\bottomrule
\end{tabular}

\vspace{2pt}
{\footnotesize Molecule pool: __POOL__.}
\end{table}
"""
    return (tmpl.replace("__NDESC__", str(n_desc))
                .replace("__NTEST__", "{:,}".format(n_test))
                .replace("__HEAD__", head)
                .replace("__BODY__", body)
                .replace("__POOL__", pool_name))


if __name__ == "__main__":
    main()
