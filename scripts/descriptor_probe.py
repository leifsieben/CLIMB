"""Can each representation predict RDKit descriptors? A linear probe, runnable on a laptop.

    python3 scripts/descriptor_probe.py --n 10000

WHAT IT ASKS. The 217 RDKit descriptors are deterministic functions of structure, so this is not a
prediction task in the usual sense -- it is a DECODABILITY test. Fit a ridge head from each frozen
representation to the descriptor vector and ask how much of it survives the encoding.

WHY THE ARMS ARE WHAT THEY ARE:

    ECFP4              the honest baseline, and the one to beat. Many descriptors are close to
                       linear functions of substructure counts, which is exactly what a Morgan
                       fingerprint is. If ECFP4 decodes a descriptor, no embedding deserves credit
                       for decoding it too.
    CLIMB unsup        the actual question: does masked-language modelling on SMILES learn
                       descriptor-relevant structure WITHOUT being told to?
    CLIMB sup, desc    trained on precisely these 217 targets, so it is a MANIPULATION CHECK, not
                       a finding. If it does not win here, something is wrong with the probe.
                       What is informative is HOW MUCH it retains once its regression head is
                       discarded and only the encoder is kept.
    random encoder     the floor. An untrained network of the same width still projects structure
                       into 512 dimensions, and a ridge head can read a surprising amount out of a
                       random projection. Anything a representation scores above this is what
                       PRETRAINING bought; the rest is architecture and dimensionality.

THE HEAD IS TRAINED FROM SCRATCH FOR EVERY ARM (Leif 2026-08-26: "obviously use the models only as
embedding and we train our own heads each time from scratch"). Same molecules, same split, same
alpha grid, same target scaling -- the representation is the only thing that varies.

TARGETS ARE Z-SCORED WITH THE PRETRAINING CORPUS STATISTICS, not with this sample's. Using the
sample's own mean and SD would make R^2 a statement about this 10k rather than about the target
space CLIMB was trained on, and would silently change what "sup, desc was trained on these targets"
means. configs/descriptor_stats.json is the same file the pretraining runs load.

CAVEAT THIS SCRIPT CANNOT FIX BY ITSELF -- see --smiles. The default molecule pool is the
downstream evaluation molecules, which are NOT guaranteed absent from the pretraining corpus. For
`sup, desc` specifically, a molecule seen during pretraining came WITH its descriptor targets, so
part of any advantage could be memorisation rather than encoding. Pass --smiles with a file of
molecules drawn from outside the pretraining corpus to remove that; the run prints which pool it
used so a result can never be quoted without it.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
STATS = ROOT / ".hf_staging" / "pretrain" / "configs" / "descriptor_stats.json"
TOKENIZER = ROOT / ".hf_staging" / "pretrain" / "tokenizer"
DEFAULT_POOL = (ROOT / "figure_data" / "climb_v2_phase2" / "skip_dense_8M" /
                "moleculenet_cv" / "test_predictions.csv")

# arm -> encoder directory. ECFP4 has no encoder and is handled separately.
ENCODERS = {
    "CLIMB unsup":      "unsup_8M",
    "CLIMB sup, desc":  "skip_dense_8M",
    "CLIMB random":     "random_baseline_00",
}

# Descriptor groups, by name prefix, coarsest-useful. The point of grouping is that the SIMPLE
# counts (MolWt, NumHDonors, ring counts) are near-deterministic functions of substructure and any
# fingerprint will decode them -- so a single macro R^2 is dominated by the descriptors that carry
# no information about the comparison. The groups that separate methods are the shape, topology and
# electronic ones.
GROUPS = [
    ("Chi (connectivity)",   lambda n: n.startswith("Chi")),
    ("Kappa (shape)",        lambda n: n.startswith("Kappa") or n == "HallKierAlpha"),
    ("BCUT (eigenvalue)",    lambda n: n.startswith("BCUT")),
    ("EState",               lambda n: "EState" in n),
    ("PEOE/SMR/SlogP VSA",   lambda n: "VSA" in n),
    ("Fragment counts (fr_)", lambda n: n.startswith("fr_")),
    ("Information (Ipc/SPS)", lambda n: "Ipc" in n or n in ("SPS", "Phi")),
    ("Simple counts / bulk", lambda n: True),          # catch-all, evaluated last
]


def group_of(name):
    for label, pred in GROUPS:
        if pred(name):
            return label
    return "other"


def descriptors(smiles, names):
    """(n_mol, 217) descriptor matrix in the SAME ORDER as the fitted statistics.

    Order matters and is asserted rather than assumed: z-scoring column i by descriptor j's mean
    produces a trained model and a meaningless one. This is the fig_F/rdkit-shadowing failure in
    miniature -- the count can match while the mapping does not.
    """
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
            # token_type_ids is emitted by the tokenizer and NOT accepted by this encoder; passing
            # everything the tokenizer returns is the obvious thing and it raises.
            e = {k: v.to(dev) for k, v in e.items() if k in ("input_ids", "attention_mask")}
            h = model(**e).last_hidden_state
            mask = e["attention_mask"].unsqueeze(-1).float()
            out.append(((h * mask).sum(1) / mask.sum(1)).float().cpu().numpy())
    return np.vstack(out)


def probe(X, Y, n_train, alphas=(1.0, 10.0, 100.0, 1000.0)):
    """Ridge from X to every column of Y; per-column R^2 on the held-out tail.

    One alpha for the whole target block, chosen on a validation slice of the TRAINING half only.
    Per-column alpha selection would let each descriptor pick its own capacity and quietly turn a
    representation comparison into a regularisation search.
    """
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
    # R^2 against the TEST block's own variance, per descriptor.
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
                     "corpus; see the module docstring")
    idx = rng.permutation(len(pool))[:args.n]
    smiles = [pool[i] for i in idx]
    print(f"pool: {pool_name}\n{len(smiles)} molecules, seed {args.seed}")

    t0 = time.time()
    Y_raw = descriptors(smiles, names)
    print(f"  descriptors {time.time() - t0:.0f}s")

    # Drop molecules RDKit could not parse, and descriptors that are constant or unusable on this
    # sample. Reported, never silent: a probe scored on a shrinking target set while claiming 217
    # is the same defect as a caption that keeps its old count.
    ok_mol = np.isfinite(Y_raw).all(1)
    Y = (Y_raw[ok_mol] - mu) / np.where(sd < 1e-9, 1.0, sd)
    # CLIPPED AT +-10, WHICH IS THE PRETRAINING CONVENTION AND NOT A CONVENIENCE. Ipc and its
    # relatives are exponential in molecular size, so a single large molecule sends a z-score past
    # float32 range -- the first run of this script died on exactly that. More importantly, CLIMB's
    # descriptor objective was trained against CLIPPED targets, so probing against unclipped ones
    # would score the representations on a target space no arm ever saw.
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
    # MEDIAN IS NOT DECORATION HERE. Ipc and its relatives are exponential in molecular size, so
    # after clipping their test variance is carried by a handful of extreme molecules and a single
    # bad prediction drives R^2 to large negatives. The mean over 212 descriptors inherits that;
    # the median does not. Report both and say which the sentence is about.
    worst = df.groupby("descriptor").r2.mean().nsmallest(5)
    print("\nworst-decoded descriptors (mean over arms) -- these drive the mean:")
    print("   " + ", ".join(f"{n} {v:.2f}" for n, v in worst.items()))

    tex = Path(args.out).with_suffix(".tex")
    tex.write_text(latex_table(df, order, len(smiles) - n_train, pool_name))
    print(f"wrote {tex}")


# Short column headers, and the ORDER IS THE ARGUMENT: the four families a graph-based encoder
# should be good at first, the four a substructure fingerprint should be good at second. A reader
# who sees the blocks in that order gets the finding from the layout; alphabetical order hides it.
TEX_COLS = [
    ("Chi (connectivity)",    "Chi"),
    ("Kappa (shape)",         "Kappa"),
    ("BCUT (eigenvalue)",     "BCUT"),
    ("Information (Ipc/SPS)", "Ipc/SPS"),
    ("EState",                "EState"),
    ("PEOE/SMR/SlogP VSA",    "VSA"),
    ("Fragment counts (fr_)", "fr\_*"),
    ("Simple counts / bulk",  "Bulk"),
]


def latex_table(df, arm_order, n_test, pool_name):
    """Transposed probe table: models as rows, descriptor groups as columns (Leif 2026-08-26).

    MEDIAN, not mean. Kappa3 and Ipc are exponential in molecular size and reach large negative
    R^2 after the +-10 clipping the pretraining objective uses, so the mean over ~212 descriptors
    is carried by two pathological columns. The caption states the choice rather than leaving a
    reader to wonder which statistic was taken and why.
    """
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
