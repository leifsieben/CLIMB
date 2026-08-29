"""Can a TREE tell an edited molecule from its parent? -- the fig_G resolution metric, as a
held-out classification AUC rather than a magnitude ratio.

WHY THIS EXISTS
---------------
fig_G reports response as an RMS per-dimension shift divided by that arm's own shift when a
different compound of matched MW is substituted. Leif's objection to a magnitude axis: XGBoost
splits on SINGLE dimensions, so a model does not need a large shift, it needs ONE coordinate that
separates the pair. A magnitude axis systematically understates a representation whose information
is concentrated -- one flipped bit in 2048 is a negligible norm change however decisive it is.

A dimension COUNT was proposed for this in 2026-08-25 and rejected on measurement
(notes/figG-resolution-metric.md): it is threshold-critical exactly where the CLM arms live, and
for a continuous embedding the natural calibration is degenerate. This is a third option and
inherits neither problem.

THE MEASUREMENT. For one (arm, edit): label every parent molecule 0 and its edited partner 1, fit
a gradient-boosted tree, report held-out ROC-AUC. 0.5 = cannot tell them apart; 1.0 = separates
every held-out pair. No threshold, no normalisation, no free parameter, and it is scale-free, so a
2048-bit fingerprint and a 512-d transformer sit on one axis with nothing to calibrate.

SPLIT BY PAIR, NEVER BY MOLECULE. Both members of a pair go to the same side, so the model is never
shown a molecule in training and asked about its partner at test. A molecule-level split lets
memorisation pass for resolution -- the model learns "this scaffold is class 1" rather than any
signature of the edit. Asserted below rather than intended: the two halves share no molecule.

THE DIRECTIONALITY CAVEAT, WHICH IS NOT RESOLVED AND MUST BE READ WITH THE NUMBERS. This metric is
well-posed only when the A/A' label means something consistent across pairs. `mode_stereo_flip`
inverts the FIRST stereocentre of whatever configuration a molecule happens to have, so A is "as
found" and A' is "inverted" -- not a consistent R->S. Across scaffolds the label is then partly
arbitrary, and an arm can read ~0.5 because the question is ill-posed rather than because it
cannot resolve the edit. The same applies to ring_size (cyclopentyl <-> cyclohexyl). c_to_n,
add_methyl, add_fluorine, isotope_13c and ch2_homologue ARE directional and
serve as the controls that separate "ill-posed" from "unresolved": if a directional edit scores
high for an arm and a symmetric one scores 0.5, the 0.5 is about the label, not the arm.

matched_descriptors is expected at ~0.5 for every arm BY CONSTRUCTION -- with two unrelated
molecules the A/A' assignment is arbitrary. It is the sanity check on the harness, not a result,
and fig_G does NOT draw it: on a [0.5, 1] axis an empty bar there would mean "ill-posed" where
everywhere else it means "not resolved", and one bar cannot carry two meanings. matched_mw, the
other such mode, was removed from the pair set entirely (Leif 2026-08-28: "100% not needed").

Class B (smiles_enumeration, kekule, symmetry_equivalent) INVERTS THE READING: these are pairs
that denote the same molecule, so a HIGH score is a failure -- the arm is reading notation.

Replicates are over the SPLIT SEED. Every representation here is deterministic (fig_G verifies
128/128 bit-exact re-embedding), so there is nothing else stochastic: the arm has no training seed
to vary, and the spread reported is the spread of the estimate, not of the model.

Writes: figure_data/embedding_resolution/separability_auc.csv
Run:  python3 scripts/resolution_separability.py
"""
from __future__ import annotations
import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "figure_data" / "embedding_resolution"
EMB = OUT / "embeddings"

# Same convention fig_G asserts: class A is a chemistry question and is asked on canonical input;
# class B IS the notation question and must be asked as written. Drawing either on the other's
# input inverts the answer, so the suffix is derived from the class and never passed in.
INPUT_OF = {"A": "_canonical", "B": ""}

# embedding name -> fig_G's `short`. Copied from scripts/resolution_relative_response.py; an arm
# absent here is still measured and written with short = "", so this is a labelling decision.
SHORT = {"ECFP4+stereo": "ECFP", "ECFP4+desc": "ECFP+d",
         "Morgan r3-counts": "r3fp", "Morgan r3-cnt+desc": "r3fp+d",
         "CLIMB unsup (enum-aug)": "uns-ENUM", "CLIMB unsup (canon-ctrl)": "uns-CANON",
         "CLIMB sup": "sup", "CLIMB unsup": "uns-MAIN",
         "random encoder": "rand", "ECFP4 stereo-blind": "ECFP-blind",
         "CheMeleon": "chemeleon",
         # The free-information floor: char n-grams of the SMILES, no chemistry. Whatever it
         # reaches on a panel is available to any string model for free.
         "notation (char n-gram)": "notation"}

# Edits whose A/A' direction is chemically consistent. Recorded per row so a reader can separate
# "scores 0.5 because the arm cannot resolve it" from "scores 0.5 because the label is arbitrary"
# without going back to the pair generator.
DIRECTIONAL = {"c_to_n", "add_methyl", "add_fluorine", "isotope_13c", "ch2_homologue",
               "regioisomer",   # para is always A: the template fixes the direction
               "smiles_enumeration", "kekule", "symmetry_equivalent"}
# Pairs of unrelated molecules: ~0.5 by construction, the harness's own control.
NULL_MODES = {"matched_mw", "matched_descriptors"}

SEEDS = [0, 1, 2, 3, 4]
TEST_FRAC = 0.2
PARAMS = dict(n_estimators=200, max_depth=4, learning_rate=0.3, n_jobs=8,
              tree_method="hist", verbosity=0)


def emb_file(name: str, suffix: str) -> Path:
    return EMB / f"{name.replace(' ', '_').replace('+', '_')}{suffix}.npz"


def load_lookup(name: str, suffix: str):
    """{smiles: vector} for one arm on one input convention."""
    f = emb_file(name, suffix)
    if not f.exists():
        return None
    z = np.load(f, allow_pickle=True)
    smi, X = z["smiles"], z["X"]
    assert len(smi) == len(X), f"{f.name}: {len(smi)} smiles vs {len(X)} vectors"
    return {s: i for i, s in enumerate(smi)}, X


def _groups(a, b, keep):
    """Pair indices grouped into connected components that SHARE A MOLECULE.

    Splitting on the pair index is not enough: molecules recur across pairs -- a handful in
    c_to_n, regioisomer and ring_size, and 835 in the since-removed matched_mw -- so pair 3 and 47 can
    hold the same molecule and a pair-level split puts it on both sides. That is the leak the
    assertion below caught on the first full run. Whole components go to one side, which is the
    only split under which "the model never saw this molecule" is actually true.
    """
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[rx] = ry

    for k in keep:
        union(("m", a[k]), ("m", b[k]))
        union(("p", k), ("m", a[k]))
    comps = {}
    for k in keep:
        comps.setdefault(find(("p", k)), []).append(k)
    return list(comps.values())


def run_cell(X_lookup, pairs, seed):
    """Held-out ROC-AUC for one (arm, mode) at one split seed.

    Returns (auc, n_pairs, n_groups, n_degenerate, largest_group). auc is None when the cell
    cannot be scored.
    """
    idx, X = X_lookup
    a = [idx.get(s) for s in pairs.smiles_a]
    b = [idx.get(s) for s in pairs.smiles_b]
    keep = [k for k in range(len(pairs)) if a[k] is not None and b[k] is not None]
    # DEGENERACY IS A PROPERTY OF THE VECTORS, NOT OF THE STRINGS. Comparing embedding ROW ids
    # only catches pairs written identically; what matters is pairs the ARM cannot possibly
    # distinguish because it maps them to the same point. The two differ enormously: ECFP4 maps
    # 952 of 1,000 isotope pairs and 65 of 1,000 stereo pairs to bit-identical vectors, none of
    # which share a SMILES. Without this, ECFP4's 0.503 on isotopes reads as "failed to learn"
    # when it is "cannot possibly differ" -- a collision claim that needs no threshold and no
    # metric, and the strongest statement this pipeline can make.
    #
    # Degenerate pairs are KEPT in the fit. They are real evidence and they correctly drag the
    # score toward 0.5; the count is reported beside the score so a reader can see how much of a
    # low AUC is collision rather than weak signal.
    degenerate = int((~(X[[a[k] for k in keep]] != X[[b[k] for k in keep]]).any(axis=1)).sum())
    if len(keep) < 50:
        return None, len(keep), 0, degenerate, 0

    comps = _groups(a, b, keep)
    # A mode whose molecules recur heavily can collapse into one giant component, and then no
    # honest 20% split exists. Reported rather than assumed: the worst case measured was
    # matched_mw (835 molecules in >1 pair), which still yielded 165 components with the largest
    # at 3.7% of pairs -- but the number belongs in the CSV, not in a memory of having checked
    # once, because the next mode added may not be so well behaved.
    largest = max(len(c) for c in comps)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(comps))
    target = max(1, int(round(TEST_FRAC * len(keep))))
    test_pairs, got = set(), 0
    for ci in order:
        if got >= target:
            break
        test_pairs.update(comps[ci])
        got += len(comps[ci])

    rows_tr, y_tr, rows_te, y_te = [], [], [], []
    for k in keep:
        dst_rows, dst_y = ((rows_te, y_te) if k in test_pairs else (rows_tr, y_tr))
        dst_rows.append(a[k]); dst_y.append(0)
        dst_rows.append(b[k]); dst_y.append(1)
    # Kept even though the component split makes it true by construction: it is the difference
    # between a generalisation test and a memorisation test, and a leak would raise the AUC, so
    # the failure mode looks like a better result.
    assert not (set(rows_tr) & set(rows_te)), "pair split leaked a molecule across the boundary"

    if len(set(y_te)) < 2 or not rows_tr:
        return None, len(keep), len(comps), degenerate, largest
    clf = xgb.XGBClassifier(**PARAMS, random_state=seed)
    clf.fit(X[rows_tr], np.array(y_tr))
    p = clf.predict_proba(X[rows_te])[:, 1]
    return float(roc_auc_score(y_te, p)), len(keep), len(comps), degenerate, largest


def main() -> int:
    pairs_by_input = {s: pd.read_csv(OUT / f"pairs{'_canonical' if s else ''}.csv")
                      for s in set(INPUT_OF.values())}
    arms = sorted({f.name.replace("_canonical", "").replace(".npz", "")
                   for f in EMB.glob("*.npz")})
    # Recover the display names from SHORT so the CSV joins to fig_G, but measure every arm on
    # disk -- an arm absent from SHORT gets short="" rather than being filtered out.
    disk_to_name = {k.replace(' ', '_').replace('+', '_'): k for k in SHORT}
    rows = []
    for arm_disk in arms:
        name = disk_to_name.get(arm_disk, arm_disk)
        for klass, suffix in INPUT_OF.items():
            look = load_lookup(name, suffix)
            if look is None:
                continue
            pt = pairs_by_input[suffix]
            for mode in sorted(set(pt.loc[pt.klass == klass, "mode"])):
                sub = pt[(pt["mode"] == mode) & (pt.klass == klass)]
                aucs, n_used, n_grp, n_deg, n_big = [], 0, 0, 0, 0
                for seed in SEEDS:
                    auc, n_used, n_grp, n_deg, n_big = run_cell(look, sub, seed)
                    if auc is not None:
                        aucs.append(auc)
                if not aucs:
                    print(f"  SKIP {name} / {mode}: only {n_used} usable pairs")
                    continue
                rows.append(dict(
                    embedding=name, short=SHORT.get(name, ""), klass=klass, mode=mode,
                    input="canonical" if suffix else "as_written",
                    auc_mean=round(float(np.mean(aucs)), 4),
                    auc_sd=round(float(np.std(aucs, ddof=1)), 4) if len(aucs) > 1 else 0.0,
                    auc_min=round(float(np.min(aucs)), 4),
                    auc_max=round(float(np.max(aucs)), 4),
                    n_seeds=len(aucs), n_pairs=n_used, n_groups=n_grp,
                    n_degenerate=n_deg, largest_group=n_big,
                    directional=int(mode in DIRECTIONAL),
                    null_control=int(mode in NULL_MODES)))
                r = rows[-1]
                print(f"  {name:26s} {mode:20s} AUC {r['auc_mean']:.3f} "
                      f"+/- {r['auc_sd']:.3f}  (n={n_used}, groups={n_grp}"
                      f"{f', DEGENERATE {n_deg}' if n_deg else ''})")

    dst = OUT / "separability_auc.csv"
    fields = ["embedding", "short", "klass", "mode", "input", "auc_mean", "auc_sd",
              "auc_min", "auc_max", "n_seeds", "n_pairs", "n_groups", "n_degenerate",
              "largest_group", "directional", "null_control"]
    with dst.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {dst}  {len(rows)} rows  (xgboost {xgb.__version__}, "
          f"{len(SEEDS)} split seeds, {int(TEST_FRAC * 100)}% held out by PAIR)")

    # The null modes are the harness's own control. Say what they read rather than assuming they
    # came out right -- if these are far from 0.5 the split is leaking and every other number here
    # is suspect.
    nul = [r for r in rows if r["null_control"]]
    if nul:
        v = [r["auc_mean"] for r in nul]
        print(f"NULL CONTROLS (unrelated molecules, expect ~0.5): "
              f"{min(v):.3f}-{max(v):.3f} over {len(v)} cells")
    return 0


if __name__ == "__main__":
    sys.exit(main())
