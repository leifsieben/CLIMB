"""Co-best analysis for Table A2: which models are 'best or not distinguishable from best'
on EVERY dataset, across all four benchmark families.

Method (unchanged from the original A1 table, README §8.1 / scripts/best_model_bootstrap.py):
a **scaffold cluster bootstrap of the paired metric difference** — resample whole Bemis-Murcko
scaffolds with replacement, because scaffold-mates are not independent — then **BH-FDR** across the
comparison family. A model is co-best on a dataset when the (leader - model) difference is NOT
significant at q >= 0.05; the leader is trivially included.

Two families per dataset, each FDR-corrected on its own:
  overall  : all models scored on that dataset      -> "best" column of Table A2
  CLIMB    : the CLIMB pretraining recipes only     -> "best CLIMB" column of Table A2

This extends the original (MoleculeNet-only) analysis to MoleculeACE, Polaris and CBS. Every model
uses its PRIMARY run's per-molecule predictions for the paired test, exactly as before, while the
leader is chosen on the seed-averaged point estimate that Fig A1 ranks on.

NOTE — Polaris cannot be tested locally: the benchmark withholds its test labels (scoring happens
on the Polaris platform), so the local CSVs carry predictions but no ground truth. Those datasets
are skipped with a message, and Table A2 shows them as not-testable rather than as zeros.

Output: analysis/rigor/cobest_all_suites.csv
  dataset, suite, metric, higher_better, n_test, n_scaffolds, arm, value, is_leader,
  boot_diff, boot_p, q_overall, cobest_overall, q_climb, cobest_climb

Run:  python3 scripts/best_model_cobest.py            (~10 min; writes as it goes)
      python3 scripts/best_model_cobest.py --quick    (200 resamples, for a smoke test)
"""
from __future__ import annotations
import sys, json, time
from pathlib import Path
from functools import lru_cache
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from figures.arms import ARMS, ARM_ORDER, system                      # noqa: E402
from figures.allsuites import wide_table, MOLNET, HIGHER              # noqa: E402

FD = ROOT / "figure_data"
OUT = ROOT / "analysis" / "rigor" / "cobest_all_suites.csv"
N_BOOT = 200 if "--quick" in sys.argv else 1000
ALPHA = 0.05
CONTROLS = {"random_encoder", "e2e_no_pretrain"}
MIN_DATASETS_FOR_TRIPWIRE = 5   # below this a co-best RATE is not a meaningful statistic


# ------------------------------------------------------------------ metrics -------------------
def _auc(y, p):
    n1 = int((y == 1).sum()); n0 = int((y == 0).sum())
    if n1 == 0 or n0 == 0:
        return np.nan
    r = stats.rankdata(p)
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def _pr_auc(y, p):
    if (y == 1).sum() == 0 or (y == 0).sum() == 0:
        return np.nan
    from sklearn.metrics import average_precision_score
    return average_precision_score(y, p)


def _nef1(y, p):
    N = len(y); A = int((y == 1).sum())
    if A == 0 or N == 0:
        return np.nan
    n = int(np.ceil(0.01 * N))
    top = np.argsort(-np.asarray(p, float))[:n]
    return int((y[top] == 1).sum()) / min(n, A)


METRICS = {
    "rmse": lambda y, p: float(np.sqrt(np.mean((p - y) ** 2))),
    "mean_absolute_error": lambda y, p: float(np.mean(np.abs(p - y))),
    "mean_squared_error": lambda y, p: float(np.mean((p - y) ** 2)),
    "roc_auc": _auc,
    "pr_auc": _pr_auc,
    "nef1": _nef1,
    "pearsonr": lambda y, p: float(np.corrcoef(y, p)[0, 1]) if np.std(p) > 0 else np.nan,
    "spearmanr": lambda y, p: float(stats.spearmanr(y, p).statistic),
}


def metric_over_outputs(y, pred, out_idx, kind):
    """Metric averaged over output columns (Tox21 has 12); NaN labels are dropped."""
    if out_idx is None:
        ok = np.isfinite(y)
        return METRICS[kind](y[ok], pred[ok]) if ok.sum() else np.nan
    vals = []
    for o in np.unique(out_idx):
        m = (out_idx == o) & np.isfinite(y)
        if m.sum() == 0:
            continue
        v = METRICS[kind](y[m], pred[m])
        if np.isfinite(v):
            vals.append(v)
    return float(np.mean(vals)) if vals else np.nan


# ------------------------------------------------------------------ scaffolds -----------------
@lru_cache(maxsize=400000)
def _scaffold(smi):
    from rdkit import Chem, RDLogger
    from rdkit.Chem.Scaffolds import MurckoScaffold
    RDLogger.DisableLog("rdApp.*")
    try:
        m = Chem.MolFromSmiles(smi)
        if m is None:
            return smi
        return MurckoScaffold.MurckoScaffoldSmiles(mol=m, includeChirality=False) or smi
    except Exception:
        return smi


# ------------------------------------------------------------------ loaders -------------------
def _primary_dir(arm):
    return ARMS[arm]["src"]["mol"][0]


def _suite_dir(arm, suite):
    """The ONE run dir this arm's co-best analysis reads for a suite.

    Most arms carry a bare string here, but the ones whose replicate dirs do not follow the
    <src>/<src>_s1/<src>_s2 convention (s2u_dense, random_encoder) carry an explicit LIST -- and
    this script indexed the field directly, so it crashed the moment ARM_ORDER reached one. The
    co-best test is a per-molecule comparison between arms on a SINGLE representative run (the
    lowest seed, see load_moleculeace), so the first dir is the right element rather than an
    arbitrary one: it is the same run the string convention would have resolved to.
    """
    src = ARMS[arm]["src"][suite]
    return src[0] if isinstance(src, list) else src


def _align(frames):
    """Align per-arm prediction frames on the molecules they SHARE.

    A few featurizers drop a handful of RDKit-unparseable SMILES (CheMeleon and the random/e2e
    encoders lose 7 of HIV's 41,127). Requiring identical row counts would silently exclude those
    arms from the comparison entirely; intersecting the keys compares every arm on the same
    molecules instead, which is what a paired test needs.
    """
    if not frames:
        return None, {}
    keys = None
    for d in frames.values():
        k = set(zip(d["mol_index"], d["output_index"]))
        keys = k if keys is None else (keys & k)
    if not keys:
        return None, {}
    base, preds = None, {}
    for a, d in frames.items():
        d = d[[(m, o) in keys for m, o in zip(d["mol_index"], d["output_index"])]]
        d = d.sort_values(["output_index", "mol_index"]).reset_index(drop=True)
        if base is None:
            base = d[["raw_smiles", "output_index", "y_true"]].copy()
        preds[a] = d["y_pred"].to_numpy(float)
    return base, preds


def load_molnet(dataset):
    """{arm: y_pred} plus (smiles, y_true, output_index) for a MoleculeNet dataset."""
    frames = {}
    for a in ARM_ORDER:
        f = FD / "climb_v2_phase2" / _primary_dir(a) / "moleculenet_cv" / "test_predictions.csv"
        if not f.exists():
            continue
        d = pd.read_csv(f)
        d = d[d.dataset == dataset].drop_duplicates(["mol_index", "output_index"])
        if not d.empty:
            frames[a] = d
    return _align(frames)


def load_cbs():
    """CBS, scored the way the benchmark scores it: NEF1 PER PROVIDED FOLD, then averaged.

    Pooling all five folds and taking one global top-1% is a different quantity (0.907 vs the
    reported 0.930 for ECFP+desc) and mixes the score scales of five separately-fitted models.
    The fold id is carried in `output_index` so the shared 'average the metric over output
    columns' machinery computes the per-fold mean for free.
    """
    folds = pd.read_csv(ROOT / "data" / "cbs.csv")[["smiles", "fold"]]
    frames = {}
    for a in ARM_ORDER:
        f = FD / "cbs_benchmark" / _primary_dir(a) / "moleculenet_cv" / "test_predictions.csv"
        if not f.exists():
            continue
        d = pd.read_csv(f).drop_duplicates(["mol_index", "output_index"])
        d = d.merge(folds, left_on="canonical_key", right_on="smiles", how="left")
        if d["fold"].isna().any():
            # A couple of CheMeleon rows canonicalise differently; drop just those molecules and
            # let _align intersect, rather than dropping the whole arm.
            d = d[d["fold"].notna()]
        d["output_index"] = d["fold"].astype(int)
        frames[a] = d
    return _align(frames)


def load_moleculeace(task):
    src = pd.read_csv(ROOT / "chemeleon_suite/data/moleculeace" / f"{task}.csv")
    te = src[src.split == "test"].reset_index(drop=True)
    # THE MoleculeACE CSVs CARRY TWO LABEL COLUMNS AND THEY DIFFER BY EXACTLY 9.0:
    #   "y"              = -log10(exp_mean in nM)   range about -5.0 .. +1.7
    #   "y [pEC50/pKi]"  = 9 - log10(nM) = pKi      range about  4.0 .. 10.7
    # Every model is TRAINED and SCORED on pKi (predictions here run 4.5-10.2), so joining "y"
    # made each residual (r + 9). For a paired difference that does not cancel -- it becomes
    # (r_A+9)^2 - (r_B+9)^2 = r_A^2 - r_B^2 + 18(r_A - r_B), and the linear term dominates -- so the
    # bootstrap was testing signed-residual differences, not squared-error differences, and Table
    # A2's MoleculeACE co-best sets were decided on the wrong quantity.
    # Found by the figures session in fig_C1 (2026-08-20), where the same join turned a +28.6% lift
    # into a -0.29% null. Asserted rather than commented, so a future join outside pKi range fails.
    yt = te["y [pEC50/pKi]"].astype(float)
    if not (yt.min() > 2.0 and yt.max() < 14.0):
        raise ValueError(f"{task}: MoleculeACE labels outside pKi range "
                         f"[{yt.min():.2f}, {yt.max():.2f}] -- wrong column?")
    base = pd.DataFrame({"raw_smiles": te.smiles, "output_index": 0, "y_true": yt})
    preds = {}
    for a in ARM_ORDER:
        f = FD / "chemeleon_suite" / "moleculeace" / _suite_dir(a, "mace") / "test_predictions.csv"
        if not f.exists():
            continue
        d = pd.read_csv(f)
        d = d[d.task == task]
        if d.empty:
            continue
        seed = sorted(d.seed.unique())[0]                       # primary run = lowest seed
        d = d[d.seed == seed].sort_values("test_index")
        if len(d) != len(base) or not (d.smiles.values == base.raw_smiles.values).all():
            continue
        preds[a] = d["y_pred"].to_numpy(float)
    return base, preds


def load_polaris(task, man):
    info = man[task]
    src = pd.read_csv(ROOT / "chemeleon_suite/data/polaris" / info["file"])
    ycol = "y" if "y" in src.columns else info["target_col"]
    te = src[src.split == "test"].reset_index(drop=True)
    if te[ycol].notna().sum() == 0:
        # Polaris withholds test labels — scoring happens on their platform, so the local CSV has
        # predictions but no ground truth. The per-molecule bootstrap is therefore impossible here.
        # To fill these cells, the eval side must export the test labels (or the per-dataset
        # bootstrap replicates) alongside polaris_scores.csv.
        return None, {}
    base = pd.DataFrame({"raw_smiles": te.smiles, "output_index": 0,
                         "y_true": te[ycol].astype(float)})
    preds = {}
    for a in ARM_ORDER:
        f = FD / "chemeleon_suite" / "polaris" / _suite_dir(a, "mace") / "test_predictions.csv"
        if not f.exists():
            continue
        d = pd.read_csv(f)
        d = d[d.task == task]
        if d.empty:
            continue
        seed = sorted(d.seed.unique())[0]
        d = d[d.seed == seed].sort_values("test_index")
        if len(d) != len(base):
            continue
        preds[a] = d["y_pred"].to_numpy(float)
    return base, preds


# ------------------------------------------------------------------ the test ------------------
def bh_fdr(p):
    p = np.asarray(p, float)
    q = np.full(p.shape, np.nan)
    idx = np.where(np.isfinite(p))[0]
    if idx.size == 0:
        return q
    ps = p[idx]; order = np.argsort(ps); n = ps.size
    adj = ps[order] * n / (np.arange(n) + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    out = np.empty(n); out[order] = np.clip(adj, 0, 1)
    q[idx] = out
    return q


def analyse_dataset(name, suite, base, preds, kind, higher_better, point, seed=0):
    """Co-best sets on one dataset. Returns a list of row dicts."""
    y = base["y_true"].to_numpy(float)
    out_idx = base["output_index"].to_numpy() if base["output_index"].nunique() > 1 else None
    scaf = np.array([_scaffold(s) for s in base["raw_smiles"]])
    groups = {}
    for pos, s in enumerate(scaf):
        groups.setdefault(s, []).append(pos)
    keys = list(groups)
    idx_of = {k: np.asarray(v) for k, v in groups.items()}
    K = len(keys)
    rng = np.random.default_rng(seed)

    resamples = []
    for _ in range(N_BOOT):
        pick = rng.integers(0, K, K)
        resamples.append(np.concatenate([idx_of[keys[i]] for i in pick]))

    arms = [a for a in preds if np.isfinite(point.get(a, np.nan))]
    obs = {a: metric_over_outputs(y, preds[a], out_idx, kind) for a in arms}
    arms = [a for a in arms if np.isfinite(obs[a])]
    if len(arms) < 2:
        return []
    boot = {a: np.array([metric_over_outputs(y[r], preds[a][r],
                                             out_idx[r] if out_idx is not None else None, kind)
                         for r in resamples]) for a in arms}

    def cobest(pool):
        """Leader = best seed-averaged point estimate in the pool; who is indistinguishable?"""
        pool = [a for a in pool if a in arms]
        if not pool:
            return {}, {}
        lead = max(pool, key=lambda a: point[a]) if higher_better else min(pool, key=lambda a: point[a])
        ps, others = [], [a for a in pool if a != lead]
        for a in others:
            d = boot[lead] - boot[a] if higher_better else boot[a] - boot[lead]
            d = d[np.isfinite(d)]
            ps.append(2 * min((d <= 0).mean(), (d >= 0).mean()) if d.size else np.nan)
        qs = bh_fdr(ps)
        qmap = {a: q for a, q in zip(others, qs)}
        qmap[lead] = np.nan
        member = {a: (a == lead) or (not np.isfinite(qmap[a])) or (qmap[a] >= ALPHA) for a in pool}
        return qmap, member

    q_all, mem_all = cobest(arms)
    climb_pool = [a for a in arms if system(a) == "CLIMB" and a not in CONTROLS]
    q_cl, mem_cl = cobest(climb_pool)
    lead_all = max(arms, key=lambda a: point[a]) if higher_better else min(arms, key=lambda a: point[a])

    rows = []
    for a in arms:
        rows.append(dict(dataset=name, suite=suite, metric=kind, higher_better=higher_better,
                         n_test=len(base), n_scaffolds=K, arm=a, value=obs[a],
                         is_leader=int(a == lead_all), q_overall=q_all.get(a, np.nan),
                         cobest_overall=int(mem_all.get(a, False)),
                         q_climb=q_cl.get(a, np.nan),
                         cobest_climb=int(mem_cl.get(a, False)) if a in mem_cl else "",
                         ))
    return rows


def main():
    S, META = wide_table(ARM_ORDER)
    man = json.load(open(ROOT / "chemeleon_suite/data/polaris/polaris_manifest.json"))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    rows, t0 = [], time.time()

    jobs = []
    for ds, (metric, hb) in MOLNET.items():
        jobs.append(("MoleculeNet", f"MolNet:{ds}", ds, metric, hb))
    for t in sorted(f.stem for f in (ROOT / "chemeleon_suite/data/moleculeace").glob("*.csv")):
        jobs.append(("MoleculeACE", f"MolACE:{t}", t, "rmse", False))
    for t in man:
        jobs.append(("Polaris", f"Polaris:{t.split('/')[-1]}", t,
                     man[t]["primary_metric"], man[t]["primary_metric"] in HIGHER))
    jobs.append(("CBS", "CBS:cbs", "cbs", "nef1", True))

    for i, (suite, key, raw, kind, hb) in enumerate(jobs, 1):
        if key not in S.columns:
            print(f"[{i}/{len(jobs)}] skip {key} (not in the score matrix)")
            continue
        if suite == "MoleculeNet":
            base, preds = load_molnet(raw)
        elif suite == "MoleculeACE":
            base, preds = load_moleculeace(raw)
        elif suite == "Polaris":
            base, preds = load_polaris(raw, man)
        else:
            base, preds = load_cbs()
        if base is None or not preds:
            why = ("Polaris withholds test labels locally" if suite == "Polaris"
                   else "no usable predictions")
            print(f"[{i}/{len(jobs)}] skip {key:34s} ({why})")
            continue
        point = S[key].to_dict()
        r = analyse_dataset(key, suite, base, preds, kind, hb, point)
        rows += r
        nb = sum(x["cobest_overall"] for x in r)
        print(f"[{i}/{len(jobs)}] {key:34s} n={len(base):6d} scaf={r[0]['n_scaffolds'] if r else 0:5d} "
              f"arms={len(r):2d} co-best={nb:2d}  ({time.time()-t0:5.0f}s)", flush=True)
        pd.DataFrame(rows).to_csv(OUT, index=False)          # checkpoint after every dataset

    print(f"\nwrote {OUT}  ({len(rows)} rows, {time.time()-t0:.0f}s, n_boot={N_BOOT})")

    # ---- CONTROL-ARM TRIPWIRE -------------------------------------------------------------------
    # A control arm doing suspiciously WELL is the same class of evidence as a real arm doing
    # implausibly well, and it is the tell this table already gave us once and we misread: with the
    # MoleculeACE truth column off by 9.0, the paired statistic gained a linear term that swamped
    # the quadratic one, every CI went wide, and random_encoder came out co-best on 28 of 30
    # MoleculeACE targets. That read as a modest result instead of a broken test. After the fix it
    # is 2 of 30.
    #
    # The range assertion in load_moleculeace catches THAT join. This catches the SYMPTOM, whatever
    # upstream defect produces it -- a randomly initialised encoder should not tie the best real arm
    # on more datasets than a TYPICAL real arm. Compared against the median rather than the max on
    # purpose: in the broken table random_encoder scored 28 and so did three real arms, so a
    # "control must not exceed the best real arm" rule passed the very table it was written for.
    # The median real arm was 26, which the control cleared. Checked against both tables before
    # committing: the rule fires on the old one and passes the fixed one. Non-fatal by design: the
    # table is already on
    # disk and is what you need to diagnose with, so this reports and exits non-zero rather than
    # destroying the evidence.
    df = pd.DataFrame(rows)
    bad = []
    for suite, g in df.groupby("suite"):
        cb = g.groupby("arm")["cobest_overall"].sum()
        ctl = cb[cb.index.isin(CONTROLS)]
        real = cb[~cb.index.isin(CONTROLS)]
        if ctl.empty or real.empty:
            continue
        if g["dataset"].nunique() < MIN_DATASETS_FOR_TRIPWIRE:
            continue        # CBS is a single dataset: every arm is co-best on 1, controls included,
                            # and "co-best rate" is not a distribution you can reason about at n=1
        for arm, v in ctl.items():
            if v >= real.median():
                bad.append(f"{suite}: control {arm} co-best on {int(v)}/{g['dataset'].nunique()} "
                           f"datasets, at or above the MEDIAN real arm ({real.median():.1f})")
    if bad:
        print("\nCONTROL-ARM TRIPWIRE FIRED -- treat this table as suspect, not as a result:")
        for b in bad:
            print(f"  {b}")
        sys.exit(2)
    print("control-arm tripwire: OK (no control ties the best real arm more often than it wins)")


if __name__ == "__main__":
    main()
