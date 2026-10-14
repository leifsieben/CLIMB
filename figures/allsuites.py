from __future__ import annotations
import csv
import json, statistics as st, collections
from pathlib import Path
import numpy as np
import pandas as pd

from figures.sixpanel import NATIVE_SUBDIRS
from figures.arms import ARMS, ARM_ORDER

ROOT = Path(__file__).resolve().parent.parent
FD = ROOT / "figure_data"


SUITES = ["MoleculeNet", "MoleculeACE", "Polaris", "CBS", "Wong", "FartDB"]


MOLNET_ROOTS = ["climb_v2_phase2", "climb_v2_h1", "climb_v2_vocab"]


def molnet_dir(prefix, subdir="moleculenet_cv"):

    for root in MOLNET_ROOTS:
        d = FD / root / prefix / subdir
        if (d / "reference_scoring.json").exists():
            continue
        if (d / "moleculenet_summary.csv").exists() or (d / "suite_summary.json").exists():
            return d
    return None


MOLNET = {"BACE": ("roc_auc", True), "BBBP": ("roc_auc", True), "HIV": ("roc_auc", True),
          "Tox21": ("roc_auc", True), "ESOL": ("rmse", False), "QM7": ("rmse", False)}


HIGHER = {"pearsonr", "spearmanr", "r2", "roc_auc", "pr_auc", "accuracy", "explained_var",
          "f1", "mcc", "cohen_kappa"}


def _polaris_manifest():
    return json.load(open(ROOT / "benchmarks/data/polaris/polaris_manifest.json"))


def _molnet(dirs, metrics=None):

    out = {}
    metrics = metrics or {}
    for ds, (default_metric, _) in MOLNET.items():


        metric = metrics.get(f"MolNet:{ds}", default_metric)

        chosen, usable = None, []
        for sub in NATIVE_SUBDIRS.get(ds, ("moleculenet_cv",)):
            usable = [d for d in dirs if molnet_dir(d, sub) is not None]
            if usable:
                chosen = sub
                break
        if not chosen:
            continue
        vals = []
        for d in usable:
            md = molnet_dir(d, chosen)
            f = md / "moleculenet_summary.csv"
            if f.exists():
                vals += [float(r["main_value"]) for r in csv.DictReader(open(f))
                         if r["dataset"] == ds and r["main_metric"] == metric
                         and r["head_seed"] not in ("MEAN", "STD")
                         and r["main_value"] not in ("", "nan")]
                continue
            j = md / "suite_summary.json"
            if j.exists():
                v = {k.lower(): v for k, v in json.load(open(j)).items()}.get(f"{ds}_mean".lower())
                if v is not None:
                    vals.append(float(v))
        if vals:
            out[ds] = st.mean(vals)
    return out


def _moleculeace(dirs):
    """{target: mean overall RMSE over eval seeds and pretraining-seed dirs}."""
    per = collections.defaultdict(list)
    for d in dirs:
        f = FD / "chemeleon_suite" / "moleculeace" / d / "results.csv"
        if not f.exists():
            continue
        for r in csv.DictReader(open(f)):
            if r["subset"] == "overall" and r["metric"] == "rmse":
                try:
                    per[r["task"]].append(float(r["value"]))
                except ValueError:
                    pass
    return {t: st.mean(v) for t, v in per.items()}


def _polaris_summary():

    f = ROOT / "chemeleon_suite" / "summaries" / "polaris_summary.csv"
    out = {}
    if f.exists():
        for r in csv.DictReader(open(f)):
            if r.get("source", "ours") == "ours":
                out[(r["model"], r["task"], r["metric"])] = float(r["mean"])
    return out


def _polaris(dirs, man, metrics=None):

    per = collections.defaultdict(lambda: collections.defaultdict(list))
    present = []
    for d in dirs:
        f = FD / "chemeleon_suite" / "polaris" / d / "polaris_scores.csv"
        if not f.exists():
            continue
        present.append(d)
        for r in csv.DictReader(open(f)):
            per[r["task"]][r["metric"]].append(float(r["value"]))
    out = {}
    metrics = metrics or {}
    for t, m in per.items():
        pm = metrics.get(f"Polaris:{t.split('/')[-1]}") or man.get(t, {}).get("primary_metric")
        if pm in m:
            out[t] = st.mean(m[pm])
    if len(present) <= 1 and len(out) < len(man):
        summ = _polaris_summary()
        for t, meta in man.items():
            pm = meta.get("primary_metric")
            key = (present[0] if present else dirs[0], t, pm)
            if t not in out and key in summ:
                out[t] = summ[key]
    return out


def _cbs_value(mol_src):

    if not mol_src:
        return None
    dirs = list(mol_src) if isinstance(mol_src, (list, tuple)) else [mol_src]
    vals = []
    for d in dirs:
        for cand in (d, f"{d}_s0"):
            base = FD / "cbs_benchmark" / cand / "moleculenet_cv"
            f = base / "moleculenet_summary.csv"
            if f.exists():
                vals += [float(r["main_value"]) for r in csv.DictReader(open(f))
                         if r["main_metric"] == "nef1" and r["head_seed"].startswith("fold")]
                break


            pf = base / "per_fold.csv"
            if pf.exists():
                vals += [float(r["nef1"]) for r in csv.DictReader(open(pf))
                         if r.get("nef1") not in (None, "", "nan")]
                break


            fv = base / "fold_values.csv"
            if fv.exists():
                vals += [float(r["value"]) for r in csv.DictReader(open(fv))
                         if r.get("metric") == "nef1" and r.get("value") not in (None, "", "nan")]
                break
        else:


            present = [c for c in (d, f"{d}_s0") if (FD / "cbs_benchmark" / c).is_dir()]
            assert not present, (
                f"CBS: {present} exists but holds none of moleculenet_summary.csv / per_fold.csv "
                f"/ fold_values.csv under moleculenet_cv/. A new runner output format must be "
                f"added here rather than read as 'not run'.")
    return sum(vals) / len(vals) if vals else None


def _flat_suite(dirs, tree, task, metric):

    vals = []
    for d in dirs:
        base = FD / tree / d
        vj, rc = base / "verified.json", base / "results.csv"
        if not rc.exists():
            continue
        if vj.exists():
            declared = json.load(open(vj)).get("metric_primary")
            assert declared in (None, metric), (
                f"{tree}/{d}: verified.json says the primary metric is {declared!r} but this "
                f"figure reads {metric!r}. One of them has moved; do not average across both.")
        hit = [float(r["mean"]) for r in csv.DictReader(open(rc)) if r["metric"] == metric]
        assert len(hit) == 1, (
            f"{tree}/{d}: {len(hit)} rows for primary metric {primary!r} in results.csv")
        vals.append(hit[0])
    return {task: sum(vals) / len(vals)} if vals else {}


def _seed_dirs(base):

    if isinstance(base, (list, tuple)):
        return list(base)
    if base.endswith("_00"):
        return [base, base[:-3] + "_01", base[:-3] + "_02"]
    return [base, f"{base}_s1", f"{base}_s2"]


def wide_table(arms=None, metrics=None):

    arms = arms or ARM_ORDER
    metrics = metrics or {}
    man = _polaris_manifest()
    rows, meta = {}, {}
    for a in arms:
        src = ARMS[a]["src"]
        vals = {}
        for ds, v in _molnet(src["mol"], metrics).items():
            key = f"MolNet:{ds}"
            met = metrics.get(key, MOLNET[ds][0])
            vals[key] = v
            meta[key] = ("MoleculeNet", met,
                         MOLNET[ds][1] if met == MOLNET[ds][0] else met in HIGHER)

        for tree, task, key, metric in (("wong_saureus", "Wong", "Wong:wong", "nef1"),
                                        ("fartdb", "FartDB", "FartDB:fartdb", "macro_ovr_auc")):
            got = _flat_suite(src["mol"], tree, task, metric)
            if got:
                vals[key] = got[task]
                meta[key] = ("Wong" if task == "Wong" else "FartDB", metric, True)
        if src.get("mace"):
            mace_dirs = _seed_dirs(src["mace"])
            for t, v in _moleculeace(mace_dirs).items():
                vals[f"MolACE:{t}"] = v
                meta[f"MolACE:{t}"] = ("MoleculeACE", "rmse", False)


            pol_dirs = _seed_dirs(src["mace"])
            for t, v in _polaris(pol_dirs, man, metrics).items():
                key = f"Polaris:{t.split('/')[-1]}"
                pm = metrics.get(key) or man[t]["primary_metric"]
                vals[key] = v
                meta[key] = ("Polaris", pm, pm in HIGHER)
        cbs_v = _cbs_value(src.get("mol"))
        if cbs_v is not None:
            vals["CBS:cbs"] = cbs_v
            meta["CBS:cbs"] = ("CBS", "nef1", True)
        rows[a] = vals
    S = pd.DataFrame(rows).T.reindex(index=arms)
    M = pd.DataFrame(meta, index=["suite", "metric", "higher_better"]).T
    M = M.loc[[c for c in S.columns]]
    order = sorted(S.columns, key=lambda c: (SUITES.index(M.loc[c, "suite"]), c))
    return S[order], M.loc[order]


def effective_n(R, M, max_pairs=4000):

    import itertools
    from scipy.stats import spearmanr
    out = {}
    for s in SUITES:
        cols = [c for c in R.columns if M.loc[c, "suite"] == s and R[c].notna().sum() >= 8]
        if len(cols) < 2:
            out[s] = float(len(cols))
            continue
        pairs = list(itertools.combinations(cols, 2))
        if len(pairs) > max_pairs:
            pairs = [pairs[i] for i in np.linspace(0, len(pairs) - 1, max_pairs).astype(int)]
        vals = []
        for a, b in pairs:
            d = R[[a, b]].dropna()
            if len(d) >= 8:
                r = spearmanr(d[a], d[b]).statistic
                if np.isfinite(r):
                    vals.append(r)
        rho = float(np.mean(vals)) if vals else 0.0
        n = len(cols)
        out[s] = n / (1 + (n - 1) * max(rho, 0.0))
    return out


def wide_ranks(arms=None, per_suite_equal=False):

    S, M = wide_table(arms)
    N = len(S)
    R = pd.DataFrame(index=S.index, columns=S.columns, dtype=float)
    for c in S.columns:
        col = S[c].dropna()
        if len(col) < 2:
            continue
        r = col.rank(ascending=not M.loc[c, "higher_better"])
        R.loc[r.index, c] = 1 + (N - 1) * (r - 1) / (len(col) - 1)
    out = pd.DataFrame(index=S.index)
    for s in SUITES:
        cols = [c for c in S.columns if M.loc[c, "suite"] == s]
        out[s] = R[cols].mean(axis=1)
        out[s + "_n"] = R[cols].notna().sum(axis=1)
    if per_suite_equal:
        out["mean_rank"] = out[SUITES].mean(axis=1)
        out["se_rank"] = out[SUITES].std(axis=1, ddof=1) / np.sqrt(out[SUITES].notna().sum(axis=1))
        out["n_units"] = out[SUITES].notna().sum(axis=1)
    else:
        out["mean_rank"] = R.mean(axis=1)
        out["se_rank"] = R.std(axis=1, ddof=1) / np.sqrt(R.notna().sum(axis=1))
        out["n_units"] = R.notna().sum(axis=1)


    ne = effective_n(R, M)
    deff = out["n_units"] / max(sum(ne.values()), 1e-9)
    out["se_rank_naive"] = out["se_rank"]
    out["se_rank"] = out["se_rank"] * np.sqrt(deff.clip(lower=1.0))
    out["n_eff"] = sum(ne.values())
    return out.sort_values("mean_rank"), R, M
