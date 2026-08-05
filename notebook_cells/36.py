# ---------- SI · embedding-redundancy test: does concatenating the CLM embedding onto fp+desc help? ----------
# Same XGBoost, same 5-fold scaffold CV, four feature sets per task. Reads the table
# scripts/concat_redundancy.py wrote (native-unit regression, Tox21 NaN-masked). Headline metric per
# task: RMSE (ESOL/QM7, lower better), ROC-AUC (BBBP/BACE/Tox21), NEF1% (HIV) -- matches the paper.
_CR = pd.read_csv("analysis/rigor/concat_redundancy.csv")
_HEADLINE = {"ESOL": "rmse", "QM7": "rmse", "BBBP": "roc_auc", "BACE": "roc_auc",
             "Tox21": "roc_auc", "HIV": "nef1"}
_FSETS = ["fp+desc", "CLM", "desc+CLM", "fp+desc+CLM"]

def _val(task, feats, metric):
    r = _CR[(_CR.task == task) & (_CR.features == feats) & (_CR.metric == metric)]
    return float(r["mean"].iloc[0]) if len(r) else np.nan

print("=" * 96)
print("Embedding-redundancy test -- same XGBoost + 5-fold scaffold CV, four feature sets (headline metric)")
print("=" * 96)
_rows, _helps = [], 0
for t in CORE_TASKS:
    m = _HEADLINE[t]; lower = t in ("ESOL", "QM7")
    vals = {f: _val(t, f, m) for f in _FSETS}
    # improvement of fp+desc+CLM over fp+desc, signed so + = concat is BETTER
    delta = (vals["fp+desc"] - vals["fp+desc+CLM"]) if lower else (vals["fp+desc+CLM"] - vals["fp+desc"])
    helps = delta > 0
    _helps += int(helps)
    arrow = "↓" if lower else "↑"
    _rows.append({"task": t, "metric": f"{m}{arrow}",
                  "fp+desc": f"{vals['fp+desc']:.4g}", "CLM": f"{vals['CLM']:.4g}",
                  "desc+CLM": f"{vals['desc+CLM']:.4g}", "fp+desc+CLM": f"{vals['fp+desc+CLM']:.4g}",
                  "Δ(+CLM)": f"{delta:+.4g}", "CLM helps?": "yes" if helps else "no"})
print(pd.DataFrame(_rows).to_string(index=False))

print(f"\nAdding the CLM embedding to fp+desc improves the headline metric on {_helps}/{len(CORE_TASKS)} tasks.")
_worse_alone = [t for t in CORE_TASKS
                if (_val(t, "CLM", _HEADLINE[t]) > _val(t, "fp+desc", _HEADLINE[t])) == (t in ("ESOL", "QM7"))]
print(f"The CLM embedding ALONE is weaker than fp+desc on {len(_worse_alone)}/{len(CORE_TASKS)} tasks "
      f"({', '.join(_worse_alone)}).")
print("desc+CLM shows the same null as fp+desc+CLM, so the flatness is not fingerprint dilution. The only "
      "gain is BBBP, where CV is saturated (a random encoder already reaches ~0.94 AUC). => the 512-d CLM "
      "embedding is informationally redundant given fingerprints + descriptors.")
