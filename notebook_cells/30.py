# ---------- Head ablation · FFN vs XGBoost on frozen CLIMB embeddings ----------
# Only --head changes (mlp -> xgb); everything else (5-fold scaffold CV, pool=mean, zscore
# standardize) is the shared eval_v2 protocol. FFN + Morgan anchors are read from the production
# figure_data runs; XGBoost-on-embeddings from the committed analysis/head_ablation/ dir (generated
# by scripts/run_head_ablation.sh, encoder re-run on MPS -> <=0.01 device offset vs the FFN rows).
from pathlib import Path as _Path
HA_TASKS = [("ESOL", "RMSE", False), ("QM7", "RMSE", False), ("BBBP", "ROC-AUC", True),
            ("BACE", "ROC-AUC", True), ("Tox21", "ROC-AUC", True), ("HIV", "ROC-AUC", True)]
def _ha(src, task, suf="_MEAN"):
    p = _Path(src) / "moleculenet_cv" / "suite_summary.json"
    return json.loads(p.read_text()).get(task + suf) if p.exists() else None
_PH = DATA_ROOT / "climb_v2_phase2"; _HA = _Path("analysis/head_ablation")
# (featurizer, head, source dir). FFN + anchors live in figure_data; the two XGB-on-embeddings runs
# are committed with the repo so this table reproduces without a private data snapshot.
HA_ROWS = [("Morgan ECFP4",       "XGBoost", _PH / "ecfp4_anchor"),
           ("Morgan+desc",        "XGBoost", _PH / "fp_desc_anchor"),
           ("CLIMB unsup_only",   "FFN",     _PH / "unsup_8M"),
           ("CLIMB unsup_only",   "XGBoost", _HA / "unsup_8M_xgb"),
           ("CLIMB dense+sparse", "FFN",     _PH / "skip_dense_plus_sparse_8M"),
           ("CLIMB dense+sparse", "XGBoost", _HA / "skip_dense_plus_sparse_8M_xgb")]
_num = {(f, h): {t: _ha(s, t) for t, _, _ in HA_TASKS} for f, h, s in HA_ROWS}
for f, h, s in HA_ROWS:
    _num[(f, h)]["HIV_nef1"] = _ha(s, "HIV", "_nef1_MEAN")

_rows = []
for f, h, s in HA_ROWS:
    r = {"featurizer": f, "head": h}
    for t, m, hb in HA_TASKS:
        v = _num[(f, h)][t]; r[f"{t}({m[:3]}{'↑' if hb else '↓'})"] = f"{v:.3f}" if isinstance(v, (int, float)) else "n.d."
    v = _num[(f, h)]["HIV_nef1"]; r["HIV(NEF↑)"] = f"{v:.3f}" if isinstance(v, (int, float)) else "n.d."
    _rows.append(r)
HA = pd.DataFrame(_rows)
print("=" * 118)
print("Head ablation — FFN vs XGBoost on FROZEN embeddings (pooled 5-fold scaffold CV mean; only the head differs)")
print("=" * 118)
print(HA.to_string(index=False))
print("\nESOL/QM7 = RMSE (lower better); BBBP/BACE/Tox21/HIV = ROC-AUC (higher better); "
      "HIV(NEF↑) = NEF1%, A1's HIV ranking metric.")

# --- did swapping FFN -> XGBoost help, on the same embeddings? (signed toward 'better') ---
def _better_delta(xgb, ffn, hb):
    if not (isinstance(xgb, (int, float)) and isinstance(ffn, (int, float))): return None
    return (xgb - ffn) if hb else (ffn - xgb)     # + => XGBoost is better than the FFN
print("\nHead effect on the CLIMB encoders (XGBoost minus FFN, + = XGBoost better):")
for enc in ("CLIMB unsup_only", "CLIMB dense+sparse"):
    parts = []
    for t, m, hb in HA_TASKS:
        d = _better_delta(_num[(enc, "XGBoost")][t], _num[(enc, "FFN")][t], hb)
        if d is not None: parts.append(f"{t} {d:+.3f}")
    print(f"   {enc:20} " + "   ".join(parts))

# --- does XGBoost-on-embeddings let CLIMB beat the best Morgan anchor per task? ---
print("\nBest per task, and whether any CLIMB+XGBoost arm is the winner:")
for t, m, hb in HA_TASKS:
    best_lbl, best_v = None, None
    for f, h, s in HA_ROWS:
        v = _num[(f, h)][t]
        if not isinstance(v, (int, float)): continue
        if best_v is None or (v > best_v if hb else v < best_v): best_lbl, best_v = f"{f} + {h}", v
    print(f"   {t:6} best = {best_lbl:34} ({best_v:.3f})")
