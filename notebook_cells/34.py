# ---------- SI · variance decomposition (pretraining-seed / fold / head-seed), 4 principal 8M arms ----------
# Reads the tables scripts/variance_decomp.py wrote from the existing per-(seed,fold) CV `_cell` rows
# (no new compute). Printed, not plotted -- the numbers go into an SI subsection; this cell keeps them
# traceable to figure_data. Components: sd_pretrain (s0/s1/s2 encoders), sd_fold (5 scaffold folds),
# sd_head (MLP init, always averaged); pct_* are the share of total score variance.
_VD  = pd.read_csv("analysis/rigor/variance_decomposition.csv")
_VDD = pd.read_csv("analysis/rigor/variance_decision.csv")

print("=" * 104)
print("Variance decomposition of the pooled 5-fold scaffold-CV score -- 4 principal 8M arms (no new compute)")
print("=" * 104)
_disp = _VD.copy()
for c in ("mean", "sd_pretrain", "sd_fold", "sd_head"):
    _disp[c] = _disp[c].map(lambda v: f"{v:.4g}")
for c in ("pct_pretrain", "pct_fold", "pct_head", "pct_resid"):
    _disp[c] = _disp[c].map(lambda v: f"{v:.1f}")
print(_disp.to_string(index=False))

# average share of total variance per component (across the 4 arms x 6 tasks), with where each peaks
print("\nShare of total CV variance (mean over the 4 arms x 6 tasks, and the single largest cell):")
for comp, lab in (("pct_head", "head-seed"), ("pct_fold", "fold / split"), ("pct_pretrain", "pretraining-seed")):
    _i = _VD[comp].idxmax()
    print(f"   {lab:16} {_VD[comp].mean():5.1f}%   (peaks at {_VD.loc[_i, comp]:.1f}% on "
          f"{_VD.loc[_i, 'arm']} / {_VD.loc[_i, 'task']})")

# per-task decision: is the unsup_only vs sup_only:dense regime gap resolvable on ONE pretraining seed?
print("\nIs the unsup_only vs sup_only:dense regime gap larger than pretraining-seed noise?")
print("   reliable_on_one_seed = |gap| > sigma_pretrain;  False => the gap is WITHIN seed noise")
print(_VDD.to_string(index=False))
_within = _VDD.loc[~_VDD.reliable_on_one_seed, "task"].tolist()
print(f"\n   Head-seed averaging is justified (head-seed share ~0-1%); fold variance dominates; "
      f"pretraining-seed variance is small overall but reaches double digits on ESOL/BACE.")
print(f"   Regime gap WITHIN pretraining-seed noise (needs >1 seed to claim): "
      f"{', '.join(_within) if _within else '(none)'}.")
