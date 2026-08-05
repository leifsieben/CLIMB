# Handoff → notebook session: regenerate the significance table/figures with the tightened stats

**From:** compute session · **Date:** 2026-08-05
**Why:** a reviewer flagged the statistical protocol (multiplicity, scaffold non-independence,
NEF1%-vs-AUC mismatch, single-seed error bars). `scripts/compare_models.py` is updated to fix this;
the headline table (T1) and any significance annotations should be regenerated from it. README §8.1/§10
already describe the new protocol.

## What changed in `scripts/compare_models.py` (backward-compatible)
- **New:** `bh_fdr(pvals)` and `compare_many(pairs, tasks, n_boot=…)` — Benjamini–Hochberg FDR across
  the whole (arm × task) family, returned as `point_q` / `boot_q`.
- **New:** `cluster_bootstrap_diff(...)` and `compare(..., n_boot=1000)` — a **scaffold cluster-bootstrap
  CI** on the metric difference (`ci_lo`, `ci_hi`, `boot_p`, `n_scaffolds`). This is the **headline
  uncertainty** now; it also gives **NEF1% its own CI** (don't test NEF via AUC).
- `compare()` with the default `n_boot=0` is unchanged and fast — existing calls still work; the new
  columns are just NaN until you pass `n_boot>0`.
- Molecule-level Wilcoxon/DeLong are relabelled `Wilcoxon~`/`DeLong~` and demoted to *indicative*
  (anti-conservative). Stop calling them "rigorous."

## To regenerate T1 (and any p-value annotations)
```python
from scripts.compare_models import compare_many
pairs = [("unsup_8M","fp_desc_anchor"), ("unsup_8M","skip_dense_8M"),
         ("u2s_dense_from8M","skip_dense_8M"), ("u2s_dense_from8M","unsup_8M")]
tasks = [("ESOL",False),("QM7",False),("BBBP",True),("BACE",True),("Tox21",True),
         ("HIV",True),("HIV",True,"nef1")]
df = compare_many(pairs, tasks, n_boot=1000)     # ~a few min (HIV bootstrap is the slow part)
```
Report **`delta` + `[ci_lo, ci_hi]`** as the headline, `boot_q` (FDR) as the p-value, and keep
`point_q` only as an indicative footnote. A CI spanning 0 = "no difference detectable" — never phrase
it as "equivalent".

Precomputed results (so you don't have to rerun): **`analysis/rigor/comparison_fdr_bootstrap.{csv,md}`**.

## Pretraining-seed error bars (the principal 8M arms have 3 seeds)
`unsup_8M`, `skip_dense_8M`, `skip_mixed_8M`, `u2s_dense_from8M` each have seeds s0/s1/s2, fully CV'd.
Add a seed error bar (or an SI panel) — **at least for ESOL**, where across-seed std ≈ 0.016–0.026
(on the order of the between-arm gaps). Numbers: **`analysis/rigor/seed_variance.{csv,md}`**
(`scripts/rigor_report.py` regenerates both this and the comparison table).

## Two results that MOVE (don't miss these when you redo the figure)
- **HIV NEF1%, `unsup_8M` vs `fp_desc`**: molecule DeLong p=1.4e-9 but that's the AUC endpoint; the
  NEF1% cluster CI is `[-0.071,+0.012]` (spans 0). The "classical beats CLM at enrichment" annotation
  must go — it was a wrong-endpoint artifact.
- **`u2s_dense_from8M` vs `skip_dense_8M` on BACE**: molecule p=0.037 → FDR q=0.073, boot_p=0.13. Not
  significant; drop any "differs on BACE" claim.

Nothing here overturns the thesis — most weak differences wash out under FDR+clustering, which supports
the "pretraining doesn't generally help" story. Ping the compute session if you want more pairs added.
