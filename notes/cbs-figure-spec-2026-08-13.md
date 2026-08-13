# Fig CBS (mainline) — external CBS-inhibitor VS benchmark: spec

**Status:** DRAFT SPEC. Data not yet complete (anchors `ecfp4`/`fp_desc` + `e2e` still running on the
box; no tidy `cbs_nef1_summary.csv` yet). Do not build until the battery lands and the
Morgan+XGBoost verification (below) passes. Placement: **mainline** (user, 2026-08-13).

## Source paper (read 2026-08-13)
Truong et al. 2026, *J. Cheminformatics* s13321-026-01262-x. Dataset = **43 actives / 10,402
inactives = 10,445** (195 exp. inactives + 10,207 property-matched DCM decoys), 5-fold UMAP-cluster
split, max inter-fold Tanimoto < 0.70. Metric = **NEF1%** (EF1% normalized by max attainable).

### ⚠️ Reference numbers — the peer's handoff had the label INVERTED
- **0.764 ± 0.191 = Truong's CBS-SPECIFIC, TARGET-TRAINED models** (RF/XGBoost/SVM/MLP on **PLEC
  docking-pose fingerprints** — structure-based). This is the *upper* reference, NOT the generic one.
  Best individual configs reach ~0.9–1.0; 0.764 is the mean over their 16 CBS-specific configs.
- **SOTA generic (16 docking/co-folding pipelines)** = the *low* bars, ~0–0.4; **all 16 = 0.000 on
  Fold 3**. Exact aggregate/best value is in their **SI `Data S1` / `Table S2`** (NOT in the main-text
  PDF) — MUST fetch before drawing this line. Generic set: Smina/Smina, Smina/RF-Score-VS, GNINA 1.3,
  DeepDock, Interformer, SurfDock, KarmaDock, Boltz-2, Boltz-2x, AlphaFold 3.
- **Truong has NO CLMs.** So "our CLIMB vs their CLMs" has no counterpart, and "their XGBoost" is
  docking-based (sees the complex), not comparable to our ligand-only Morgan+XGBoost.
- **Truong's ligand-only baseline** (RDKit physicochemical descriptors + logistic regression) =
  **NEF1% 0.000–0.125** — low BY DESIGN: decoys were property-matched to actives on 8 physchem props.

## Verification GATES before this goes mainline
1. **Anchors present**: `ecfp4_anchor`, `fp_desc_anchor` scored (currently absent on S3).
2. **Morgan+XGBoost sanity (critical).** Preliminary ≈0.79 (ligand-only ECFP4) is ~6× Truong's
   ligand-only baseline (0.125). Plausible (fingerprints separate structures that property-matched
   descriptors can't; folds are scaffold-separated so it's cross-scaffold generalization, not trivial
   dup leakage) — but verify before headlining: (a) we ran on the benchmark's OWN provided folds
   (`cv_scheme=provided`), (b) no active leaks across folds (reconfirm max inter-fold Tanimoto < 0.70
   on OUR fold assignment), (c) NEF1% via `heads_v2.compute_nef`, EF1% normalized by max attainable,
   same as Truong. If it survives, the story is strong and *contradicts Truong's "you need structure"*
   conclusion; if not, it's an eval artifact.
3. **Exact SOTA-generic value** pulled from Truong SI `Data S1`/`Table S2` (or their Zenodo/GitHub).
4. **Reproduction chain**: confirm `cbs_benchmark/*` raw evals are on S3 (yes) AND published to HF
   `climb-results` (`publish_to_hf.py` executed) — same audit as expA/expB.

## Figure
- **Type:** horizontal bar chart (`barh`), one bar per arm, NEF1% on the x-axis, range [0, 1].
- **Arms & order:** `A1_ORDER` from `notebook_cells/08.py`, same labels + `rc_color` colours:
  `ecfp4`, `fp_desc`, `no_pretrain`, `no_pretrain_e2e`, `unsup_only`,
  `sup_only:{dense,sparse_all,dense_plus_sparse}`, `unsup2sup:{dense,sparse_all,dense_plus_sparse}`.
- **Error bars:** ±1 sd over the **3 pretraining seeds** where they exist (unsup/sup/u2s), else over
  the **5 provided folds** (ecfp4, fp_desc, no_pretrain, e2e). Note which is which in the caption.
- **Reference overlays (vertical):**
  - **Truong target-specific: dashed line at 0.764**, shaded ±0.191 band. Label: "Truong 2026
    target-specific (docking-based) — 0.764 ± 0.191".
  - **SOTA generic: dashed line** at the best generic pipeline's NEF1% (value from SI), labelled with
    the method; optionally a faint band [0, best] for the full generic spread. Label: "SOTA generic
    VS (docking/co-folding), Truong 2026".
- **Secondary metric (ROC-AUC):** near-ceiling here, less informative — put it in a small companion
  panel or a printed table under the figure, NOT as the headline.
- **Highlight:** the two ligand-only anchors (ecfp4, fp_desc) are the crux — annotate that these are
  ligand-only, no docking. If Morgan+XGBoost lands near 0.764, that's the headline.

## Caption must state (correctly)
1. NEF1% = normalized enrichment factor of true actives at top-1%; benchmark's own leakage-controlled
   5-fold split (max inter-fold Tanimoto < 0.70).
2. Reference lines are from Truong 2026: 0.764 ± 0.191 is their **target-trained, docking-based**
   models (upper); the generic docking/co-folding pipelines are the lower line.
3. **Modality caveat:** Truong's models use the protein + docking poses (PLEC features); ours are
   **ligand-only from SMILES**. Not a head-to-head on identical inputs — it's "where do cheap
   ligand-only models land vs structure-based tools on an external target".
4. **Decoy caveat:** the DCM decoys are physicochemical-property-matched to the actives (which is why
   Truong's ligand-descriptor baseline ≈0), so fingerprint enrichment reflects structural
   active/decoy separation; our numbers are on the scaffold-separated folds, so cross-scaffold, not
   near-duplicate, generalization.

## Data / cells
- Input: `analysis/rigor/cbs_nef1_summary.csv` (arm, metric, mean, std_over_seeds, n_seeds) +
  `cbs_per_run.csv` (fold-level). Commit both to the repo (like expA/expB CSVs).
- New mainline cell pair (`save_fig("figCBS_external_validation")`). Placement in the notebook TBD at
  build time (insert in the mainline block with renumber, or append + mark mainline).
- Do NOT touch any existing figure/dataset (user directive).
