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
1. **Anchors present**: `ecfp4_anchor`, `fp_desc_anchor` scored (still finishing on the box).
2. **Morgan+XGBoost sanity (critical) — ✅ PASSED (compute session, 2026-08-13).** Preliminary ≈0.79
   (ligand-only ECFP4) is ~6× Truong's ligand-only baseline (0.125). Checks:
   - **Fold provenance**: our `fold` column matches Truong Table 1 EXACTLY (per-fold actives
     9/10/8/8/8, inactives 2080/2079/2080/2083/2080) — we are literally on their published provided folds.
   - **Leakage**: all 10,445 unique; 0 active duplicates across folds; 0 label collisions.
   - **Normalization faithful (the decisive one)**: reproduced THEIR ligand-only baseline
     (rdkit_desc + linear) inside OUR pipeline on the provided folds → NEF1% **0.070 ± 0.058**, dead
     inside their reported 0.000–0.125. So our NEF1% normalization + decoy handling are not inflating —
     if they were, this descriptor baseline would be inflated too.
   - **Per-fold Morgan+XGBoost** (3 head seeds): 0.78 / 0.80 / **1.00** / 0.63 / 0.75 — high on every
     fold; fold-id 3 = 1.00 where all 16 generic pipelines scored 0.000 (confirm fold-id↔paper-Fold-3
     alignment when Data S1 lands).
   => Reading #1 (legitimate) holds: ligand-only Morgan+XGBoost matches Truong's target-trained
   structure-based models (0.764) and beats every generic pipeline (~0), with no docking/3D.
3. **Exact SOTA-generic value** from Truong SI `Data S1`/`Table S2` (compute session pulling from their
   GitHub+Zenodo). Treat generic as ≈0 until then.
4. **Reproduction chain**: `cbs_benchmark/*` raw evals on S3 (yes) AND published to HF `climb-results`
   (`publish_to_hf.py` executed) — same audit as expA/expB. Compute session to confirm on completion.

## Figure
- **Type:** horizontal bar chart (`barh`), one bar per arm, NEF1% on the x-axis, range [0, 1].
- **Arms & order:** `A1_ORDER` from `notebook_cells/08.py`, same labels + `rc_color` colours:
  `ecfp4`, `fp_desc`, `no_pretrain`, `no_pretrain_e2e`, `unsup_only`,
  `sup_only:{dense,sparse_all,dense_plus_sparse}`, `unsup2sup:{dense,sparse_all,dense_plus_sparse}`.
- **Error bars:** ±1 sd over the **3 pretraining seeds** where they exist (unsup/sup/u2s), else over
  the **5 provided folds** (ecfp4, fp_desc, no_pretrain, e2e). Note which is which in the caption.
- **Reference overlays (vertical) — 3 lines, from `experiment_cbs/cbs_reference_lines.csv`:**
  - **Truong target-specific (upper): dashed line at 0.764**, shaded ±0.191 band. Label: "Truong 2026
    target-specific (docking-based) — 0.764 ± 0.191".
  - **Truong ligand-only baseline: dashed line** at 0.000–0.125 (draw at ~0.06 or as a thin band).
    Label: "Truong ligand-only (RDKit descriptors + logreg)". This is the crux — it's the ligand-only
    method that FAILS, so our ligand-only anchors sitting way above it is the point.
  - **SOTA generic (lower, ≈0): dashed line** at the best generic pipeline's NEF1% (value from SI),
    labelled with the method; optionally a faint band [0, best] for the full generic spread. Label:
    "SOTA generic VS (docking/co-folding), Truong 2026".
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
4. **Decoy/analogue caveat (state it plainly):** the standard retrospective-VS limitation (Sieg 2019;
   Chen 2019) — property-matching controls physicochemical but NOT structural provenance, so ligand
   ML can in principle exploit ChemDiv-decoy-vs-active substructure differences. Verified faithful to
   the published benchmark on identical folds, so **"matches structure-based methods on THIS
   benchmark" is solid; "great prospective CBS screener" is not fully established.** Structure-based
   methods are less exposed to this bias, which is why a ligand-only method topping them earns the
   caveat. OPTIONAL robustness (compute session offered): a provenance probe — train a decoy-source
   classifier and check its correlation with our active-ranking.

## Framing (the story)
Agreement: we **beat the SOTA generic pipelines**. Disagreement with Truong's implicit "you need
structure": their ligand-only baseline (physicochemical descriptors) fails BY CONSTRUCTION; they never
tested ligand-only **structural fingerprints**, which do NOT fail. Ligand-only Morgan+XGBoost matches
their target-trained structure-based models with no docking — the on-thesis "cheap baseline is hard to
beat" result for CLIMB, with the decoy caveat above.

## Data / cells
- Input: `experiment_cbs/cbs_nef1_summary.csv` (arm, metric, mean, std_over_seeds, n_seeds) +
  `cbs_per_run.csv` (fold-level) + `cbs_reference_lines.csv` (correctly-labelled reference values, so
  the 0.764=generic error can't reach a caption). Commit all to the repo (like expA/expB CSVs).
- New mainline cell pair (`save_fig("figCBS_external_validation")`). Placement in the notebook TBD at
  build time (insert in the mainline block with renumber, or append + mark mainline).
- Do NOT touch any existing figure/dataset (user directive).
