# BBBP: is the XGBoost anchor really that bad? — independent re-implementation, 2026-08-16

**Question.** BBBP is the only panel of the six where the classical anchors lose to every
transformer arm, and where an *untrained* random encoder (0.941 pooled-OOF ROC-AUC) beats a tuned
XGBoost on ECFP+descriptors (0.904). That looked like a bug.

**Check.** `scripts/verify_bbbp_anchor.py` rebuilds the anchor from scratch — fresh featurization
(Morgan/ECFP4 2048@r2 + the 217 RDKit descriptors) and a fresh XGBoost fit — sharing nothing with
`eval_v2.py` except the 2039 molecules and the deterministic Bemis-Murcko fold rule.

| model | pooled OOF ROC-AUC |
|---|---|
| pipeline's stored ECFP+desc predictions | 0.9043 |
| **re-implemented XGBoost · ECFP4 + descriptors** | **0.9055** |
| XGBoost · descriptors only | 0.9042 |
| XGBoost · ECFP4 only | 0.8882 |
| XGBoost · ECFP+desc, 2000 trees @ lr 0.02 | 0.9099 |
| RandomForest · ECFP+desc | 0.9126 |
| MLP · ECFP+desc (z-scored) | 0.8641 |
| MLP · descriptors only | 0.8636 |
| LogisticRegression · ECFP+desc | 0.8409 |
| XGBoost · MolWt + logP + TPSA only | 0.8219 |

**Verdict — the anchor is correct.** The re-implementation reproduces the pipeline to within 0.001.
Roughly 0.91 is the ceiling for fingerprint/descriptor features on this split no matter the model
family or budget, while the frozen-embedding arms reach 0.92–0.95.

**But BBBP does not measure pretraining.** The random encoder (an untrained transformer, mean-pooled,
z-scored, MLP head) scores 0.941 — above every classical model and above most pretrained arms. It is
also not a head artifact: an MLP on the same classical features scores *worse* (0.864), not better.
What wins on BBBP is the dense low-dimensional embedding + MLP combination, trained or not.

**Consequences for the paper.**
1. BBBP separates featurization style, not pretraining quality. Say so, and do not read a BBBP win
   as evidence that pretraining helped.
2. BBBP is also the most compressed panel: across the top eight models the whole field spans 1.8%
   of ROC-AUC (vs 22% on MoleculeACE, 18% on CBS). Mean *rank* therefore over-weights it badly —
   ECFP+desc ranks 14th there while trailing by only 5.7%. Fig A1 headlines mean **shortfall**
   (% behind the panel's best model) for this reason; the rank view is kept as a variant.
3. Head to head, ECFP+desc beats the best CLIMB arm on 5 of 6 panels, by 14.6% on MoleculeACE and
   9.1% on CBS. The mean-rank near-tie (3.83 vs 3.87) was an artifact and must not be reported.

Other checks worth knowing: standardizers are fit on train indices only (`eval_v2.py:493`), so
there is no fold leakage; BBBP is 76.5% positive; fold sizes are 408/408/408/408/407.
