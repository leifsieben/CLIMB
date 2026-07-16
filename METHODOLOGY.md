# CLIMB — Methodology & Design

Companion to the living [RESULTS_DOSSIER](experiments/climb_v2_phase2/RESULTS_DOSSIER.md).
This document is the *design of record*: what we are asking, how the model and training
are built and why, how every experiment is run, what is weak, and what artefacts we owe.

---

## 1. Hypotheses to be answered (full list)

**Core**
- **H1 — Does unsupervised (MLM) pretraining help at all?** Compare, per task, three regimes
  at matched compute: *skip-unsup* (random init + SFT), *pure-unsup* (MLM only, frozen),
  *unsup→sup* (MLM then SFT).
- **H2 — Prior vs endpoint.** Is unsup merely a *better initialization* (faster convergence)
  that SFT-from-scratch catches up to if trained long enough, or does unsup reach a genuinely
  *better endpoint* that no amount of SFT-only compute matches?
- **H3 — Does the SFT label type decide it?** dense (RDKit-descriptor MTR) vs sparse (assay:
  PCBA/L1000/WONG) vs dense+sparse. Sub-question: can **dense compensate for missing unsup**
  where **sparse cannot**?
- **H4 — Plateaus.** Where (in compute / #molecules) do the pure-unsup and skip-unsup curves
  flatten? Is there a point past which more unsup stops helping?

**Validity / confounds (must resolve before trusting H1–H4)**
- **H5 — Is "SFT ≤ MLM base" real or an artefact?** Two suspects: (a) the frozen probe is too
  low-ceiling to resolve encoder quality; (b) warm-start SFT at LR 2e-4 destroys MLM features.
  Tested by the finetune eval-ceiling (Fig D) and the SFT-LR sweep.
- **H6 — Leakage.** Are results inflated by molecule (InChIKey) overlap between the eval **test**
  splits and the pretraining corpus and/or the SFT assay data?

**Secondary / bonus**
- **H7 — Catastrophic forgetting.** Does running MLM *after* SFT erase the supervised signal?
- **H8 — Beyond one epoch (side-story).** Past 124M molecules, does SMILES enumeration
  (fresh augmentation each pass) beat exact canonical repetition?
- **H9 — Is the model big enough / is mean-pool the right readout?** (sanity, not headline).

---

## 2. Model architecture (and why)

**Encoder: ModernBERT, ~41.4M params** — hidden 512, 12 layers, 8 heads, intermediate 1536,
max positions 256, vocab 1000 (byte-level BPE), SDPA attention.

Rationale:
- **Encoder, not decoder.** The task is representation → property prediction; a bidirectional
  masked encoder is the right inductive setup (ChemBERTa, MoLFormer are encoders).
- **~41M is deliberately "believable-SOTA but single-GPU".** Matches MoLFormer-base (44M) and
  ChemBERTa-2 (46M), so a null result on unsup can't be dismissed as a toy model, yet it trains
  on one A10G. (Size sufficiency is itself checked, H9.)
- **ModernBERT** for RoPE + GLU + no-bias + unpadding efficiency (modern, fast, stable at 30%
  masking). flash-attn unavailable in our build → SDPA; `reference_compile=False` (toolchain).
- **Byte-level BPE, vocab 1000** — small chemistry alphabet; avoids OOV on rare SMILES tokens.
- **128-token training cap** — >99% of PubChem SMILES fit; bounds padding memory, keeps
  throughput ~750 seq/s (a known limitation for large molecules — see §5).

**Readout / heads:**
- **Frozen featurizer = masked-mean pool → per-feature z-score (train-fit) → small MLP**,
  trained with 3 head seeds. This mirrors how molecular foundation models are actually deployed
  (frozen embeddings + light head) and is our **primary** eval.
- **SFT heads:** per-family 2-layer MLP heads (MiniMol style) + one always-on MTR head for the
  217 RDKit descriptors. Losses combined with **Kendall homoscedastic uncertainty weighting**
  (learned per-family log-variance) so near-unlearnable families (L1000) can't dominate.
- **Anchors:** untrained-encoder random floor (×3 seeds) and ECFP4 + XGBoost, through the same
  head pipeline, for an honest lower/again-classical bar.

---

## 3. Training methodology

- **Objectives** are sampled per batch from a weighted mix of `{mlm, mtr, supervised}`; a run
  declares its mix (e.g. `{mlm:1}`, `{mtr:1}`, `{supervised:1}`, `{mtr:0.5, supervised:0.5}`).
- **MLM:** 30% masking (ModernBERT regime), dense per-token cross-entropy.
- **MTR (dense):** regress 217 deterministic RDKit descriptors, mean/std-normalized (stats fit
  once on a fixed 20k-SMILES sample); MSE. Structure-derived, so no assay-label leakage.
- **Supervised (sparse):** per-family MLP heads; MAE for quantum/L1000 regression, BCE for
  binary assays; **stratified per-family loading with caps** (routes each block-sparse row to its
  family bucket so no family starves — fixes the earlier PCQM-collapse bug).
- **Warm-start:** `init_encoder_path` loads a saved MLM encoder → *unsup→sup* (sequential,
  realistic deployment), vs random init → *skip-unsup*.
- **Optimisation:** AdamW, LR 2e-4, warmup 5%, weight-decay 0.01, grad-clip 1.0, batch 256,
  bf16. ⚠️ Same LR for scratch and warm-start — the LR sweep (§4) checks this is fair.
- **Compute accounting:** everything measured in **forward passes** (molecule-presentations).
  Within one epoch of the 124M corpus, forward passes = #unique molecules; they diverge only
  past 124M (the side-story regime).
- **Canonical SMILES** for all primary runs (one presentation per molecule); enumeration is the
  beyond-1-epoch side-story lever.
- **Durability:** periodic encoder checkpoints (`save_every_steps`) + a 10-min S3-sync sidecar,
  so a spot reclaim loses at most one checkpoint interval.

---

## 4. Methodology of all experiments

**Shared eval protocol.** Frozen featurizer, **scaffold splits** (Bemis-Murcko, via DeepChem),
absolute **per-task** metrics (never z-scored/aggregated across tasks): ESOL & QM7 = RMSE↓,
BBBP/BACE/Tox21/HIV = ROC-AUC↑. Standardizer fit on **train only**. Value reported = mean over
3 head seeds. "Lift" = improvement over the random-encoder floor.

1. **Round-1 exploratory (done).** unsup_only / sup_only / mixed at 2M FP + a unique-molecule
   scaling sweep. Result: unsup helps; sup-only(broken)≈random; frozen scaling looked saturated.
2. **Dense-vs-sparse ablation (done).** One MLM base; 6 arms warm-started (mtr / pcba / l1000 /
   sparse_all / dense_plus_sparse / pcqm) at 2M SFT. Verdict: SFT rebuild works (all beat random)
   but no arm beats the MLM base under the frozen probe; dense ≳ sparse mildly.
3. **Phase-2 scaling (running).** The H1–H4 core. Three blocks:
   - *pure-unsup ladder* — MLM from scratch at 2M→8M→24M→48M FP (→96M if still rising).
   - *unsup→sup[W]* — warm-start each ladder checkpoint on W at a fixed 2M SFT.
   - *skip-unsup[W]* — random init + SFT[W] at a per-W budget ladder (dense→96M; sparse/both→48M).
   For W ∈ {dense, sparse_all, dense_plus_sparse}. Single pretraining seed. Plotted vs **compute
   and #molecules**. **Stop-when-flat**: extend any ladder only while it moves >~2%/doubling.
4. **SFT-LR sweep (running, H5).** unsup→sup[dense] and [sparse_all] from a fixed base at LR
   {2e-4, 1e-4, 5e-5, 2e-5}. If a lower LR lifts SFT above the base, the ablation finding was an
   LR artefact.
5. **Leakage audit (running, H6).** InChIKey (2D-skeleton) overlap of each eval **test** split
   vs a 2M pretrain sample and each SFT family. Feeds an **InChIKey blocklist** that is stripped
   from the SFT data (and disclosed for pretraining).
6. **Eval-ceiling / Fig D (to build, H5).** Frozen probe vs **end-to-end finetune** per ladder
   checkpoint, plus a harder/larger task (HIV ~41k). If finetuning separates encoders the frozen
   probe couldn't, the probe was the ceiling — and the finetune comparison becomes primary.
7. **Headline bars / Fig A (to build).** random · ECFP4 · skip-unsup · pure-unsup · unsup→sup
   per W, compute-matched, **3 pretraining seeds** with CIs (the only statistically-defensible
   head-to-head).
8. **Catastrophic forgetting (bonus, H7).** Take an SFT'd encoder, continue with MLM, re-eval.
9. **Side-story (H8).** Past 124M FP: canonical repetition vs enumerated augmentation.

**Planned data enrichment (before re-running the sparse arms):** add **WONG** (already tokenized,
4 assays), raise the PCBA cap (currently 500k of ~1.5M), resolve the L1000 shortfall (root cause
pending the audit's per-family count), and **dedup eval-test InChIKeys** out of all SFT data.

---

## 5. Limitations, open problems, concerns

**Blocking (could change conclusions):**
- **Leakage (H6).** Nothing currently dedups pretrain/SFT against eval test sets; PubChem
  contains ~all MoleculeNet molecules and PCBA/Tox21 share bioassay space with eval tasks. Being
  quantified now; SFT will be deduped.
- **SFT-LR confound (H5).** "SFT ≤ MLM base" may be a too-high warm-start LR, not a real effect.
  Being swept now.
- **Single pretraining seed** on all scaling curves → no error bars; the ablation's dense-vs-
  sparse gap (6.6 vs 4.6%) is within plausible seed noise. Headline (Fig A) needs ≥3 seeds + CIs.
- **Frozen-probe ceiling.** The probe under-resolves encoder quality (MLM loss 0.14 vs 0.39 →
  same downstream), so "skip ≈ unsup" risks a Type-II error. Fig D must be promoted to primary.

**Scale & scope:**
- **Undertraining.** ≤96M FP is <1 epoch of 124M and 40–500× below MoLFormer's token budget.
  The *relative* claim is compute-matched and safe; *absolute-SOTA* and "unsup plateaus globally"
  claims are scoped to "this model, ≤96M FP" (the beyond-1-epoch story is separate).
- **Model size** (41M) may be below where unsup benefits fully emerge; H9 caveats this.

**Data quality / representativeness (point 6):**
- SFT is **PCQM-dominated** (~3.8M of 5.38M); **L1000 loads only ~1.6k** (root cause pending) and
  is near-unlearnable; Kendall weighting can drive L1000 toward zero → "sparse" ≈ "PCBA+WONG".
- Eval is **5–6 small MoleculeNet tasks** with documented label noise (esp. BBBP) and high-
  variance scaffold-test sets; QM7 (quantum) is a poor probe for SMILES structural pretraining.

**Methodology fine print:**
- Only **masked-mean pooling** tested ([CLS]/max/last-layer untested — could mask quality).
- **128-token cap** silently drops large molecules from pretraining *and* eval.
- **Descriptor stats auto-fit per box** (deterministic sample, but not a single shared file) —
  minor normalization drift risk.
- **Cross-wave absolute comparisons** (round-1 base vs phase-2 ladder) are only approximate;
  compare within a wave.

---

## 6. Data, tables, and figures to be produced

**Datasets (documented in dossier §2):** unsup PubChem (~124M); SFT wide parquet (5.38M:
PCQM/PCBA/L1000/WONG); MoleculeNet eval (ESOL/BBBP/BACE/Tox21/QM7 + HIV for ceiling).

| Artefact | Content | Status |
|---|---|---|
| **Fig A — headline bars** | random · ECFP4 · skip · pure-unsup · unsup→sup, per W, 3 seeds + CI | 🔲 build |
| **Fig B(compute)** | 3 lines/task (pure-unsup, unsup→sup, skip) vs forward passes, ×3 W | ⏳ collecting |
| **Fig B(data)** | same, x = #unique molecules | ⏳ collecting |
| **Fig C — ablation** | dense-vs-sparse lift table (7 arms) | ✅ done |
| **Fig D — eval-ceiling** | frozen vs finetune per checkpoint + HIV | 🔲 build |
| **Table — leakage audit** | eval-test ∩ {pretrain, PCBA, L1000, PCQM, WONG} InChIKey % | ⏳ running |
| **Table — SFT-LR sweep** | dense/sparse lift vs base at 4 LRs | ⏳ running |
| **Table — catastrophic forgetting** | metric before/after post-hoc MLM | 🔲 bonus |
| **Fig (side-story)** | beyond-124M: canonical vs enumerated | 🔲 future |
| **Plateau/knee summary** | budget where each ladder Δ<2%/doubling | ⏳ derived |
| **Datasets table + Model card** | sizes, splits, metrics; architecture/hparams | ✅ in dossier/§2 |

Each table/figure is regenerated from the per-run `moleculenet/moleculenet_summary.csv`;
placeholders live in the dossier keyed to run IDs.
