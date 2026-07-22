# CLIMB — Does unsupervised pretraining help a chemical language model?

CLIMB is a controlled study of **whether, and how much, unsupervised (masked-language-model)
pretraining on SMILES improves a chemical language model** — or whether you can skip it and
train supervised-from-scratch instead. Everything in the pipeline exists to isolate the causal
effect of *pretraining strategy* on downstream molecular-property prediction, holding the model,
tokenizer, optimizer, and evaluation fixed.

This README is the single source of truth for the methodology. It is written to be exacting
enough to serve as the methods section of the paper: it states the idea and hypotheses, lists
every experiment and figure, and then documents the architecture, data, curation/deduplication,
training, and evaluation in full detail.

> **Status (2026-07-20):** v2.1 pipeline. A ~41M-parameter ModernBERT encoder, frozen-featurizer
> evaluation, and a phase-2 scaling matrix on AWS. A molecule-level leakage audit is complete and
> dedup applied. A batch of long runs was **truncated by an old 12h watchdog cap** (compounded by an
> unwired descriptor precompute); the harness is now **hardened** — throughput-based cap, completion
> judged by verified forward-passes (not file existence), a preflight gate, and direct box→email
> (SNS) alerts on START/COMPLETE/TRUNCATION/STALL. The recovery wave + the expanded experiment set
> (E12–E13, mechanism figures) below are queued short-first. Live experiment state and result
> placeholders live in `experiments/climb_v2_phase2/RESULTS_DOSSIER.md` (a scratch storyboard).

---

# Part I — The study

## 1. Introduction: the idea

The conventional recipe for a chemical language model is: pretrain a transformer encoder on a huge
corpus of unlabeled SMILES with a masked-language-modeling (MLM) objective, then use the frozen
encoder (or fine-tune it) for downstream property prediction. The premise is that unsupervised
pretraining learns a transferable representation of chemical space.

The original motivating intuition for this project (see the project's founding note, now folded
into §12) was skeptical: **if you ultimately have supervised data, is the unsupervised phase doing
anything the supervised phase could not?** The design philosophy has been constant throughout:

- **One encoder, many heads.** A shared encoder trunk feeds task-specific heads; MLM is treated as
  just one more head. Heads are discarded after pretraining — only the encoder is kept and evaluated.
- **Isolate the variable of interest.** Fix the architecture, the BPE tokenizer, and (via a single
  up-front hyperparameter search) the optimizer settings, then vary *only* the pretraining
  strategy. Any downstream difference is then attributable to pretraining, not confounds.
- **Measure on a fixed, external benchmark.** Downstream performance is always MoleculeNet, under a
  frozen-featurizer protocol that mirrors how these models are actually deployed.

Over the project this sharpened into a **three-regime comparison**, evaluated per task at matched
compute. **Canonical regime vocabulary (used identically in the README and every figure):**

| Regime | Definition | Question it answers |
|---|---|---|
| **random** | dumb chance model — 0.5 ROC-AUC / predict-the-mean RMSE (≈1.0 on DeepChem-normalized targets). A reference, not a trained model. | The floor any real model must clear. |
| **no_pretrain** | random-init ModernBERT, **no pretraining**, **frozen** features → head trained on the eval set. (Formerly the "random floor" — but it is a real ~41M model, often non-trivially above chance.) | What do random encoder features alone give? |
| **no_pretrain_end_to_end** | random-init ModernBERT **finetuned end-to-end** (encoder unfrozen) on each eval task. The from-scratch finetune baseline (E1). *Not yet run — reserved as a pending placeholder bar in Fig A1.* | Does finetuning a random encoder beat freezing it? |
| **sup_only** | random init → supervised fine-tune (SFT). Never sees MLM. *(formerly "skip-unsup")* | Can you skip unsupervised pretraining entirely? |
| **unsup_only** | MLM only, frozen features. *(formerly "pure-unsup")* | What does the unsupervised prior give on its own? |
| **unsup→sup** | MLM, then SFT. *(formerly "mixed")* | The realistic deployment recipe. |
| **sup→unsup** | SFT, then MLM (the forgetting direction — H7). | Does post-hoc MLM erase the supervised signal? |

crossed with **what you supervise on** — dense computed labels (RDKit descriptors, à la ChemBERTa-2)
vs sparse assay labels (PCBA/L1000/PCQM/WONG, à la MiniMol) vs both.

### Evolution from the original design (what changed and why)

The founding design (v1) used a **13M-parameter RoBERTa**, measured a single **aggregate score**
across all MoleculeNet tasks, budgeted compute in **tokens**, and planned a 3-D
unsupervised×supervised surface. v2 revised this after early results and reviewer-style scrutiny:

- **Model:** 13M RoBERTa → **~41M ModernBERT** (RoPE, GeGLU, pre-norm), matching MoLFormer-base
  (44M) / ChemBERTa-2 (46M) so a null result cannot be dismissed as a toy model.
- **Metric:** single aggregate z-scored score → **absolute per-task metrics** (RMSE/AUC), never
  aggregated across heterogeneous tasks, because z-scoring across tasks obscured effects and made
  the number sensitive to the task mix.
- **Compute axis:** tokens → **forward passes** (molecule-presentations), which is the quantity the
  scaling ladders actually vary and is comparable across objectives.
- **Supervised recipe:** rebuilt from a single linear head to **per-dataset MLP heads with Kendall
  homoscedastic uncertainty weighting**, after diagnosing that the v1 supervised phase was broken
  (see §6.3, the PCQM-collapse bug).
- **Leakage:** the v1 README flagged possible PCBA train/eval overlap as an open action item. v2
  **measured it at the molecule level** and found it material (§6.5), then implemented dedup.

---

## 2. Hypotheses

**Core**
- **H1 — Does unsupervised pretraining help at all?** Compare sup_only vs unsup_only vs unsup→sup
  per task at matched compute.
- **H2 — Which mechanism? (initialization vs regularization vs added information).** If unsup helps,
  *why*? The earlier "better initialization vs better endpoint" framing silently merged two distinct
  mechanisms — regularization *also* yields a better endpoint sup_only can't reach — so it misfiled
  regularization as "adds information." Three testable mechanisms, each with a distinctive signature:
  - **(a) initialization** — same reachable endpoint, reached faster/cheaper; sup_only catches up
    given enough SFT compute. *Signature:* a left-shift on the label/compute axis that **closes at
    full labels / high compute**.
  - **(b) regularization** — constrains SFT toward a more generalizable endpoint sup_only can't reach;
    benefit **concentrated in the low-label regime** with a **smaller train–test gap**. *Signature:*
    gain only at small label fractions + reduced overfitting, vanishing at 100% labels.
  - **(c) adds information** — injects content sup_only has no access to; benefit **persists as
    labels grow** and is **content-dependent**. *Signature:* a persistent vertical gap at 100% labels;
    provable only against a content-free pretraining control (E13 / C10).

  These are separated by the label-efficiency curves (§4/C7) and the corrupted-pretraining control.
- **H3 — Does the SFT label type decide it?** Dense (RDKit-descriptor MTR) vs sparse (assays) vs
  both. Sub-question: can dense labels compensate for missing unsup where sparse cannot?
- **H4 — Plateaus.** Where (in compute / molecules) do the unsup_only and sup_only curves flatten?

**Validity / confounds (must resolve before trusting H1–H4)**
- **H5 — Is "SFT ≤ MLM base" real or an artifact?** Two suspects: the frozen probe is too
  low-ceiling to resolve encoder quality; or warm-start SFT at the pretraining LR destroys MLM
  features. Tested by the finetune eval-ceiling (Fig D) and the SFT-LR sweep.
- **H6 — Leakage.** Are results inflated by molecule (canonical-SMILES) overlap between eval **test**
  splits and the pretraining / SFT data? **Measured — material; dedup applied (§6.5).**

**Secondary / bonus**
- **H7 — Catastrophic forgetting.** Does running MLM *after* SFT erase the supervised signal?
- **H8 — Beyond one epoch.** Past one pass of the corpus, does SMILES enumeration (fresh
  augmentation per pass) beat exact canonical repetition?
- **H9 — Representation vs memorization.** Do novel molecules benefit from pretraining as much as
  molecules the model has actually *seen during unsupervised pretraining*? Studied **entirely on the
  production (deduped) models — no un-deduplicated model is trained**, so no leakage is reintroduced.
  The key is that H9 asks about the **pretraining** corpus, and the SFT dedup (§6.6) only removes eval
  overlap from the **supervised** data — the pretraining overlap is a *different*, disclosed 0–7%
  (§10) that is deliberately kept. Two leakage-free handles: (i) that **0–7% pretraining overlap** is a
  genuine "seen-in-pretraining" group already present in the clean model; (ii) **Tanimoto distance to
  the nearest training molecule** (ECFP4, binned) gives a continuous interpolation↔extrapolation
  dose–response that subsumes the binary seen/not-seen split (at distance ≈0 it already probes the
  near-memorization limit). Exact memorization from *supervised* labels is intentionally **not** tested
  — that would require reintroducing the leakage we removed.

**Content (added 2026-07-20)**
- **H10 — Domain-matched transfer.** Transfer from an SFT family to a downstream task is governed by
  molecular / domain **content** similarity (bioassay families → BBBP/BACE/Tox21/HIV; quantum PCQM →
  QM7), *not* by label type or label distribution per se. Sub-question: how best to *measure*
  similarity — molecular content (nearest-neighbor Tanimoto / scaffold overlap) vs label-space
  similarity. Built from the existing E1 single-family arms (transfer matrix, §4 / C8).

*(No formal scaling-law hypothesis. Model-size and compute-optimal scaling are **out of scope for
this paper**; a purely descriptive compute/data plot is recycled from runs collected anyway — §4.)*

A hypothesis-resolution matrix (positive vs negative evidence per sub-question) is in §11.

---

## 3. Experiments

All runs use the shared eval protocol (§8): frozen featurizer, scaffold split, absolute per-task
metrics, lift over a random-encoder floor.

| # | Experiment | Purpose | Status |
|---|---|---|---|
| E0 | **Round-1 exploratory** | unsup_only / sup_only / mixed at a fixed budget + a unique-molecule sweep | ✅ done |
| E1 | **Dense-vs-sparse ablation + transfer-matrix source** | 6 SFT arms warm-started from one MLM base (mtr / pcba / l1000 / sparse_all / dense+sparse / pcqm). **Retain per-task (not just per-arm) metrics** as the source for the H10 task-transfer matrix (Fig, C8) | ✅ done; re-run deduped |
| E2 | **Phase-2 scaling matrix** | the H1–H4 core: unsup_only ladder + unsup→sup + sup_only ladders × 5 SFT recipes | ⏳ running |
| E3 | **SFT-LR sweep (H5)** | warm-start SFT at LR {2e-4,1e-4,5e-5,2e-5} for dense + sparse | ⏳ (base refit pending) |
| E4 | **Leakage audit (H6)** | canonical-SMILES overlap of eval test vs pretrain + each SFT family | ✅ done |
| E5 | **Eval-ceiling / Fig D (H5)** | frozen probe vs end-to-end finetune per checkpoint + HIV | 🔲 planned |
| E6 | **Headline bars / Fig A** | random · no_pretrain · ECFP4 · sup_only · unsup_only · unsup→sup per recipe (**single seed for now; 3-seed CIs deferred**) | 🔲 planned |
| E7 | **Catastrophic forgetting (H7)** | SFT encoder → continue MLM → re-eval | 🔲 planned |
| E8 | **Beyond-one-epoch (H8)** | canonical repetition vs enumerated augmentation past one pass | 🔲 planned |
| E9 | **Molecule-overlap matrix (H9)** | performance by (seen-in-unsup × seen-in-sup) group | 🔲 planned |
| ~~E10~~ | **Model-size / Chinchilla scaling — DROPPED** | model-size scaling (13M/50M/100M/200M) and any compute-optimal / IsoFLOP scaling-law analysis are **out of scope for this paper**; no new runs. A descriptive compute/data plot is recycled from existing runs (§4) | ❌ dropped |
| E12 | **Label-efficiency sweep (H2 mechanism)** | retrain the frozen probe on 5/10/25/50/100% of each eval **train** split, per regime + per task; report the train–test gap. No new pretraining (cached embeddings) | 🔲 planned |
| E13 | **Corrupted-pretraining control (H2c)** | two arms at the 8M matched budget: `corrupt_mlm_8M` (**shuffled-token MLM**) and `corrupt_mtr_8M` (**shuffled-target MTR**) — same objective/compute, zero chemical content. See §7.1. | ✅ **done** — both arms trained and verified 2026-07-22 |

The **5 SFT recipes** used in E2 (each gets a full sup_only + unsup→sup ladder):
`dense` (RDKit-MTR) · `sparse_all` (PCBA+L1000) · `dense_plus_sparse` · `minimol_full`
(PCQM+PCBA+L1000+WONG, the faithful MiniMol LargeMix + Wong) · `mixed` (descriptors + minimol_full).

### Compute ladders (E2)
- **unsup_only ladder:** MLM from scratch at **2M → 8M → 24M → 48M** forward passes (→96M if still
  rising, stop-when-flat: extend while Δ > ~2% per doubling).
- **unsup→sup[recipe]:** warm-start each ladder checkpoint on the recipe with a fixed 2M-FP SFT.
- **sup_only[recipe]:** random init + SFT at a per-recipe budget ladder; `dense` goes to **96M**
  (the catch-up candidate for H2), the others to 48M.

Because the corpus is ~12M molecules (§6.1), forward-pass budgets ≥ 12M are **multi-epoch**
(24M ≈ 2 epochs, 96M ≈ 8), which is why H8 (repetition vs augmentation) is a distinct question.

---

## 4. Figures / results produced

**Figure-ID convention.** Each **hypothesis** gets a **letter** (A = H1 … J = H10); each **plot** for
that hypothesis gets a **number**; `S` = supplementary/descriptive (no single hypothesis); `T` =
paired-significance **table** (meta-prefix, like `S`). All figures and tables
are generated by `climb_figures.ipynb` (one global style block → uniform, journal-ready formatting;
exports 300-dpi PNG + vector PDF to `figures_out/`). The `Data` column is the current collection
state; ❌-dummy figures are drawn as labelled empty-axis placeholders (legend + "what's missing"), no
invented data.

| ID | Hyp. | Content | x-axis / form | Data |
|---|---|---|---|---|
| **A1** | H1 | random (chance) · no_pretrain · no_pretrain_end_to_end · Morgan+XGBoost · unsup_only · sup_only ×5 · unsup→sup | compute-matched bars (8M FP) | ✅ (unsup→sup + no_pretrain_end_to_end pending placeholders) |
| **A2** | H1·H4 | unsup_only + 5 sup_only lines/task, vs no_pretrain (dashed) & random (dotted) references; plateau = knee | vs forward passes (log) | ✅ (unsup→sup pending; 7 truncated excluded) |
| **A3** | H1 | round-1 exploratory: no_pretrain · unsup_only · sup_only · unsup→sup · Morgan+XGBoost | grouped bars | ✅ |
| **B1** | H2 | label-efficiency: Morgan+XGBoost · no_pretrain · sup_only · unsup_only vs #labels (100/300/1k/3k/full) | vs label count (log) | ✅ (no unsup→sup; no train–test gap) |
| **B2** | H2c | three bars per task: **no_pretrain · corrupted pretraining · real pretraining** (MLM arm vs `unsup_only`; MTR arm vs `sup_only:dense`) — separates H2(c) "adds information" from H2(a/b) "init/regularization"; see §7.1 | control bars | ✅ real data (both control arms verified 2026-07-22) |
| **C1** | H3 | dense-vs-sparse ablation, per-arm lift over no_pretrain (6 unsup→sup arms + base) | bars + heatmap | ✅ (pre-dedup ‡ assay arms) |
| **E1** | H5 | eval-ceiling: frozen probe vs finetuned per ckpt (+HIV) | vs compute | ❌ dummy (no finetune runs) |
| **E2** | H5 | SFT-LR sweep: dense/sparse lift vs unsup_only base at 4 LRs | bars | ❌ dummy (8 runs trained, not evaluated) |
| **F1** | H6 | eval-test overlap % with pretrain / L1000 / PCBA / WONG / PCQM | heatmap (§6.5) | ✅ |
| **G1** | H7 | metric before/after post-hoc MLM (sup→unsup forgetting) | bars | ❌ dummy (no sup→unsup run) |
| **H1** | H8 | canonical vs enumerated (unsup_only): downstream lift *(MLM val/test loss panel missing)* | vs unique-mol fraction | ✅ (downstream only) |
| **I1** | H9 | Panel 1 lift seen-vs-not-seen (pretrain overlap); Panel 2 lift vs Tanimoto distance | 2 panels | ❌ dummy (no per-mol dumps / fingerprints) |
| **J1** | H10 | rows = single-family unsup→sup arms, cols = eval task, cell = lift over no_pretrain; domain tags | heatmap | ✅ partial (no Tanimoto overlay) |
| **S1** | — | every collected point: metric vs forward passes, coloured by regime/recipe — descriptive, **no fitted law** | scatter | ✅ |
| **T1** | H1 | CLM vs toughest classical baseline (fp_desc = Morgan+descriptors→XGBoost): `dense` (trained on descriptors) + `unsup_only` (control, never saw descriptors), each with per-task Δ, fold-t, and the rigorous point test (Wilcoxon sq-err for RMSE / DeLong paired-AUC for classification) — protocol in §8.1 | paired-significance table | ✅ (8M single-seed×5-fold; refreshes on the 3-seed×5-fold pass) |
| **T2** | H1 | `unsup→sup` (MLM→SFT) decomposed at the 8M base: **Q1** vs `sup_only` (does the MLM base help the SFT?) → adds ~0, 0/5 tasks significant, worse on `dense_plus_sparse`; **Q2** vs `unsup_only` (does SFT help on top of MLM?) → significantly helps regression (ESOL/QM7), hurts bioactivity (BBBP/BACE). Reinforces "skip unsupervised pretraining." | paired-significance table | ✅ (8M) |

**Two distinct baselines (do not conflate):** **random** is the dumb chance model — 0.5 ROC-AUC, or
predict-the-mean RMSE (≈1.0 on DeepChem-normalized targets); it is a reference line, not a trained
run. **no_pretrain** is a random-parameter ModernBERT run through the *same* frozen-feature + head
pipeline as every other arm (neither MLM nor SFT) — a real ~41M model that is often **non-trivially
above chance** (e.g. BBBP no_pretrain ≈0.695 vs Morgan+XGBoost ≈0.657, both well above the 0.5 random
line). The separate **no_pretrain_end_to_end** baseline (random encoder **unfrozen**, finetuned
directly on each eval task) is the eval-ceiling (E1) case — **not yet run**, and reserved as a hatched
pending-placeholder bar in Fig A1 so it drops straight in once collected.

Metric conventions: ESOL/QM7/Lipophilicity = RMSE (lower better); BBBP/BACE/Tox21 = ROC-AUC (higher
better); **HIV = NEF1% (top-1% enrichment, higher better)** as the virtual-screening headline (ROC-AUC
secondary; see §6.5). In the per-task figures (A1/A2/…) HIV's bar/line uses NEF1%. "Lift" = improvement
over the **no_pretrain** floor. **All figures are currently single
pretraining seed** (3-seed replication with CIs on the bar figures is deferred — see §10); each eval
still averages **3 head seeds**, and scaling curves use the stop-when-flat rule in place of plateau
error bars.

**Data-collection gaps** (why the ❌-dummy figures exist): the whole **unsup→sup ladder** (A1/A2),
the **finetune eval-ceiling** (E1), the **SFT-LR evaluation** (E2 — encoders trained but never
evaluated), the **forgetting** run (G1), the **MLM val/test-loss** readout (H1), and the
**per-molecule prediction / ECFP4-fingerprint dumps** (I1) were never emitted. Seven phase-2 runs were
truncated by the old 12h cap and are excluded (`unsup_48M`, `skip_dense_{24M,48M,96M}`,
`skip_mixed_24M`, `skip_minimol_full_48M`, `skip_dense_plus_sparse_48M`) — note `skip_dense_24M` /
`skip_mixed_24M` report `status="ok"` but only reached 40% / 37% of budget.

---

# Part II — Methods (exacting detail)

## 5. Model architecture

A **ModernBERT** encoder (HuggingFace `ModernBert*`), constructed in exactly one place
(`config_v2.build_modernbert_config`) so pretraining, evaluation, and the random baseline
instantiate an identical architecture.

| Hyperparameter | Value |
|---|---|
| Backbone | ModernBERT (RoPE, pre-norm, GeGLU, alternating global/local attention) |
| hidden_size | 512 |
| num_hidden_layers | 12 |
| num_attention_heads | 8 |
| intermediate_size (GeGLU post-gate) | 1536 |
| max_position_embeddings (RoPE cap) | 256 |
| global_attn_every_n_layers | 3 |
| local_attention window | 128 |
| dropouts (attention / mlp / embedding) | 0.0 / 0.0 / 0.0 |
| norm_eps | 1e-5 |
| vocab_size | 1000 |
| **Total encoder params** | **~41.4M** (verified empirically) |
| attention implementation | SDPA (flash-attn unavailable in the build) |
| reference_compile | False (portable across CPU test boxes + GPU workers; toolchain-safe) |

**Special tokens.** The SMILES tokenizer has no CLS/SEP; position-0 (`<s>`/bos) doubles as the
pooled "cls". Ids: pad=1, bos=0, eos=2, cls=bos, sep=eos (all < vocab_size). Pooling for evaluation
is masked-mean, not CLS (see §8).

**Objective heads (discarded after pretraining, only the encoder is kept):**
- **MLM head** — standard masked token prediction.
- **MTR head** — a 2-layer MLP regressing 217 normalized RDKit descriptors (dense targets).
- **Supervised multi-head** — one `Linear→GELU→Linear` head *per dataset family*, with per-family
  loss type (MAE for regression families, BCE for binary assays) combined by **Kendall
  homoscedastic uncertainty weighting** (learned per-family log-variance: `exp(-s)·L_f + s`), so
  near-unlearnable families cannot dominate and no manual loss weights are needed.

**Why this size.** ~41M is deliberately "believable-SOTA but single-GPU": it matches MoLFormer-base
(44M) and ChemBERTa-2 (46M), trains on one A10G (g5.2xlarge), and keeps a large ablation matrix
affordable. SMILES are short (median ~30–50 tokens), so token-level modeling does not need large
capacity. Generalization to larger models is a stated limitation and is **out of scope for this
paper** (model-size scaling / E10 dropped).

---

## 6. Datasets (provenance, curation, deduplication)

### 6.1 Unsupervised corpus (MLM / MTR source)
- **Source:** PubChem (derived from the PubChem-124M SMILES/SELFIES/InChI/IUPAC dataset), filtered.
- **As used:** `s3://climb-s3-bucket/tokenized_sources/pubchem_filtered/` — **12 parquet shards ×
  1,000,000 rows = ~12M molecules**, column `SMILES_canonical`. (A pre-tokenized pickle mirror
  exists at `.../pubchem_filtered_tokenized_pkl/` for the fast canonical MLM stream.)
- **Implication:** one epoch = ~12M molecules. Forward-pass budgets above 12M are multi-epoch.
- Streaming is deterministic given `subset_seed` (worker-sharded, hash-based subset membership), so
  every run sees the same molecule order and the ladders are nested subsets.
- ⚠️ **Not implemented: held-out MLM val/test loss.** An earlier plan (C5) called for a hash-based
  corpus holdout giving per-run MLM validation/test loss, which would have exposed the H8 mechanism
  directly (canonical repetition overfitting while enumeration keeps generalizing). **No such holdout
  or loss exists in the code** — the training loop logs train loss only. H8 is therefore adjudicated
  on downstream lift alone, which the frozen-probe ceiling (§10) may compress.

### 6.2 Tokenizer
- **Byte-level BPE, vocab 1000**, artifact `s3://climb-s3-bucket/tokenizer_10M/` (`tokenizer.json`).
- **Fixed across every run** (HPO, pretraining, SFT, evaluation) so vocabulary/segmentation never
  confounds a comparison. Training sequences are capped at 128 tokens (>99% of SMILES fit; bounds
  padding memory), separate from the 256 RoPE position cap. Tokenizer training is reproducible via
  `train_tokenizer.py` (provenance only; the artifact is prebuilt).

### 6.3 Supervised corpus (SFT source) — the "wide" parquet
- **Path:** `s3://climb-s3-bucket/tokenized/supervised_wide_parquet/`.
- **Shape:** **5,382,243 rows × 3,322 columns**. Columns: `smiles_canon`, `input_ids`,
  `attention_mask`, and one label column per assay endpoint, prefixed by family.
- **Families and sizes (molecules present, measured 2026-07-16):**

  | Family | Type | Label cols | Molecules |
  |---|---|---|---|
  | PCQM | quantum regression | 31 | 3,810,323 |
  | PCBA | bioassay (binary) | 1,328 | ~1,517,000 |
  | WONG | antibiotic screen | 4 | 34,652 |
  | L1000_MCF7 | gene-expression regression | 978 | 11,718 |
  | L1000_VCAP | gene-expression regression | 978 | 7,800 |

  This is the MiniMol **LargeMix** (PCQM4M + PCBA_1328 + L1000_VCAP + L1000_MCF7) plus **WONG**.
- **Layout caveat — multi-family rows.** The wide table is a join, not exclusive per-family blocks:
  a molecule assayed in several families carries several families' labels in one row (NaN elsewhere).
  This drove the L1000 loader bug in §6.4.
- **Family selection rationale.** MiniMol's Table 6 shows PCQM *negatively* transfers as a single
  fine-tuning source, so v1 dropped it; v2 keeps it inside the multitask mixture (where uncertainty
  weighting neutralizes it) because that is MiniMol's actual recipe and because QM7 is an eval task.

### 6.4 Supervised loading and the L1000 routing fix
`data_v2.load_supervised_inram_stratified` builds a family-balanced in-RAM training set:
- **Stratified per-family caps** (PCBA 500k, L1000 100k each, PCQM 200k, WONG 100k) so no family
  starves another. (v1 bug: the old loader took the first ~1M rows of a PCQM-first table → 100%
  PCQM → the head collapsed to ~25 quantum tasks; this is why v1 supervised ≈ random.)
- **Scarce-family-first routing.** Because rows are multi-family, first-match routing in PCBA-first
  order let PCBA *steal* L1000's molecules (only ~1.6k of ~11.7k L1000 loaded). Routing smallest-cap
  families first recovers the full L1000 set. Each retained row keeps its full label vector, so a
  molecule still contributes to *every* family head it has labels for.
- **Density filter** drops label columns with < 0.1% coverage within the retained rows.
- **Leakage dedup** (see 6.5): rows whose `smiles_canon` is on the eval blocklist are dropped.

### 6.5 Downstream evaluation datasets
MoleculeNet via DeepChem, **scaffold split** (Bemis–Murcko), loaded in `eval_v2._load_moleculenet`.
Pre-registered 7-task suite, chosen as one dataset per common cheminformatics use case and **not
revised post-hoc** (`config_v2.MOLECULENET_TASKS_V2`; evaluated by default on every run). Counts below
are computed from the same loaders (`featurizer="Raw"`, scaffold split), summed over train+val+test:

| Dataset | Type | Metric | Description | Positives / Negatives | Total (molecules) |
|---|---|---|---|---|---|
| ESOL | regression | RMSE ↓ | log aqueous solubility (log mol/L) | n.a. | 1,128 |
| BBBP | classification | ROC-AUC ↑ | binary — does the molecule cross the blood–brain barrier | 1,560 / 479 | 2,039 |
| BACE | classification | ROC-AUC ↑ | binary — β-secretase 1 (BACE-1) inhibition | 691 / 822 | 1,513 |
| Tox21 | classification (12 tasks) | ROC-AUC ↑ | 12 binary toxicity assays (nuclear-receptor & stress-response) | 5,858 / 72,006 † | 7,823 |
| QM7 | regression | RMSE ↓ | atomization energy (quantum, DFT) | n.a. | 6,838 |
| **HIV** | classification (**virtual screening**) | **NEF1% ↑** (primary), ROC-AUC ↑ (secondary) | binary — inhibits HIV replication; large + **highly imbalanced (3.5% active)** → a genuine early-enrichment / hit-retrieval task | 1,443 / 39,677 | 41,120 |
| Lipophilicity | regression | RMSE ↓ | octanol/water distribution coefficient (logD 7.4) | n.a. | 4,200 |

† Tox21 positives/negatives are **label counts summed over its 12 binary tasks** (missing labels
excluded), not molecule counts — hence the total exceeds the 7,823 molecules.

**HIV as the virtual-screening task (added 2026-07-20).** Virtual screening — ranking a large library
so true actives surface in the tiny top slice a chemist can actually test — is a primary downstream use
of molecular foundation models, and none of the small MoleculeNet sets exercises it. HIV does: 41k
molecules, 3.5% active. Its headline metric is therefore **NEF1%** (normalized enrichment factor of true
actives in the top 1%), following Truong et al. 2026 (*J. Cheminform.*, s13321-026-01262-x) and the
LIT-PCBA definition (Tran-Nguyen, Jacquemard & Rognan, *JCIM* 2020):

$$\text{NEF}_{1\%}=\frac{\text{EF}_{1\%}}{\text{EF}_{1\%}^{\max}}=\frac{H_a}{\min(n,\,A)},\qquad n=\lceil 0.01\,N\rceil$$

where *N* = #compounds in the (held-out) test fold, *A* = #actives, *n* = size of the top-1% slice,
*H_a* = actives found in it. It is precision@top-1%, capped by recall when actives are scarce; range
[0, 1], with 1 = a perfect ranking that put every retrievable active on top. Computed **per held-out
fold**, then reported as mean ± std across folds (same protocol as every other metric). ROC-AUC is kept
as a secondary metric. NEF1% is only *informative* on imbalanced retrieval — on class-balanced sets
(e.g. BBBP, ~76% positive) the top 1% is trivially all-active and NEF1% saturates at 1.0, which is why
it is the headline metric for HIV specifically and reported-but-not-headline elsewhere. Implemented in
`heads_v2.compute_nef` and emitted for every classification task by `eval_v2` (as `main_metric="nef1"`
rows; the primary metric keeps the bare `<ds>_MEAN` suite key, NEF is namespaced `<ds>_nef1_MEAN`).

### 6.6 Molecule-level leakage audit and dedup (H6)
`scripts/leakage_audit.py` computes a canonical key for every eval molecule and measures overlap
with a pretrain sample and each SFT family. **Key = RDKit canonical SMILES of the largest fragment**
(`MolFromSmiles → keep largest '.'-fragment (salt strip) → MolToSmiles`) — ~100× faster than
InChIKey and adequate for 2-D structure identity.

**Result (% of each eval TEST split found in each source, 2026-07-16):**

| eval | pretrain | L1000_MCF7 | PCBA | WONG | PCQM |
|---|---|---|---|---|---|
| ESOL | 7.1% | 18.6% | 19.5% | 21.2% | 0 |
| BBBP | 5.9% | **42.6%** | 12.3% | 15.2% | 0 |
| BACE | 0% | 0 | 0.7% | 0 | 0 |
| Tox21 | 3.7% | **24.7%** | 10.2% | 11.1% | 0 |
| QM7 | 0.9% | 0.1% | 2.3% | 1.0% | 0 |
| HIV | 0.3% | 1.1% | **37.0%** | 1.6% | 0 |

Interpretation: **pretrain overlap is low (0–7%, the standard/disclosed kind that biases *toward*
"unsup helps"); SFT-assay overlap is large** (BBBP 43% via L1000, HIV 37% via PCBA). Therefore every
arm that trains on assay labels must be deduped and re-run; unsup_only, dense-MTR (descriptors are
not eval labels), and random arms are essentially unaffected. PCQM overlaps 0% (training on quantum
does not leak into QM7); BACE is clean.

**Dedup (`scripts/make_eval_blocklist.py`):** compute canonical keys for **all** eval molecules
(train+val+test, all 6 tasks) = **58,008 keys**; scan the parquet's `smiles_canon` once and collect
the exact strings that match = **34,301 leaked rows**; the supervised loader drops these (direct
string membership, no RDKit in the hot path). Blocklist:
`s3://climb-s3-bucket/configs/eval_blocklist.json`.

⚠️ **The blocklist predates Lipophilicity and does not cover it.** The S3 artifact was built on
**2026-07-16** over the then-6-task suite; **Lipophilicity was added to the eval suite on 2026-07-18**
(commit `6f939c0`) and the blocklist has not been rebuilt since (34,301 entries, unchanged). No SFT
arm is therefore deduplicated against Lipophilicity's eval molecules. Lipophilicity is **not** one of
the six core tasks any figure plots, but it *is* scored by every 7-task evaluation, so **no
Lipophilicity number may be reported without first rebuilding the blocklist and re-running the
affected SFT arms**.

⚠️ **The blocklist is applied only to supervised objectives.** `pretrain_v2` loads and applies it
only when a run's objective mix contains `supervised`, so pure-MLM and pure-MTR runs never consult
it. That is correct — those objectives never touch the supervised parquet — but it means "deduped"
describes the SFT stage specifically, not the corpus.

**Two distinct keys — dedup vs "seen" (C17).** ⚠️ *Corrected against the code:* the two keys differ
**only** by the largest-fragment (salt-strip) step, **not** by stereochemistry. Both
`scripts/make_eval_blocklist.py` and the eval-side identity key in `eval_v2.py` call
`Chem.MolToSmiles(m)` with RDKit's default `isomericSmiles=True`, so **stereochemistry is retained in
both**; an earlier version of this section claimed the blocklist key "canonicalizes without stereo",
which the code does not do. The blocklist key additionally salt-strips to the largest fragment
(selected by *string length*, not heavy-atom count), which conservatively over-removes — the right
bias for a **safety** dedup. The H9 "seen" axis uses the un-stripped key, so a salt form and its
parent count as distinct there while the blocklist collapses them. Both keys are recorded per eval
molecule (§9.5) so a reviewer can see which molecules each classifies as overlapping.

### 6.7 Dense descriptor targets (MTR) and precompute
- **Targets:** 217 RDKit `Descriptors.descList` values, NaN-safe, **z-normalized** with mean/std fit
  once on a deterministic 20k-molecule sample (`subset_seed=0`); stats cached to
  `configs/descriptor_stats.json` (+ S3). Descriptors are structure-invariant, so they are valid
  under SMILES enumeration.
- **Precompute (`scripts/precompute_descriptors.py`).** On-the-fly descriptor computation is
  16 ms/mol → it starved the GPU (dense ran at ~120 vs 748 seq/s). Since the corpus is only ~12M
  molecules, descriptors are computed **once per shard**, normalized, and written as row-aligned
  float16 companions (`descriptors_shard_NNNNN.npy`) to
  `s3://climb-s3-bucket/tokenized_sources/pubchem_descriptors/`. The MTR streaming path reads these
  instead of calling RDKit, making dense runs GPU-bound. (This also avoids recomputing descriptors
  every epoch in the multi-epoch ladder points.)

---

## 7. Training methodology

- **Objective sampler.** Each run declares a per-batch objective mix over `{mlm, mtr, supervised}`
  (e.g. `{mlm:1}`, `{mtr:1}`, `{supervised:1}`, `{mtr:0.5, supervised:0.5}`); one optimizer step per
  batch, objective sampled by weight (`data_v2.MultiObjectiveBatchIterator`).
- **MLM:** 30% masking (ModernBERT regime), dense per-token cross-entropy.
- **MTR:** masked MSE on the 217 normalized descriptors.
- **Supervised:** per-family MLP heads; MAE (quantum/L1000 regression) or BCE (PCBA/WONG binary);
  Kendall uncertainty weighting across heads.
- **Warm-start.** `init_encoder_path` loads a saved MLM encoder → unsup→sup (sequential); absent →
  sup_only (random init).
- **Optimizer.** AdamW, LR **2e-4**, warmup ratio **0.05**, weight-decay **0.01**, grad-clip **1.0**,
  batch **256**, bf16. ⚠️ The same LR is used for scratch and warm-start; the SFT-LR
  sweep (E3) tests whether this is fair for the warm-start phase.
- **Seeding.** The governing seed is `selection.pretraining_seed` (0 for the primary runs; replicates
  use 1 and 2), which fixes weight init, corpus-subset membership, objective sampling and the
  corruption RNG. `TrainingConfigV2.seed = 42` is only a fallback and is *not* what the runs used.
- **MLM corruption detail.** Of the 30% selected positions, the standard **80/10/10** split applies:
  80% replaced with `[MASK]`, 10% with a random token, 10% left unchanged
  (`data_v2.mlm_mask_tokens`); loss is computed on the selected positions only.
- **Supervised label handling.** Every non-binary supervised column is **z-scored** over the loaded
  subset (mean/std computed on that subset, values clipped to ±10), and a column is treated as
  **classification iff its unique valid values are a subset of {0,1}** — task type is *inferred*
  from the labels, not declared (`data_v2.py`). MAE/BCE in the bullet above therefore act on
  z-scored and inferred-type targets respectively.
- **Compute accounting.** Everything is measured in **forward passes** (molecule-presentations).
  Within one epoch (≤12M) forward passes = #unique molecules; beyond that they diverge (repetition).
- **Canonical vs enumerated.** Primary runs use canonical SMILES (one presentation per molecule);
  enumeration (on-the-fly RDKit randomization) is the H8 lever for the beyond-one-epoch regime.
  ⚠️ **Known limitation — the enumeration RNG is not captured, so enumerated runs are not
  bit-reproducible.** `smiles_augment.randomize_smiles` accepts an `rng` argument but ignores it and
  calls `Chem.MolToSmiles(mol, doRandom=True)`, which draws atom orderings from RDKit's *global* RNG;
  the two collators that use it are likewise unseeded on this path (`RawSmilesMLMCollator` builds a
  bare `random.Random()`, `MTRCollator` passes no rng at all). Re-running an enumerated arm therefore
  reproduces the *distribution* of randomized SMILES but not the exact sequence. This is accepted
  rather than fixed: the arms affected are the H8 canonical-vs-enumerated sweep, where the three
  pretraining seeds still give genuine independent draws, so the reported spread remains a valid
  measure of run-to-run variability — only exact replay is unavailable. Everything else (weight init,
  subset membership, objective sampling, corruption) *is* seeded and reproducible.
- **HPO policy.** A single up-front MLM hyperparameter search fixes the recipe; the same settings are
  reused for every subsequent run (no per-experiment tuning) so optimization never confounds a
  comparison. The exception is the deliberate SFT-LR ablation (E3).
- **Durability.** Periodic encoder checkpoints (`save_every_steps`) plus a 10-minute S3-sync sidecar
  in the worker script survive spot reclaim (current boxes are on-demand, so this is belt-and-braces).

---

### 7.1 Corrupted-pretraining control (E13 → Fig B2, hypothesis H2c)

The label-efficiency curves (B1) can show *that* pretraining helps but cannot say *why*: a
gain could come from the chemistry the corpus contains (**information**, H2c) or merely from
running a self-supervised objective at all — the optimization/regularization effect of the
task's structure (**initialization/regularization**, H2a/b). The corrupted control separates
them by holding **everything** fixed except chemical content:

| Held fixed | Destroyed |
|---|---|
| objective + loss shape, compute (8M FP), schedule, model, batch size, mask rate, sequence lengths, token / target **distributions** | SMILES grammar & atom ordering (MLM arm); the molecule→descriptor **mapping** (MTR arm) |

Two arms, each matched to its real counterpart at the 8M budget (`scripts/make_e13_manifest.py`):

- **`corrupt_mlm_8M`** — `corruption: shuffle_tokens`. Permutes the *interior* token positions
  of every sequence (CLS/SEP stay pinned), applying the **same permutation to `input_ids` and
  `labels`** so each masked slot still asks for its own original token — only the surrounding
  context is scrambled. Controls for `unsup_only` (real MLM).
- **`corrupt_mtr_8M`** — `corruption: shuffle_targets`. Permutes the descriptor target *rows*
  across the batch, so each molecule is regressed onto **another** molecule's descriptors.
  Controls for `sup_only:dense` (real dense MTR).

Implemented as `data_v2.CorruptedCollator` (wraps any pretraining collator; deterministic per
seed with a per-dataloader-worker offset), selected via `selection.corruption` in the run config
and recorded in the run metadata (`corruption` field) so every run is auditable. Invariants are
regression-tested in `tests/test_data_v2.py` (CLS/SEP pinned, interior is a true permutation,
input↔label pairing survives, targets permuted and never the identity).

**How to read Fig B2** (three bars per task — `no_pretrain` · corrupted · real):

- corrupted ≈ `no_pretrain`, and real **>** both → the gain is **chemical information** (H2c). ✔
- corrupted ≈ real, and both **>** `no_pretrain` → the gain is the **objective's structure**
  (initialization/regularization, H2a/b) — pretraining is not teaching chemistry.

This is the experiment that decides whether the persistent part of B1's `unsup_only` advantage
(clearest on ESOL/BACE) is real chemistry or an optimization artifact.

## 8. Evaluation protocol (frozen featurizer)

The **primary** protocol mirrors real deployment of molecular foundation models: freeze the encoder,
extract one embedding per molecule, train a small head on those embeddings.

- **Featurizer:** frozen encoder → **masked-mean pooling** over token states (not CLS — v1's
  CLS-linear-probe was pathological). Encoder features are computed once per run and sliced per fold;
  there is no persistent on-disk cache (`EvalConfigV2.cache_encoder_forward` is declared but unused).
- **Standardization:** per-feature **z-score fit on the train split only**, applied to val/test (no
  standardization leakage). **Exception:** the classical anchors (`ecfp4`, `rdkit_desc`, `fp_desc`)
  are forced to `std_method="none"` — z-scoring sparse binary fingerprint bits is meaningless and
  hurts the tree models.
- **Head:** small **MLP** by default (also `linear`, `xgb`), trained with **3 head seeds**.
  Full hyperparameters (`heads_v2.HEAD_HPARAMS`) — **MLP:** hidden 256, dropout 0.2, Adam lr 1e-3,
  weight-decay 1e-4, batch 64, ≤100 epochs, early-stopping patience 15 on val.
  **XGBoost:** 600 estimators, max_depth 6, lr 0.08, subsample 0.8, colsample_bytree 0.8,
  min_child_weight 2, early_stopping_rounds 40, one model per output column (sklearn
  HistGradientBoosting fallback if xgboost is unavailable). **ECFP4** = Morgan radius 2, **2048 bits**.
- ⚠️ **How the 3 head seeds enter the number differs by scheme, and the two are not the same
  quantity.** In the **single-split** path a metric is computed *per seed* and the reported
  `MEAN`/`STD` are over the 3 seeds. In the **CV** path the 3 seeds' predictions are **averaged into
  one prediction vector per fold** and a single metric is computed from it, so `MEAN`/`STD` are over
  the **5 folds** and head-seed variance never enters the error bar.
- **Train metrics.** `eval_v2` additionally emits a `<metric>_train` row (the same metric evaluated
  on the head's own training molecules) **in the single-split path only**; the CV path emits no train
  rows. Fig B1p1's fit-vs-generalize panel reads these, which is why it uses the single-split scheme.
- **Splits — two schemes, both reported.** Every figure has a **default** readout and an optional
  **tougher SI variant**, from the same encoder features:
  1. **Scaffold 5-fold cross-validation — DEFAULT (`--cv_folds 5`).** Partition molecules into **5
     scaffold-disjoint folds** (each Bemis–Murcko scaffold in exactly one fold; ring-less molecules
     are singletons so they distribute and keep folds balanced), greedily balanced by size. Each fold
     is tested against a head trained on the other four (with a 10% validation carve-out for early
     stopping); report **mean ± std across the 5 folds**. This is the primary error bar because the
     dominant uncertainty on these ~1–8k-molecule tasks is *which molecules land in test*, and CV
     measures exactly that. Nearly free — encoder features are computed once and sliced per fold (no
     re-pretraining) — and it yields a **complete out-of-fold per-molecule prediction set** (feeds
     H9/mechanism figures, C16).
  2. **DeepChem single scaffold hold-out — TOUGHER SI VARIANT (default splitter, no `--cv_folds`).**
     One 80/10/10 split where DeepChem sorts scaffolds by frequency and sends the **rarest** scaffolds
     to test — a deliberately adversarial "generalize to novel chemistry" stress test. Systematically
     **lower and noisier** than CV (one draw; its error bar is head-seed spread, not split variance).
  **Fold pairing.** The CV partition and the 10% validation carve-out are both drawn from
  `--subsample_seed` (default **0**). Every run that must be compared molecule-for-molecule — including
  the end-to-end arm — has to use the same value, or the folds do not align and the paired tests in
  §8.1 silently compare different molecule sets.
  **Convention (revised 2026-07-22).** The **DeepChem single scaffold hold-out is the headline
  scheme**, and CV-5 is reported alongside it. The convention was originally the other way round and
  was reversed on evidence: the balanced CV split saturates — an *untrained* random encoder reaches
  ≈0.94 ROC-AUC on BBBP under CV versus ≈0.70 in the literature — and compresses the between-regime
  gaps by 2-3× (BACE: +0.056 CV vs +0.146 hold-out), i.e. it suppresses exactly the effect the paper
  measures. CV is retained for the two jobs it does better: it is the only scheme whose error bar
  reflects **split variance**, and the only one that yields a **complete out-of-fold per-molecule
  prediction set** (required by Fig I1). The code default follows this — `eval_v2.py`'s `--cv_folds`
  defaults to `None` (hold-out) and the wave launcher never passes it; CV is produced by explicit
  after-the-fact passes (`scripts/cv_eval_local.py`, `scripts/h1_cv_eval.sh`).
  The two are stored side by side (`moleculenet_cv/` vs `moleculenet/`) and are **never mixed within
  one panel** — every bar in a figure uses the same scheme. Absolute numbers differ markedly between schemes (e.g. BBBP ≈0.95
  CV vs ≈0.74 single-split), which is expected and disclosed; model *rankings* are what transfer.
- **Metrics:** absolute **per task** — RMSE for ESOL/QM7/Lipophilicity, ROC-AUC for BBBP/BACE/Tox21,
  and **NEF1% (top-1% normalized enrichment) as the headline for HIV** (the virtual-screening task; ROC-AUC
  kept secondary) — see §6.5 for the NEF1% definition. Never z-scored or averaged across tasks.
- **Anchors (classical baselines, `--head xgb`, all through the same eval pipeline):** an
  untrained-encoder **random floor** (3 seeds); **`ecfp4`** = Morgan ECFP4 + XGBoost; **`rdkit_desc`**
  = 217 RDKit descriptors + XGBoost (the classical control for the dense-MTR arm — *implemented in
  `eval_v2.py` but never actually run: no wave or script invokes it, so no `rdkit_desc` results
  exist*); and **`fp_desc`** =
  **Morgan fingerprints ++ RDKit descriptors concatenated → XGBoost** — the *toughest* classical
  baseline (both substructure bits and computed physchem), which a CLM must beat to justify itself
  (e.g. it already gets ESOL ≈0.35 CV-RMSE, ahead of every neural regime at 8M). "Lift" is improvement
  over the random floor.
- **Eval-ceiling (H5, Fig E1):** the same encoders are additionally **fine-tuned end-to-end** on
  BACE/BBBP/ESOL + HIV, to test whether the frozen probe under-resolves encoder quality. Driven by
  `scripts/run_eval_ceiling.py` (+ `scripts/run_e1_gpu.sh` for the unsup ladder and
  `scripts/run_e1_sup_gpu.sh` for the sup_only ladder), which fine-tunes via **`finetune_e2e_v2`**
  (§8.2). ⚠️ The legacy `finetune_v2.py` is *not* the path used for any current figure: it raises on
  multi-output tasks, emits no per-molecule predictions, no CV and no NEF1%.
- **Per-molecule prediction dump (C16, blocking).** `eval_v2` writes **`(canonical_key, y_true,
  y_pred, task, split)` per eval molecule**, not only aggregate RMSE/AUC. This is required to build
  both Fig H9 panels (C6) and to bin the label-efficiency curves (C7); without it H9 is unbuildable
  without re-running eval. *Caveat:* per-cell / per-bin **ROC-AUC is unstable** on small subsets —
  prefer per-molecule residual/rank, or run the mechanism panels on the RMSE tasks (ESOL, QM7).
- ⚠️ **Not implemented: persisted fingerprints (C19).** The plan was to save the ECFP4 fingerprints
  computed for the anchor. `eval_v2` writes only `test_predictions.csv`, `moleculenet_summary.csv`
  and `suite_summary.json`; no fingerprints are persisted. The Tanimoto analyses (Figs I1, C1J1)
  therefore **re-derive** fingerprints from the dumped SMILES in
  `scripts/compute_tanimoto_novelty.py` and `scripts/compute_family_task_similarity.py`.
- **Mechanism figures through finetune (C20).** For H9, label-efficiency, and the transfer matrix,
  report at least a subset **through end-to-end finetune** (`finetune_v2`) as well as the frozen
  probe — or caveat explicitly with the frozen-probe ceiling (H5 / §10). A ceiling-compressed probe
  flattens the very differences these figures rely on (Type-II risk), so the mechanism story must not
  rest solely on the probe H5 says may be under-resolving.

### 8.1 Model-vs-model comparison protocol (headline tables)

Every pairwise claim ("model A beats baseline B") is backed by the same protocol. The comparison
table (see `scripts/compare_models.py`, reproduced in the notebook) reports:

⚠️ **Corrected against the code.** This section previously claimed "3 seeds × 5 scaffold-CV folds =
15 metric evaluations" with the error bar taken "across the 15 points". That is not what `eval_v2`
computes: in the CV path the 3 head seeds' predictions are averaged into one prediction per fold
before the metric is taken, so there are **5 metric values per model**, and the reported std is
**fold (split) variance only** — head-seed variance is averaged away and never appears. Pretraining-seed
variance is a *separate* axis, obtained by re-running pretraining under seeds 0/1/2 and aggregating
across runs (see §9.6); where a figure's error bar uses that axis instead, its caption says so.

| Element | Specification |
|---|---|
| Points per model | **5** — one metric per scaffold-CV fold, each computed from the 3-head-seed-averaged prediction |
| Error bar | mean ± std across the **5 folds** = scaffold-split variance. Head-seed variance is inside the averaged prediction; pretraining-seed variance is a separate axis (§9.6) |
| Pairing | all models share one scaffold fold partition per seed → fold- and molecule-paired |
| Effect | Δ(metric) and relative % |
| **RMSE tasks** (ESOL/QM7/Lipo) | rigorous test = molecule-level **paired Wilcoxon** on per-molecule squared error over the pooled out-of-fold predictions (large n) |
| **AUC tasks** (BBBP/BACE/Tox21) | rigorous test = **DeLong paired-AUC** on the pooled OOF scores (per label column for multi-task Tox21, then summarised) |
| **HIV (virtual screening)** | headline metric = **NEF1%** (top-1% enrichment), reported as mean ± std across folds; rigorous paired test = **DeLong paired-AUC** on the pooled OOF scores (a rank test that tracks early enrichment). Select NEF1% in `compare_models.compare` by passing the task as `('HIV', True, 'nef1')` |
| Fold-level test | paired t across folds — reported but **flagged anti-conservative** (CV folds share training data; Bengio & Grandvalet 2004), so the molecule-level Wilcoxon/DeLong is the test of record |

> **The error bar and the test answer different questions — overlapping bars do NOT mean "tied".**
> This trips up every reader who checks a figure against a table, so it is stated here rather than
> left to be inferred. The **error bar** is the spread of the whole-dataset average across the 5
> scaffold splits: it asks *how stable is this model's score if the folds had been drawn
> differently?* That variation is driven by which scaffolds landed in which fold, and it hits every
> model **identically**. The **test** is paired per molecule: it takes the difference in error
> between two models on the *same* molecule and asks whether those differences sit systematically
> on one side of zero — so the shared split difficulty cancels exactly, and a molecule that is hard
> for everyone contributes nothing.
>
> Two runners on five courses: their finishing-time distributions across courses can overlap
> heavily while one still beats the other on every single course. Concretely, on QM7
> `fp_desc` (0.819) vs `sup_only:dense` (0.851) have visibly overlapping fold error bars and a
> paired p of 7.7e-17; on BACE, ECFP4 (0.882) vs `unsup_only` (0.866) likewise overlap with
> p = 0.012.
>
> The corollary for anyone tempted to "make the table match the figure": don't. A fold-level test
> (n = 5) is both underpowered and anti-conservative, as the row above says. Read the error bar for
> split stability and the table for whether one model actually beats another.

The canonical instance is **"does a CLM beat the toughest classical baseline (`fp_desc`)?"** run for
each regime; a **non-descriptor CLM (`unsup_only`) is included as a control** so the descriptor-trained
`dense` CLM's gap to `fp_desc` can be read against a CLM that never saw descriptors (isolates whether
descriptor pretraining actually transfers descriptor information).

> **⚠️ Descriptor-favorable tasks — interpret the `fp_desc` gap with care (ESOL especially).** The
> physchem/quantum *regressions* are structurally biased toward the classical descriptor baselines.
> ESOL's canonical model (Delaney) is a near-linear function of LogP / MW / rotatable-bonds /
> aromatic-fraction — all RDKit descriptors — so `fp_desc` has *near-oracle* features and beating the
> CLM on ESOL (≈0.35 vs ≈0.43 CV-RMSE) is **expected, not evidence the CLM is weak**. Lipophilicity
> (logD ≈ LogP) shares this; QM7 partly (atomization energy tracks composition). We **keep ESOL** as a
> deliberate *descriptor-optimal reference point*, but the CLM-vs-descriptor question is adjudicated
> primarily on the **bioactivity / virtual-screening** tasks (BACE, Tox21, HIV) where structure→property
> is not a simple descriptor. The cleanest *positive* signal is internal to the CLMs: the
> descriptor-trained `dense` CLM closing ESOL's gap over the non-descriptor `unsup_only` control is
> evidence that descriptor pretraining transferred descriptor-relevant information.

Replication / error bars: the primary error bar is **scaffold k-fold CV** (mean ± std across folds,
above) — it captures the split variance that dominates on these small tasks and costs no extra
training. Pretraining-seed replication (3-seed CIs on the bar figures) is a *separate* axis, deferred
(§10); the plan is CV first, then seed replicates only for the key headline arms if the CV bars leave
a contrast ambiguous. Each eval also averages **3 head seeds**; scaling curves use stop-when-flat in
place of plateau error bars.

### 8.2 End-to-end fine-tuning protocol (`finetune_e2e_v2`)

The frozen-probe protocol above is the primary readout. A second, **separate** protocol unfreezes the
encoder; it produces Fig E1 (eval ceiling) and Fig B1p1's `no_pretrain_end_to_end` series. It is a
different protocol, not a variant of §8, and its numbers are **not** comparable bar-for-bar with the
frozen ones.

| Element | Frozen probe (`eval_v2`) | End-to-end (`finetune_e2e_v2`) |
|---|---|---|
| Encoder | frozen; features extracted once | **unfrozen**, reloaded fresh per seed/fold |
| Pooling | masked-mean | masked-mean (same) |
| Head | MLP 256, dropout 0.2 | **single `Linear(hidden, n_outputs)`** (`linear_e2e`) |
| Feature standardization | z-score on train | **none** |
| Target scaling | task-native | DeepChem's own `NormalizationTransformer`; no extra rescaling |
| Optimizer | Adam 1e-3, wd 1e-4 | AdamW **lr 2e-5**, wd 0.01 |
| Schedule | ≤100 epochs, patience 15, batch 64 | **20 epochs, patience 5, batch 32**, max_length 256, bf16 autocast |
| Seeds | 3 head seeds | 3 seeds `[0,1,2]`, each a full re-finetune |
| Multi-output | supported | supported, per-column masked loss (raw logits, never sigmoided) |
| Splits | hold-out + CV | hold-out + CV, sharing `_scaffold_kfold_indices` and RNG draw order with `eval_v2` so folds pair molecule-for-molecule |
| Outputs | `eval_v2` schema | **same schema**, so rows merge for the paired tests of §8.1 |

Because the fine-tuned arm re-randomises the whole encoder optimisation while the frozen arm only
re-randomises a head, **their error bars are not the same quantity** and the fine-tuned band is
expected to be wider; figure captions state this explicitly.

The legacy `finetune_v2.py` predates this module, is single-seed, hold-out-only, single-output, and
emits neither per-molecule predictions nor NEF1%. It is retained for provenance and is not used by
any current figure.

---

## 9. Reproducibility

### 9.1 Code map (what each module does)
| File | Role |
|---|---|
| `config_v2.py` | single source of truth: `ModelConfigV2`, `build_modernbert_config`, `TrainingConfigV2`, `EvalConfigV2`, task list, supervised families/groups/weights/caps |
| `data_v2.py` | streaming MLM/raw-SMILES datasets, `MTRCollator` (+ precomputed descriptors), stratified supervised loader (dedup + scarce-first routing), objective iterator |
| `descriptors_v2.py` | 217 RDKit descriptors, fit/normalize/save stats |
| `pretrain_v2.py` | `ClimbV2Model` (MLM/MTR/supervised heads), training loop, warm-start, checkpointing |
| `eval_v2.py` | frozen-featurizer MoleculeNet evaluation, scaffold hold-out + k-fold CV, standardization; **per-molecule prediction dump (C16)**; `<metric>_train` rows (hold-out path only) |
| `featurize_v2.py` / `heads_v2.py` | pooling + standardizer; downstream heads + metrics. ⚠️ **head hyperparameters live in `heads_v2.HEAD_HPARAMS`, not `config_v2.py`** |
| `smiles_augment.py` | randomized-SMILES enumeration (the H8 lever). ⚠️ unseeded — see §7 |
| `finetune_e2e_v2.py` | **end-to-end fine-tuning used by every current figure** (Fig E1, Fig B1p1 e2e series) — see §8.2 |
| `finetune_v2.py` | legacy end-to-end fine-tuning; single-seed, single-output, superseded by `finetune_e2e_v2.py` |
| `random_baseline_v2.py` | untrained-encoder + ECFP4 anchors |
| `experiment_v2.py` | manifest generator for every wave (ablation / compute_scaling / phase2) |
| `data.py`, `storage_utils.py`, `utils.py`, `token_budget.py` | shared streaming base + S3 helpers (dependencies of the above) |
| `train_tokenizer.py` | tokenizer training (provenance; artifact is prebuilt) |
| `scripts/` | launch/split/deploy, leakage audit, blocklist + descriptor precompute, report/figure builders (see §9.3) |

**Scripts that produced published numbers** (each is the *only* way to regenerate its artifact):

| Script | Produces |
|---|---|
| `scripts/launch_v2_wave.py` + `scripts/phase2_worker.sh` | every pretraining wave |
| `scripts/unattended_guard.sh` | the runner used for unattended waves: saves to S3 on every exit path, then stops the box; `POST_HOOK` runs a second stage (e.g. CV) before the stop |
| `scripts/cv_eval_local.py`, `scripts/cv_all_budgets.sh`, `scripts/h1_cv_eval.sh` | all 5-fold CV numbers and the `fp_desc` anchor |
| `scripts/reeval_7task.py` | the 7-task (incl. HIV NEF1%) re-scoring of existing encoders |
| `scripts/run_eval_ceiling.py`, `scripts/run_e1_gpu.sh`, `scripts/run_e1_sup_gpu.sh` | Fig E1 |
| `scripts/b1_e2e_v2.sh`, `scripts/run_b1_e2e_cell.py`, `scripts/run_label_efficiency.py` | Fig B1p1 |
| `scripts/run_e2e_random.py`, `scripts/run_e2e_wave.sh` | the e2e random-init replicates |
| `scripts/make_e13_manifest.py` | Fig B2 corrupted controls |
| `scripts/build_h1_rescale_manifest.py` | Fig H1 (3-seed retrain) |
| `scripts/compute_tanimoto_novelty.py`, `scripts/compute_family_task_similarity.py` | Figs I1, C1J1 |
| `scripts/compare_models.py`, `scripts/verify_e2e_pairing.py` | §8.1 paired tests and their pairing check |
| `scripts/backfill_verified.py`, `scripts/reproducibility_audit.py`, `scripts/gen_readme_inventory.py` | completion markers, the audit, and §9.6 |

### 9.1b Completion is proven, never assumed

A truncated run still writes a well-formed summary, so file existence cannot be used to decide
whether a run finished — doing so is how half-trained encoders reached figures earlier in this
project. The rule, applied everywhere:

- A run is complete **iff** its achieved forward passes reach **≥98%** of its declared budget
  (tolerance for the final partial batch). Only then is `verified.json` written
  (`scripts/launch_v2_wave.py::_write_verified_marker`), recording `budget_fp`, `final_fp`,
  `fraction` and a UTC timestamp.
- Every "is this done?" decision — skip logic, downstream consumption, shutdown gating, figure
  inclusion — reads that marker or recomputes achieved work; precedence is local marker → S3 marker
  → (for anchors, which have no FP budget) presence of `suite_summary.json` → else the ≥98% check
  (`scripts/launch_v2_wave.py::_is_complete`).
- Runs predating the marker system are retro-marked from their `metrics.jsonl` by
  `scripts/backfill_verified.py`, which refuses to mark anything below 98%.
- The evaluation side has its own variant: a cell is verified only when every requested task has a
  finite `MEAN` row for **both** `<metric>` and `<metric>_train`
  (`scripts/run_b1_e2e_cell.py`, `scripts/run_e2e_random.py`).
- `scripts/reproducibility_audit.py` reports, per run, which of {checkpoint, training curve,
  completion proof, each evaluation artifact} exist in S3 and locally; §9.6 is generated from it.

### 9.1c Environment pinning (⚠️ load-bearing)

The **RDKit version is part of the experiment definition**, not an implementation detail.
`descriptors_v2.py` derives the descriptor list from the installed RDKit
(`len(Descriptors.descList)`), so the version fixes:

- the MTR target dimension (**217** with the RDKit used for these runs, 2025.09.2),
- the contents of `configs/descriptor_stats.json` and every precomputed descriptor shard,
- the feature width of the `rdkit_desc` / `fp_desc` classical anchors.

⚠️ `requirements.txt` currently pins `rdkit-pypi==2022.9.5`, whose `descList` length **differs** from
217. A reproducer installing from `requirements.txt` would silently build a different-width MTR head
and different anchors. This pin must be corrected to the version actually used before release.

### 9.2 Configs
`configs/v2_phase2.yaml` (the 5-arm scaling matrix), `configs/v2_ablation.yaml` (dense-vs-sparse),
`configs/v2_compute.yaml`, `configs/v2_headline.yaml`. The whole `configs/` dir is gitignored by
default; the experiment configs are force-added so definitions stay in git.

### 9.3 How to run a wave
```bash
# 1. resolve a spec into a manifest
python experiment_v2.py --spec configs/v2_phase2.yaml --output experiments/climb_v2_phase2/manifest.json
# 2. split across N workers (stage 1 = ladder+skip, stage 2 = warm-start u2s)
python scripts/split_manifest.py --manifest .../manifest.json --workers 4 --stages ladder skip --with_anchors --out_dir .../manifests/stage1
# 3. deploy code to a box and run a worker manifest (self-stops + S3-syncs when done)
scripts/deploy_to_ec2.sh <ip>
ssh <box> 'nohup bash scripts/phase2_worker.sh <worker.json> w0 &'
# one-time prerequisites
python scripts/make_eval_blocklist.py --out configs/eval_blocklist.json --s3_out s3://climb-s3-bucket/configs/eval_blocklist.json
python scripts/precompute_descriptors.py --shard_range 0-11 --out_s3 s3://climb-s3-bucket/tokenized_sources/pubchem_descriptors/
```
Env on GPU boxes: `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, `TORCHDYNAMO_DISABLE=1`.

### 9.4 Artifacts (S3)
| Path | Contents |
|---|---|
| `s3://climb-s3-bucket/tokenizer_10M/` | byte-BPE tokenizer (vocab 1000) |
| `.../tokenized_sources/pubchem_filtered/` | ~12M unsup SMILES (12 shards) |
| `.../tokenized_sources/pubchem_descriptors/` | precomputed 217-descriptor companions |
| `.../tokenized/supervised_wide_parquet/` | 5.38M-row supervised wide table |
| `.../configs/eval_blocklist.json` | 34,301 leaked-molecule dedup blocklist |
| `.../configs/descriptor_stats.json` | descriptor normalization stats |
| `.../experiments/climb_v2*/` | run outputs (encoders, metrics, MoleculeNet CSVs, **per-molecule predictions C16**) |
| `.../configs/eval_fingerprints/` | persisted ECFP4 fingerprints for eval molecules + a pretraining sample (C19) |

### 9.5 Reporting fields for the paper appendix
exact S3 prefixes; git commit hash; config path(s); tokenizer id; descriptor-stats + blocklist
hashes; per-run forward-pass budget; head/pretraining seeds; **the enumeration seed (C18)**; the
frozen-vs-finetune eval flag; **both leakage keys per eval molecule — the fuzzy dedup key and the
exact-identity "seen" key (C17)**.

---

<!-- BEGIN MODEL INVENTORY (generated by scripts/reproducibility_audit.py) -->

### 9.6 Model inventory (what exists, and what we hold for it)

**137 runs** across the four waves the paper draws on. Regenerate with
`python scripts/reproducibility_audit.py --listing <s3 listing> --out audit/`; the per-run
breakdown is `paper_artifacts/INVENTORY.md`.

| Pretraining type | Budgets | Seeds | Runs | ckpt | curve | proof | single-split | 5-fold CV | CV preds |
|---|---|---|--:|--:|--:|--:|--:|--:|--:|
| `corrupted control (mlm: content destroyed)` | 8M | 0 | 1 | 1 | 1 | 1 | 1 | 0 | 0 |
| `corrupted control (mtr: content destroyed)` | 8M | 0 | 1 | 1 | 1 | 1 | 1 | 0 | 0 |
| `no_pretrain (random init, frozen)` | — | 0,1,2 | 6 | 6 | n/a | n/a | 6 | 3 | 3 |
| `sup_only: dense` | 2M, 8M, 24M, 48M, 96M | 0,1,2 | 7 | 7 | 7 | 7 | 7 | 4 | 4 |
| `sup_only: dense_plus_sparse` | 2M, 8M, 24M, 48M | 0,1,2 | 6 | 6 | 6 | 6 | 6 | 4 | 4 |
| `sup_only: minimol_full` | 2M, 8M, 24M, 48M | 0,1,2 | 6 | 6 | 6 | 6 | 6 | 4 | 4 |
| `sup_only: mixed` | 2M, 8M, 24M, 48M | 0,1,2 | 6 | 6 | 6 | 6 | 6 | 4 | 4 |
| `sup_only: sparse_all` | 2M, 8M, 24M, 48M | 0,1,2 | 6 | 6 | 6 | 6 | 6 | 4 | 4 |
| `unsup->sup (ablation): dense_plus_sparse` | 2M+2M | 0 | 1 | 1 | 1 | 1 | 1 | 0 | 0 |
| `unsup->sup (ablation): l1000` | 2M+2M | 0 | 1 | 1 | 1 | 1 | 1 | 0 | 0 |
| `unsup->sup (ablation): mtr` | 2M+2M | 0 | 1 | 1 | 1 | 1 | 1 | 0 | 0 |
| `unsup->sup (ablation): pcba` | 2M+2M | 0 | 1 | 1 | 1 | 1 | 1 | 0 | 0 |
| `unsup->sup (ablation): pcqm` | 2M+2M | 0 | 1 | 1 | 1 | 1 | 1 | 0 | 0 |
| `unsup->sup (ablation): sparse_all` | 2M+2M | 0 | 1 | 1 | 1 | 1 | 1 | 0 | 0 |
| `unsup->sup: dense` | 2M+2M, 8M+2M, 24M+2M, 48M+2M | 0,1,2 | 6 | 6 | 6 | 6 | 6 | 3 | 3 |
| `unsup->sup: dense_plus_sparse` | 2M+2M, 8M+2M, 24M+2M, 48M+2M | 0,1,2 | 6 | 6 | 6 | 6 | 6 | 3 | 3 |
| `unsup->sup: minimol_full` | 2M+2M, 8M+2M, 24M+2M, 48M+2M | 0,1,2 | 6 | 6 | 6 | 6 | 6 | 3 | 3 |
| `unsup->sup: mixed` | 2M+2M, 8M+2M, 24M+2M, 48M+2M | 0,1,2 | 6 | 6 | 6 | 6 | 6 | 3 | 3 |
| `unsup->sup: sparse_all` | 2M+2M, 8M+2M, 24M+2M, 48M+2M | 0,1,2 | 6 | 6 | 6 | 6 | 6 | 3 | 3 |
| `unsup_only (MLM)` | 2M, 8M, 24M, 48M | 0,1,2 | 6 | 6 | 6 | 6 | 6 | 4 | 4 |
| `unsup_only, canonical SMILES` | 2M @ frac0p3 | 2 | 1 | 0 | 0 | 0 | 0 | 0 | 0 |
| `classical: Morgan+XGBoost` | — | 0 | 2 | n/a | n/a | n/a | 2 | 1 | 1 |
| `classical: Morgan+desc+XGBoost` | — | 0 | 1 | n/a | n/a | n/a | 1 | 1 | 1 |
| `label-efficiency probe: random` | n=0, n=100, n=1000, n=300, n=3000 | 0,1,2 | 13 | n/a | n/a | n/a | 13 | 0 | 0 |
| `label-efficiency probe: sup` | n=0, n=100, n=1000, n=300, n=3000 | 0,1,2 | 13 | n/a | n/a | n/a | 13 | 0 | 0 |
| `label-efficiency probe: unsup` | n=0, n=100, n=1000, n=300, n=3000 | 0,1,2 | 13 | n/a | n/a | n/a | 13 | 0 | 0 |
| `label-efficiency probe: unsup2sup` | n=0, n=100, n=1000, n=300, n=3000 | 0,1,2 | 13 | n/a | n/a | n/a | 13 | 0 | 0 |

**81 encoder checkpoints, 13.4 GB**, indexed in `paper_artifacts/checkpoints.csv`
with `fetch_checkpoint.sh <run>` to pull one. They are not mirrored to laptops.

**Columns.** `ckpt` = encoder weights in S3 · `curve` = `metrics.jsonl` training curve ·
`proof` = `verified.json`, written only once achieved forward passes reach ≥98% of budget ·
`single-split` = DeepChem scaffold hold-out evaluation · `5-fold CV` = scaffold CV ·
`CV preds` = per-molecule predictions (needed by Fig I1). `n/a` marks a column that cannot
apply: classical anchors have no encoder, label-efficiency probes are *evaluations* of an
existing 8M encoder at different label budgets rather than models of their own, and a
random-init baseline has a checkpoint but no training curve and no forward-pass budget.

**Evaluation.** Every model is scored on the same 7 tasks (ESOL, Lipophilicity, QM7, BBBP,
BACE, Tox21, HIV) with 3 head seeds, under two schemes: the DeepChem scaffold hold-out
(headline; rarest scaffolds in test) and 5-fold scaffold CV (split-variance error bars, and
the only source of per-molecule predictions). Both emit train and test metrics; HIV also
carries NEF1% (top-1% enrichment).

**Storage.** Working bucket `s3://climb-s3-bucket`; independent versioned backup at
`s3://climb-paper-backup-075120018132` (no expiry lifecycle, copy-only). The old
`experiments/robust_matrix` wave (3.7 TB, 98.9% of it per-epoch fine-tuning checkpoints, read
by no figure) is lifecycled to Glacier Deep Archive: ~$86/month → ~$4/month, data retained.

<!-- END MODEL INVENTORY -->

---

## 10. Limitations and known issues
- **Leakage (H6):** measured and material for assay-label arms; dedup applied and those arms re-run.
  Pretrain overlap (0–7%) is disclosed, not removed (standard for the field, and conservative).
- **SFT-LR confound (H5):** warm-start uses the pretraining LR; the E3 sweep tests whether the
  "SFT ≤ MLM base" ablation finding is an LR artifact.
- **Error bars / seeds.** Bar figures now carry **scaffold k-fold CV** error bars (fold spread, §8),
  which capture the split variance that dominates on these small tasks. **Pretraining-seed** replication
  (3 seeds) is a separate, still-deferred axis — so any gap smaller than the CV band, or plausibly
  within pretraining-seed noise, is not yet claimed (e.g. the dense-vs-sparse ablation gap). Plan: CV
  first, targeted seed replicates only where CV leaves a headline contrast ambiguous.
- **Frozen-probe ceiling:** the probe under-resolves encoder quality (MLM loss 0.14 vs 0.39 → same
  downstream), so "sup_only ≈ unsup_only" risks a Type-II error; Fig D is the test.
- **Scale:** ladders reach ≤8 epochs of a ~12M corpus, well below MoLFormer's token budget; absolute
  and "global plateau" claims are scoped to this model/compute. **Model-size scaling is out of scope
  for this paper** (E10 dropped); the only scaling view is a descriptive recycling plot (§4) from
  runs collected anyway, which makes no scaling-law claim.
- **SFT data quality:** PCQM-dominated; L1000 small and near-unlearnable; Kendall weighting can drive
  L1000 toward zero, so "sparse" is effectively PCBA(+WONG).
- **Eval breadth:** 5–6 small MoleculeNet tasks with known label noise (esp. BBBP) and high-variance
  scaffold-test sets; QM7 is quantum and a weak probe for structural pretraining.
- **Pooling / length:** only masked-mean tested; the 128-token cap drops large molecules from both
  pretraining and eval.
- **Fixed unlabeled corpus (C21).** Only the *labeled* side of the "unlabeled × labeled combinations"
  question is varied; the unlabeled corpus is a single fixed PubChem set. Any claim about *which
  unlabeled data* matters is out of scope unless a second, domain-shifted unsupervised corpus is added
  (relates to the corrupted-pretraining control, E13 / C10).
- **All models are deduped (C22).** There is no un-deduplicated model. H9 (memorization) is answered
  on the production deduped models using the disclosed 0–7% *pretraining* overlap plus a Tanimoto
  dose–response (§2 H9), so no leaked artifact is ever trained. Exact memorization from *supervised*
  labels is deliberately out of scope — testing it would mean reintroducing the leakage the dedup
  removed, which would contaminate the model set for no proportional insight.

---

## 11. Hypothesis-resolution matrix

| Sub-question | Data source | Positive evidence | Negative evidence |
|---|---|---|---|
| Does any unsupervised pretraining help? | Fig A/B vs random | curve above floor CI at some budget | overlaps floor throughout |
| Does more unsup monotonically help? | Fig B shape | log-linear rise then flatten | flat / non-monotone |
| Can you skip unsup (SFT-only match)? | sup_only vs unsup→sup lines | sup_only reaches unsup→sup endpoint | sup_only plateaus below |
| Mechanism (a) initialization? | H2a: label-efficiency + sup_only trained long | left-shift that **closes at 100% labels / high compute** | gap persists at full labels |
| Mechanism (b) regularization? | H2b: label-efficiency + train–test gap | gain **only at small label fractions** + smaller overfit gap | no low-label-specific gain |
| Mechanism (c) adds information? | H2c: label-efficiency + corrupted control (E13) | **persistent gap at 100% labels**, absent for content-free pretraining | content-free pretraining helps equally |
| Does label type decide it? | dense vs sparse sup_only lines | dense compensates, sparse doesn't (or vice versa) | all recipes equal |
| Domain-matched transfer? (H10) | transfer matrix (E1) + Tanimoto/domain overlay | lift tracks content similarity, not label type | transfer unrelated to content similarity |
| Robust to training order? | forgetting (H7) | reversed ≈ standard within CI | large degradation |
| Representation vs memorization? (H9) | deduped model: pretrain-overlap (0–7%) seen axis + Tanimoto dose–response | novel ≈ seen; benefit flat across Tanimoto distance | benefit decays sharply with distance from training data |
| Is the frozen-probe result real? | Fig D (H5) | finetune agrees with frozen | finetune flips it |
| Is eval leaked? | leakage audit (H6) | small/uniform overlap | large systematic overlap (→ dedup) |

---

## 12. Appendix — history

- **v1 (RoBERTa-13M, token-budgeted):** archived under `archive/v1/` and `archive/v1_root/`. The
  founding design note (one-encoder-many-heads, fixed-HPO-and-tokenizer, isolate pretraining,
  MoleculeNet aggregate score, 3-D unsup×sup surface, Chinchilla Stage 2) is preserved in git
  history and its enduring ideas are folded into §1, §3, and E9–E10 above.
- **Operational rules (carried over):** ask before spinning up EC2 (16 vCPU account cap); keep local
  / git / EC2 code consistent; prefer standard packages (transformers, torch, rdkit, scipy); macOS
  bash 3.2 has no associative arrays — use `case`, never `declare -A`.
- **v2 milestones:** exploratory wave → dense-vs-sparse ablation → leakage audit + dedup →
  descriptor precompute → phase-2 scaling matrix → **12h-cap truncation incident + harness hardening
  (verified-completion, preflight, SNS alerts)** → recovery + expanded mechanism experiments (v2.1).
- **v2.1 changes (2026-07-20, from the C1–C22 hand-off + follow-up discussion):** H2 split into
  init/regularization/added-information; added H10 (domain-matched transfer, from existing E1 data).
  Reframed H9 to run **on the deduped models** via the disclosed 0–7% pretraining overlap + a Tanimoto
  dose–response — the earlier proposal to train an un-deduped model (E11) was **dropped** to avoid
  reintroducing leakage (Tanimoto already supplies the memorization gradient leak-free). New
  experiments E12 (label-efficiency) and E13 (corrupted control, optional); new figures
  (label-efficiency, transfer matrix, corrupted control). **Scaling laws dropped:** the proposed H11 /
  E10 model-size & Chinchilla data×compute analysis is **out of scope**; only a descriptive
  compute/data recycling plot (existing runs, no new compute, no fitted law) remains. Bar figures are
  **single pretraining seed for now** (3-seed CIs deferred). Data-collection changes: per-molecule
  predictions, persisted fingerprints, MLM val/test loss, enumeration seed.

---

## 13. Operational state & session recovery (LIVE — updated 2026-07-20)

> **Volatile section** (unlike the methods above): live infrastructure + run state so a fresh session
> can resume with zero context loss. Update as runs land.

### 13.1 Access — buckets, machines, credentials
- **S3 bucket `s3://climb-s3-bucket/`** — everything lives here:
  - `experiments/climb_v2_phase2/<run>/` → `encoder/` (weights), `moleculenet/` (single-split eval),
    `moleculenet_cv/` (**5-fold CV** eval), `metrics.jsonl`, `config.yaml`, `run_status.json`,
    **`verified.json`** (written ONLY when a run reached ≥98% of its FP budget — the source of truth
    for "done", not file existence).
  - `tokenizer_10M/`, `tokenized_sources/pubchem_filtered/` (~12M unsup shards),
    `tokenized_sources/pubchem_descriptors/` (precomputed 217-descriptor companions),
    `tokenized/supervised_wide_parquet/` (SFT table), `configs/eval_blocklist.json` (34,301-string
    dedup), `configs/descriptor_stats.json`.
  - Prior waves: `experiments/{climb_v2_ablation,climb_v2_headline,climb_v2_labeleff,climb_v2_lrsweep,climb_v2}/`.
- **EC2:** 5× g5.2xlarge (A10G, **us-east-1d**, on-demand). Instance-ID → worker:
  `i-02dfaa83dae4ad937`=w0 · `i-03b11b7ddd885c65d`=w1 · `i-0b59865cc08ef390c`=w2 ·
  `i-0dbc751470e2108d5`=w3 · `i-07486d883063b0925`=w4. IPs change on restart (query them). SSH
  `ssh -i climb-gpu-key.pem ec2-user@<ip>`; repo `~/CLIMB`; venv `/home/ec2-user/venvs/climb/bin/python`.
  **Capacity caveat:** us-east-1d intermittently returns InsufficientInstanceCapacity on start —
  retry until it clears (minutes–hours).
- **SNS (direct box→email, no Claude in loop):** topic
  `arn:aws:sns:us-east-1:075120018132:climb-experiments`, email **sieben.leif@gmail.com**
  (must be confirmed). Boxes email START/COMPLETE/TRUNCATED/STALL/heartbeat via `scripts/notify.sh`.
- **Git:** branch **`v2-redux`** on `github.com/leifsieben/CLIMB`. Boxes deploy via
  `git fetch && git reset --hard origin/v2-redux`.
- **Local ML env (this Mac):** `.venv_sanity/bin/python` = torch/transformers/deepchem/rdkit/xgboost
  (CPU) — used for local CV eval + figure rendering.

### 13.2 Data local on THIS machine (no re-download needed)
- `experiments/climb_v2_phase2/results_20260720_104231/` — result files (metrics/summaries/configs,
  no weights) for all 28 phase-2 runs.
- `figure_data/climb_v2_phase2/<run>/moleculenet/` (single-split) + `.../moleculenet_cv/` (**5-fold
  CV**, the 10 Fig A1 models). `figure_data/_tokenizer/`, and downloaded `.../<run>/encoder/`.
- `figures_out/figA1_H1_headline_bars_{cv,SI}.{png,pdf}` — rendered.
- `climb_figures.ipynb` (repo root) — figure notebook (global STYLE; `DF` single-split + `DF_CV`;
  `plot_A1()` renders both schemes). Render headlessly: exec code cells 2,3,5,7 with matplotlib Agg.
- `experiments/climb_v2_phase2/download_valid_data.sh` — pulls valid results from S3.

### 13.3 Run inventory — the full program
**A. COMPLETE & VALID (in S3, reached budget):** phase-2 single-split ladder — `unsup_{2M,8M,24M}`,
`skip_sparse_all_{2M,8M,24M,48M}`, `skip_mixed_{2M,8M,48M}`, `skip_minimol_full_{2M,8M,24M}`,
`skip_dense_plus_sparse_{2M,8M,24M}`, `skip_dense_{2M,8M}` + anchors `ecfp4_anchor`,
`random_baseline_{00,01,02}` (18 pretrains + 4 anchors; the four `*_48M` are 7-task, rest 5-task).
**5-fold CV done locally** for the 10 Fig A1 models. Prior waves in S3: `climb_v2_ablation` seq_* arms
(**pre-dedup**, transfer-matrix source), `climb_v2_labeleff`, `climb_v2_headline`, `climb_v2` (E0).
`climb_v2_lrsweep` = 8 runs TRAINED but NOT evaluated.

**B. TRUNCATED (being re-run in wave 1):** unsup_48M, skip_dense_{24M,48M,96M},
skip_minimol_full_48M, skip_dense_plus_sparse_48M, skip_mixed_24M (root cause: old 12h cap +
unwired precompute — both fixed; see §12 + harness in scripts/).

**C. WAVE 1 — RUNNING** (manifest `experiments/climb_v2_phase2/manifest_wave1.json`, split into
`manifests/wave1/worker{0..4}.json`; workers 1 & 4 relaunched from `~/CLIMB/w{1,4}_ordered.json`
= short-first reorder so u2s land first). Contents: the 7 recovery redos (minus u2s_from48M) + the
15 u2s runs warm-started from clean 2M/8M/24M encoders. **Lesson baked in:** the worker manifest must
live OUTSIDE the S3-synced `experiments/` tree, else the startup `aws s3 sync` clobbers a reorder.

**D. WAVE 2 — QUEUED:** u2s_*_from48M (5, after unsup_48M verifies) · 7-task re-eval of clean
encoders · descriptor-XGB + dummy anchors · eval the 8 lrsweep runs (E3) · E1 deduped re-run · E7
forgetting · E8 enumerated · E12 label-efficiency · E13 corrupted control. **Code prereqs still TODO:**
C5 (MLM val/test loss), C18 (enum seed), E13 (corrupted objective), C19 (fingerprints).

**E. FINAL robustness pass (after waves 1 & 2) — the strong error bars:**
**Retrain every headline model at 3 pretraining seeds, then 5-fold CV each → 15 (metric) points per
bar** (3 seeds × 5 folds), so the error bar folds together the two dominant noise sources —
pretraining-seed AND scaffold-split variance — into one honest interval. CV is ~30 min/model, so the
extra ~1.5h/model is cheap for a much stronger claim. **This supersedes the "single pretraining seed"
caveat throughout** once done.

### 13.4 Current live status (2026-07-21)
**RUNNING** (each box now owns exactly one run; see the ownership rule in §13.7):
`i-02dfaa83dae4ad937` (w0) = `skip_dense_96M` ~69% · `i-0b59865cc08ef390c` (w2) = `unsup_48M` ~36% ·
`i-03b11b7ddd885c65d` **repurposed as seed-1 worker** = `skip_dense_8M_s1` ~34% ·
`i-07486d883063b0925` **repurposed as seed-2 worker** = `skip_dense_8M_s2` ~34%.
**SFT-LR sweep (E3 / Fig E2) RELAUNCHED 2026-07-21** on the two previously-stopped boxes:
`i-03b11b7ddd885c65d` = lr0 (`lrsweep_worker0.json`) · `i-0089f074cd2749635` = lr1
(`lrsweep_worker1.json`), 4 runs each, ~43 min/run at a verified ~760 fp/s → ~3h total.
Manifests staged at `~/CLIMB/lrsweep_worker{0,1}.json`, OUTSIDE the synced tree.
**These 8 runs had never trained** — see §13.8. IPs change on every start — always re-query.

### 13.9 Deduped ablation recreation — QUEUED behind the lrsweep (2026-07-21)
`climb_v2_ablation` cannot be re-evaluated in place: **no encoder survives** for it (nor for
`climb_v2_labeleff`, `climb_v2_headline`, or `climb_v2` round-1 — that is a systemic
weight-retention failure across every pre-phase-2 wave). Its assay arms were also run PRE-dedup
and carry an eval-test leak, which is why they are daggered in Fig C1/J1. Re-running fixes both.

Manifests: `experiments/climb_v2_ablation_dedup/manifests/ablation_dedup_worker{0,1}.json`
(built by `scripts/build_ablation_dedup_manifests.py`). 10 runs — 6 pretrains (2M FP each) +
4 eval-only anchors — 12M FP ≈ **4.42 GPU-h, ~2.21h wall on two boxes**. Three repairs, all
verified present in the built manifest: live warm-start base (`climb_v2_phase2/unsup_2M`, the
original `climb_v2/unsup_only_seed0/encoder` is gone), `eval_blocklist_path` injected into every
arm (**this is the de-leaking step**), and `descriptor_precompute_dir` wired via
`finalize_manifest.py`. Writes to a NEW prefix so the pre-dedup numbers survive as before/after
leakage evidence for H6.

Launch is chained, not manual: `scripts/chain_wave.py` polls for the 8 lrsweep `verified.json`
markers, then restarts `i-03b11b7ddd885c65d` / `i-0089f074cd2749635` (they self-stop on
completion), redeploys, and launches workers `ab0`/`ab1`. **After it lands, Fig C1 and J1 must be
re-pointed** from `climb_v2_ablation` to `climb_v2_ablation_dedup` (notebook cell 23 sets
`ABL=`), and the `‡` leakage tags dropped for the re-run arms.

**Waves deliberately NOT recreated:** `climb_v2_labeleff` (already redone against phase-2 8M
encoders → `figure_data/climb_v2_labeleff_rep/`, which cell 19 reads), `climb_v2_headline`
(A1 reads phase-2's own anchors; nothing depends on it), and `climb_v2` round-1 (10.3 GPU-h,
exploratory; Fig H1 is MLM-only so the leak does not affect it).

### 13.8 The lrsweep runs never trained (diagnosed 2026-07-21)
`climb_v2_lrsweep` was recorded as "8 runs TRAINED but NOT evaluated". They were not: each run
dir held only `config.yaml` + `run_status.json`, no metrics.jsonl and no encoder, with elapsed
times of 13-990 s and status `failed`/`stalled`. **There was nothing to evaluate.**

Cause: every run warm-starts from `experiments/climb_v2/unsup_only_seed0/encoder`, and **no
`climb_v2` round-1 run has a surviving encoder anywhere** — that wave kept metrics and evals but
never its weights. A missing warm-start base kills the run at startup, which is why all 8 failed
instantly and identically.

Repair (`scripts/build_lrsweep_manifests.py`, commit 373a247): re-point the base at
`climb_v2_phase2/unsup_2M` — MLM-only, 1,999,872 achieved fp against the original base's
1,999,872, an identical budget, and leakage-deduped where round-1 was not — then route through
`finalize_manifest.py` to wire `descriptor_precompute_dir` into the 4 MTR arms (unwired, they
recompute descriptors on the fly at ~6x slowdown). Verified in flight: logs show
`MTR using PRECOMPUTED descriptors` and `warm-starting encoder from .../unsup_2M/encoder`, at
~760 fp/s versus the 105-141 fp/s collapse signature.

**What lands automatically:** the worker runs `_run_eval` after each pretrain, giving the
**single-split** `moleculenet/` eval (3 head seeds). The DEFAULT scaffold **5-fold CV** panel is
NOT produced box-side — run it locally over the 8 encoders afterwards, as for every other wave.

The 3-seed robustness pass is UNDER WAY: seed manifests hold 6 runs each
(`unsup_8M`, `skip_dense_8M`, `skip_sparse_all_8M`, `skip_dense_plus_sparse_8M`,
`skip_minimol_full_8M`, `skip_mixed_8M`), written to `<run>_s1` / `<run>_s2` output dirs.
`unsup_8M_s{1,2}` are verified complete; `skip_dense_8M_s{1,2}` in flight; 4 more per seed.

**Nothing else needs re-training.** Every other phase-2 run is verified complete (74 markers in
S3). The "INCOMPLETE" alerts w3/w4 emailed on 2026-07-21 were FALSE — caused by the sync bug in
§13.7 corrupting the metrics.jsonl that the completion gate recomputes achieved-FP from, not by
missing work. Verify against `verified.json` before ever redoing a run on the strength of an alert.

### 13.7 The cross-box S3 clobbering bug (found + fixed 2026-07-21)
**Symptom:** completed runs silently reverted in S3 — `verified.json` said 100% while
`metrics.jsonl` showed the old truncated run. Still active when caught (a run completed at
11:39 was reverted at 11:43).

**Cause:** every worker downloaded the ENTIRE wave tree at startup and pushed the ENTIRE tree
back every 10 min, so each box held copies of runs owned by *other* boxes and re-uploaded its
stale copy forever. `aws s3 sync` decides by size-or-newer-mtime: metrics.jsonl/run_status.json
differ in SIZE from the good copy so they were always pushed, while encoders are
same-size-and-older so sync skipped them — **which is the only reason model weights survived.**
The startup download then propagated the clobbered files back DOWN onto the box that had
produced the good copy, destroying the last good original.

**Blast radius:** 5 runs reverted. Encoders, `moleculenet*/` evals and `verified.json` intact
throughout — **no scientific artifact was lost.** Training logs restored for
`skip_dense_24M`, `skip_mixed_24M`, `skip_dense_48M`; **permanently short** for
`skip_minimol_full_48M` (32.1M of 48M logged) and `u2s_mixed_from2M` (269K of 2M logged) — those
two runs are complete and their encoders/evals are valid, only the loss curve is truncated.
One CV had to be redone: `skip_mixed_24M`'s encoder had been downloaded mid-training
(md5-distinct from the final), so its CV had scored the wrong model. Both that re-run and a
first-time CV for `skip_dense_48M` (verified complete, never previously evaluated) completed
cleanly on 2026-07-21 via `scripts/cv_repair.sh` — 7-task, 5-fold, 3 head seeds, incl. NEF1%.

**Fix (commit 686049f):** a run has exactly one owner — the box whose manifest lists it. Uploads
are scoped to owned runs; downloads pull owned runs in full (needed to resume) but only
`encoder/` for the rest (u2s warm-start bases). **Ownership keys off `output_dir`, NOT `run_id`:**
seed manifests reuse the base `run_id` ("unsup_8M") for a run that lives at "unsup_8M_s1", so
scoping by `run_id` would push seed results straight over the original completed runs.

**Standing rules this bought:**
- Never let two boxes hold write authority over one `output_dir`. Before offloading tail work to
  a spare box, remember a running worker's queue cannot be edited — an overlap is a collision.
- Keep scratch/quarantine dirs OUTSIDE `experiments/` — a `_quarantine_stale/` placed inside it
  was promptly synced to S3 (7.5 GB / 412 objects of duplicated run dirs). Purged 2026-07-21
  after verifying no quarantined copy was ahead of its live counterpart. The one that looked
  ahead — `unsup_48M` at 22.5M vs 18.9M live — was the DEAD 12h-truncated run, while the live
  one was the fresh rerun still climbing to 48M: **higher forward-pass count, wrong run.** That
  is why box→S3 authority is ownership-based and never "highest fp wins".
- Deleting under versioning needs `--version-id`: these objects predated versioning (VersionId
  `null`) and the lifecycle keeps the 3 newest noncurrent versions indefinitely, so a plain
  `aws s3 rm` would have added delete markers and freed nothing. See
  `scripts/` history / session notes for the batched versioned-purge pattern.
- **S3 bucket versioning is now ENABLED** (2026-07-21) — it was off, which is why the overwrites
  were unrecoverable. A clobber is now a rollback:
  ```bash
  # list versions of a file that got overwritten, newest first
  aws s3api list-object-versions --bucket climb-s3-bucket \
    --prefix experiments/climb_v2_phase2/<run>/metrics.jsonl \
    --query 'Versions[].[LastModified,Size,IsLatest,VersionId]' --output text
  # restore a prior version
  aws s3api get-object --bucket climb-s3-bucket \
    --key experiments/climb_v2_phase2/<run>/metrics.jsonl --version-id <VID> restored.jsonl
  ```
  Verified end-to-end (write → clobber → recover prior version) on 2026-07-21.
  Paired with a lifecycle policy so versions cannot accumulate unbounded on a 3.8 TB bucket:
  noncurrent versions expire after **14 days** but the **3 most recent are always kept**, and
  incomplete multipart uploads abort after 7 days. Note versioning changes delete semantics —
  `aws s3 rm` now leaves a delete marker; purging data for real needs `--version-id`.

### 13.5 Figures — done vs pending
- **DONE:** Fig A1 (both **CV default** + **single-split SI**), leakage table (§6.6).
- **PENDING** (need wave-1/2 data): A2 scaling, Fig B, Fig C ablation, Fig D eval-ceiling (finetune —
  not run), label-efficiency, transfer matrix, forgetting, H8 repetition, H9 memorization/Tanimoto,
  compute/data recycling. The notebook carries DUMMY placeholder cells for the uncollected ones (nb §11).

### 13.6 Resume-from-scratch commands
```bash
# live state
aws ec2 describe-instances --instance-ids i-02dfaa83dae4ad937 i-03b11b7ddd885c65d i-0b59865cc08ef390c i-0dbc751470e2108d5 i-07486d883063b0925 --query 'Reservations[].Instances[].[InstanceId,State.Name,PublicIpAddress]' --output text
for r in $(aws s3 ls s3://climb-s3-bucket/experiments/climb_v2_phase2/ | awk '{print $2}' | tr -d /); do aws s3 ls s3://climb-s3-bucket/experiments/climb_v2_phase2/$r/verified.json >/dev/null 2>&1 && echo VERIFIED $r; done
# deploy + (re)launch a worker (manifest-first, manifest OUTSIDE the synced tree):
ssh -i climb-gpu-key.pem ec2-user@<ip> 'cd ~/CLIMB && git fetch && git reset --hard origin/v2-redux && chmod +x scripts/*.sh && nohup bash scripts/phase2_worker.sh <manifest.json> wN > phase2_wN.log 2>&1 &'
# local CV eval → moleculenet_cv/, then render figures (exec nb code cells 2,3,5,7, matplotlib Agg)
.venv_sanity/bin/python scripts/cv_eval_local.py
```
