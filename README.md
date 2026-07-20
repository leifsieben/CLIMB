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
| E13 | **Corrupted / irrelevant-pretraining control (H2c)** | pretrain on a domain-mismatched corpus **or** a corrupted objective (shuffled-token MLM / shuffled-target MTR); 1–2 points vs random init and real pretraining | 🔲 planned *(optional, new compute)* |

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
| **B2** | H2c | corrupted / domain-mismatch / real pretrain vs no_pretrain | control bars | ❌ dummy (E13 unrun) |
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
- **Held-out MLM val / test split (C5, for H8).** A fixed hash-based holdout of the corpus (never
  streamed into training) provides MLM **validation and test** loss, logged per run. This exposes the
  H8 mechanism directly: canonical repetition should overfit (train↓, val/test↑) while enumeration
  keeps generalizing — a readout the frozen-probe downstream lift can mask (the ceiling of §10). The
  holdout membership is derived from the same deterministic `subset_seed` hashing so it is disjoint
  from every ladder's training subset.

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

**Two distinct keys — dedup vs "seen" (C17).** The blocklist key above is deliberately *fuzzy*: it
salt-strips to the largest fragment and canonicalizes without stereo, which conservatively
over-removes (salt forms / stereo-variants collapse together) — the right choice for a **safety**
dedup. It is **not** used to define the H9 "seen" axis. For H9 (memorization), "seen" = **true
exact-molecule identity**: the full canonical SMILES *including stereochemistry*, with **no**
salt-strip / largest-fragment collapse. Keeping the two keys separate means dedup stays conservative
while the memorization axis is never mislabelled by merged salt/stereo variants. Both keys are
recorded per eval molecule (§9.5) so a reviewer can see which molecules each classifies as overlapping.

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
  batch **256**, bf16, seed **42**. ⚠️ The same LR is used for scratch and warm-start; the SFT-LR
  sweep (E3) tests whether this is fair for the warm-start phase.
- **Compute accounting.** Everything is measured in **forward passes** (molecule-presentations).
  Within one epoch (≤12M) forward passes = #unique molecules; beyond that they diverge (repetition).
- **Canonical vs enumerated.** Primary runs use canonical SMILES (one presentation per molecule);
  enumeration (on-the-fly RDKit randomization) is the H8 lever for the beyond-one-epoch regime.
  **Enumeration RNG is logged (C18):** the randomized-SMILES enumeration seed is recorded per run
  (separate from `subset_seed`, which does not capture the augmentation RNG), so an enumerated run is
  reproducible bit-for-bit.
- **HPO policy.** A single up-front MLM hyperparameter search fixes the recipe; the same settings are
  reused for every subsequent run (no per-experiment tuning) so optimization never confounds a
  comparison. The exception is the deliberate SFT-LR ablation (E3).
- **Durability.** Periodic encoder checkpoints (`save_every_steps`) plus a 10-minute S3-sync sidecar
  in the worker script survive spot reclaim (current boxes are on-demand, so this is belt-and-braces).

---

## 8. Evaluation protocol (frozen featurizer)

The **primary** protocol mirrors real deployment of molecular foundation models: freeze the encoder,
extract one embedding per molecule, train a small head on those embeddings.

- **Featurizer:** frozen encoder → **masked-mean pooling** over token states (not CLS — v1's
  CLS-linear-probe was pathological). Optional cache of encoder forwards for speed.
- **Standardization:** per-feature **z-score fit on the train split only**, applied to val/test (no
  standardization leakage).
- **Head:** small **MLP** (also supports linear / XGBoost), trained with **3 head seeds**; the
  reported value is the mean over seeds. Early stopping on val (patience 15, ≤100 epochs).
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
     **lower and noisier** than CV (one draw, no error bar), so it lives in the SI, not the headline.
  **Convention:** CV-5 is the default scheme for **all** figures; the harder single-split is added to
  the SI where the novel-scaffold stress is worth showing (currently Fig A1). The two are stored side
  by side (`moleculenet_cv/` vs `moleculenet/`) and are **never mixed within one panel** — every bar
  in a figure uses the same scheme. Absolute numbers differ markedly between schemes (e.g. BBBP ≈0.95
  CV vs ≈0.74 single-split), which is expected and disclosed; model *rankings* are what transfer.
- **Metrics:** absolute **per task** — RMSE for ESOL/QM7/Lipophilicity, ROC-AUC for BBBP/BACE/Tox21,
  and **NEF1% (top-1% normalized enrichment) as the headline for HIV** (the virtual-screening task; ROC-AUC
  kept secondary) — see §6.5 for the NEF1% definition. Never z-scored or averaged across tasks.
- **Anchors (classical baselines, `--head xgb`, all through the same eval pipeline):** an
  untrained-encoder **random floor** (3 seeds); **`ecfp4`** = Morgan ECFP4 + XGBoost; **`rdkit_desc`**
  = 217 RDKit descriptors + XGBoost (the classical control for the dense-MTR arm); and **`fp_desc`** =
  **Morgan fingerprints ++ RDKit descriptors concatenated → XGBoost** — the *toughest* classical
  baseline (both substructure bits and computed physchem), which a CLM must beat to justify itself
  (e.g. it already gets ESOL ≈0.35 CV-RMSE, ahead of every neural regime at 8M). "Lift" is improvement
  over the random floor.
- **Eval-ceiling (E5):** the same encoders are additionally **fine-tuned end-to-end** (`finetune_v2`)
  on a few tasks + HIV, to test whether the frozen probe under-resolves encoder quality (H5).
- **Per-molecule prediction dump (C16, blocking).** `eval_v2` writes **`(canonical_key, y_true,
  y_pred, task, split)` per eval molecule**, not only aggregate RMSE/AUC. This is required to build
  both Fig H9 panels (C6) and to bin the label-efficiency curves (C7); without it H9 is unbuildable
  without re-running eval. *Caveat:* per-cell / per-bin **ROC-AUC is unstable** on small subsets —
  prefer per-molecule residual/rank, or run the mechanism panels on the RMSE tasks (ESOL, QM7).
- **Persisted fingerprints (C19).** The **ECFP4 fingerprints** already computed for the ECFP4 anchor
  are saved for all eval molecules **and a pretraining-corpus sample**, so nearest-neighbor Tanimoto
  distance (Fig H9 panel 2) and the C8 similarity overlay need no recomputation.
- **Mechanism figures through finetune (C20).** For H9, label-efficiency, and the transfer matrix,
  report at least a subset **through end-to-end finetune** (`finetune_v2`) as well as the frozen
  probe — or caveat explicitly with the frozen-probe ceiling (H5 / §10). A ceiling-compressed probe
  flattens the very differences these figures rely on (Type-II risk), so the mechanism story must not
  rest solely on the probe H5 says may be under-resolving.

### 8.1 Model-vs-model comparison protocol (headline tables)

Every pairwise claim ("model A beats baseline B") is backed by the same protocol. **Every model —
CLM *and* classical baseline (Morgan/desc/fp_desc XGBoost) — is run at 3 seeds × 5 scaffold-CV folds
= 15 (metric) evaluations.** The comparison table (see `scripts/compare_models.py`, reproduced in the
notebook) reports:

| Element | Specification |
|---|---|
| Points per model | **3 training seeds × 5 scaffold-CV folds = 15** metric evaluations (CLMs: pretraining seed; XGBoost: model seed) |
| Error bar | mean ± std across the 15 points — folds in **both** pretraining-seed and scaffold-split variance |
| Pairing | all models share one scaffold fold partition per seed → fold- and molecule-paired |
| Effect | Δ(metric) and relative % |
| **RMSE tasks** (ESOL/QM7/Lipo) | rigorous test = molecule-level **paired Wilcoxon** on per-molecule squared error over the pooled out-of-fold predictions (large n) |
| **AUC tasks** (BBBP/BACE/Tox21) | rigorous test = **DeLong paired-AUC** on the pooled OOF scores (per label column for multi-task Tox21, then summarised) |
| **HIV (virtual screening)** | headline metric = **NEF1%** (top-1% enrichment), reported as mean ± std across folds; rigorous paired test = **DeLong paired-AUC** on the pooled OOF scores (a rank test that tracks early enrichment). Select NEF1% in `compare_models.compare` by passing the task as `('HIV', True, 'nef1')` |
| Fold-level test | paired t across folds — reported but **flagged anti-conservative** (CV folds share training data; Bengio & Grandvalet 2004), so the molecule-level Wilcoxon/DeLong is the test of record |

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

---

## 9. Reproducibility

### 9.1 Code map (what each module does)
| File | Role |
|---|---|
| `config_v2.py` | single source of truth: `ModelConfigV2`, `build_modernbert_config`, `TrainingConfigV2`, `EvalConfigV2`, task list, supervised families/groups/weights/caps |
| `data_v2.py` | streaming MLM/raw-SMILES datasets, `MTRCollator` (+ precomputed descriptors), stratified supervised loader (dedup + scarce-first routing), objective iterator |
| `descriptors_v2.py` | 217 RDKit descriptors, fit/normalize/save stats |
| `pretrain_v2.py` | `ClimbV2Model` (MLM/MTR/supervised heads), training loop, warm-start, checkpointing |
| `eval_v2.py` | frozen-featurizer MoleculeNet evaluation, scaffold splits, standardization; **per-molecule prediction dump (C16)** and **ECFP4 fingerprint persistence (C19)** |
| `featurize_v2.py` / `heads_v2.py` | pooling + standardizer; downstream heads + metrics |
| `finetune_v2.py` | end-to-end fine-tuning (eval-ceiling) |
| `random_baseline_v2.py` | untrained-encoder + ECFP4 anchors |
| `experiment_v2.py` | manifest generator for every wave (ablation / compute_scaling / phase2) |
| `data.py`, `storage_utils.py`, `utils.py`, `token_budget.py` | shared streaming base + S3 helpers (dependencies of the above) |
| `train_tokenizer.py` | tokenizer training (provenance; artifact is prebuilt) |
| `scripts/` | launch/split/deploy, leakage audit, blocklist + descriptor precompute, report/figure builders (see §9.3) |

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

### 13.4 Current live status (2026-07-20)
w0=`skip_dense_96M` · w1=`u2s_minimol_full_from24M` · w2=`skip_dense_plus_sparse_48M` ·
w3=`skip_minimol_full_48M` · w4=`u2s_dense_plus_sparse_from2M`. 4 u2s verified so far. ETAs from
launch: u2s ~1 day, recovery 48M ~18h, 96M ~35h.

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
