# Handoff → notebook session: add the vocab-size scaling SI figure(s) to `climb_figures.ipynb`

**From:** compute session · **Date:** 2026-08-04
**Why you:** you own `climb_figures.ipynb` / `notebook_cells/`. The compute session ran the experiment,
has the data + a working standalone plot, and wrote README §7.2 (method). What's left is folding a
figure into the generated notebook, which is your lane.

---

## 1. What the experiment is (for the caption)

SI vocabulary-size scaling law, wave `climb_v2_vocab` (README **§7.2**). Two tokenizer families —
**byte-level BPE** (the main-paper family) and **Unigram-LM** — each at four *reachable, distinct*
vocab sizes (SMILES tokenization saturates). One **MLM-only** encoder per tokenizer, **2M forward
passes**, same corpus, same eval as everything else; embedding auto-sizes to the vocab so params grow
with vocab (≈41.0M→47.1M) — the one dimension deliberately allowed to move.

**Finding (near-null — this IS the result, state it plainly):** across the reachable range,
vocabulary size changes downstream frozen-probe CV scores by **less than the fold-to-fold noise on
almost every task**, and BPE ≈ Unigram at matched vocab. The **character-level floor (vocab 261, no
merges) is already competitive** (best or tied on several tasks). The only >~1σ move is **ESOL, where
BPE degrades with vocab (0.454→0.507 RMSE) while Unigram does not (0.440→0.429)** — worth one hedged
sentence, not a headline.

Suggested caption:
> **Fig S? — Vocabulary size barely affects unsupervised SMILES pretraining.** Frozen-probe 5-fold
> scaffold-CV performance for MLM encoders pretrained at matched 2M-FP compute, as a function of the
> (actual) tokenizer vocabulary, for byte-level BPE and Unigram-LM. Error bars are ±1 fold std. Across
> the reachable SMILES vocab range the effect is within fold noise on all tasks except a mild ESOL/BPE
> degradation; a character-level tokenizer (vocab 261) is already competitive. Vocabulary size and
> embedding-parameter count are confounded by construction (larger vocab ⇒ more params); disclosed, not
> removed.

---

## 2. Where the data is (already local + in S3, backed up)

- **Per-run CV summaries (what to plot from):** `figure_data/climb_v2_vocab/<run>_cv.csv`
  for the 8 runs `bpe_{261,1000,3000,12000}` and `unigram_{261,700,1200,3000}`.
  Aggregate rows are `main_metric ∈ {rmse, roc_auc, nef1}` with `head_seed ∈ {MEAN, STD}` (mean/std
  across the 5 folds of the 3-head-seed-averaged prediction — identical scheme to every other CV panel).
- **Tidy long table (easiest to load):** `figure_data/climb_v2_vocab/vocab_cv_summary.csv`
  columns `family, actual_vocab, run, dataset, metric, mean, fold_std`.
- **S3:** `s3://climb-s3-bucket/experiments/climb_v2_vocab/<run>/moleculenet_cv/`; encoders + tokenizers
  also there and mirrored to the backup bucket.

**Actual (measured) vocab = the x-axis** — NOT the nominal target:
BPE `261, 1000, 3000, 12000`; Unigram `261, 700, 876, 3000` (the `unigram_1200` run resolved to 876).
The tidy CSV already carries `actual_vocab`.

---

## 3. Reference implementation you can lift

`scripts/fig_vocab_scaling.py` (compute session) already renders the two panels below to
`figures_vocab/` (PNG+PDF). It's standalone matplotlib and reads the CSVs above — copy its
`series()` / panel loop straight into a cell. I put the outputs in **`figures_vocab/` on purpose, NOT
`figures_out/`, so I would not desync your notebook↔figures_out check.** When you add the cell, save
into `figures_out/` via your `save_fig` helper and delete `figures_vocab/`.

- **`figSV_vocab_scaling`** — 2×3 task panels (ESOL, QM7, BBBP, BACE, Tox21, HIV-NEF1), metric vs
  actual vocab (log x), BPE vs Unigram, ±fold-std bars. This is the main SI figure.
- **`figSV_vocab_effect`** — one summary panel: largest-vocab effect vs the vocab-261 baseline in
  fold-std units, with a shaded ±1σ band. Optional but it makes the null legible; keep it or drop it.

---

## 4. How to wire it into the generated notebook (your workflow)

1. Add a new cell pair in `notebook_cells/` after the current last cell (`30.py`): `31.md` (spec
   header, same style as other figure specs) + `32.py` (the plot).
2. In `32.py` reuse the global **`STYLE`/`PALETTE`** (cells 02/03) and call **`save_fig(fig, name)`**
   so it lands in `figures_out/` as PNG+PDF. Suggested names following the `fig<ID>_<name>` scheme:
   `figS?_vocab_scaling` and (optional) `figS?_vocab_effect` — pick the next free **S-index** (S1 =
   compute/data scatter already exists). Suggested family colors: BPE `PALETTE["blue"]` (#4477AA),
   Unigram a warm contrast (`#EE7733`).
3. Rebuild + verify: `python scripts/build_figure_notebook.py` → execute → `python
   scripts/verify_notebook_sync.py` (must pass before commit — on-disk ipynb == cells == figures_out).
4. **Regenerate `figure_data_manifest.json`** so it includes the new `figure_data/climb_v2_vocab/*`
   files (otherwise the sync/traceability snapshot is stale).

---

## 5. Gotchas (don't trip these)

- **CV only.** Plot the 5-fold CV numbers; do not mix in a hold-out point. All 8 runs have CV, none
  were given a single-split headline pass — so a CV-only panel is apples-to-apples.
- **Lipophilicity excluded** everywhere (blocklist predates it). The six tasks above are the set.
- **HIV headline = NEF1%** (virtual screening); its ROC-AUC exists too (`roc_auc` rows) if you want a
  secondary panel, but NEF1 is the one to show.
- **Error bars = fold std** (split variance), same quantity as every other CV figure — say so in the
  caption; it is NOT head-seed or pretraining-seed variance.
- **One seed.** These are single-pretraining-seed runs (seed 0), unlike the 3-seed main arms — note it
  if a reviewer asks about pretraining-seed variance here.
- **Vocab↔params confound** is inherent and disclosed (README §7.2); the caption line above covers it.

Data, figures, and the generator script are committed by the compute session; the notebook integration
is all yours. Ping back if you want the effect-panel dropped or a different task layout.
