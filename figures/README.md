# `figures/` — the paper figure pipeline (v2, six-panel suite)

Every figure in the paper is produced by a script in this directory. No notebook, no hidden state:
edit a script, run it, the PNG/PDF in `figures_v2/` changes. The old `climb_figures.ipynb` is left
untouched as a reference and is **not** part of this pipeline.

## Layout

| File | Role |
|---|---|
| `arms.py` | **single source of truth** — model names, colours, and the 6 benchmark panels |
| `style.py` | matplotlib rcParams, figure sizes, `save()` |
| `sixpanel.py` | loaders for `figure_data/six_panel/*` + the ranking maths |
| `fig_<ID>_<name>.py` | one script per paper figure; each renders several visual variants |

## Run

```bash
python3 scripts/six_panel_aggregate.py     # refresh the results tables + STATUS board
python3 -m figures.fig_A1                  # render Fig A1 into figures_v2/
```

## Nomenclature (fixed — use verbatim, never invent a new label)

`ECFP` · `ECFP+desc` · `supervised, <readout>` · `unsupervised` · `unsup→sup, <readout>` ·
`sup→unsup, <readout>` · `random encoder` · `no pretrain, end2end` · `CheMeleon / end2end`

Readouts: `dense`, `dense+sparse`, `mixed`, `sparse`, `MiniMol tasks`. ONLY the two-phase recipes
are abbreviated (`unsup→sup`, `sup→unsup`); single-phase arms are written out in full.

## Floors — two different ones, on purpose

`fig_C2`, `fig_D` lift over **no pretrain, end2end** (is the frozen pipeline worth it at all?).
`fig_E` lifts over **no pretrain, frozen** (what did the pretraining OBJECTIVE contribute, holding
architecture and probe protocol fixed?). These answer different questions — never "unify" them.

## Seed rule

Every reported number is the **mean over all replicate seeds** — never a single seed. MolNet values
average 3 pretraining seeds x 3 head seeds x 5 folds; CBS/MoleculeACE/Polaris average their 3 seeds.
Where an analysis structurally needs one seed's per-molecule predictions (the co-best paired
bootstrap), that is stated explicitly beside the number.

## Colours

orange = XGBoost anchors · red = supervised · blue = unsupervised · green = unsup → supervised ·
purple = CheMeleon · grey = no pretrain, end2end · black = random encoder. Shades within a family
run dark (headline recipe) → light (peripheral recipe).

## The six panels

MoleculeACE (macro RMSE, 30 targets) · CBS (NEF1%) · BACE (ROC-AUC) · **hERG (ROC-AUC)** ·
Tox21 (mean ROC-AUC, 12 assays) · QM7 (RMSE). BBBP was replaced by hERG on 2026-08-16; HIV and
ESOL/Lipophilicity are dropped.
CheMeleon appears in A1/A2 only — it is excluded from every ablation and scaling figure.

## Where the numbers come from

`figure_data/six_panel/` — see `figure_data/six_panel/STATUS.md` for the coverage board
(which model × benchmark cells exist, which compute waves are still running).

## Figure roster

**One script → one figure.** While iterating we may render alternatives, but the committed state of
each script outputs exactly one file. No `v1/v2/...` suffixes in the final version.

**Never draw the caption into the image.** Captions belong in the LaTeX `\caption{}`; write the
caption source in the script's module docstring instead. There is deliberately no `caption()`
helper in `style.py`.

**All text is black** (`#000000`) — no grey text anywhere. Row labels are two lines: the model
system in **bold** (`XGBoost` / `CLIMB` / `CheMeleon`) over the recipe in regular weight.

| ID | Figure | Script | Output | Status |
|---|---|---|---|---|
| A1 | mean rank across all 66 datasets (4 suites) | `fig_A1.py` | `figA1` | final |
| A2 | the 6 panels, 8M mainline, sd_total bars | `fig_A2.py` | `figA2` | final |
| B | pretraining scaling ladders (x = tokens) | `fig_B.py` | `figB` | final (clean variant) |
| C1 | molecular similarity, unsupervised (memorization vs representation) | `fig_C1.py` | `figC1` | final |
| C2 | molecular similarity, supervised (H10 test) | `fig_C2.py` | `figC2` | final |
| D | task similarity (bars + transfer matrix + descriptor mapping) | `fig_D.py` | `figD` | final |
| C+D | assembled a–f: C1+C2 top row, D bottom row | `fig_C_D.py` | `fig_C_D` | final; composes the three `compute()`+`draw()` pairs, no re-analysis |
| E | corrupted objectives — 2 panels: (a) supervised real vs permuted targets, (b) unsupervised ladder real/shuffled/bigram/unigram/wiki | `fig_E.py` | `figE` | final on the 6 MoleculeNet tasks (5-fold CV); input table built by `scripts/build_figE_table.py`. Corrupted + synthetic arms have MoleculeNet evals ONLY, so a canonical-panel version needs MoleculeACE/CBS/hERG runs of 7 encoders |
| F | where end2end overtakes a pretrained frozen encoder (absolute performance vs labelled training size) | `fig_F.py` | `figF` | built; input table from `scripts/build_figF_table.py`. MoleculeACE/CBS/hERG drawn EMPTY — the label-fraction sweep was only ever run on MoleculeNet; evals requested 2026-08-17 |
| SI a–e | e2e necessity · vocab · featurization cost · redundancy · canonical vs augmented | — | — | |
