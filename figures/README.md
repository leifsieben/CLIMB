# `figures/` — the paper figure pipeline (v2, six-panel suite)

Every figure in the paper is produced by a script in this directory. No notebook, no hidden state:
edit a script, run it, the PNG/PDF in `figures_v2/` changes. The old `climb_figures.ipynb` is left
untouched as a reference and is **not** part of this pipeline.

## Layout

| File | Role |
|---|---|
| `arms.py` | **single source of truth** — model names, colours, and the 6 benchmark panels |
| `style.py` | matplotlib rcParams, figure sizes, `save()` (incl. the page-width check and the `subdir=` used by component panels) |
| `sixpanel.py` | loaders for `figure_data/six_panel/*` + the ranking maths |
| `fig_<ID>.py` / `SI_fig_<id>.py` | one script per paper figure. Naming is fixed: main text `fig_*`, supplementary `SI_fig_*`, and the artefact in `figures_v2/` carries the SAME name as its script |

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

## What lands where

`figures_v2/` holds ONLY paper deliverables — the numbered figures, plus `SI_fig_c.csv/.tex`
because that supplementary item IS a table. Component panels that exist only to be assembled into
another figure (C1, C2, D → fig_C_D) render to `figures_v2/panels/` via `save(..., subdir="panels")`.
Per-figure data records go to `figure_data/<fig>/`, never to `figures_v2/`.

## Page width

Every figure is authored at `STYLE["col2"]` = **6.69 in**, the 170 mm text block of A4 with 20 mm
margins, so figures go into LaTeX at `width=\textwidth` with NO downscaling and the point sizes in
`FS` are the sizes that print. `save()` measures the width actually written and WARNS when it
deviates >5%: `savefig(bbox_inches="tight")` trims slack margins and, worse, GROWS past the canvas
when a legend or title is anchored outside the axes. Known offenders today: `fig_C1` (-14%, slack
internal margins), `fig_C2` (+6%) and `fig_D` (+9%). Fix them by making the axes fill the canvas /
moving anchored content inside — NOT by rescaling `figsize`, which tight-bbox simply re-trims.

## Colours

orange = XGBoost anchors · red = supervised · blue = unsupervised · green = unsup → supervised ·
purple = CheMeleon · grey = no pretrain, end2end · black = random encoder. Shades within a family
run dark (headline recipe) → light (peripheral recipe).

## The six panels

MoleculeACE (macro RMSE, 30 targets) · CBS (NEF1%) · BACE (ROC-AUC) · **Ames (ROC-AUC)** ·
Tox21 (mean ROC-AUC, 12 assays) · QM7 (RMSE). BBBP was replaced by hERG on 2026-08-16, and hERG
by **Ames** on 2026-08-17 (hERG's 132 test molecules gave only ~5.6 SE of headroom against Ames'
~12.2 at n=1457); HIV and
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
| A1 | mean rank across all 66 datasets (4 suites) | `fig_A1.py` | `fig_A1` | final |
| A2 | the 6 panels, 8M mainline, sd_total bars | `fig_A2.py` | `fig_A2` | final |
| B | pretraining scaling ladders (x = tokens) | `fig_B.py` | `fig_B` | final (clean variant) |
| C1 | molecular similarity, unsupervised (memorization vs representation) | `fig_C1.py` | `panels/fig_C1` | final |
| C2 | molecular similarity, supervised (H10 test) | `fig_C2.py` | `panels/fig_C2` | final |
| D | task similarity (bars + transfer matrix + descriptor mapping) | `fig_D.py` | `panels/fig_D` | final |
| C+D | assembled a–f: C1+C2 top row, D bottom row | `fig_C_D.py` | `fig_C_D` | final; composes the three `compute()`+`draw()` pairs, no re-analysis. **This is the paper figure**; C1/C2/D are its components |
| E | corrupted objectives — 2 panels: (a) supervised real vs permuted targets, (b) unsupervised ladder real/shuffled/bigram/unigram/wiki | `fig_E.py` | `fig_E` | final on the 6 MoleculeNet tasks (5-fold CV); input table built by `scripts/build_fig_E_table.py`. Corrupted + synthetic arms have MoleculeNet evals ONLY; a canonical-panel version needs MoleculeACE/CBS/Ames runs of 13 encoders (requested 2026-08-17) |
| F | are CLIMB embeddings redundant to classical features? (concatenation test) | `fig_F.py` | `fig_F` | built; 3/6 panels — the concatenation test only ever ran on MoleculeNet. Promoted from SI d 2026-08-17; CheMeleon arm + the 3 missing panels requested |
| SI a | do you need end2end training on downstream data? | `SI_fig_a.py` | `SI_fig_a` | built; slope plot, 5/6 panels (CBS has no e2e run of a pretrained encoder). Protocol differs BETWEEN panels — compare within a panel only |
| SI b | tokenizer family / vocabulary size | `SI_fig_b.py` | `SI_fig_b` | built; 5/6 panels — CBS landed, Ames pending the Polaris re-score. Near-null result, so it carries error bars |
| SI c | featurization cost, descriptors vs transformer | `SI_fig_c.py` | `SI_fig_c.csv/.tex` | built (table) |
| SI d | canonical vs augmented SMILES | `SI_fig_d.py` | `SI_fig_d` | built; 5/6 panels (Ames pending). Was 2/6 until a WRONG-ROOT fix: climb_v2 is the round-1 wave, climb_v2_h1 is the retrained one. Was SI e |
| SI e | where end2end overtakes a pretrained frozen encoder | `SI_fig_e.py` | `SI_fig_e` | built; 3/6 panels — label-fraction sweep never run on MoleculeACE/CBS/Ames. Was SI f |
