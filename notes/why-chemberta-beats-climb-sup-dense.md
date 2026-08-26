# Why ChemBERTa-2 outranks CLIMB supervised-desc (2026-08-26)

Leif asked. It is not the objective, not the architecture, and not the compute. It is the
**pretraining corpus**, and our own scaling ladder demonstrates it.

## The two models are the same recipe

| | CLIMB `sup_dense` (`skip_dense_8M`) | ChemBERTa-2 (`ChemBERTa-77M-MTR`) |
|---|---|---|
| objective | MTR, 217 RDKit descriptors | MTR, ~200 RDKit descriptors |
| encoder params | **41.4 M** | **3.4 M** |
| molecules seen | 8 M presentations | ~77 M |
| corpus | **12 M distinct molecules** | ~77 M |
| MoleculeACE macro RMSE | 0.7741 | **0.7446** |

Same objective family. We are **12x larger** and still lose, so this is not a capacity problem.

## Compute on our corpus does not close it -- it saturates and then reverses

`sup_dense` ladder, MoleculeACE macro RMSE (lower better):

    2M   0.09B tok   0.7923
    8M   0.34B tok   0.7741      <- the arm in fig_A
    24M  1.03B tok   0.7687
    48M  2.06B tok   0.7674      <- best ever
    96M  4.12B tok   0.7748      <- WORSE than the 24M rung

A 12x increase in tokens buys 0.0067 and then goes backwards. The best MTR model we have ever
trained on the 12M corpus is 0.7674, still 0.023 behind ChemBERTa.

## Corpus size DOES close it, at matched compute

The two `unsup` rungs marked `big_corpus` use a larger corpus than the 12M one below them.
Comparing at essentially identical compute -- 48M vs 50M forward passes:

| panel | metric | unsup_48M (12M corpus) | unsup_50M (larger) | unsup_100M (larger) |
|---|---|---|---|---|
| MoleculeACE | macro RMSE | 0.7820 | 0.7474 | **0.7307** |
| CBS | NEF1% | 0.6456 | 0.8350 | 0.8100 |
| Ames | ROC-AUC | 0.8015 | 0.8299 | 0.8261 |
| Tox21 | ROC-AUC | 0.7908 | 0.8164 | 0.8206 |
| HIV | NEF1% | 0.6482 | 0.6819 | 0.6627 |
| QM7 | RMSE | 198.48 | 194.72 | 194.94 |
| BACE | ROC-AUC | 0.8535 | 0.8477 | 0.8448 |

Six of seven panels improve, several substantially, for the SAME number of forward passes. The only
CLIMB models that beat ChemBERTa-2 on MoleculeACE are the two big-corpus rungs: **0.7474 and
0.7307 against 0.7446.**

## So the answer

ChemBERTa-2 is not a better recipe. It saw more distinct molecules. Our ladder shows that more
GRADIENT STEPS on 12M molecules saturates, while more MOLECULES at the same step count keeps
paying -- which is the same statement from two directions, and it identifies the binding
constraint as corpus diversity rather than compute or capacity.

Note also that the arm which overtakes ChemBERTa is **unsupervised MLM**, not MTR. At the 12M
corpus MTR beats MLM (0.7741 vs 0.7766, a tie); with a larger corpus MLM pulls clear (0.7307).

## What this means for fig_A, and it belongs in the caption

Every CLIMB arm in fig_A is an 8M-presentation model on the 12M corpus. ChemBERTa-2 placing second
is therefore a comparison of **what each project shipped**, not of which recipe is better. A reader
will take it as the latter. The honest caption sentence is that the CLIMB arms are held at a fixed
pretraining budget by design -- the figure varies the OBJECTIVE, not the data scale, which is
fig_B's axis -- and that our own ladder puts a larger-corpus CLIMB model ahead of ChemBERTa-2.

**Open item:** `unsup_100M` exists on the seven scaling panels only, not across all 65 ranking
datasets, so it cannot currently be drawn in fig_A. If the paper wants to make the "CLIMB at a
comparable corpus beats ChemBERTa-2" claim in the ranking rather than only in fig_B, that arm needs
the full suite. Worth costing before deciding.
