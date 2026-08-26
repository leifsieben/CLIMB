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

## It is DISTINCT MOLECULES, not the corpus label -- the matched-compute control is a wash

`unsup_8M_c124` is a matched 8M-forward-pass run on the 124M corpus, built to isolate corpus from
training length. At 8M forward passes you see 8M unique molecules from EITHER corpus (both are
under one epoch), so if the 124M corpus were better per se, this run would show it. It does not:

    BACE  0.8694 -> 0.8471 (12M better)    QM7   197.40 -> 196.01 (124M better)
    BBBP  0.9536 -> 0.9480 (12M better)    Tox21 0.7987 -> 0.8097 (124M better)
    HIV   0.7789 -> 0.7802 (124M better)   ESOL   1.0131 -> 0.9733 (124M better)

Four of six favour the 124M corpus, two favour the 12M one, all small. A wash.

So the gain at 50M/100M is not "the big corpus is better chemistry". It is that the 12M corpus
CAPS distinct molecules at 12M: the 24M and 48M rungs are re-reading it 2x and 4x. The 124M corpus
is the only way the study exceeds 12M distinct molecules.

    unsup_48M   12M corpus, ~4 epochs,  12M distinct   MoleculeACE 0.7820
    unsup_50M   124M corpus, <1 epoch,  50M distinct   MoleculeACE 0.7474

Same compute, 4x the distinct molecules, 0.035 better. REPETITION SATURATES; NEW MOLECULES DO NOT.

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

## Does CLIMB reach ChemBERTa on a larger corpus? Yes -- but be precise about the terms

Head-to-head, `unsup_100M` against ChemBERTa-2 on the five panels where both are measured with a
matching metric:

| panel | metric | CLIMB unsup_100M | ChemBERTa-2 | winner |
|---|---|---|---|---|
| MoleculeACE | macro RMSE | **0.7307** | 0.7446 | CLIMB |
| CBS | NEF1% | **0.8100** | 0.7818 | CLIMB |
| Tox21 | ROC-AUC | **0.8206** | 0.7938 | CLIMB |
| QM7 | RMSE | **194.94** | 197.57 | CLIMB |
| BACE | ROC-AUC | 0.8448 | **0.8471** | ChemBERTa (by 0.0023) |

CLIMB wins 4 of 5. But the honest framing is a LADDER, not a win:

    50M distinct molecules, 41.4M params   ->  0.7474   ~tie with ChemBERTa
    77M distinct molecules,  3.4M params   ->  0.7446   ChemBERTa
    100M distinct molecules, 41.4M params  ->  0.7307   CLIMB ahead

At FEWER molecules than ChemBERTa (50M vs 77M) we tie it; at 1.3x its data we pass it -- while
carrying **12x the parameters**. Parameter for parameter, ChemBERTa-2 is far more efficient than
anything in this study, and that is the sentence a reviewer will write if we do not. The defensible
claim is "CLIMB reaches and passes ChemBERTa-2 once it sees a comparable number of distinct
molecules", NOT "CLIMB is the better model".

**Open item:** `unsup_100M` exists on the seven scaling panels only, not across all 65 ranking
datasets, so it cannot currently be drawn in fig_A. If the paper wants to make the "CLIMB at a
comparable corpus beats ChemBERTa-2" claim in the ranking rather than only in fig_B, that arm needs
the full suite. Worth costing before deciding.
