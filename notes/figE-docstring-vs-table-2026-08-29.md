# fig_E: the docstring contradicts its own table on the paper's most quotable sentence

**Status: FOR THE FIGURES SESSION.** Found 2026-08-29 by the AWS session. Nothing in `figures/` or
`figure_data/` has been changed — this is a report, not a fix.

## 1. The claim that is wrong

`figures/fig_E.py`'s module docstring says Wikipedia "beats the real corpus on MoleculeACE".
`figure_data/fig_E/fig_E_lift.csv` says the opposite:

| MoleculeACE (RMSE, lower better) | mean | sd | lift |
|---|---|---|---|
| real (unsup_8M) | 0.778069 | 0.001237 | **+6.39%** |
| wiki (wiki_real_8M) | 0.781823 | 0.003270 | +5.94% |

Real is better. It is not only MoleculeACE: **real beats wiki on all six panels**, and real is the
best rung on 5 of 6 (QM7 goes to `shuffled`, +3.85 vs real +2.86).

The docstring's other claim — shuffled is "the BEST rung on 3 of 6 panels" — **is** defensible read
as *best among the corrupted rungs* (BACE, Tox21, QM7). Not flagged.

A defensible replacement headline, still striking and referee-proof: **zero-chemistry Wikipedia
recovers ~60% of real chemistry's benefit** (mean lift over six panels 4.44 vs 7.53).

`_audit_docstring()` re-derives the counts in that docstring but evidently not this comparison —
worth extending, since this is the sentence a referee checks first. See
[[absence-claims-go-stale-silently]] for the same shape: prose asserting a relation that cannot
fail loudly when the table moves under it.

## 2. Where wiki is actually good

Wiki's lift as a fraction of real's, per panel:

| MoleculeACE | QM7 | HIV | BACE | Tox21 | Ames |
|---|---|---|---|---|---|
| 93% | 99% | 70% | 30% | 14% | −3% |

The zero-chemistry benefit is concentrated on the two RMSE panels and HIV, and is essentially
absent on the small binary classification sets.

## 3. HIV: wiki's advantage survives the metric change, shuffled's does not

expA/expB carry ROC-AUC for HIV as well as NEF1. Same runs, floor 0.7499:

| arm | HIV ROC-AUC | lift | % of real |
|---|---|---|---|
| real | 0.7689 | +2.53% | 100% |
| wiki | 0.7624 | +1.67% | 66% |
| bigram | 0.7549 | +0.67% | 26% |
| shuffled | 0.7522 | +0.31% | 12% |
| unigram | 0.7471 | −0.38% | −15% |

On NEF1 wiki and shuffled tie at 70% of real; on ROC-AUC wiki keeps 66% and shuffled collapses to
12%. If the ladder's story is "shuffled tokens keep most of the benefit", that is **metric-dependent
on HIV**.

Also worth a caption line: HIV's bars dwarf every other panel largely because NEF1 is a top-1%
count metric. The same runs move 0.3–2.5% on ROC-AUC versus 5–22% on NEF1.

## 4. A near-miss, recorded because it nearly went the other way

In the table, wiki and shuffled both show HIV mean `0.624900` — which looks exactly like the
provenance collisions found repeatedly this week. **It is not one.** expA's `shuffle_tokens` is
0.6248995984; wiki's own-tree mean is 0.6248996667. Different numbers agreeing to six decimals.
NEF1 on HIV is a small-count ratio, so it lives on a lattice of rationals and near-collisions are
far likelier than for a continuous metric. Stopping at the rounded display would have produced a
confident false bug report.
