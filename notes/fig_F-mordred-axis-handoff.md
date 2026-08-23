# fig_F: add Mordred as a second descriptor axis

Handoff for whoever owns the figures next. Leif asked for Mordred alongside RDKit on fig_F's
x-axis, with **the figure's logic unchanged** — hold one block fixed, add one thing — and an
apples-to-apples comparison. The data is computed and committed; one piece is in flight.

## Why Mordred deserves its own axis

CheMeleon is a D-MPNN pretrained to regress exactly the **1,613 Mordred 2D descriptors**. Against
RDKit descriptors, "CheMeleon is redundant to fp+desc" compares two unrelated feature families and
is only suggestive. Against Mordred it is the sharp version: *if CheMeleon adds nothing on top of
its own pretraining target, it compressed that target rather than learning past it.*

`Calculator(descriptors, ignore_3D=True)` yields 1613 — that set exactly, not an approximation.

## The result

Deltas in units of the fold SD; positive means the embedding helped.

| onto Mordred | CLIMB-unsup | CLIMB-sup | CheMeleon |
|---|---|---|---|
| **BBBP** | **+1.8 sd** | **+1.6 sd** | **+0.1 sd** |
| ESOL, QM7, BACE, Tox21, HIV | all within ±0.6 sd, signs mixed | | |

On top of `fp+Mordred` the picture is the same, and **CheMeleon never exceeds +0.7 sd on any task
against either baseline.**

Two findings:

- **CheMeleon adds essentially nothing to Mordred, anywhere.** It behaves like a compression of its
  own pretraining target rather than a representation that learned beyond it.
- **BBBP is the sole exception in the entire grid.** Both CLIMB encoders — independently pretrained
  — add ~0.05 AUC at 1.6–1.8 sd, where CheMeleon adds 0.003. Whatever BBBP rewards, Mordred does
  not encode it and CheMeleon did not learn it.

## THE APPLES-TO-APPLES PROBLEM — read before plotting

**Do not draw the Mordred column against the *published* RDKit column.** They come from different
environments.

Tested on the one block that shares identical code and contains neither descriptors nor an
embedding — plain `fp`:

    fp, QM7 rmse:   tonight 216.1573    published 216.6266    |Δ| 0.4693  = 0.22 fold SD
    across all 10 fp cells:  0.01 – 0.22 fold SD

That is small, and ~10× below the BBBP effect, so it does not threaten the finding. But it is a
pure environment term that would sit inside every RDKit-vs-Mordred difference the figure draws, and
the figure's logic is *change exactly one thing*. Changing the descriptor family **and** the
environment is two things.

So the RDKit arm is being regenerated in the same environment — same script, same folds, same
seeds, same machine, same hour, only `CONCAT_DESC` differs (`scripts/rdkit_sameenv_run.sh`).

**Pair these, not the published tables:**

    analysis/rigor/concat_mordred_{CLMunsup,CLMsup,CheMel}.csv          committed 8a71189
    analysis/rigor/concat_rdkit_sameenv_{CLMunsup,CLMsup,CheMel}.csv    same-env RDKit arm

## Gotcha: tag mismatch

My unsup tables use `CLMunsup`; the published table uses `CLM`. Same arm
(`figure_data/climb_v2_phase2/unsup_8M/encoder`), different string — and fig_F's lattice keys on
these strings directly. **Rename on read rather than re-running.** Do not remap `CLMsup` or
`CheMel`; those already match.

## Caveat to carry into the figure

BBBP sits at 0.95 AUC on a scaffold split, and the pipeline reports `nef1` of **exactly 1.0** there
(visible in the `fp` block too). A perfect enrichment score is worth auditing on its own terms
before that +1.8 sd gap is read as representation quality. It may be a split artefact — and it is
the single cell carrying the only positive result in the grid.

## Side observation, free with the data

Mordred alone beats `fp+Mordred` on 4 of 6 tasks, and beats plain ECFP by a wide margin (ESOL 0.72
vs 1.51). Once 1,613 descriptors are present, the fingerprint is mostly dilution. If fig_F has
room, `desc` alone vs `fp+desc` is a cheap and slightly awkward panel for the fingerprint.

## Method

The table is 59,665 × 1,613, built once by `scripts/compute_mordred.py` and reused by all three
models. It is keyed on the **reference environment's** SMILES with a **strict** lookup that raises
on a miss rather than mean-filling — which is why this experiment stays local: a box whose deepchem
parses Tox21 as 7,831 rather than 7,823 molecules would raise loudly instead of quietly scoring a
different molecule set.

36 of 1613 columns are entirely NaN (descriptors undefined on every molecule); 11.6% of entries are
NaN overall. XGBoost consumes NaN natively, as the RDKit block already relies on.

Nothing here is wired into any figure yet.
