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


---

# PASS A COMPLETE — 2026-08-24 01:00 UTC

All twelve tables, all six fig_F panels, **one environment** (box `i-05dafbb7bd11cdc7f`, m5.8xlarge).

    s3://climb-s3-bucket/experiments/_figF/
      concat_{rdkit_sameenv,mordred}_{CLMunsup,CLMsup,CheMel}.csv          6 MolNet datasets
      concat_panels_{rdkit_sameenv,mordred}_{CLMunsup,CLMsup,CheMel}.csv   MoleculeACE + Ames

Verified 12/12 by counted tasks. Every table says `desc` in both families — the rename to `mdesc`
belongs to the figures session alone.

## One environment, proved rather than asserted

`fp` and `desc` contain no embedding, so within a descriptor family they must be identical across
all three tags. They are, bit-for-bit:

| family | scope | shared cells | disagree |
|---|---|---|---|
| rdkit_sameenv | MolNet | 20 | **0** |
| rdkit_sameenv | panels | 4 | **0** |
| mordred | MolNet | 10 | **0** |
| mordred | panels | 2 | **0** |

This is the check that *failed* at 0.22 fold SD between the published tables and a fresh run, which
is why the RDKit arm was regenerated rather than reused.

## The sharpest form of the result — Ames

| | RDKit desc | Mordred desc |
|---|---|---|
| baseline | 0.8430 | 0.8628 |
| + CLIMB unsup | 0.8211 | 0.8543 |
| + CLIMB sup | 0.8283 | 0.8507 |
| **+ CheMeleon** | **0.8586** | **0.8619** |

CheMeleon adds **+0.0156** on top of RDKit descriptors and **+0.0009** — nothing — on top of
Mordred. Same model, same panel, same folds. It carries information beyond RDKit descriptors and
essentially none beyond its own pretraining target. This result only exists because Mordred is on
the axis.

## Two bugs fixed mid-run

**Ames predictions were overwriting each other.** The panels script keyed its prediction directory
on the embedding *family* (`concat_climb`), not the table, so all four CLIMB panel runs wrote the
same file. Confirmed by inspection — the survivor held only `['desc','desc+CLMunsup']`. Every table
still written, every count still passing. Fixed at source (directory is now 1:1 with the output
table); lost predictions regenerated.

**The completion gate used `want=2` for panels.** Ames yields no in-process rows because Polaris
withholds test labels, so the gate called *complete* tables incomplete and refused to upload them —
the same "header-only by design" trap that withheld six finished Polaris dirs earlier that day. The
true count went 4/12 → 10/12 on fixing the **check**, with no change to the data.

## Caveats for the figure

- **15 molecules (0.015%) have all-NaN CheMeleon vectors.** The chemprop venv's newer RDKit rejects
  them — exotic organometallics, Al/Ge/B at unusual valences — while the loader's RDKit accepts
  them. Dropping them would have given the CheMeleon arm a *different molecule set* from the CLIMB
  arms, so they are NaN (XGBoost consumes as missing) and listed in
  `figure_data/_chemeleon_nan_molecules.json`.
- **Tox21 parses as 7,831 here vs 7,823 on the laptop.** Expected, and harmless because every
  fig_F cell now comes from this one box — but it is why local tables must not be mixed in.

## Pass B

Surplus blocks are running into `*_EXTRA.csv`, which cannot touch what the figure reads. Nothing in
the figure depends on them.

---

# V2 — 2026-08-24 23:04 UTC — READ BEFORE RE-RENDERING fig_F

Box `i-0bad233198a94150e` (m5.8xlarge), 16 cells, self-terminated with all five gates green.
Tables in `analysis/rigor/figF_v2/`, S3 `experiments/_figF_v2/`, environment recorded in
`_figF_v2/_environment.json` and `figure_data/_figF_v2_environment.json`.

## v1 and v2 CANNOT BE MIXED

Measured on the 30 embedding-free cells, which contain no embedding and must therefore be equal:

    27 of 30 differ.  median 0.38 fold SD, max 1.82 (BBBP fp roc_auc 0.8475 -> 0.8822)
    the lifts fig_F draws:  0.1 - 0.4 fold SD

The environment shift is the same size as or larger than the effect being plotted. **Re-render
everything from v2** — means, SDs, per-fold, Ames. Do not pair v2 folds against v1 means.

Nothing about the experiment changed. `eval_v2.py`, `heads_v2.py`, `descriptors_v2.py` are
byte-identical between the runs; same folds (seed 0, deterministic), same seeds, same npz tables,
same encoders, same S3 objects. The one uncontrolled variable is the **xgboost version**, unpinned
anywhere in the repo and pip-installed on a box that no longer exists.

## The v1 Ames headline does not replicate

|  CheMeleon lift over descriptors | RDKit | Mordred |
|---|---|---|
| v1 (published, in the caption) | +0.0156 | +0.0009 |
| **v2** | **+0.0052** | **+0.0075** |
| one Hanley–McNeil SE | 0.0097 | 0.0097 |

v1 read as *"real information beyond RDKit, essentially none beyond its own pretraining target"* —
a 17× asymmetry. In v2 the two lifts are the same size, in the opposite order, and **both sit
inside one standard error**. That sentence was not measuring CheMeleon; it was measuring the gap
between two environments. **It must come out of the caption.**

## What v2 adds

- **2,400 per-fold rows** in `*_folds.csv` — one row per (task, features, metric, fold). MoleculeACE
  keys on the TARGET name, its actual replicate unit. Ames emits none, correctly.
- **The Ames ECFP4+desc ticks v1 lost with its box.** On `fp+desc`, CheMeleon adds +0.0041 (RDKit)
  and **+0.0138 (Mordred)** while both CLIMB arms subtract. The Mordred figure is above one SE.
- **Full 15-block grid in both families**, `fp+desc+<TAG>` included.

## v2 is one environment, proved

`fp` is plain ECFP4 and must be identical across descriptor families: **11 shared cells, 0
disagreements.** On Ames the family-independent blocks agree exactly as well (fp 0.8378, CheMel
0.8640, fp+CheMel 0.8724 in both).

## Provenance gotcha worth carrying forward

**Two rdkit distributions are installed** — `rdkit 2025.9.2` and `rdkit-pypi 2022.9.5` — and the
one that *imports* is `rdkit-pypi`. Every ECFP4 fingerprint and RDKit descriptor came from
**2022.09.5** while `pip list` leads with 2025.9.2. Recorded as `rdkit_EFFECTIVE`. It comes from
the AMI so v1 shared it — not part of the v1→v2 shift — but anyone reconstructing either
environment from `pip list` would write down the wrong version and conclude they matched.
