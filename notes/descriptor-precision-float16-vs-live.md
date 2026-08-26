# The notation bridge computes descriptors LIVE; the rung it bridges read them PRECOMPUTED

2026-08-26. Measured by the compute session, recorded here because it is a real difference between
two arms of one contrast and the decision not to erase it should be findable later.

## The asymmetry

    skip_dense_8M        pubchem_filtered   descriptors PRECOMPUTED, stored float16
    skip_dense_8M_c124   pubchem_124m_full  descriptors computed LIVE, float32

Live computation is forced, not chosen: the two corpora name their shards identically, so a shared
precompute directory would attach the wrong molecules' descriptors to the c124 shards. That is a
worse failure than a precision difference -- silent and total rather than small and bounded.

## The size of it, measured rather than assumed

1,000 molecules of pubchem_filtered, live float32 against the July float16 precompute:

    NaN pattern           IDENTICAL
    median |dz|           3.4e-5
    99.9th percentile     0.0017
    cells > 0.01 z        11 of 216,818  (0.005%)
    worst cell            AvgIpc, 0.98 z

The tail is the Ipc family specifically: those values are large, and float16 keeps ~3 significant
digits. So it is a SYSTEMATIC effect on one descriptor family, not random noise -- but it lands on
roughly 1% of molecules in one target of 217.

## Decision: one SI sentence, no footnote on the figure, no recompute

The bridge exists to isolate NOTATION (0.3% vs 86.4% lowercase-aromatic SMILES). A corpus-level
contrast cannot see 0.005% of target cells, and re-precomputing 124 shards to erase it costs ~24
CPU-hours to buy nothing measurable.

Write it as EVIDENCE rather than as a limitation, because that is what it actually is: it shows the
current rdkit reproduces the July fleet's descriptor VALUES and not merely its descriptor NAMES.
The name check alone would pass an environment that computed different numbers.

## What made this findable at all

The same launch caught a shadowed `rdkit-pypi 2022.9.5` exposing 208 of 217 descriptors. That one
was loud only because the COUNT differed and numpy refused to broadcast. A version that merely
REORDERED the list would have z-scored every descriptor by another descriptor's statistics, trained
happily, and produced a number silently incomparable to the rung it bridges -- the fig_F v1/v2
shape exactly, one computation split across two environments.

The gate now compares descriptor names IN ORDER against the fitted normalization stats, and the
width against what the run template RECORDED rather than against a constant. Same principle as
[[absence-claims-go-stale-silently]]: assert the condition, do not quote the number.

---

# The normalizer question, and the bug it found one script over (2026-08-26)

I asked whether the bridge z-scores its live float32 values with the JULY-FITTED stats or refits on
c124, because a refit normalizer is a second difference inside a one-difference control and nothing
would error. Answer: **July stats**, established four ways rather than asserted --

    digest   stats file on the box sha1[:12] 1a77d80d06d8 == canonical S3 object, byte identical
    mtime    both stamped 2026-07-16T12:17:21Z; the refit branch calls save_stats(), which
             REWRITES that file, so an unchanged mtime is positive evidence it never ran
    log      the refit branch's "fitting descriptor stats..." line appears 0 times
    path     bridge config.yaml `descriptor_stats_path` is the same string as skip_dense_8M's

`metadata.json` now carries `mtr_stats_sha1` beside `mtr_n_desc` -- a digest doing for the
normalizer's VALUES what the width field already did for the descriptor COUNT. Runs after that
commit state it in their own artefact; for the bridge, the four items above are the record.

## The live one it turned up

`precompute_descriptors.py` refit the stats whenever the file was merely absent LOCALLY, then
uploaded that fit OVER the canonical S3 copy with `check=True` -- silently renormalizing the target
space for every run in the project. A fresh box with an empty `configs/` was the whole trigger.
Now fixed: fetch canonical first, refit only if nothing canonical exists anywhere.

**The blast radius is closed by evidence already in hand, in one direction.** The canonical object's
LastModified is unchanged since 2026-07-16T12:17:21Z, so no PUT has landed on it since -- the bug
cannot have fired after that date. What that timestamp does NOT settle is whether it fired ON that
date, overwriting an earlier canonical file. That only matters for a rung that began before it;
every rung in the ladder is later, so the question is closed, but it closed on a date comparison
rather than on the mtime alone.

Worth keeping as the shape: the evidence gathered to answer one question retired a second, larger
question that was not asked. [[absence-claims-go-stale-silently]]
