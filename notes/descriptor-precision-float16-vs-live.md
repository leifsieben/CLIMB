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
