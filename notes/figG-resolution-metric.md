# fig_G: how the resolution metric was chosen, and why it changed twice

**STATUS, 2026-08-29.** fig_G no longer reports a ratio. It reports HELD-OUT CLASSIFICATION AUC:
label every parent molecule 0 and its edited partner 1, fit a gradient-boosted tree, report ROC-AUC
on molecules the model never saw. This note is kept because THE MEASUREMENT BELOW STILL STANDS --
it is the reason a dimension COUNT was rejected, and it is still the reason. What it got wrong is
narrower: it treated "not a count" as "therefore the ratio", and a third option existed that
neither the proposal nor this note considered.

Why the ratio fell, on the same standard of evidence this note demanded of the count:

* **Leif's objection, 2026-08-28.** XGBoost splits on SINGLE dimensions, so a model does not need a
  large shift, it needs ONE coordinate that separates the pair. A magnitude axis systematically
  understates a representation whose information is concentrated -- one flipped bit in 2048 is a
  negligible norm change however decisive it is. That argument is the same one this note quotes
  approvingly in favour of a ratio over a norm; carried one step further it disqualifies the ratio
  too.
* **It changed conclusions, not just presentation.** The HUME figures session measured CheMeleon at
  0.544 on stereo -- barely above chance -- where the ratio axis had called it strongly
  stereo-sensitive. The magnitude was real movement in an inconsistent direction.
* **Measured on our own data.** The random encoder moves FURTHER on para-vs-meta (cosine 0.0127)
  than on stereo (0.0042), so the ratio axis ranks it more regioisomer-sensitive than
  stereo-sensitive. Classification says the opposite -- 0.533 against 0.624 -- and is right: the
  displacement on para-vs-meta is arbitrary in direction and generalises to nothing.
* **Classification inherits neither defect this note identified in the count.** No threshold, and
  no degenerate calibration, because it divides by nothing.

The ratio's one real advantage, which the AUC gives up: it was interpretable without a reference
line, since 1.0 meant "as far as a different molecule". The AUC replaces that with a bound the
ratio never had -- a character n-gram floor drawn in every panel, so a reader can see what the
edit hands over for free.

WHAT SURVIVES FROM 2026-08-25, unchanged and still load-bearing: everything below. The count is
still rejected, for the reasons measured here.

---

# (2026-08-25) why the unit is a ratio and not a count

**2026-08-25.** A rewrite of the fig_G methodology was proposed that replaced the reported unit
with a **count of representation dimensions displaced by at least half of that dimension's
background standard deviation**, `n_res(A,A') = |{ j : |x_j(A) - x_j(A')| >= sigma_j/2 }|`, reported
as median and IQR on a logarithmic axis, with the ratio `d(A,A')/d(A,A_MW)` alongside.

Four of its five components were adopted (see below). The metric swap was **not**, and this note
records the measurement that decided it, so the question does not get re-opened from prose.

The proposal's own justification for a count is good and is now quoted in the methods: gradient
boosted trees subsample features at each split, so a change confined to one coordinate is often
never offered to the learner while a change spread over many is reliably found; and injectivity
is not a discriminating criterion because every deterministic representation satisfies it. Those
arguments motivate **a ratio rather than a norm**. They do not select a count.

## What the count actually measures

Computed on the existing pair set (100 pairs/mode, canonical input, same embeddings the figure
draws), median `n_res` at sigma/2:

| mode | ECFP4 | r3fp | CLIMB sup | CLIMB uns-canon |
|---|---|---|---|---|
| add_methyl | 15 | 32 | 56 | 152 |
| isotope_13c | 0 | 0 | 46 | 182 |
| **matched_mw (the reference)** | **28** | **34** | **364** | **418** |

ECFP4's response to a **completely different compound** is 28 of 2048 bits -- 1.4% of its
dimensions. CLIMB's is 418 of 512 -- 82%. Raw counts therefore report that CLIMB resolves an
unrelated molecule fifteen times better than ECFP4 does, and that one added methyl moves CLIMB ten
times further than it moves ECFP4. Both are artefacts of **density**: fingerprints are sparse by
construction, and the count is bounded by how many substructures an edit touches, not by how well
the edit is resolved. Scaling each dimension by sigma_j makes the *dimensions* commensurable; it
does nothing about how many dimensions a representation spends per edit.

## The threshold is a free parameter exactly where the argument lives

Median `n_res` for one added methyl, at three thresholds:

| arm | sigma/3 | sigma/2 | sigma |
|---|---|---|---|
| ECFP4 | 15 | 15 | 15 |
| Morgan r3-counts | 33 | 32 | 30 |
| CLIMB sup | 140 | 56 | 1 |
| CLIMB unsup (canonical) | 250 | 152 | 18 |

Binary fingerprint bits sit nowhere near any threshold (a flipped bit is >= 2 sigma for any
`p_j`), so their counts are flat. The CLM displacements sit **on** the knife edge: a 3x move in an
arbitrary constant moves the CLM by 140x. This is the same failure the current unit was written to
escape -- `scripts/resolution_effect_size.py` records that the previous threshold-calibrated
version "drove every class-A cell to 0%".

## The log axis cannot be drawn

36 of 91 cells have median `n_res` = 0, including **every** stereo, E/Z and ring-size cell for all
three CLM arms -- which is the figure's headline. The ratio unit puts those at 0.009-0.030 against
0.679-0.920, a measurable 20-100x. The count turns it into "0 versus 8": an undefined ratio, on an
axis that can render neither endpoint. Half of fig_G's cells are structural zeros (a fingerprint
cannot see a re-written string; Morgan invariants cannot see an isotope) and the figure labels them
explicitly for that reason.

## What was adopted

- **n = 1,000 pairs per mode**, up from 100. Feasible on every mode (pool: 44,667 unique).
- **Background = 10,000 unedited molecules**, up from ~1,000, for the sigma_j estimate, and now
  explicitly disjoint from every molecule under test.
- **Median + IQR per cell**, plus a joint percentile bootstrap of the median that resamples the
  matched-MW reference as well. The figure previously reported one number per cell with **no**
  dispersion at all, which left its arm-vs-arm claims unfalsifiable as drawn.
- **Paired arm-vs-arm contrasts** (`resolution_contrasts.csv`): every arm sees the identical pair
  list, so pair difficulty cancels and the marginal IQRs understate what the data supports.
- The **injectivity** and **tree-subsampling** arguments, in the methods, as the justification for
  reporting a ratio rather than a norm.

The one sentence not adopted as written is "computed on the edited form exactly as supplied,
without re-canonicalisation". The pipeline already does this and `figures/fig_G.py` asserts it per
class. The edits are made on the RDKit **mol object** and written out once, so class-A pairs are
canonical without anything re-canonicalising them; class B is emitted and read as written, which is
what keeps the notation controls from collapsing to no-ops. Applying "as supplied" to class A would
require string-surgery edits, which mix an uncontrolled notation change into every chemical one and
would flatter the CLM arms.

## If the count is wanted anyway

The form that works is `n_res(A,A') / n_res(A,A_MW)` -- the same count, normalised by the same
model's own reference, which cancels the density term. On stereo that reads ECFP4 0.29, r3fp 0.59,
CLIMB 0.00, consistent with the figure. It belongs in the SI as a threshold-based corroboration,
with its threshold sensitivity stated, not as the main axis.
