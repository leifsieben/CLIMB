# The scaling ladders do NOT scale unique molecules for the supervised objectives

**2026-08-26, Leif:** *"Take this insight as a criticism of our scaling laws: we do scale in terms
of unique molecules as well right? For both unsup and sup?"*

**No.** Only the unsupervised ladder does. This is a live confound in the scaling result, not a
caveat, and it is reviewer-facing.

## What each ladder actually sees

The base corpus is 12M filtered molecules. A rung at more than 12M forward passes is RE-READING it.
Only two rungs in the entire study draw from the 124M full corpus, and both are `unsup`:

| ladder | rungs | on the 124M corpus | max unique molecules |
|---|---|---|---|
| `unsup` | 2M, 8M, 24M, 48M, **50M**, **100M** | **2 of 6** | **100M** |
| `sup_dense` | 2M, 8M, 24M, 48M, 96M | **0 of 5** | **12M** |
| `sup_dense_sparse` | 2M, 8M, 24M, 48M | 0 of 4 | 12M |
| `sup_sparse` | 2M, 8M, 24M, 48M | 0 of 4 | 12M |
| `u2s_dense` | 2M, 8M, 24M, 48M | 0 of 4 | 12M |

On the unique-molecule axis every supervised ladder has **three real points (2M, 8M, 12M) and then
stops**, while `unsup` has five (2M, 8M, 12M, 50M, 100M).

## The two claims this breaks

**1. "The supervised objective saturates."** It saturates exactly where its unique-molecule count
stops moving, and nowhere else. `sup_dense` on MoleculeACE:

    unique  2M  -> 0.7923
    unique  8M  -> 0.7741
    unique 12M  -> 0.7687 (24M FP), 0.7674 (48M FP), 0.7748 (96M FP)   <- three points, one x

Its top three rungs are the same amount of data read 2x, 4x and 8x. Against forward passes that
looks like a plateau; against unique molecules it is a single point with a repetition sweep through
it, and the last real step (8M -> 12M) was still improving. **We have not shown MTR saturates on
data. We have shown it saturates on repetition.**

**2. "MLM scales better than MTR."** At MATCHED unique molecules the ordering is the opposite:

    2M unique   MTR 0.7923   MLM 0.7826   MLM ahead
    8M unique   MTR 0.7741   MLM 0.7766   MTR ahead
    12M unique  MTR 0.7674   MLM 0.7774   MTR ahead

MLM only overtakes at 50M and 100M unique -- rungs MTR was never given. The comparison at the top
of the ladder is objective CONFOUNDED WITH corpus, and the confound runs in MLM's favour.

## The figure does not expose this

README section 3 states the scaling figures plot against **both** forward passes (A2.a) and unique
molecules seen (A2.b) "precisely so the two regimes are never conflated". `figures/fig_B.py` plots
pretraining TOKENS only; there is no unique-molecule axis in fig_B or fig_A2. The safeguard the
design called for was never built, and the x-axis that is drawn is the one that hides the problem:
`skip_dense_96M` sits at 4.12B tokens, far to the right, while representing the same 12M molecules
as the rung three positions to its left.

## Verified from run artifacts, not from the README (compute session, 2026-08-26)

I had taken the corpus split and the no-repetition claim from README section 3. Both were checked
against the runs themselves and both hold:

- **Corpus split** is in each rung's own `config.yaml`: `unsup_50M` and `unsup_100M` carry
  `unsupervised_data_paths -> pubchem_124m_full`; `unsup_8M` and every `skip_dense_*` rung carry
  `pubchem_filtered`.
- **Sampling is without replacement and the stream does not cycle**: `data.py:__iter__` yields from
  the iterator and RETURNS on StopIteration rather than re-creating it, so within one pass a
  molecule cannot repeat. 50M/100M forward passes from a 124M-molecule corpus really are 50M/100M
  DISTINCT molecules. The unique-molecule axis is correct as drawn.
- **Repetition above the corpus size is real and comes from the epoch loop**, not the sampler:
  `skip_dense_96M` reports `final_fp 95,994,624` against a 12M-molecule corpus, `skip_dense_48M`
  reports `47,997,696`. A single pass cannot yield either. ~8x and ~4x.

## Cost of the fix, from measured throughput

Supervised rungs run at **1,316 s per 1M forward passes** (skip_dense_48M 63,177 s / 48M and
skip_dense_96M 126,295 s / 96M -- the two agree to three figures). MLM is cheaper per pass at
~850 s/1M (unsup_50M 43,617 s / 50M, unsup_100M 83,251 s / 100M), which is consistent with MTR
also computing 217 descriptor targets.

    skip_dense_50M_c124    ~18.3 GPU-hours    tests the claim
    skip_dense_100M_c124   ~36.6 GPU-hours    makes the ladder symmetric, changes no conclusion

This is PRETRAINING, an order of magnitude above the frozen-probe runs, so it is a spend decision
rather than a queue item.

## What would fix it, cheapest first

1. **Draw the unique-molecule axis** (fig_B panel b, as A2.b already specifies). Costs nothing --
   the numbers exist -- and it makes the cap visible instead of hiding it. It also makes the honest
   version of the saturation claim legible: repetition past ~12M buys nothing.
2. **Mark the capped rungs** in fig_B (open vs filled, as the big-corpus rungs already are) so a
   reader cannot mistake 96M forward passes for 96M molecules.
3. **Run `skip_dense_50M_c124`** -- one supervised rung on the 124M corpus, matching `unsup_50M`.
   That single run converts "MLM scales better" from confounded to tested. It is the same shape as
   `unsup_8M_c124`, which already exists as the matched control at the bottom of the ladder.

Until (3) exists, the paper should not claim the unsupervised objective scales better with data;
it can only claim it was the one given more data.

Related: [[why-chemberta-beats-climb-sup-dense]] -- the same cap is why ChemBERTa-2 outranks
`sup_dense` in fig_A, and `unsup_8M_c124` is the control showing the corpus itself is a wash at
matched unique molecules.

---

# I walked into this note's own trap, and then over-read a variance (2026-08-26)

Recorded because both halves were mine, both looked rigorous, and neither would have failed.

## Half one: I compared across two axes at once

Wong landed for unsup_100M and I wrote that its +0.0754 over the `unsup` arm was worth putting in
the text. The compute session checked both runs' CONFIGS rather than the arm table:

    unsup_8M, _s1, _s2     pubchem_filtered      8M FP     0.3% lowercase-aromatic, ~12M molecules
    unsup_100M             pubchem_124m_full   100M FP    86.4% lowercase-aromatic, 123.4M molecules

Different budget AND different corpus AND different notation. So the gap is scale plus corpus plus
notation -- which is exactly the confound the rest of this note is about. I had written the note,
cited it in three figures that week, and still made the comparison, because the two numbers came
out of one table looking like siblings.

THE TABLE IS WHAT DID IT. `wide_table` returns one value per (arm, dataset), and nothing in that
shape records that two rows differ in their pretraining corpus. A confound that is documented in
prose and absent from the data structure will be re-made by whoever reads the structure -- which is
eventually always someone, including its author.

The clean pairs, once unsup_8M_c124 is scored:

    unsup_8M_c124  vs  unsup_100M     same corpus, 8M -> 100M FP     = BUDGET
    unsup_8M       vs  unsup_8M_c124  same budget, both corpora      = CORPUS + NOTATION

Any text claim about the 100M rung rests on the first row, not on 0.4034 vs 0.3280.

## Half two: a variance from three runs is barely a variance

Having measured pretraining-seed SD on Wong at 0.0118 from the three `unsup` dirs, I wrote that the
gap was "6.4x the pretraining-seed SD". The arithmetic is right and the inference is not: an SD
estimated from n = 3 has an enormous sampling interval. Chi-square, 2 df, recomputed here rather
than taken on trust:

    s          0.0118
    95% CI     sigma in [0.0062, 0.0744]
    gap        0.0754
    gap / s    point 6.4x,  95% range [1.0x, 12.2x]

The upper end of the interval says the gap is ONE pretraining SD. The data cannot distinguish
"clearly outside seed noise" from "exactly seed noise".

What survives is the measurement WITH ITS N: "pretraining-seed SD on Wong is 0.0118 from three runs
(95% CI 0.006-0.074)". That lets a reader see the estimate and how little it rests on. The ratio
hides the second half behind a single confident number.

## The shape, which is the reusable part

Both halves are the same species as the SELFIES-TED note earlier the same day: **the evidence was
real, the conclusion may well be right, and the reasoning had a gap.** Nothing errors, no check
fires, and the output looks like the output of sound work. It is worse than an ordinary bug for
exactly that reason -- a later reader inherits the METHOD, and the method is the broken part.

Two guards that would have caught these, in order of how much they cost:

  * Before comparing two arms' VALUES, diff their configs, not their names. Cheap, mechanical, and
    it is the check that caught this one.
  * Before quoting a ratio to a spread, ask how many samples the spread came from. Under about 10,
    quote the interval instead of the ratio.

Related: [[numeric-repro-bounds]], [[anchors-need-model-seeds]], [[replicate-axis-depends-on-arm]].

---

# The c124 tokenizer is fitted to the OTHER corpus, and what that does is measurable

2026-08-26. Confirmed by the compute session from the configs, not the naming: `tokenizer_10M`,
vocab 1000, hidden 512, 12 layers -- IDENTICAL across unsup_8M, unsup_8M_c124, unsup_50M and
unsup_100M. So both clean contrasts above have exactly two axes and not a third.

## But the tokenizer was fitted on the small corpus

`tokenizer_10M` was fit on a 10M sample of pubchem_filtered, which is 0.3% lowercase-aromatic. The
c124 runs then train on text that is 86.4% lowercase-aromatic. Every arm on both sides of both
contrasts uses the same tokenizer, so this is NOT a confound between arms -- but it means the
CORPUS row measures "different corpus, read through a vocabulary fitted to the other one", and a
penalty there has tokenizer mismatch as a live explanation rather than an excluded one.

## The obvious symptom is absent, measured

A vocabulary with no merges for lowercase aromatics would fragment `c1ccccc1` toward single
characters and inflate sequence length. Tokens per forward pass, from each run's own trainer count:

    unsup_2M     filtered    42.93
    unsup_8M     filtered    42.91
    unsup_24M    filtered    42.91
    unsup_48M    filtered    42.91
    unsup_50M    c124        40.55
    unsup_100M   c124        40.38

The four small-corpus rungs agree to four significant figures, which is the sanity check on the
measurement. The c124 rungs are ~5.5% SHORTER, not longer -- the opposite direction from severe
fragmentation.

This BOUNDS the concern rather than eliminating it: the two corpora hold different molecules, so
part of that 5.5% is a size distribution rather than tokenization quality. What it rules out is the
large effect. A vocabulary genuinely unable to represent the majority notation would not produce
sequences 5% shorter than the corpus it was fitted on.

## Which way it cuts, and it is not the flattering direction by accident

unsup_100M places 2nd of 14 in fig_A while being tokenized by a vocabulary fitted to a different
notation. Whatever that costs, the arm is carrying it. So the big-corpus result is CONSERVATIVE --
the honest framing is "despite", not "with an advantage". Worth one clause if the 100M rung is
claimed, alongside the fact that it is a single pretraining run.

## The durable fix for the confound above, from the compute session

My guard was "diff the configs before comparing two arms", and their reply is that a discipline is
the thing that fails. The structural version: carry corpus and budget as FIELDS ON THE ROW rather
than as knowledge about the row, so two rows differing on more than the axis under comparison are
visibly different at the point of comparison. Then the guard is in the shape, not in the reader.

Not today's work. The moment it stops being hypothetical is when fig_B's ladder gains a c124 row
beside a pubchem_filtered one.
