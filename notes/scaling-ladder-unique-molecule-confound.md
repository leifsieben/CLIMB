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
