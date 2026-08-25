# Failures that produce a complete, well-formed, wrong artifact

Three from the fig_A build, 2026-08-25. They share a shape: **every gate passed, nothing errored,
and the output looked exactly like success.** They are recorded because the fix in each case was
not "be careful" but "test the property the pipeline is assumed to have".

## 1. A featurizer that returned the same vector for every molecule

`selfies-ted`'s tokenizer expects **space-separated** SELFIES — `[C] [C] [O]`, not `[C][C][O]`.
Fed the unspaced string it emitted **three tokens for every molecule regardless of size**, and
every embedding came out near-identical:

| input | tokens | per-dim sd | mean off-diagonal cosine |
|---|---|---|---|
| unspaced | 3 | 0.0000 | **1.0000** |
| space-separated | 26 | 0.2444 | 0.6239 |

It produced a well-formed npz, a complete 30/30 MoleculeACE run, uploaded artifacts, and a score of
**1.2171 macro RMSE — last place by a wide margin.**

**A representation that cannot distinguish any two molecules is, from the outside, indistinguishable
from a representation that is simply bad.** And "the new model came last" is exactly the result
nobody interrogates. It was caught only because the number was *too* bad to believe.

*Fix:* `extract_clm_embeddings.py` now **measures separation** — median per-dimension sd, dead
dimensions, and mean off-diagonal cosine over a 512-molecule sample — and refuses to write a table
whose vectors do not distinguish molecules.

Related: the model is a BART, so `AutoModel(...).last_hidden_state` is the **decoder** output, not
the representation the model exists to expose. `--encoder_only` uses the encoder, which separates
molecules better (0.62 vs 0.76 cosine).

## 2. A fix for one code path that silently opened the same hole in another

`verified.json` was fixed to record `hf_model` and `hf_revision`, because a runtime pin that is not
written down is unrecoverable from the artifact. Two commits later, routing arms through
`--featurizer npz` **bypassed the place that record is written**: the file said `featurizer: "npz"`
and a path — strictly *less* than the direct path had just been taught to say.

**This failure mode has no natural detector.** `npz` was a legitimate route, nothing about the
resulting artifact looks wrong, and the earlier fix was real and still worked on its own path.

*Fix:* the npz carries its own provenance — model, revision, tokenizer source and revision, pooling,
`encoder_only`, `max_length`, hidden size, **parameter count**, library versions, and the two
separation statistics — and the runner reads it into `verified.json` under `npz_provenance`.
Provenance travels with the vectors instead of depending on which route reaches the writer.

## 3. A gate that checked one of the two properties it implied

The skip test for an existing feature table checked **coverage only**. A table extracted before the
meta blob existed covers all 177,922 molecules and still cannot say what produced it — so it would
be skipped forever **while reporting `SKIP`, which reads as success.**

Coverage and provenance are independent properties. *A gate that checks one while implying both is
worse than no gate, because it is trusted.* It now tests both, and on the next pass it correctly
re-extracted exactly the two tables that had coverage but no meta.

## A distinction worth keeping: which environment splits are fatal

Not every "two environments" is the fig_F v1/v2 mistake, and reflexively avoiding all of them would
have blocked three arms.

| | what happened | verdict |
|---|---|---|
| **fig_F v1 vs v2** | the *same* feature blocks (`fp`, `desc`, `fp+desc`) computed under two environments and compared against each other | **fatal** — 27 of 30 shared cells moved, median 0.38 fold SD |
| **fig_A featurization** | each arm's vectors come from *its own model's* forward pass; no shared computation exists for a library version to perturb | **safe** |

What must be common is the **probe**: featurize wherever the model loads, hand over an npz, and
score every arm in the pinned venv through the same MLP head. Then the comparison stays a
comparison of representations.

The one case that *is* the fig_F shape: three literature CLMs whose vectors came from two different
transformers versions, ranked against each other. That is a split **within** a compared group, and
it was fixed by re-extracting all three in one environment.
