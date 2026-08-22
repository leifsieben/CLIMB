# What actually moves a number when nothing about the experiment changed

Three effects were measured on 2026-08-21, each by rebuilding a published cell and diffing it
against its own published copy. Same code, same seeds, same encoder weights, same data every time.
The only difference is the thing named.

| effect | max \|Δ\| | median \|Δ\| | measured on |
|---|---|---|---|
| Python environment (fresh venv, same pins) | **0.2196** | — | `unsup_8M`, encoder + MLP, 630 cells |
| Python environment, different head | 0.2339 | — | `unsup_8M__xgb`, encoder + XGB, 630 cells |
| CPU instruction set (c5 Intel vs g5 AMD) | **0.0308** | 0.000149 | `unsup_8M__xgb`, 630 cells, 168 exact |
| chemprop/CheMeleon featurizer, fresh venv | 0.0027 | — | `chemeleon_frozen__xgb`, 630 cells |

For scale, the thing these can contaminate:

| signal | typical size |
|---|---|
| pretraining-seed spread (inter-directory distance, MoleculeACE) | **~0.03** |
| gap between the #1 and #2 ranked arms | 0.020 of a rank |

## The two conclusions that follow

**A seed spread must be built on one instruction set.** The ISA maximum, 0.0308, is the same size
as the pretraining-seed effect a replicate trio exists to measure. This retires the older working
assumption — "~0.95% cross-architecture, probably fine" — with an actual bound. It is not a
correction to apply; it is a constraint on how a trio is produced. When the 12-cell `__xgb` rebuild
was split six ways for wall-clock, every box had to be the same instance type, and the only reason
mixed types were avoidable was that the standard on-demand vCPU limit forced all six onto g5 rather
than some onto c5.

**The drift is in the encoder forward pass, not in any head.** Two rebuilds settle it by
elimination:

    encoder + MLP   0.2196     no xgboost anywhere
    encoder + XGB   0.2339     no MLP anywhere
    chemprop + XGB  0.0027

If the drift lived in XGBoost the first would have reproduced; if it lived in the MLP the second
would have. Both moved by the same amount, so it is the component they share — the ModernBERT
forward pass under torch/transformers. `chemeleon_frozen` is featurized by chemprop, and its drift
is ~11× *below* the seed signal it would sit inside, so that arm needed no in-env rebuild.

## What is NOT here, and why the absence matters

No artifact in this repo records the environment it was produced in. `verified.json` carries
track, model, featurizer, head, seeds and n_tasks — no library versions, no instance type, no
image id. So none of the above is recoverable after the fact from the artifacts themselves; every
number in the table above had to be obtained by *rebuilding*, at roughly 40 minutes a cell.

That is the real cost. Whether the 08-13 bases and their 08-17 replicates shared an environment
could not be answered from provenance and had to be inferred statistically instead — from the fact
that a 0.22 shift would have stood out by an order of magnitude against a 0.03 inter-directory
distance, and did not appear.

## How to reproduce any row

Rebuild the published cell into a scratch directory at *its own* published seeds, and diff the
shared `(task, seed, subset, metric)` cells:

```bash
ARM=unsup_8M bash scripts/env_drift_locate_run.sh    # tests A and B
```

Report the **distribution**, not one statistic. An earlier version of that script printed only a
max and then deleted its scratch directories on the reasoning that a diagnostic produces answers
rather than artifacts. When the per-cell distribution behind test B's 0.0027 was wanted, it was
gone, and the conclusion survived only because a max happens to bound every cell — luck about
which statistic had been printed, not a property of the design. For a diagnostic whose *output is
the finding*, the rebuilt cells are the evidence. They are now preserved under `_diagnostics/`.

## Preserved inputs

    _diagnostics/c5_intel_unsup_8M__xgb/            the c5 cell behind the ISA row
    _preserved/published_xgb_bases_20260821/        suite bases, pre-rebuild
    _preserved/published_xgb_molnet_cbs_20260821/   MolNet/CBS bases, pre-rebuild
    _preserved/polaris_scores_PREREBUILD_20260819/  the stale score files
