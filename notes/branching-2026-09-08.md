# Work on `main` from now on

**2026-09-08.** `main` and `v2-redux` are identical as of `692889e`. Leif's instruction: all further
work goes on `main`.

## Why this changed

`main` had been 494 commits behind `v2-redux` — the entire v2 history, including `figures/`, the
evaluation battery and `METHODS.md`, existed only on the branch. That was invisible from outside:
anyone cloning the public repository got `main` and therefore got none of it, and the paper's Code
and Data Availability statement pointed at a repository URL that resolved to stale code.

It is now fast-forwarded. Both branches point at the same commit and there is no divergence to
reconcile.

## What this means in practice

- Commit to `main`. Do not start new work on `v2-redux`.
- The Hugging Face cards link to `https://github.com/leifsieben/CLIMB/blob/main/...`. Those links
  break if a file they name is only ever committed to a branch.
- The paper cites a commit on `main`. Tag a release when the figures are final, so the citation
  names a fixed tree rather than a moving branch tip.

## Related state as of this date

All three Hugging Face repositories are public and current:

| Repo | Contents |
|---|---|
| [`lsieben/climb-encoders`](https://huggingface.co/lsieben/climb-encoders) | 157 encoders, including the c124 rungs and `skip_dense_100M_c124` |
| [`lsieben/climb-results`](https://huggingface.co/datasets/lsieben/climb-results) | per-run evaluation outputs, including this week's five arms |
| [`lsieben/climb-pretrain-data`](https://huggingface.co/datasets/lsieben/climb-pretrain-data) | pretraining corpora, including the wiki, bigram and unigram ladder corpora |

Note that `scripts/publish_to_hf.py` uploads only the waves in its `PAPER_WAVES`. Work produced
outside those waves is silently absent from the Hub even after a successful `--execute`, which is
how five arms and three corpora came to exist on S3 only. Verify by listing the repository
afterwards; the upload call returns success either way.
