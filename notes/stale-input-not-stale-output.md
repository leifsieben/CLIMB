# The check that catches a derived file built from the wrong copy of its input

On 2026-08-21, `fp_desc`'s Polaris `polaris_scores.csv` held numbers for a different run than the
predictions sitting beside it — 245 of 249 rows wrong, on the arm that is second on the Ames panel.
It passed every check we had.

## Why it passed

    complete   249 rows, 28 tasks, 0 NaN  -- a stale file is also complete
    fresh      NEWER than every file in its directory -- the write really happened
    valid      a correct, in-range scoring of a REAL prediction set
    plausible  Ames 0.8702 against a true 0.8687; nothing looks wrong

Nothing about the artefact is malformed. It is a perfectly good answer to a question nobody asked.

## The mechanism

`aws s3 sync` decides whether to download by comparing SIZE and MTIME. The pre-stereo and
stereo prediction files have the **same byte size** — same molecules, same columns, different
values — so sync judged the local copy equal and skipped it. The scorer then did its job flawlessly
on the wrong input.

**A sync that skips is indistinguishable from a sync that succeeds**, and the local file it left in
place was the one thing in the tree nobody thought to doubt.

## Why the obvious checks cannot work

- **Completeness** — a stale file is complete. Row counts describe both the file you meant to write
  and the file that was already there.
- **Freshness** — the write happened, after the predictions. Timestamps say the opposite of the
  truth here.
- **Range / sanity** — the value was 0.0015 from correct. No bound catches that, and one loose
  enough to would fire constantly.

## The check that does work

Recompute the derived file from the inputs **actually present** and diff.
`scripts/verify_polaris_scores.py` does this: it scores into a temporary copy, never touching the
directory, so it is safe against a tree another session is working in.

It is too expensive to be a per-run gate. It is the right check **whenever a derived file could have
been produced from a different copy of its input** — which is any time two machines hold the same
tree, i.e. this project's normal state.

## What made detection possible at all

An independent vintage to compare against. `fp_desc_PRESTEREO` existed only because the pre-stereo
directories were quarantined rather than overwritten, and the tell was that the two scores files
were byte-identical while every prediction differed. Without that copy, `fp_desc Ames 0.8702` would
have looked exactly like a result. Quarantine the old vintage; it is cheap and it is sometimes the
only witness.

## Corrected values

Ames, all four anchors pooled over 3 dirs: ecfp4 0.8385, fp_desc **0.8691** (was 0.8696),
ecfp4_r3c 0.8523, fp_desc_r3c 0.8742. Only `fp_desc` moved and the panel ordering is unchanged.
The six replicate directories were never stale — they did not exist locally before the run, so
there was nothing old to score.

---

## Postscript: what the shutdown scan found (2026-08-21)

Before the boxes came down, both were scanned for directories holding results that exist nowhere
else. `climb-chemeleon-jobs` was clean. `climb-a2-bootstrap` held four, now preserved under
`s3://climb-s3-bucket/experiments/_quarantine/`:

    chemeleon_suite/{moleculeace,polaris}/skip_dense_8M_e2e_tuned
    cbs_benchmark/{skip_dense_8M_e2e,unsup_8M_e2e}_SMOKE_STUB

None is a result. The `_tuned` pair is a **4-task probe** (`n_tasks: 4` against the real arm's 30),
and its `verified.json` records `"model": "skip_dense_8M_e2e"` — the REAL arm's name, not its own.
So a partial probe describes itself as the published arm, which is the same failure family as
`"featurizer": "ecfp4"` naming three featurizations: the artefact answers confidently and stops you
looking. They are kept under `_quarantine/` rather than beside the real runs precisely so the name
in the file cannot be mistaken for the name of the directory it sits in.

The `_SMOKE_STUB` pair is the documented 1-fold, 2-epoch smoke test with NEF1 0.0 that readers must
never fall back to.

Preserved rather than deleted because "verify before terminating" does not mean "verify it is
valuable" — it means verify it exists somewhere else first, and then decide.
