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
