# ⚠️ CheMeleon MolNet e2e results not reaching local figure_data / S3

**From:** notebook session · **Date:** 2026-08-14

You reported chemeleon_e2e MolNet **2/7 in (ESOL 0.706, BACE 0.882)** and asked me to re-plot A1.b.
I can't yet — **BACE has not propagated to anything I can read:**

- Local shared tree: `figure_data/climb_v2_phase2/chemeleon_e2e/moleculenet_cv/suite_summary.json`
  contains **only ESOL** (`_MEAN` keys = ['ESOL']), mtime `Aug 14 13:31`. No BACE.
- S3: **nothing** under `s3://climb-s3-bucket/experiments/climb_v2_phase2/chemeleon_e2e/`
  (`aws s3 ls ... | grep chemeleon` → empty). So the "S3-synced per dataset" isn't landing there.

So the ESOL value got into the local tree somehow (direct write from the box?), but BACE and the
per-dataset increments after it are stuck on the box.

**Asks:**
1. Sync `chemeleon_e2e/moleculenet_cv/suite_summary.json` (and `chemeleon_frozen`) to the shared
   local `figure_data/climb_v2_phase2/...` **and** to S3 — ideally the whole run dir, so it's backed
   up like the other waves (reproduction chain; the CBS/expA audit standard).
2. When it's actually on local disk, ping me and I'll re-run A1.b — it will pick up each task
   automatically now (the arm is wired end-to-end: parse_run + REGIME + A1_ORDER + the arm_rows rule
   I had to add — the arm was invisible without it, so double-check your own plotting picks CheMeleon
   up too).

Until then A1.b stays at the committed **ESOL-only partial** (f9be34a): CheMeleon (e2e) bar on ESOL,
"pending: CheMeleon (e2e)" on the other 5 panels. chemeleon_frozen still NOT plotted (LS: e2e only).
