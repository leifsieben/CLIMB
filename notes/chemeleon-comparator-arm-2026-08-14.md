# CheMeleon comparator arm (e2e only) → CBS + A1.a/A1.b + A1 table

**Scope (user, 2026-08-14):** add **`chemeleon_e2e` ONLY** (NOT `chemeleon_frozen`, NOT `chemprop_e2e`)
as a comparator arm to figCBS + Fig A1.a + Fig A1.b + the A1 table. "Across all of them for now."
Native chemprop `--from-foundation CHEMELEON`, scored through eval_v2 (same scaffold folds seed 0,
compute_metric/compute_nef) so 1:1 with our arms. **HOLD until peer pings data has landed** (order:
frozen → e2e → HIV last; wait for ALL 6 core tasks incl HIV before rebuilding A1a/A1b).

## Data (incoming, peer)
- MolNet: `figure_data/climb_v2_phase2/chemeleon_e2e{,_s1,_s2}/moleculenet_cv/` (schema identical to
  our arms: `<DS>_MEAN/_STD`, `<DS>_nef1_MEAN/_STD`). 3 seeds via the `_s1/_s2` suffix.
- CBS: `experiment_cbs/cbs_nef1_summary.csv` gets a `chemeleon_e2e` row (build_cbs_summary.py extended).
  (It will also emit chemeleon_frozen / chemprop_e2e rows — those are NOT plotted because they are not
  in A1_ORDER.)

## The 3 infra edits (all my lane; each auto-flows)
1. **`notebook_cells/05.py` `parse_run`** — add before the fallthrough:
   `if name=="chemeleon_e2e": return dict(seed=0,regime="chemeleon_e2e",recipe=None,budget_label=None,budget_fp=np.nan)`
   (the `_s(\d+)$` regex at the top already aggregates `_s1/_s2`). → `build_table` picks it up.
2. **`notebook_cells/03.py` REGIME** — add:
   `REGIME["chemeleon_e2e"]=("<distinct published colour>","o","-","CheMeleon (e2e)")`
   → gives `rc_color`/`rc_label`. Pick a colour distinct from our arms (e.g. a purple/violet).
3. **`notebook_cells/08.py` A1_ORDER** — insert `"chemeleon_e2e"`. Placement TBD at build: after
   `fp_desc` (group the strong external comparators) is the leading candidate; confirm visually.

## Auto-flow (no extra edits)
- **A1.a / A1.b bars** (cell 08) iterate `A1_ORDER` → CheMeleon bar appears once in A1_ORDER + data present.
- **A1 table** (cell 10 via `build_table`, cell 05) → row appears once `parse_run` recognizes the name.
- **figCBS** (cell 42): `_cbs_arms=[a for a in A1_ORDER if ...]` → CheMeleon bar appears if the CBS CSV
  has a `chemeleon_e2e` row. Its A1_ORDER position also sets its bar order there.

## Build checklist (when data lands)
1. Confirm run dirs + CBS CSV row present (frozen→e2e→HIV all in). 2. Make the 3 edits. 3. Rebuild +
execute; check A1a/A1b/table/CBS all show CheMeleon (e2e), no "pending" slot. 4. Revert PDF churn,
regen manifest (new chemeleon_e2e raw evals in figure_data → manifest), verify_notebook_sync OK.
5. Commit + push. 6. Docs: note CheMeleon comparator in README A1/CBS rows. Confirm reproduction
(chemeleon_e2e on HF/S3).
