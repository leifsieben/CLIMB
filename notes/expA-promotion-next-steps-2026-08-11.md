# Experiment A — promotion checklist (HF checkpoints + docs + friend zip)

User (2026-08-11) may promote the synthetic-statistics ladder to a **main result**, and asked, once
the **bigram** data is in, to: (1) upload checkpoints to HF, (2) update all docs, (3) package a zip
for a collaborator. All tooling is committed; this is the run-order for the next session.

## 0. Prerequisite — confirm bigram + finalize are done
- Bigram wave writes `EXPA_BIGRAM_DONE` (SNS email "ExpA bigram wave COMPLETE").
- `scripts/finalize_expA.sh` (armed on the box) then syncs the 8 new encoders to
  `s3://climb-s3-bucket/experiments/climb_v2_expA/<run>/encoder/` and STOPS the box (SNS email
  "ExpA FINALIZED — checkpoints on S3"). Verify:
  `aws s3 ls s3://climb-s3-bucket/experiments/climb_v2_expA/bigram_8M/encoder/model.safetensors`
- Box `i-0c51bc3895a269f60` will be STOPPED. Everything below runs LOCALLY from S3 — no box needed.
  (Restart the box only to re-run training.)

## 1. Rebuild the ladder (now includes bigram) + hand to ipynb
```
python scripts/build_expA_ladder_summary.py      # reads native _baselines + all arms incl. bigram
```
- Sanity-check the printout. **Prediction: bigram ≈ unigram** (near floor). If bigram lands high
  (near shuffle/real), that's the surprise — local adjacency suffices; flag it.
- Commit `analysis/rigor/expA_ladder_{per_run,summary}.csv`.
- Message the ipynb session (`local_ef125a27`, "CLIMB results ipynb") that the bigram rows are in;
  cell 38 (figSA) auto-slots bigram_resample when the CSV has its rows — just re-exec the notebook.

## 2. HF checkpoints + results  (needs `hf auth login` first)
`climb_v2_expA` is already in `PAPER_WAVES` (publish_to_hf.py), and `_baselines/` is auto-skipped
(no encoder there). Dry-run, then execute:
```
python scripts/publish_to_hf.py encoders            # dry-run: lists the 8 expA encoders
python scripts/publish_to_hf.py encoders --execute  # -> lsieben/climb-encoders (climb_v2_expA/<run>/)
```
Results: expA per-run eval lives on S3, but `stage_results` reads LOCAL `figure_data/`. To include
per-run raw eval on HF, first pull them local (encoders excluded), then publish:
```
aws s3 sync s3://climb-s3-bucket/experiments/climb_v2_expA/ figure_data/climb_v2_expA/ --exclude "*/encoder/*"
python scripts/publish_to_hf.py results --execute   # ladder CSVs (experiment_a/) + per-run eval
```
Then refresh the cards to mention Experiment A: `hf/model_card.md`, `hf/dataset_card_results.md`
(and `python scripts/publish_to_hf.py cards --execute`).

## 3. Docs
- `README.md`: add an Experiment A section — the ladder (real/shuffle/bigram/unigram/no_pretrain),
  the arms table (see notes/…-2026-08-10 / methodology), the result (order & marginal don't matter;
  composition does), and that §3.7 "relative token frequencies" is superseded by "composition".
- `REPRODUCE.md`: repro path = build_synthetic_corpus.py (unigram/bigram) → build_expA_manifest.py →
  expA_run.sh / expA_bigram_run.sh (pretrain+CV) → expA_baselines_native_eval.sh (native baselines) →
  build_expA_ladder_summary.py. Wave `climb_v2_expA`; native units; frozen-probe 5-fold CV.
- If PROMOTED to main text (not SI figSA): coordinate the figure move with the ipynb session.
- Run `scripts/verify_notebook_sync.py` if any notebook/manifest touched (ipynb's lane).

## 4. Friend zip
```
python scripts/package_expA_bundle.py --out dist/experiment_a_bundle.zip
```
Bundles ladder CSVs + per-run CV predictions + configs + corruption diagnostics + a checkpoints.csv
(S3/HF URIs; weights NOT inlined). Send `dist/experiment_a_bundle.zip`. If the friend wants weights,
point them at `lsieben/climb-encoders` (private — grant access) or offer a separate weights tarball.

## Housekeeping / provenance
- IAM: box role `ChemTFMDownloader` has an added `ClimbBucketReadWrite` statement (climb-s3-bucket).
  Reversible — remove that statement if locking down.
- Synthetic corpora on S3: `tokenized_sources/pubchem_filtered_{unigram,bigram}_pkl` (gen-seed 12345).
- Units: expA is NATIVE regression; the phase2 moleculenet_cv is normalized — never mix. The native
  re-evals live under `climb_v2_expA/_baselines/` (build_expA_ladder_summary.py already points there).
