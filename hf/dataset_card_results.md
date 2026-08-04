---
license: cc-by-4.0
pretty_name: CLIMB raw evaluation results
task_categories:
  - tabular-regression
  - tabular-classification
tags:
  - chemistry
  - molecular-property-prediction
  - benchmark-results
  - reproducibility
  - moleculenet
---

# CLIMB — raw evaluation results

The exact per-model evaluation outputs behind every figure and table in the CLIMB study (does
unsupervised SMILES pretraining help a chemical language model?). This is the `figure_data/` snapshot:
feed it to the figure notebook and you regenerate the paper's figures **byte-for-byte**.

- 📄 Paper: `<CITATION / arXiv link>`
- 💻 Code + `REPRODUCE.md`: `https://github.com/<org>/CLIMB`
- 🧠 Checkpoints: [`<org>/climb-encoders`](https://huggingface.co/<org>/climb-encoders)
- 🧪 Pre-training data: [`<org>/climb-pretrain-data`](https://huggingface.co/datasets/<org>/climb-pretrain-data)

## What's inside

One directory per run, per wave, mirroring the checkpoints repo:

```
<wave>/<run>/moleculenet/suite_summary.json        # single scaffold hold-out: per-task metrics
<wave>/<run>/moleculenet/test_predictions.csv      # per-molecule predictions (hold-out)
<wave>/<run>/moleculenet_cv/suite_summary.json     # pooled 5-fold scaffold CV: per-task metrics
<wave>/<run>/moleculenet_cv/test_predictions.csv   # per-molecule OOF predictions (CV)
<wave>/<run>/metrics.jsonl                          # training curve + token counts
```

- **Tasks (6):** ESOL, BBBP, BACE, Tox21, QM7, HIV — RMSE for regression, ROC-AUC / NEF1% for
  classification & virtual screening.
- **`suite_summary.json`** keys: `<TASK>_MEAN`, `<TASK>_STD` (and `<TASK>_nef1_MEAN` for HIV).
- **`test_predictions.csv`** columns: `dataset, task_type, mol_index, canonical_key, raw_smiles,
  output_index, y_true, y_pred` — enough to recompute every metric and every paired significance test.

## How the figures are regenerated

```bash
git clone https://github.com/<org>/CLIMB && cd CLIMB
# place this dataset at figure_data/
python scripts/build_data_manifest.py --check     # confirm your copy == the paper snapshot
python scripts/build_figure_notebook.py
jupyter nbconvert --to notebook --execute --inplace climb_figures.ipynb
python scripts/verify_notebook_sync.py            # all-green = reproduced
```

`figure_data_manifest.json` in the repo is a content fingerprint of this exact snapshot; the checker
above names any per-file difference instead of silently producing different numbers. See `REPRODUCE.md`
for the figure → data → command map.

## Provenance & notes

- Evaluation is a frozen featurizer (masked-mean-pooled CLIMB embeddings → z-score → head), plus
  Morgan+XGBoost / Morgan+desc+XGBoost classical anchors. Protocol: paper §8, `eval_v2.py`.
- Downstream tasks are MoleculeNet (DeepChem loaders); all training molecules are deduplicated against
  the eval sets at the InChIKey level (34,301-molecule blocklist, shipped with the pre-training data).

## Citation

```bibtex
<BIBTEX — fill in>
```

License: CC-BY-4.0 for these derived results. Downstream label sources are MoleculeNet / public assays;
please confirm attribution requirements before release.
