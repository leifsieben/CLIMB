# CLIMB

Code for **Does Pretraining Teach Chemical Language Models Chemistry?**
Leif Sieben and Yoel Zimmermann, 2026.

Pretraining, evaluation on 64 molecular property datasets, and paper figures.

## Setup

Use Python 3.9 and run commands from the repository root. Paper-scale pretraining uses a CUDA GPU; figure scripts use Arial.

```bash
python3.9 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Data and checkpoints

- [Pretrained encoders and tokenizer](https://huggingface.co/lsieben/climb-encoders)
- [Evaluation results](https://huggingface.co/datasets/lsieben/climb-results)
- [Pretraining data](https://huggingface.co/datasets/lsieben/climb-pretrain-data)

Download paths retain the published archive names.

```bash
hf download lsieben/climb-encoders --local-dir checkpoints \
  --include 'climb_v2_phase2/unsup_8M/*' 'tokenizer/*'
hf download lsieben/climb-results --repo-type dataset --local-dir figure_data \
  --exclude 'embedding_resolution/*'
```

Set local data and tokenizer paths in `configs/*.yaml` before pretraining. The full 124M-molecule corpus comes from [PubChem-124M](https://huggingface.co/datasets/hheiden/PubChem-124M-SMILES-SELFIES-InChI-IUPAC). Polaris scoring uses a separate environment with `requirements-polaris.txt` because its test labels are withheld.

## Experiments

```bash
python pretrain.py --config configs/unsup_8M.yaml --run_dir experiments/unsup_8M
python evaluate.py --encoder checkpoints/climb_v2_phase2/unsup_8M \
  --tokenizer checkpoints/tokenizer --output_dir results/unsup_8M --cv_folds 5
python evaluate.py --featurizer fp_desc --head xgb \
  --output_dir results/fp_desc --cv_folds 5
```

`finetune.py` runs end-to-end fine-tuning. `scripts/evaluate_benchmarks.py` runs MoleculeACE and Polaris; `cbs_run.py`, `wong_run.py`, and `fartdb_multiclass.py` cover the remaining benchmarks. Use `--help` for options.

## Paper figures

Main figures are `figures/figure_1.py` through `figure_6.py`; supplementary figures are `figure_C1.py`, `figure_E1.py`, `figure_F1.py`, `figure_G1.py`, `figure_H1.py`, and `figure_I1.py`. Run them as modules; plots are saved to `plots/`:

```bash
python -m figures.figure_5
```

Figure 5 uses the bundled tables. For the other figures, rebuild the archived six-panel summaries with `scripts/six_panel_aggregate.py` and `scripts/six_panel_scaling.py`, then run the corresponding `scripts/build_figure_*_table.py` where present.

For Figure 6, set checkpoint paths in `scripts/embed_resolution_pairs.py`, run it once normally and once with `RESOLUTION_INPUT=canonical`, then run `scripts/resolution_separability.py`. These final resolution inputs are not included in the public results archive, whose older pair files are excluded above.

Code: [Apache-2.0](LICENSE). Upstream datasets retain their respective licenses.
