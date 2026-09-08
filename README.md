# CLIMB

A controlled study of whether unsupervised pretraining on SMILES teaches a chemical language model
chemistry, or whether its benefit comes from something else. Model, tokenizer, optimizer, data
budget and evaluation are held fixed across arms so that the pretraining corpus is the only thing
that varies.

## Artifacts

| What | Where |
|---|---|
| Encoder weights for every run, plus the vocab-1000 byte-BPE tokenizer | [`lsieben/climb-encoders`](https://huggingface.co/lsieben/climb-encoders) |
| Evaluation results, including the per-molecule test predictions every figure is computed from | [`lsieben/climb-results`](https://huggingface.co/datasets/lsieben/climb-results) |
| Pretraining corpora, 217-descriptor targets, supervised table, leakage blocklist | [`lsieben/climb-pretrain-data`](https://huggingface.co/datasets/lsieben/climb-pretrain-data) |

The 124M-molecule corpus is not re-hosted; it derives from
[`hheiden/PubChem-124M-SMILES-SELFIES-InChI-IUPAC`](https://huggingface.co/datasets/hheiden/PubChem-124M-SMILES-SELFIES-InChI-IUPAC)
and `scripts/download_pubchem_full.sh` rebuilds the exact copy used here.

## Contents

| Path | Contents |
|---|---|
| `METHODS.md` | full methodology: hypotheses, experiments, architecture, data curation, training, evaluation |
| `REPRODUCE.md` | figure-by-figure reproduction, each mapped to its data and exact command |
| `figures/` | one script per paper figure; no notebook and no hidden state |
| `scripts/` | training, evaluation and analysis entry points |
| `hf/` | the Hugging Face repo cards and how they are published |
| `notes/` | dated records of corrections and decisions |

## Reproducing a figure

```bash
python3 scripts/six_panel_aggregate.py        # refresh the results tables
python3 -m figures.fig_A                      # render one figure into figures_v2/
python3 scripts/audit_figure_consistency.py   # run before shipping any figure
```

`REPRODUCE.md` covers the full path from a fresh clone, including which Hugging Face artifacts to
download and how to verify them.

## Citation

```bibtex
@misc{climb2026,
  title  = {Does Pretraining Teach Chemical Language Models Chemistry?},
  author = {Sieben, Leif and Zimmermann, Yoel},
  year   = {2026},
  note   = {Preprint, arXiv},
  url    = {https://github.com/leifsieben/CLIMB}
}
```

## License

Code Apache-2.0. Weights Apache-2.0. Data CC-BY-4.0.
