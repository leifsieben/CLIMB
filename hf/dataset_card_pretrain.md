---
license: cc-by-4.0
pretty_name: CLIMB pre-training data
task_categories:
  - fill-mask
tags:
  - chemistry
  - smiles
  - pubchem
  - pretraining
  - cheminformatics
size_categories:
  - 10M<n<100M
---

# CLIMB — pre-training data

Everything needed to (re)train the CLIMB encoders: the tokenized PubChem corpus for masked-language
modeling, the descriptor targets for multi-task regression, the supervised fine-tuning table, and the
eval-leakage blocklist. Companion to the checkpoints and results repos.

- 📄 Paper: `<CITATION / arXiv link>`
- 💻 Code (`pretrain_v2.py`, `finetune_v2.py`) + `REPRODUCE.md`: `https://github.com/<org>/CLIMB`
- 🧠 Checkpoints: [`<org>/climb-encoders`](https://huggingface.co/<org>/climb-encoders)
- 📊 Raw results: [`<org>/climb-results`](https://huggingface.co/datasets/<org>/climb-results)

## Components

| Path | Contents |
|---|---|
| `tokenized_sources/pubchem_filtered/` | ~12M unique filtered PubChem SMILES for MLM (12 parquet shards) |
| `tokenized_sources/pubchem_descriptors/` | precomputed 217 RDKit descriptors per molecule (MTR targets) |
| `tokenized/supervised_wide_parquet/` | ~5.38M-row supervised "wide" table (PCBA, L1000, PCQM, WONG assays) |
| `tokenizer/` | byte-level BPE tokenizer, **vocab 1000**, zero-UNK on SMILES |
| `configs/eval_blocklist.json` | 34,301 molecules leaked into eval sets — excluded from all training |
| `configs/descriptor_stats.json` | descriptor normalization statistics |

## Provenance & curation

- **Source:** PubChem (public). SMILES were filtered (validity, size, element set) and RDKit-canonical
  normalized; see paper §6.1 and `scripts/` for the exact pipeline.
- **Leakage control (important):** every downstream eval molecule was removed from the training data at
  the InChIKey level. The `eval_blocklist.json` above is the exact exclusion list; training on it would
  invalidate the study. Details in paper §6.6.
- **Descriptor targets:** 217 RDKit descriptors, standardized with `descriptor_stats.json`, used as the
  multi-task-regression (MTR / "dense") pretraining signal.

## How to use

```python
from datasets import load_dataset
mlm = load_dataset("<org>/climb-pretrain-data", data_dir="tokenized_sources/pubchem_filtered", split="train")
```

To pretrain end-to-end, point `pretrain_v2.py` at these paths (see `REPRODUCE.md` §3 and README §7).
The tokenizer here is the same one shipped with the checkpoints repo.

## Citation

```bibtex
<BIBTEX — fill in>
```

License: CC-BY-4.0 for the derived/tokenized corpus. PubChem source data is public; please confirm
redistribution terms for the assay sources (PCBA/L1000/PCQM/WONG) before release.
