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
| `tokenized_sources/pubchem_filtered/` | ~12M unique filtered PubChem SMILES for MLM (12 parquet shards) — the base scaling ladder (2M–48M FP) |
| `tokenized_sources/pubchem_descriptors/` | precomputed 217 RDKit descriptors per molecule (MTR targets) |
| `tokenized/supervised_wide_parquet/` | ~5.38M-row supervised "wide" table (PCBA, L1000, PCQM, WONG assays) |
| `tokenizer/` | byte-level BPE tokenizer, **vocab 1000**, zero-UNK on SMILES |
| `configs/eval_blocklist.json` | 34,301 molecules leaked into eval sets — excluded from all training |
| `configs/descriptor_stats.json` | descriptor normalization statistics |

## The full ~124M corpus (long scaling runs) — linked, not re-hosted

The 50M/100M scaling runs (and the `*_c124` controls) draw from the **full ~124M-molecule PubChem
set**, which is **not re-hosted here**. It is a re-canonicalized derivative of the upstream dataset
[`hheiden/PubChem-124M-SMILES-SELFIES-InChI-IUPAC`](https://huggingface.co/datasets/hheiden/PubChem-124M-SMILES-SELFIES-InChI-IUPAC);
rebuild our exact copy with `scripts/download_pubchem_full.sh` (RDKit re-canonicalization kept on, to
match the tokenizer). The ~12M `pubchem_filtered/` corpus above — used by every headline run — **is**
shipped here so those results are self-contained.

## Provenance & curation

- **Source:** PubChem (public), via `hheiden/PubChem-124M-SMILES-SELFIES-InChI-IUPAC`. SMILES were
  filtered (validity, size, element set) and RDKit-canonical normalized; see paper §6.1 and `scripts/`
  for the exact pipeline.
- **Leakage control (important):** every downstream eval molecule was removed from the supervised
  training table by **RDKit canonical SMILES of the largest fragment** (salt-stripped). The
  `eval_blocklist.json` above is the exact exclusion list; training on it would invalidate the study.
  (Applied to the supervised objective; pure-MLM/MTR runs never touch the supervised table.) Details in
  paper §6.6.
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
