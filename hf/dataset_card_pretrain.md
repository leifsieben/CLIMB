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

# CLIMB pretraining data

The corpora, descriptor targets and auxiliary tables used to pretrain the CLIMB encoders.

## Layout

```
tokenized_sources/pubchem_filtered/            source SMILES after filtering
tokenized_sources/pubchem_filtered_*_pkl/      tokenized pretraining corpora (see below)
tokenized_sources/pubchem_descriptors/         217 RDKit descriptor targets, sharded
tokenized/supervised_wide_parquet/             supervised fine-tuning table
tokenizer/, tokenizers_vocab/                  the vocab-1000 tokenizer and the vocab-sweep variants
configs/descriptor_stats.json                  descriptor names and normalisation statistics
```

## Pretraining corpora

| Corpus | Contents |
|---|---|
| `pubchem_filtered_tokenized_pkl` | the real PubChem SMILES corpus |
| `pubchem_filtered_bigram_pkl` | sequences resampled from the corpus bigram statistics: local adjacency only |
| `pubchem_filtered_unigram_pkl` | sequences resampled from the corpus unigram marginal: no sequential structure |
| `pubchem_filtered_wiki_pkl` | English Wikipedia text, tokenized with the same tokenizer: no chemistry |

The token-shuffled control is applied as a training-time transform of the real corpus and has no
separate artifact.

The 124M-molecule RDKit-canonical corpus is not re-hosted. It derives from
[`hheiden/PubChem-124M-SMILES-SELFIES-InChI-IUPAC`](https://huggingface.co/datasets/hheiden/PubChem-124M-SMILES-SELFIES-InChI-IUPAC);
[`scripts/download_pubchem_full.sh`](https://github.com/leifsieben/CLIMB/blob/v2-redux/scripts/download_pubchem_full.sh) rebuilds the exact copy used here.

## Leakage

Molecules overlapping the downstream evaluation sets are recorded in the blocklist and excluded;
the audit procedure is described in [`METHODS.md`](https://github.com/leifsieben/CLIMB/blob/v2-redux/METHODS.md).

## Related

- Code: [github.com/leifsieben/CLIMB](https://github.com/leifsieben/CLIMB)
- Encoders: [`lsieben/climb-encoders`](https://huggingface.co/lsieben/climb-encoders)
- Results: [`lsieben/climb-results`](https://huggingface.co/datasets/lsieben/climb-results)
- Pretraining data: [`lsieben/climb-pretrain-data`](https://huggingface.co/datasets/lsieben/climb-pretrain-data)

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

CC-BY-4.0.
