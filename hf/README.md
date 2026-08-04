# Hugging Face publishing templates

Ready-to-use repo READMEs (cards) for the three public artifacts. Upload each as the `README.md` of
its Hub repo, then fill the placeholders (`<org>`, `<CITATION>`, `<BIBTEX>`, exact counts).

| File | Upload as the README of | Repo type |
|---|---|---|
| `model_card.md` | `<org>/climb-encoders` | model |
| `dataset_card_results.md` | `<org>/climb-results` | dataset |
| `dataset_card_pretrain.md` | `<org>/climb-pretrain-data` | dataset |

Before publishing, confirm: (1) `<org>` and the paper citation/BibTeX, (2) the licenses in each YAML
front-matter (Apache-2.0 for weights, CC-BY-4.0 for data — verify PubChem/assay redistribution terms),
(3) that the cross-links between the three repos and the GitHub repo resolve.

The reproduction path these cards point to is [`../REPRODUCE.md`](../REPRODUCE.md).
