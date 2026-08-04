# Hugging Face cards + publishing

Finalized repo READMEs (cards) for the three CLIMB artifacts, and how they get published.

| File | Is the README of | Repo type | Visibility |
|---|---|---|---|
| `model_card.md` | [`lsieben/climb-encoders`](https://huggingface.co/lsieben/climb-encoders) | model | private |
| `dataset_card_results.md` | [`lsieben/climb-results`](https://huggingface.co/datasets/lsieben/climb-results) | dataset | private |
| `dataset_card_pretrain.md` | [`lsieben/climb-pretrain-data`](https://huggingface.co/datasets/lsieben/climb-pretrain-data) | dataset | private |

Namespaces are filled (HF `lsieben`, GitHub `leifsieben`); licenses are Apache-2.0 (weights) /
CC-BY-4.0 (data). The citation is a `@misc` placeholder — **update the author list + swap in the real
BibTeX once the preprint exists** (search the cards for `climb2026`).

## Publishing / updating

Everything is driven by [`../scripts/publish_to_hf.py`](../scripts/publish_to_hf.py) — dry-run by
default, creates the repos **private**, refuses to run while logged out. Log in first
(`python3 -c "from huggingface_hub import login; login()"`), then:

```bash
python scripts/publish_to_hf.py --org lsieben --repo all              # dry-run plan
python scripts/publish_to_hf.py --org lsieben --repo results  --execute
python scripts/publish_to_hf.py --org lsieben --repo encoders --execute   # ~18.8 GB, staged from S3
python scripts/publish_to_hf.py --org lsieben --repo pretrain --execute   # ~9 GB, 124M corpus excluded
python scripts/publish_to_hf.py --org lsieben --repo cards     --execute  # push only the READMEs
```

Re-running an `--execute` re-syncs changed files (idempotent). To share with reviewers while private,
add them (or a share link) in each repo's **Settings → sharing** on the Hub.

The 124M pre-training corpus is **not** re-hosted — the pretrain card links to
[`hheiden/PubChem-124M-SMILES-SELFIES-InChI-IUPAC`](https://huggingface.co/datasets/hheiden/PubChem-124M-SMILES-SELFIES-InChI-IUPAC)
and `scripts/download_pubchem_full.sh` rebuilds our exact copy. The reproduction path all three cards
point to is [`../REPRODUCE.md`](../REPRODUCE.md).
