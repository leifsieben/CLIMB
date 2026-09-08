# Hugging Face cards and publishing

The three CLIMB artifacts on the Hub, and how their READMEs get published. All three are public.

| File | Is the README of | Repo type |
|---|---|---|
| `model_card.md` | [`lsieben/climb-encoders`](https://huggingface.co/lsieben/climb-encoders) | model |
| `dataset_card_results.md` | [`lsieben/climb-results`](https://huggingface.co/datasets/lsieben/climb-results) | dataset |
| `dataset_card_pretrain.md` | [`lsieben/climb-pretrain-data`](https://huggingface.co/datasets/lsieben/climb-pretrain-data) | dataset |

Licenses are Apache-2.0 for code and weights, CC-BY-4.0 for data.

## Publishing

[`../scripts/publish_to_hf.py`](../scripts/publish_to_hf.py) drives the full sync. It is dry-run by
default and refuses to run while logged out. Log in first with
`python3 -c "from huggingface_hub import login; login()"`, then:

```bash
python3 scripts/publish_to_hf.py --org lsieben --repo all                 # dry-run plan
python3 scripts/publish_to_hf.py --org lsieben --repo results  --execute
python3 scripts/publish_to_hf.py --org lsieben --repo encoders --execute
python3 scripts/publish_to_hf.py --org lsieben --repo pretrain --execute
python3 scripts/publish_to_hf.py --org lsieben --repo cards    --execute  # push only these READMEs
```

Re-running `--execute` re-syncs changed files and is idempotent.

Note that `publish_to_hf.py` uploads only the waves listed in its `PAPER_WAVES`. Result groups
outside that list are pushed by the surgical uploaders beside it
(`upload_cbs_results_hf.py`, `upload_chemeleon_molnet_hf.py`), each of which stages one tree into
`climb-results` without touching the others. Anything added outside both paths must be uploaded
explicitly; verify by listing the repo afterwards rather than trusting the upload's return value.

The 124M pretraining corpus is not re-hosted. The pretrain card links to
[`hheiden/PubChem-124M-SMILES-SELFIES-InChI-IUPAC`](https://huggingface.co/datasets/hheiden/PubChem-124M-SMILES-SELFIES-InChI-IUPAC)
and [`../scripts/download_pubchem_full.sh`](../scripts/download_pubchem_full.sh) rebuilds the exact
copy used here.

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
