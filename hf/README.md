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

### CheMeleon-suite / CBS / MoleculeNet-CheMeleon results

`publish_to_hf.py` uploads only `PAPER_WAVES`. Two result groups added 2026-08-14 are handled separately:
- **CBS battery** (`figure_data/cbs_benchmark/`, incl the CheMeleon frozen + chemprop-e2e comparators) →
  `scripts/upload_cbs_results_hf.py --org lsieben --execute` (surgical, stages `cbs_benchmark/<run>/` +
  `experiment_cbs/*.csv` into `climb-results` without touching other waves).
- **MoleculeNet CheMeleon arms** (`chemeleon_frozen`, `chemeleon_e2e`) live under `climb_v2_phase2/`, so
  the standard `publish_to_hf.py --repo results --execute` re-sync picks them up.
- **MoleculeNet CheMeleon arms + Polaris/MoleculeACE suite** →
  `scripts/upload_chemeleon_molnet_hf.py --org lsieben --execute` (surgical: stages
  `climb_v2_phase2/chemeleon_{frozen,e2e,e2e_s1,e2e_s2}/moleculenet_cv/` incl per-molecule OOF, and
  `chemeleon_suite/<track>/<model>/`, into `climb-results` without touching other waves).

> **Pushed 2026-08-15** (all in `lsieben/climb-results`, private): `cbs_benchmark/` (CBS incl CheMeleon
> comparators), `climb_v2_phase2/chemeleon_{frozen,e2e,e2e_s1,e2e_s2}/`, and `chemeleon_suite/`
> (Polaris 76 + MoleculeACE 57 files). Re-run the two uploaders to re-sync after any change (idempotent).

The 124M pre-training corpus is **not** re-hosted — the pretrain card links to
[`hheiden/PubChem-124M-SMILES-SELFIES-InChI-IUPAC`](https://huggingface.co/datasets/hheiden/PubChem-124M-SMILES-SELFIES-InChI-IUPAC)
and `scripts/download_pubchem_full.sh` rebuilds our exact copy. The reproduction path all three cards
point to is [`../REPRODUCE.md`](../REPRODUCE.md).
