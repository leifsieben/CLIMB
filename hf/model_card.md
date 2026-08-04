---
license: apache-2.0
library_name: transformers
pipeline_tag: feature-extraction
tags:
  - chemistry
  - cheminformatics
  - smiles
  - molecular-property-prediction
  - modernbert
  - masked-language-modeling
---

# CLIMB encoders — does unsupervised pretraining help a chemical language model?

Frozen-featurizer **ModernBERT encoders (~41M params)** for every run in the CLIMB study, which asks
**whether, and how much, unsupervised (MLM) pretraining on SMILES improves a chemical language model**
versus training supervised-from-scratch. Model, tokenizer, optimizer and evaluation are held fixed
across runs so the only thing that varies is the *pretraining strategy*.

- 📄 Paper: preprint in preparation (link via the GitHub repo)
- 💻 Code + full methods: `https://github.com/leifsieben/CLIMB` (see `REPRODUCE.md`)
- 📊 Raw results: [`lsieben/climb-results`](https://huggingface.co/datasets/lsieben/climb-results)
- 🧪 Pre-training data: [`lsieben/climb-pretrain-data`](https://huggingface.co/datasets/lsieben/climb-pretrain-data)

## Repository layout

One subfolder per run, mirroring the experiment waves (each has `model.safetensors` + `config.json`).
The byte-BPE tokenizer (vocab 1000) is at `tokenizer/`.

```
tokenizer/                                  # byte-level BPE, vocab 1000, zero-UNK
climb_v2_phase2/unsup_8M/                   # MLM-only, 8M forward-pass budget
climb_v2_phase2/skip_dense_8M/              # supervised-from-scratch (MTR descriptors)
climb_v2_phase2/u2s_dense_from8M/           # unsup -> sup warm-start
climb_v2_phase2/unsup_{2M,24M,48M,50M,100M} # the scaling ladder
climb_v2_ablation_dedup/seq_*/              # SFT-family ablation (leakage-deduped)
climb_v2_h1/scaling_{canonical,enumerated}_*_s{0,1,2}/   # enumeration study, 3 seeds
...
```

The classical Morgan+XGBoost baselines have **no encoder** (they are fingerprints + XGBoost); see the
results repo. `lsieben/climb-encoders` is checkpoints only — pair it with the results and data repos.

## Regimes

| regime | what it is |
|---|---|
| `no_pretrain` | randomly-initialised encoder (frozen, or fine-tuned end-to-end) |
| `unsup_only` | MLM pretraining on SMILES only |
| `sup_only` | supervised-from-scratch (recipes: dense MTR / sparse assays / combinations) |
| `unsup→sup` | MLM pretraining then a supervised warm-start |

## How to use

```python
from transformers import ModernBertModel, PreTrainedTokenizerFast
import torch

tok = PreTrainedTokenizerFast.from_pretrained("lsieben/climb-encoders", subfolder="tokenizer")
enc = ModernBertModel.from_pretrained(
    "lsieben/climb-encoders", subfolder="climb_v2_phase2/unsup_8M",
    attn_implementation="sdpa", reference_compile=False).eval()

smiles = ["CCO", "c1ccccc1O"]
ids = tok(smiles, return_tensors="pt", padding=True, truncation=True, max_length=256)
with torch.no_grad():
    h = enc(**ids).last_hidden_state          # [B, L, H]
mask = ids["attention_mask"].unsqueeze(-1)
emb = (h * mask).sum(1) / mask.sum(1)         # masked-mean pooling (the paper's featurizer)
```

The paper evaluates these as a **frozen featurizer** (masked-mean pooled → z-scored → small head).
The exact protocol and per-run eval commands are in `eval_v2.py` and `REPRODUCE.md`.

## Architecture & training

- ~41M-param ModernBERT encoder; byte-level BPE tokenizer, **vocab 1000** (zero-UNK on SMILES).
- Objectives across runs: masked-language-modeling (MLM), multi-task descriptor regression (MTR),
  and supervised assay heads; see the paper §5, §7.
- Trained on the tokenized PubChem corpus in [`lsieben/climb-pretrain-data`](https://huggingface.co/datasets/lsieben/climb-pretrain-data).
- `metrics.jsonl` (token counts, loss curves) for each run ships in the results repo.

## Intended use & limitations

Research artifact for studying pretraining strategy in molecular property prediction. Not a
production model. Frozen-featurizer performance is task-dependent; on several MoleculeNet tasks a
**Morgan+descriptor+XGBoost** baseline remains competitive or better (that comparison is the point of
the study). Do not use for clinical/safety decisions.

## Citation

```bibtex
@misc{climb2026,
  title  = {CLIMB: does unsupervised pretraining help a chemical language model?},
  author = {Sieben, Leif},          % TODO: finalize author list before the preprint
  year   = {2026},
  note   = {Preprint in preparation},
  url    = {https://github.com/leifsieben/CLIMB}
}
```

License: **Apache-2.0** (encoder weights).
