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

# CLIMB encoders

Frozen encoder weights for every run in CLIMB, a controlled study of whether unsupervised
pretraining on SMILES teaches a chemical language model chemistry. All arms share one
architecture, tokenizer and optimizer; they differ only in the pretraining corpus and budget.

## Architecture

ModernBERT, 12 layers, hidden size 512, 8 heads, ~41.4M parameters. Byte-BPE tokenizer, vocab
1000, maximum sequence length 256.

## Layout

```
<wave>/<run>/config.json
<wave>/<run>/model.safetensors
tokenizer/tokenizer.json, tokenizer/tokenizer_config.json, tokenizer/special_tokens_map.json
```

`<wave>` is the experiment group (`climb_v2_phase2`, `climb_v2_expA`, `climb_v2_expB`, and
others); `<run>` names the arm. [`METHODS.md`](https://github.com/leifsieben/CLIMB/blob/main/METHODS.md) in the code repository defines every arm.

## Use

```python
from transformers import AutoModel, PreTrainedTokenizerFast
from huggingface_hub import snapshot_download

path = snapshot_download("lsieben/climb-encoders",
                         allow_patterns=["climb_v2_phase2/unsup_8M/*", "tokenizer/*"])
model = AutoModel.from_pretrained(f"{path}/climb_v2_phase2/unsup_8M")
tok = PreTrainedTokenizerFast.from_pretrained(f"{path}/tokenizer")
```

Encoders are used frozen, mean-pooled, with a trained head. The evaluation protocol is in
[`METHODS.md`](https://github.com/leifsieben/CLIMB/blob/main/METHODS.md).

## What is not here

Training configurations and trainer metrics logs are not published in this repository.

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

Apache-2.0.
