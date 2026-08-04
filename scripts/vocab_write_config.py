"""Write one vocab-sweep run config by patching the unsup_2M template.

Only the tokenizer, the on-the-fly-canonical routing, the FP budget, and (for the 100k point) the
batch size change. Everything else is byte-identical to the main unsupervised runs, so vocab is
the only variable.
"""
import argparse, yaml
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--template", required=True)
ap.add_argument("--run-id", required=True)
ap.add_argument("--tokenizer", required=True)   # local dir or s3 uri
ap.add_argument("--vocab", type=int, required=True)
ap.add_argument("--fp", type=int, default=2_000_000)
ap.add_argument("--out", required=True)
a = ap.parse_args()

c = yaml.safe_load(Path(a.template).read_text())
c["run_id"] = a.run_id
c["tokenizer_path"] = a.tokenizer
sel = c.setdefault("selection", {})
sel["augmentation"] = "canonical_raw"           # tokenize canonical SMILES on the fly, no enum
sel["total_forward_passes"] = a.fp
sel["pretraining_seed"] = 0
sel["init_encoder_path"] = None
sel["objectives"] = {"mlm": 1.0}
c.setdefault("training", {})["augmentation"] = "canonical_raw"
# vocab 100k: the MLM softmax over 100k classes OOMs the 24GB A10G at batch 256; shrink the batch
# (FP budget is held constant, so this only costs steps, not molecules seen).
if a.vocab >= 100_000:
    c["training"]["batch_size"] = 64
Path(a.out).write_text(yaml.safe_dump(c, sort_keys=False))
print(f"wrote {a.out}: tok={a.tokenizer} vocab={a.vocab} fp={a.fp} batch={c['training']['batch_size']}")
