"""Train the tokenizers for the SI vocab-size scaling law (wave climb_v2_vocab).

Two families x four vocab sizes = eight tokenizers:
  * BPE       — byte-level BPE, the SAME family as the main paper (floor 256 → 261 = bytes only,
                no merges; larger sizes add merges).
  * Unigram   — SentencePiece-style Unigram LM, a different sub-word ALGORITHM at matched vocab.
Vocab sizes: 261, 1000, 10000, 100000.

Each is saved as a HuggingFace PreTrainedTokenizerFast with the exact special tokens the model
config expects (<s>=bos, <pad>, </s>=eos, <unk>, <mask>), so pretrain_v2 loads them unchanged and
sizes the embedding to the tokenizer's vocab automatically.

Training text is a sample of canonical SMILES pulled from the PubChem corpus parquet; a couple of
million lines is plenty to learn every merge/piece that occurs >=2 times on this alphabet.

Usage:
    python scripts/build_vocab_tokenizers.py --sample 2000000 --out experiments/_vocab_tok --s3 \
        s3://climb-s3-bucket/tokenizers_vocab
"""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import pyarrow.parquet as pq

SPECIAL = ["<s>", "<pad>", "</s>", "<unk>", "<mask>"]
# Per-family targets, chosen to be 4 DISTINCT points within each family's reachable range
# (SMILES saturates: BPE ~10k on an 8M sample with min_frequency=1, Unigram lower). Targets above
# a family's ceiling simply cap there; the actual vocab is what the run reports and plots against.
FAMILY_VOCABS = {"bpe": [261, 1000, 3000, 12000], "unigram": [261, 700, 1200, 3000]}
RAW_SMILES_S3 = "s3://climb-s3-bucket/tokenized_sources/pubchem_filtered/"


def sample_smiles(n: int, cache: Path) -> Path:
    """Pull n canonical SMILES from the corpus parquet into a flat text file (one per line)."""
    txt = cache / "smiles_sample.txt"
    if txt.exists() and txt.stat().st_size > 0:
        print(f"[tok] reusing {txt}")
        return txt
    cache.mkdir(parents=True, exist_ok=True)
    # grab enough shards to cover n (each shard ~1M rows)
    local = cache / "_shards"; local.mkdir(exist_ok=True)
    need = n // 1_000_000 + 1
    ls = subprocess.run(["aws", "s3", "ls", RAW_SMILES_S3], capture_output=True, text=True).stdout
    shards = [l.split()[-1] for l in ls.splitlines() if l.strip().endswith(".parquet")][:need]
    written = 0
    with txt.open("w") as fh:
        for s in shards:
            loc = local / s
            if not loc.exists():
                subprocess.run(["aws", "s3", "cp", RAW_SMILES_S3 + s, str(loc)], check=True,
                               capture_output=True)
            pf = pq.ParquetFile(loc)
            col = next(c for c in pf.schema_arrow.names
                       if c.lower() in ("smiles_canonical", "smiles", "canonical_smiles", "smiles_canon"))
            for batch in pf.iter_batches(columns=[col], batch_size=50_000):
                for v in batch.column(0).to_pylist():
                    if v:
                        fh.write(v + "\n"); written += 1
                        if written >= n:
                            print(f"[tok] wrote {written} SMILES -> {txt}")
                            return txt
    print(f"[tok] wrote {written} SMILES -> {txt}")
    return txt


def wrap_and_save(tk, out: Path):
    from transformers import PreTrainedTokenizerFast
    out.mkdir(parents=True, exist_ok=True)
    tk.save(str(out / "tokenizer.json"))
    hf = PreTrainedTokenizerFast(
        tokenizer_file=str(out / "tokenizer.json"),
        bos_token="<s>", eos_token="</s>", unk_token="<unk>",
        pad_token="<pad>", mask_token="<mask>",
    )
    hf.save_pretrained(str(out))
    return len(hf)


def train_bpe(txt: Path, vocab: int, out: Path) -> int:
    from tokenizers import ByteLevelBPETokenizer
    tk = ByteLevelBPETokenizer()
    tk.train(files=[str(txt)], vocab_size=vocab, min_frequency=1, special_tokens=SPECIAL)  # min_freq=1 stretches the BPE ceiling
    return wrap_and_save(tk, out)


def train_unigram(txt: Path, vocab: int, out: Path) -> int:
    from tokenizers import Tokenizer, models, trainers, pre_tokenizers
    tk = Tokenizer(models.Unigram())
    # ByteLevel pre-tokenizer keeps the alphabet closed (any byte representable), matching BPE's
    # coverage guarantee so a rare character never becomes <unk> for one family but not the other.
    tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tk.decoder = __import__("tokenizers").decoders.ByteLevel()
    trainer = trainers.UnigramTrainer(
        vocab_size=vocab, special_tokens=SPECIAL, unk_token="<unk>",
        max_piece_length=16,
    )
    tk.train([str(txt)], trainer)
    return wrap_and_save(tk, out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=8_000_000)
    ap.add_argument("--out", default="experiments/_vocab_tok")
    ap.add_argument("--s3", default="s3://climb-s3-bucket/tokenizers_vocab")
    ap.add_argument("--only", default="", help="comma list like bpe_1000,unigram_261 to (re)build a subset")
    a = ap.parse_args()
    root = Path(a.out); root.mkdir(parents=True, exist_ok=True)
    txt = sample_smiles(a.sample, root)

    only = set(x for x in a.only.split(",") if x)
    plan = [(fam, v) for fam in ("bpe", "unigram") for v in FAMILY_VOCABS[fam]]
    for fam, v in plan:
        name = f"{fam}_{v}"
        if only and name not in only:
            continue
        out = root / name
        if (out / "tokenizer.json").exists():
            print(f"[tok] {name}: exists, skipping"); continue
        print(f"[tok] training {name} ...", flush=True)
        n = train_bpe(txt, v, out) if fam == "bpe" else train_unigram(txt, v, out)
        print(f"[tok] {name}: actual vocab = {n}")
        subprocess.run(["aws", "s3", "sync", str(out), f"{a.s3}/{name}", "--only-show-errors"],
                       check=True)
        print(f"[tok] {name}: uploaded to {a.s3}/{name}")
    print("[tok] DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
