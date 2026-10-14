from __future__ import annotations

import argparse
from pathlib import Path

import pyarrow.parquet as pq

SPECIAL = ["<s>", "<pad>", "</s>", "<unk>", "<mask>"]


FAMILY_VOCABS = {"bpe": [261, 1000, 3000, 12000], "unigram": [261, 700, 1200, 3000]}


def sample_smiles(n: int, cache: Path, source: Path) -> Path:
    """Pull n canonical SMILES from the corpus parquet into a flat text file (one per line)."""
    txt = cache / "smiles_sample.txt"
    if txt.exists() and txt.stat().st_size > 0:
        print(f"[tok] reusing {txt}")
        return txt
    cache.mkdir(parents=True, exist_ok=True)
    shards = sorted(source.glob("*.parquet"))[:n // 1_000_000 + 1]
    if not shards:
        raise FileNotFoundError(f"No parquet shards in {source}")
    written = 0
    with txt.open("w") as fh:
        for s in shards:
            pf = pq.ParquetFile(s)
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
    tk.train(files=[str(txt)], vocab_size=vocab, min_frequency=1, special_tokens=SPECIAL)
    return wrap_and_save(tk, out)


def train_unigram(txt: Path, vocab: int, out: Path) -> int:
    from tokenizers import Tokenizer, models, trainers, pre_tokenizers
    tk = Tokenizer(models.Unigram())


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
    ap.add_argument("--input_dir", type=Path, required=True)
    ap.add_argument("--only", default="", help="comma list like bpe_1000,unigram_261 to (re)build a subset")
    a = ap.parse_args()
    root = Path(a.out); root.mkdir(parents=True, exist_ok=True)
    txt = sample_smiles(a.sample, root, a.input_dir)

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
    print("[tok] DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
