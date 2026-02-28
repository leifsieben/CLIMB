#!/usr/bin/env python3
"""Shard-aware SMILES tokenization

Tokenizes a large parquet/CSV/text SMILES file into multiple pickle shards
(~200–300 MB each by default) to keep training I/O manageable and streaming-
friendly. Outputs pickles with a single key `data` that is a list of dicts
with `input_ids` and `attention_mask`, matching the expectations of
train_model.py.

Example:
    python preparing_datasets/tokenizing_datasets.py \
        --tokenizer tokenizer_10M \
        --input preparing_datasets/unsup_10M.parquet \
        --output-dir tokenized/unsup_hp \
        --shard-size 250000 \
        --max-length 512
"""

import argparse
import logging
import pickle
from pathlib import Path
from typing import Iterable, List

import pyarrow.parquet as pq
from transformers import PreTrainedTokenizerFast


logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


def iter_smiles(parquet_path: Path, batch_size: int = 100_000) -> Iterable[List[str]]:
    """Stream SMILES strings from a parquet file in batches to limit memory."""
    pf = pq.ParquetFile(parquet_path)
    for batch in pf.iter_batches(batch_size=batch_size, columns=["SMILES"], use_threads=True):
        yield batch.column(0).to_pylist()


def tokenize_batch(smiles_batch: List[str], tokenizer, max_length: int):
    enc = tokenizer(
        smiles_batch,
        truncation=True,
        max_length=max_length,
        padding=False,
        return_attention_mask=True,
    )
    data = []
    for ids, mask in zip(enc["input_ids"], enc["attention_mask"]):
        data.append({"input_ids": ids, "attention_mask": mask})
    return data


def write_shard(shard_data: List[dict], out_dir: Path, shard_idx: int):
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"shard_{shard_idx:05d}.pkl"
    with out_path.open("wb") as f:
        pickle.dump({"data": shard_data}, f)
    logger.info("Wrote %s (%d samples)", out_path, len(shard_data))


def main():
    parser = argparse.ArgumentParser(description="Tokenize SMILES into sharded pickles")
    parser.add_argument("--tokenizer", required=True, help="Directory containing tokenizer.json")
    parser.add_argument("--input", required=True, help="Parquet/CSV/TXT of SMILES; must have SMILES column")
    parser.add_argument("--output-dir", required=True, help="Destination directory for shards")
    parser.add_argument("--shard-size", type=int, default=250_000, help="Rows per shard (approx)")
    parser.add_argument("--batch-size", type=int, default=100_000, help="Parquet read batch size")
    parser.add_argument("--max-length", type=int, default=512, help="Max token length")
    parser.add_argument("--limit", type=int, default=None, help="Optional total rows limit for quick tests")
    args = parser.parse_args()

    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=str(Path(args.tokenizer) / "tokenizer.json"),
        bos_token="<s>",
        eos_token="</s>",
        unk_token="<unk>",
        pad_token="<pad>",
        mask_token="<mask>",
    )

    parquet_path = Path(args.input)
    out_dir = Path(args.output_dir)

    total = 0
    shard_idx = 0
    shard_data: List[dict] = []

    logger.info("Streaming from %s", parquet_path)
    for smiles_batch in iter_smiles(parquet_path, batch_size=args.batch_size):
        if args.limit is not None and total >= args.limit:
            break

        # Clip batch if we are near the limit
        if args.limit is not None and total + len(smiles_batch) > args.limit:
            smiles_batch = smiles_batch[: args.limit - total]

        shard_data.extend(tokenize_batch(smiles_batch, tokenizer, args.max_length))
        total += len(smiles_batch)

        if len(shard_data) >= args.shard_size:
            write_shard(shard_data, out_dir, shard_idx)
            shard_data = []
            shard_idx += 1

    if shard_data:
        write_shard(shard_data, out_dir, shard_idx)

    logger.info("Done. Total samples tokenized: %d in %d shards", total, shard_idx + (1 if shard_data else 0))


if __name__ == "__main__":
    main()
