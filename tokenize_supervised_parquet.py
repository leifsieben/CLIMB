#!/usr/bin/env python3
"""Tokenize a supervised wide parquet and write tokenized parquet shards.

Outputs parquet shards that include input_ids and attention_mask columns plus
all original label columns (optionally dropping the SMILES column).
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List
import glob

import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.dataset as ds
from transformers import PreTrainedTokenizerFast


logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def detect_smiles_column(columns: List[str]) -> str:
    candidates = ["smiles_canon", "SMILES_std", "SMILES", "smiles"]
    for name in candidates:
        if name in columns:
            return name
    raise ValueError(f"No SMILES column found. Tried: {candidates}")


def open_writer(out_dir: Path, shard_idx: int, schema: pa.Schema) -> pq.ParquetWriter:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"shard_{shard_idx:05d}.parquet"
    logger.info("Writing %s", out_path)
    return pq.ParquetWriter(out_path, schema, compression="zstd")


def main() -> None:
    parser = argparse.ArgumentParser(description="Tokenize supervised wide parquet to tokenized shards")
    parser.add_argument("--tokenizer", required=True, help="Directory containing tokenizer.json")
    parser.add_argument("--input", required=True, nargs="+", help="Parquet file(s) or glob patterns")
    parser.add_argument("--output-dir", required=True, help="Output directory for tokenized shards")
    parser.add_argument("--smiles-col", default=None, help="Override SMILES column name")
    parser.add_argument("--max-length", type=int, default=512, help="Max token length")
    parser.add_argument("--batch-rows", type=int, default=4096, help="Rows per read batch")
    parser.add_argument("--shard-rows", type=int, default=200_000, help="Rows per output shard")
    parser.add_argument("--drop-smiles", action="store_true", help="Drop SMILES column in output")
    parser.add_argument("--limit", type=int, default=None, help="Optional total row limit for testing")
    args = parser.parse_args()

    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=str(Path(args.tokenizer) / "tokenizer.json"),
        bos_token="<s>",
        eos_token="</s>",
        unk_token="<unk>",
        pad_token="<pad>",
        mask_token="<mask>",
    )

    files: List[str] = []
    for pat in args.input:
        files.extend(sorted(glob.glob(pat)))
    if not files:
        raise FileNotFoundError(f"No files matched: {args.input}")

    out_dir = Path(args.output_dir)
    shard_idx = 0
    rows_in_shard = 0
    writer: pq.ParquetWriter | None = None
    total_rows = 0

    for fp in files:
        dataset = ds.dataset(fp, format="parquet")
        columns = dataset.schema.names
        smiles_col = args.smiles_col or detect_smiles_column(columns)
        logger.info("Tokenizing %s (smiles_col=%s)", fp, smiles_col)

        for batch in dataset.to_batches(batch_size=args.batch_rows):
            if args.limit is not None and total_rows >= args.limit:
                break

            table = batch.to_table()
            smiles = table[smiles_col].to_pylist()
            if args.limit is not None and total_rows + len(smiles) > args.limit:
                keep = args.limit - total_rows
                table = table.slice(0, keep)
                smiles = smiles[:keep]

            enc = tokenizer(
                smiles,
                truncation=True,
                max_length=args.max_length,
                padding="max_length",
                return_attention_mask=True,
            )
            input_ids = pa.array(enc["input_ids"], type=pa.list_(pa.int32()))
            attn_mask = pa.array(enc["attention_mask"], type=pa.list_(pa.int8()))
            table = table.append_column("input_ids", input_ids)
            table = table.append_column("attention_mask", attn_mask)

            if args.drop_smiles and smiles_col in table.column_names:
                table = table.drop([smiles_col])

            if writer is None:
                writer = open_writer(out_dir, shard_idx, table.schema)

            if rows_in_shard + table.num_rows > args.shard_rows and rows_in_shard > 0:
                writer.close()
                shard_idx += 1
                rows_in_shard = 0
                writer = open_writer(out_dir, shard_idx, table.schema)

            writer.write_table(table)
            rows_in_shard += table.num_rows
            total_rows += table.num_rows

        if args.limit is not None and total_rows >= args.limit:
            break

    if writer is not None:
        writer.close()

    logger.info("Done. Total rows tokenized: %d in %d shard(s)", total_rows, shard_idx + 1)


if __name__ == "__main__":
    main()
