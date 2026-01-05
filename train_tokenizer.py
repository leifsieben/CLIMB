# ==============================================================================
# FILE: train_tokenizer.py
# Train BPE tokenizer on SMILES data
# ==============================================================================

"""train_tokenizer.py

Usage:
    python train_tokenizer.py --input data/smiles.smi --output tokenizer/ --vocab_size 1000
"""

import argparse
import logging
from pathlib import Path
from tokenizers import ByteLevelBPETokenizer
from transformers import PreTrainedTokenizerFast
from utils import setup_logging

logger = logging.getLogger(__name__)


def train_tokenizer(
    input_files: list[str],
    output_dir: str,
    vocab_size: int = 1000,
    min_frequency: int = 2,
):
    """Train BPE tokenizer on chemical SMILES"""
    
    # Validate inputs
    for file in input_files:
        if not Path(file).exists():
            raise FileNotFoundError(f"Input file not found: {file}")
    
    logger.info(f"Training tokenizer on {len(input_files)} files")
    logger.info(f"Vocab size: {vocab_size}, Min frequency: {min_frequency}")
    
    # Initialize and train
    tokenizer = ByteLevelBPETokenizer()
    special_tokens = ["<s>", "<pad>", "</s>", "<unk>", "<mask>"]
    
    tokenizer.train(
        files=input_files,
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=special_tokens,
    )
    
    # Save
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(output_path / "tokenizer.json"))
    
    # Convert to HuggingFace format
    hf_tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=str(output_path / "tokenizer.json"),
        bos_token="<s>",
        eos_token="</s>",
        unk_token="<unk>",
        pad_token="<pad>",
        mask_token="<mask>",
    )
    hf_tokenizer.save_pretrained(str(output_path))
    
    logger.info(f"✓ Tokenizer saved to {output_dir}")
    logger.info(f"  Vocab size: {len(hf_tokenizer)}")
    
    return hf_tokenizer


def main():
    parser = argparse.ArgumentParser(description="Train BPE tokenizer")
    parser.add_argument("--input", nargs="+", required=True, help="Input SMILES files")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--vocab_size", type=int, default=1000)
    parser.add_argument("--min_frequency", type=int, default=2)
    parser.add_argument("--log_file", help="Log file path")
    args = parser.parse_args()
    
    setup_logging(args.log_file)
    train_tokenizer(args.input, args.output, args.vocab_size, args.min_frequency)


if __name__ == "__main__":
    main()
