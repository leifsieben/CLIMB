import argparse
import logging
from pathlib import Path
from tokenizers import ByteLevelBPETokenizer
from transformers import PreTrainedTokenizerFast

logger = logging.getLogger(__name__)


def setup_logging(log_file: str = None, level: str = "INFO"):
    """Setup logging configuration"""
    log_format = "%(asctime)s - %(levelname)s - %(message)s"
    handlers = [logging.StreamHandler()]

    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(level=getattr(logging, level), format=log_format, handlers=handlers)


def train_tokenizer(
    input_files: list[str],
    output_dir: str,
    vocab_size: int = 1000,
    min_frequency: int = 2,
):
    """Train BPE tokenizer on chemical SMILES"""


    for file in input_files:
        if not Path(file).exists():
            raise FileNotFoundError(f"Input file not found: {file}")

    logger.info(f"Training tokenizer on {len(input_files)} files")
    logger.info(f"Vocab size: {vocab_size}, Min frequency: {min_frequency}")


    tokenizer = ByteLevelBPETokenizer()
    special_tokens = ["<s>", "<pad>", "</s>", "<unk>", "<mask>"]

    tokenizer.train(
        files=input_files,
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=special_tokens,
    )


    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(output_path / "tokenizer.json"))


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
