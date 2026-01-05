# ==============================================================================
# FILE: preprocess_data.py
# Tokenize and save datasets
# ==============================================================================

"""preprocess_data.py

Usage:
    # Unsupervised data
    python preprocess_data.py --tokenizer tokenizer/ --input data/smiles.smi --output data/unsup.pkl
    
    # Supervised data
    python preprocess_data.py --tokenizer tokenizer/ --input data/supervised.csv --output data/sup.pkl --supervised
"""

import argparse
import logging
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from transformers import PreTrainedTokenizerFast
from utils import setup_logging

logger = logging.getLogger(__name__)


def load_smiles_file(file_path: str):
    """Load SMILES from various file formats"""
    path = Path(file_path)
    suffix = path.suffix.lower()
    
    if suffix in ['.smi', '.txt']:
        # Simple text file, one SMILES per line
        try:
            df = pd.read_csv(file_path, sep='\s+', header=None, names=['SMILES', 'Name'])
            smiles = df['SMILES'].tolist()
        except:
            # Fallback: just SMILES, no names
            with open(file_path) as f:
                smiles = [line.strip() for line in f if line.strip()]
    elif suffix == '.csv':
        df = pd.read_csv(file_path)
        smiles = df['SMILES'].tolist() if 'SMILES' in df.columns else df.iloc[:, 0].tolist()
    else:
        raise ValueError(f"Unsupported file format: {suffix}")
    
    logger.info(f"Loaded {len(smiles)} SMILES from {file_path}")
    return smiles


def load_supervised_data(file_path: str):
    """Load supervised dataset with gene expression data"""
    df = pd.read_csv(file_path)
    smiles = df['SMILES'].tolist()
    gene_columns = [col for col in df.columns if col.startswith('geneID-')]
    labels = df[gene_columns].values
    
    logger.info(f"Loaded supervised data: {len(smiles)} samples, {len(gene_columns)} genes")
    return smiles, labels, gene_columns


def tokenize_data(smiles_list: list[str], tokenizer, max_length: int = 512):
    """Tokenize SMILES strings"""
    logger.info(f"Tokenizing {len(smiles_list)} SMILES...")
    
    tokenized = []
    for smiles in smiles_list:
        encoding = tokenizer(
            smiles,
            truncation=True,
            max_length=max_length,
            padding=False,
            return_tensors=None,
        )
        tokenized.append({
            "input_ids": encoding["input_ids"],
            "attention_mask": encoding["attention_mask"],
        })
    
    seq_lengths = [len(item['input_ids']) for item in tokenized]
    logger.info(f"✓ Tokenized {len(tokenized)} samples")
    logger.info(f"  Seq length - Min: {min(seq_lengths)}, Max: {max(seq_lengths)}, Mean: {np.mean(seq_lengths):.1f}")
    
    return tokenized


def main():
    parser = argparse.ArgumentParser(description="Preprocess and tokenize data")
    parser.add_argument("--tokenizer", required=True, help="Tokenizer directory")
    parser.add_argument("--input", required=True, help="Input SMILES file")
    parser.add_argument("--output", required=True, help="Output pickle file")
    parser.add_argument("--supervised", action="store_true", help="Supervised data with labels")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--log_file", help="Log file path")
    args = parser.parse_args()
    
    setup_logging(args.log_file)
    
    # Load tokenizer
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=str(Path(args.tokenizer) / "tokenizer.json"),
        bos_token="<s>",
        eos_token="</s>",
        unk_token="<unk>",
        pad_token="<pad>",
        mask_token="<mask>",
    )
    
    # Process data
    if args.supervised:
        smiles, labels, gene_names = load_supervised_data(args.input)
        tokenized = tokenize_data(smiles, tokenizer, args.max_length)
        data = {'data': tokenized, 'labels': labels, 'gene_names': gene_names}
    else:
        smiles = load_smiles_file(args.input)
        tokenized = tokenize_data(smiles, tokenizer, args.max_length)
        data = {'data': tokenized}
    
    # Save
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'wb') as f:
        pickle.dump(data, f)
    
    logger.info(f"✓ Saved to {args.output}")


if __name__ == "__main__":
    main()