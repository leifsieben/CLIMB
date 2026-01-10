# ==============================================================================
# FILE: utils.py
# Shared utilities used across all scripts
# ==============================================================================

"""utils.py"""

import logging
import signal
import sys
import yaml
import torch
from pathlib import Path
from typing import Dict, Any
from transformers import Trainer


def setup_logging(log_file: str = None, level: str = "INFO"):
    """Setup logging configuration"""
    log_format = "%(asctime)s - %(levelname)s - %(message)s"
    handlers = [logging.StreamHandler()]
    
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(level=getattr(logging, level), format=log_format, handlers=handlers)


def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML config file"""
    with open(config_path) as f:
        return yaml.safe_load(f)


def get_device():
    """Get available device (MPS for Mac, CUDA for GPU, CPU otherwise)"""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        # Check both is_available() and is_built() for MPS
        try:
            # Test that MPS actually works by creating a small tensor
            torch.zeros(1, device="mps")
            return "mps"
        except Exception:
            pass
    return "cpu"


def register_spot_handler(trainer: Trainer, output_dir: str):
    """Register a SIGTERM handler that saves a checkpoint for spot instances."""
    def _handler(signum, frame):
        logger = logging.getLogger(__name__)
        logger.warning("Spot interruption detected (SIGTERM). Saving checkpoint before exit.")
        ckpt_dir = Path(output_dir) / "spot_checkpoint"
        trainer.save_model(str(ckpt_dir))
        trainer.save_state()
        logger.info(f"Checkpoint saved to {ckpt_dir}")
        sys.exit(0)

    signal.signal(signal.SIGTERM, _handler)
