# ==============================================================================
# FILE: utils.py
# Shared utilities used across all scripts
# ==============================================================================

"""utils.py"""

import logging
import yaml
import torch
from pathlib import Path
from typing import Dict, Any


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
