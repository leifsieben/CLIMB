"""config.py"""

import json
from dataclasses import dataclass, asdict
from typing import Optional
from pathlib import Path


@dataclass
class TokenizerConfig:
    vocab_size: int = 1000
    min_frequency: int = 2
    max_length: int = 512


@dataclass
class ModelConfig:
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1
    max_position_embeddings: int = 512
    use_flash_attention: bool = False


@dataclass
class TrainingConfig:
    output_dir: str = "./experiments"
    num_epochs: int = 10
    batch_size: int = 32
    learning_rate: float = 5e-5
    warmup_steps: int = 500
    weight_decay: float = 0.01
    mlm_probability: float = 0.15
    eval_steps: int = 500
    save_steps: int = 1000
    logging_steps: int = 100
    fp16: bool = True
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0


@dataclass
class ExperimentConfig:
    """Configuration for a specific experiment"""
    name: str
    unsupervised_weight: float  # 0.0 to 1.0
    supervised_weight: float    # 0.0 to 1.0
    task_type: str = "mixed"  # "unsupervised", "supervised", "mixed"
    pretrained_model_path: Optional[str] = None  # For fine-tuning
    
    def __post_init__(self):
        assert 0 <= self.unsupervised_weight <= 1.0
        assert 0 <= self.supervised_weight <= 1.0


class ConfigManager:
    """Manage and save/load configurations"""
    
    def __init__(self, config_dir: str = "./configs"):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)
    
    def save_config(self, config, name: str):
        """Save any config to JSON"""
        path = self.config_dir / f"{name}.json"
        with open(path, 'w') as f:
            json.dump(asdict(config), f, indent=2)
        print(f"Config saved to {path}")
    
    def load_config(self, name: str, config_class):
        """Load config from JSON"""
        path = self.config_dir / f"{name}.json"
        with open(path, 'r') as f:
            data = json.load(f)
        return config_class(**data)

