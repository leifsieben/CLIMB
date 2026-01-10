"""config.py"""

import json
from dataclasses import dataclass, asdict, field
from typing import Optional, List, Dict, Any
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


# ==============================================================================
# Multi-Task Learning Configuration
# ==============================================================================

@dataclass
class TaskConfig:
    """Configuration for a single supervised task"""
    name: str
    task_type: str  # "regression", "binary_classification", "multiclass_classification"
    num_classes: int = 1  # For multiclass; ignored for binary/regression
    loss_weight: float = 1.0
    metric: str = ""  # Primary eval metric (e.g., "roc_auc", "rmse")
    description: str = ""

    def __post_init__(self):
        valid_types = ["regression", "binary_classification", "multiclass_classification"]
        if self.task_type not in valid_types:
            raise ValueError(f"task_type must be one of {valid_types}, got '{self.task_type}'")


@dataclass
class DataSourceConfig:
    """Configuration for a data source with its associated tasks"""
    path: str  # Path to data file (pickle, csv, etc.)
    tasks: List[str]  # Task names this data source provides labels for
    smiles_column: str = "SMILES"  # Column name for SMILES strings
    split_column: Optional[str] = None  # Optional column for train/val/test splits


@dataclass
class MultiTaskTrainingConfig:
    """Training configuration for multi-task learning"""
    output_dir: str = "./experiments/multitask"
    num_epochs: int = 50
    batch_size: int = 32
    learning_rate: float = 2e-5
    warmup_ratio: float = 0.1  # Warmup as fraction of total steps
    weight_decay: float = 0.01
    eval_steps: int = 500
    save_steps: int = 1000
    logging_steps: int = 100
    fp16: bool = True
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    dataloader_num_workers: int = 0
    early_stopping_patience: int = 10  # Epochs without improvement before stopping
    freeze_encoder: bool = False  # Whether to freeze encoder weights


@dataclass
class MultiTaskConfig:
    """
    Complete configuration for multi-task learning experiments.

    Can be loaded from YAML:
        config = MultiTaskConfig.from_yaml("config.yaml")
    """
    pretrained_model_path: Optional[str] = None  # Path to pretrained MLM model (optional)
    tasks: List[TaskConfig] = field(default_factory=list)
    data_sources: List[DataSourceConfig] = field(default_factory=list)
    training: MultiTaskTrainingConfig = field(default_factory=MultiTaskTrainingConfig)
    tokenizer_path: Optional[str] = None  # If different from model path

    def __post_init__(self):
        # Convert dict configs to dataclass instances if needed
        if self.tasks and isinstance(self.tasks[0], dict):
            self.tasks = [TaskConfig(**t) for t in self.tasks]
        if self.data_sources and isinstance(self.data_sources[0], dict):
            self.data_sources = [DataSourceConfig(**d) for d in self.data_sources]
        if isinstance(self.training, dict):
            self.training = MultiTaskTrainingConfig(**self.training)

    @property
    def task_names(self) -> List[str]:
        """List of all task names"""
        return [t.name for t in self.tasks]

    def get_task(self, name: str) -> TaskConfig:
        """Get task config by name"""
        for task in self.tasks:
            if task.name == name:
                return task
        raise KeyError(f"Task '{name}' not found. Available: {self.task_names}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'pretrained_model_path': self.pretrained_model_path,
            'tokenizer_path': self.tokenizer_path,
            'tasks': [asdict(t) for t in self.tasks],
            'data_sources': [asdict(d) for d in self.data_sources],
            'training': asdict(self.training),
        }

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> 'MultiTaskConfig':
        """Create from dictionary"""
        return cls(
            pretrained_model_path=config.get('pretrained_model_path'),
            tokenizer_path=config.get('tokenizer_path'),
            tasks=[TaskConfig(**t) for t in config.get('tasks', [])],
            data_sources=[DataSourceConfig(**d) for d in config.get('data_sources', [])],
            training=MultiTaskTrainingConfig(**config.get('training', {})),
        )

    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'MultiTaskConfig':
        """Create from YAML file"""
        import yaml
        with open(yaml_path) as f:
            config = yaml.safe_load(f)
        return cls.from_dict(config)

    def save_yaml(self, yaml_path: str) -> None:
        """Save configuration to YAML file"""
        import yaml
        with open(yaml_path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)


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


@dataclass
class ComputeBudgetConfig:
    """Compute budget split between supervised and unsupervised phases."""
    total_epochs: int = 50
    supervised_fraction: float = 0.5

    def __post_init__(self):
        if self.total_epochs <= 0:
            raise ValueError("total_epochs must be > 0")
        if not (0.0 <= self.supervised_fraction <= 1.0):
            raise ValueError("supervised_fraction must be between 0 and 1")

    @property
    def unsupervised_fraction(self) -> float:
        """Fraction of budget devoted to MLM pretraining."""
        return 1.0 - self.supervised_fraction


@dataclass
class MLMTrainingConfig:
    """Configuration for the MLM (unsupervised) training phase."""
    batch_size: int = 32
    learning_rate: float = 5e-5
    warmup_steps: int = 1000
    weight_decay: float = 0.0
    mlm_probability: float = 0.15
    logging_steps: int = 100
    save_steps: int = 500
    eval_steps: int = 0
    fp16: bool = False
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    save_total_limit: int = 2
    dataloader_num_workers: int = 0
    evaluation_strategy: str = "no"
    shuffle: bool = False
    streaming_max_samples: Optional[int] = None


@dataclass
class PretrainingConfig:
    """High-level configuration for the combined pretraining pipeline."""
    name: str = "pretraining"
    output_dir: str = "./experiments/pretraining"
    tokenizer_path: str = "./tokenizer"
    model: ModelConfig = field(default_factory=ModelConfig)
    compute_budget: ComputeBudgetConfig = field(default_factory=ComputeBudgetConfig)
    mlm_training: MLMTrainingConfig = field(default_factory=MLMTrainingConfig)
    unsupervised_data: List[str] = field(default_factory=list)
    tasks: List[Dict[str, Any]] = field(default_factory=list)
    data_sources: List[Dict[str, Any]] = field(default_factory=list)
    supervised_training: MultiTaskTrainingConfig = field(default_factory=MultiTaskTrainingConfig)
    validation_fraction: float = 0.1

    def __post_init__(self):
        if isinstance(self.model, dict):
            self.model = ModelConfig(**self.model)
        if isinstance(self.compute_budget, dict):
            self.compute_budget = ComputeBudgetConfig(**self.compute_budget)
        if isinstance(self.mlm_training, dict):
            self.mlm_training = MLMTrainingConfig(**self.mlm_training)

        if not self.unsupervised_data:
            self.unsupervised_data = []

        if isinstance(self.supervised_training, dict):
            self.supervised_training = MultiTaskTrainingConfig(**self.supervised_training)

        if not (0.0 <= self.validation_fraction <= 0.5):
            raise ValueError("validation_fraction must be between 0 and 0.5")

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> 'PretrainingConfig':
        if 'tokenizer_path' not in config:
            raise ValueError("pretraining config must define 'tokenizer_path'")

        return cls(
            name=config.get('name', 'pretraining'),
            output_dir=config.get('output_dir', './experiments/pretraining'),
            tokenizer_path=config['tokenizer_path'],
            model=config.get('model', {}),
            compute_budget=config.get('compute_budget', {}),
            mlm_training=config.get('mlm_training', {}),
            unsupervised_data=config.get('unsupervised_data', []),
            tasks=config.get('tasks', []),
            data_sources=config.get('data_sources', []),
            supervised_training=config.get('supervised_training', {}),
            validation_fraction=config.get('validation_fraction', 0.1),
        )

    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'PretrainingConfig':
        import yaml
        with open(yaml_path) as f:
            config = yaml.safe_load(f)
        return cls.from_dict(config)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'output_dir': self.output_dir,
            'tokenizer_path': self.tokenizer_path,
            'model': asdict(self.model),
            'compute_budget': asdict(self.compute_budget),
            'mlm_training': asdict(self.mlm_training),
            'unsupervised_data': self.unsupervised_data,
            'tasks': self.tasks,
            'data_sources': self.data_sources,
            'supervised_training': asdict(self.supervised_training),
            'validation_fraction': self.validation_fraction,
        }

    def save_yaml(self, path: str) -> None:
        import yaml
        with open(path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)
