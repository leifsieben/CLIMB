# ==============================================================================
# FILE: tasks.py
# Task specification system for multi-task learning
# ==============================================================================

"""
Task specification system for multi-task chemical property prediction.

This module defines:
- TaskType: Enum for task types (regression, binary/multiclass classification)
- TaskSpec: Dataclass for individual task specifications
- TaskRegistry: Registry for managing multiple tasks

Example:
    registry = TaskRegistry()
    registry.register(TaskSpec(
        name="BBBP",
        task_type=TaskType.BINARY_CLASSIFICATION,
        metric="roc_auc",
    ))
    registry.register(TaskSpec(
        name="ESOL",
        task_type=TaskType.REGRESSION,
        metric="rmse",
    ))
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Callable
from enum import Enum
import torch.nn as nn


class TaskType(Enum):
    """Types of prediction tasks supported"""
    REGRESSION = "regression"
    BINARY_CLASSIFICATION = "binary_classification"
    MULTICLASS_CLASSIFICATION = "multiclass_classification"


@dataclass
class TaskSpec:
    """
    Specification for a single prediction task.

    Attributes:
        name: Unique identifier for the task
        task_type: Type of prediction (regression, classification)
        num_classes: Number of classes for classification, or output dim for regression
        loss_weight: Weight in combined loss (default 1.0 for equal weighting)
        metric: Primary evaluation metric (e.g., "roc_auc", "rmse", "accuracy")
        description: Human-readable description of the task
    """
    name: str
    task_type: TaskType
    num_classes: int = 1
    loss_weight: float = 1.0
    metric: str = ""
    description: str = ""

    def __post_init__(self):
        """Validate and adjust settings based on task type"""
        if self.task_type == TaskType.BINARY_CLASSIFICATION:
            self.num_classes = 2  # Enforce binary

        # Set default metrics if not specified
        if not self.metric:
            if self.task_type == TaskType.REGRESSION:
                self.metric = "rmse"
            else:
                self.metric = "roc_auc"

    @property
    def output_dim(self) -> int:
        """
        Output dimension for the task head.

        Returns:
            int: Number of output units needed for this task
        """
        if self.task_type == TaskType.REGRESSION:
            return self.num_classes  # For regression, num_classes is output dim
        elif self.task_type == TaskType.BINARY_CLASSIFICATION:
            return 2  # Logits for 2 classes
        else:  # MULTICLASS_CLASSIFICATION
            return self.num_classes

    def get_loss_fn(self) -> nn.Module:
        """
        Return the appropriate loss function for this task.

        All loss functions use reduction='none' to allow per-sample masking.

        Returns:
            nn.Module: Loss function instance
        """
        if self.task_type == TaskType.REGRESSION:
            return nn.MSELoss(reduction='none')
        else:  # Classification (binary or multiclass)
            return nn.CrossEntropyLoss(reduction='none')

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'name': self.name,
            'type': self.task_type.value,
            'num_classes': self.num_classes,
            'loss_weight': self.loss_weight,
            'metric': self.metric,
            'description': self.description,
        }

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> 'TaskSpec':
        """Create TaskSpec from dictionary"""
        return cls(
            name=config['name'],
            task_type=TaskType(config['type']),
            num_classes=config.get('num_classes', 1),
            loss_weight=config.get('loss_weight', 1.0),
            metric=config.get('metric', ''),
            description=config.get('description', ''),
        )


class TaskRegistry:
    """
    Registry for managing multiple task specifications.

    Provides methods to register, retrieve, and iterate over tasks.
    Can be created from YAML/dict configuration.

    Example:
        registry = TaskRegistry.from_config({
            'tasks': [
                {'name': 'BBBP', 'type': 'binary_classification'},
                {'name': 'ESOL', 'type': 'regression'},
            ]
        })

        for task in registry:
            print(f"{task.name}: {task.task_type}")
    """

    def __init__(self):
        self._tasks: Dict[str, TaskSpec] = {}

    def register(self, task: TaskSpec) -> None:
        """
        Register a task specification.

        Args:
            task: TaskSpec to register

        Raises:
            ValueError: If task name already exists
        """
        if task.name in self._tasks:
            raise ValueError(f"Task '{task.name}' already registered")
        self._tasks[task.name] = task

    def get(self, name: str) -> TaskSpec:
        """
        Get task by name.

        Args:
            name: Task identifier

        Returns:
            TaskSpec for the given name

        Raises:
            KeyError: If task not found
        """
        if name not in self._tasks:
            raise KeyError(f"Task '{name}' not found. Available: {list(self._tasks.keys())}")
        return self._tasks[name]

    def get_all(self) -> Dict[str, TaskSpec]:
        """Get all registered tasks as a dictionary"""
        return self._tasks.copy()

    @property
    def task_names(self) -> List[str]:
        """List of all registered task names"""
        return list(self._tasks.keys())

    def __len__(self) -> int:
        return len(self._tasks)

    def __iter__(self):
        return iter(self._tasks.values())

    def __contains__(self, name: str) -> bool:
        return name in self._tasks

    def to_dict(self) -> Dict[str, Any]:
        """Convert registry to dictionary for serialization"""
        return {
            'tasks': [task.to_dict() for task in self._tasks.values()]
        }

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'TaskRegistry':
        """
        Create registry from config dictionary.

        Args:
            config: Dictionary with 'tasks' key containing list of task configs

        Returns:
            TaskRegistry with all tasks registered
        """
        registry = cls()
        for task_config in config.get('tasks', []):
            task = TaskSpec.from_dict(task_config)
            registry.register(task)
        return registry

    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'TaskRegistry':
        """
        Create registry from YAML file.

        Args:
            yaml_path: Path to YAML configuration file

        Returns:
            TaskRegistry with all tasks registered
        """
        import yaml
        with open(yaml_path) as f:
            config = yaml.safe_load(f)
        return cls.from_config(config)


# Convenience function for common MoleculeNet tasks
def create_moleculenet_registry() -> TaskRegistry:
    """
    Create a TaskRegistry with common MoleculeNet benchmark tasks.

    Returns:
        TaskRegistry pre-populated with standard MoleculeNet tasks
    """
    registry = TaskRegistry()

    # Classification tasks
    classification_tasks = [
        ("BBBP", "Blood-brain barrier penetration"),
        ("BACE", "Beta-secretase 1 inhibitor"),
        ("HIV", "HIV replication inhibition"),
        ("ClinTox", "Clinical trial toxicity"),
    ]

    for name, desc in classification_tasks:
        registry.register(TaskSpec(
            name=name,
            task_type=TaskType.BINARY_CLASSIFICATION,
            metric="roc_auc",
            description=desc,
        ))

    # Regression tasks
    regression_tasks = [
        ("ESOL", "Aqueous solubility (log mol/L)"),
        ("FreeSolv", "Hydration free energy (kcal/mol)"),
        ("Lipophilicity", "Octanol/water partition coefficient"),
    ]

    for name, desc in regression_tasks:
        registry.register(TaskSpec(
            name=name,
            task_type=TaskType.REGRESSION,
            metric="rmse",
            description=desc,
        ))

    # Multi-task classification (Tox21 has 12 tasks, SIDER has 27)
    # These would typically be handled as separate binary tasks

    return registry
