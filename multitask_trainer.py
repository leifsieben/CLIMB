# ==============================================================================
# FILE: multitask_trainer.py
# Multi-task training with HuggingFace Trainer
# ==============================================================================

"""
Multi-task training using HuggingFace Trainer.

The MultiTaskModel.forward_supervised() returns 'loss' directly,
which Trainer expects. This module provides helper functions to
set up training with the right configuration.

Example:
    model = MultiTaskModel.from_pretrained_encoder(...)
    dataset = MultiTaskDataset.load(...)

    results = train_multitask(
        model=model,
        train_dataset=dataset,
        output_dir="./output",
        num_epochs=50,
    )
"""

import logging
import os
import json
from typing import Dict, Optional, Any
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments

from multitask_model import MultiTaskModel
from multitask_data import MultiTaskDataset, MultiTaskCollator

logger = logging.getLogger(__name__)


class MultiTaskTrainer(Trainer):
    """
    HuggingFace Trainer adapted for multi-task learning.

    Overrides compute_loss to handle the dict-based label format.
    """

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """
        Compute multi-task loss.

        The model expects:
        - input_ids, attention_mask
        - labels: Dict[task_name -> tensor]
        - label_masks: Dict[task_name -> tensor]
        """
        outputs = model.forward_supervised(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            labels=inputs.get('labels'),
            label_masks=inputs.get('label_masks'),
        )

        loss = outputs.get('loss')
        if loss is None:
            # No valid labels in batch - return zero loss
            loss = torch.tensor(0.0, device=inputs['input_ids'].device, requires_grad=True)

        return (loss, outputs) if return_outputs else loss


def get_training_args(
    output_dir: str,
    num_epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 2e-5,
    warmup_ratio: float = 0.1,
    weight_decay: float = 0.01,
    logging_steps: int = 100,
    save_steps: int = 1000,
    eval_steps: int = 500,
    save_total_limit: int = 3,
    fp16: bool = False,
    gradient_accumulation_steps: int = 1,
    max_grad_norm: float = 1.0,
    dataloader_num_workers: int = 0,
    eval_strategy: str = "steps",
    load_best_model_at_end: bool = True,
    metric_for_best_model: str = "loss",
    greater_is_better: bool = False,
    **kwargs,
) -> TrainingArguments:
    """
    Create TrainingArguments for multi-task training.

    Device selection is automatic (CUDA > MPS > CPU).
    """
    return TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        warmup_ratio=warmup_ratio,
        weight_decay=weight_decay,
        logging_dir=os.path.join(output_dir, "logs"),
        logging_steps=logging_steps,
        save_steps=save_steps,
        eval_steps=eval_steps if eval_strategy != "no" else None,
        save_total_limit=save_total_limit,
        fp16=fp16 and torch.cuda.is_available(),
        gradient_accumulation_steps=gradient_accumulation_steps,
        max_grad_norm=max_grad_norm,
        dataloader_num_workers=dataloader_num_workers,
        eval_strategy=eval_strategy,
        load_best_model_at_end=load_best_model_at_end if eval_strategy != "no" else False,
        metric_for_best_model=metric_for_best_model,
        greater_is_better=greater_is_better,
        remove_unused_columns=False,  # Keep all columns for multi-task
        **kwargs,
    )


def train_multitask(
    model: MultiTaskModel,
    train_dataset: MultiTaskDataset,
    val_dataset: Optional[MultiTaskDataset] = None,
    output_dir: str = "./output",
    num_epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 2e-5,
    save_encoder: bool = True,
    **kwargs,
) -> Dict[str, Any]:
    """
    Train multi-task model using HuggingFace Trainer.

    Args:
        model: MultiTaskModel instance
        train_dataset: Training dataset
        val_dataset: Validation dataset (optional)
        output_dir: Output directory for checkpoints
        num_epochs: Number of training epochs
        batch_size: Batch size per device
        learning_rate: Learning rate
        save_encoder: Whether to save encoder separately after training
        **kwargs: Additional TrainingArguments parameters

    Returns:
        Dict with training metrics
    """
    os.makedirs(output_dir, exist_ok=True)

    # Log configuration
    logger.info("=" * 80)
    logger.info("MULTI-TASK TRAINING")
    logger.info("=" * 80)
    logger.info(f"Tasks: {list(model.task_heads.keys())}")
    logger.info(f"Training samples: {len(train_dataset)}")
    if val_dataset:
        logger.info(f"Validation samples: {len(val_dataset)}")
    logger.info(f"Epochs: {num_epochs}")
    logger.info(f"Batch size: {batch_size}")
    logger.info(f"Learning rate: {learning_rate}")

    # Training arguments
    eval_strategy = "steps" if val_dataset else "no"
    training_args = get_training_args(
        output_dir=output_dir,
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        eval_strategy=eval_strategy,
        **kwargs,
    )

    # Collator
    collator = MultiTaskCollator()

    # Trainer
    trainer = MultiTaskTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collator,
    )

    # Train
    logger.info("Starting training...")
    train_result = trainer.train()

    # Save final model
    trainer.save_model(os.path.join(output_dir, "final"))

    # Save encoder separately (primary output)
    if save_encoder:
        encoder_path = os.path.join(output_dir, "encoder")
        model.save_encoder(encoder_path)
        logger.info(f"Saved encoder to {encoder_path}")

    # Save training results
    results = {
        'train_loss': train_result.training_loss,
        'train_samples': len(train_dataset),
        'num_epochs': num_epochs,
        'global_step': train_result.global_step,
    }

    with open(os.path.join(output_dir, "training_results.json"), 'w') as f:
        json.dump(results, f, indent=2)

    logger.info("=" * 80)
    logger.info("TRAINING COMPLETE")
    logger.info(f"Final loss: {train_result.training_loss:.4f}")
    logger.info("=" * 80)

    return results


def evaluate_multitask(
    model: MultiTaskModel,
    dataset: MultiTaskDataset,
    batch_size: int = 32,
) -> Dict[str, float]:
    """
    Evaluate multi-task model.

    Returns:
        Dict with evaluation metrics
    """
    training_args = TrainingArguments(
        output_dir="./tmp_eval",
        per_device_eval_batch_size=batch_size,
        remove_unused_columns=False,
    )

    trainer = MultiTaskTrainer(
        model=model,
        args=training_args,
        data_collator=MultiTaskCollator(),
    )

    metrics = trainer.evaluate(dataset)
    return metrics
