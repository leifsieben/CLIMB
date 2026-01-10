# ==============================================================================
# FILE: multitask_model.py
# Multi-task learning model with shared encoder
# ==============================================================================

"""
Multi-task learning model for chemical language pretraining.

Design Philosophy:
- The ENCODER is the primary output - task heads are training scaffolding
- Task heads are used during supervised pretraining, then discarded
- The saved encoder is used downstream (fine-tuning, inference)
- Supports mixing MLM (unsupervised) and task prediction (supervised) losses

Architecture during training:
    Input -> Encoder -> [CLS] pooled -> TaskHead(task1) -> supervised loss
                     -> MLM Head -> MLM loss (for unsupervised samples)

What we keep after pretraining:
    Input -> Encoder -> representations (for downstream tasks)

Example:
    # Create model with task heads for supervised pretraining
    model = MultiTaskModel.from_encoder_config(
        config=encoder_config,
        task_registry=registry,
    )

    # Train with supervised and/or MLM losses
    outputs = model(batch)
    loss = outputs['loss']

    # After pretraining, save only the encoder
    model.save_encoder("path/to/encoder")

    # Later, load encoder for downstream use
    encoder = RobertaModel.from_pretrained("path/to/encoder")
"""

import logging
import os
import json
from typing import Dict, Optional, Any, Union

import torch
import torch.nn as nn
from transformers import RobertaModel, RobertaConfig, RobertaForMaskedLM

from tasks import TaskType, TaskSpec, TaskRegistry

logger = logging.getLogger(__name__)


class TaskHead(nn.Module):
    """
    Task-specific prediction head.

    Architecture: pooled -> Dense -> GELU -> Dropout -> Output

    This is TRAINING SCAFFOLDING - discarded after pretraining.
    """

    def __init__(
        self,
        task_spec: TaskSpec,
        hidden_size: int,
        dropout_prob: float = 0.1,
    ):
        super().__init__()
        self.task_spec = task_spec
        self.task_type = task_spec.task_type
        self.output_dim = task_spec.output_dim

        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout_prob)
        self.output = nn.Linear(hidden_size, self.output_dim)
        self.loss_fn = task_spec.get_loss_fn()

    def forward(
        self,
        pooled: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            pooled: [batch, hidden] from encoder [CLS] token
            labels: [batch] ground truth (optional)
            mask: [batch] validity mask, 1=valid, 0=missing (optional)

        Returns:
            Dict with 'logits' and optionally 'loss', 'n_valid'
        """
        x = self.dropout(self.activation(self.dense(pooled)))
        logits = self.output(x)

        result = {'logits': logits}

        if labels is not None:
            # Prepare labels based on task type
            if self.task_type == TaskType.REGRESSION:
                pred = logits.squeeze(-1) if logits.dim() > 1 and logits.size(-1) == 1 else logits
                target = labels.float().squeeze(-1) if labels.dim() > 1 else labels.float()
            else:
                pred = logits
                target = labels.long().squeeze(-1) if labels.dim() > 1 else labels.long()

            per_sample_loss = self.loss_fn(pred, target)

            # Apply mask
            if mask is not None:
                mask_f = mask.float()
                n_valid = mask_f.sum()
                if n_valid > 0:
                    loss = (per_sample_loss * mask_f).sum() / n_valid
                else:
                    loss = torch.tensor(0.0, device=logits.device, requires_grad=True)
                result['n_valid'] = n_valid.item()
            else:
                loss = per_sample_loss.mean()
                result['n_valid'] = len(labels)

            result['loss'] = loss

        return result


class MultiTaskModel(nn.Module):
    """
    Multi-task model with shared encoder and detachable task heads.

    The encoder is the primary component. Task heads are temporary
    training scaffolding that can be discarded after pretraining.

    Supports:
    - Pure supervised: forward_supervised() with task labels
    - Pure MLM: forward_mlm() with masked tokens
    - Mixed: Both losses combined
    """

    def __init__(
        self,
        encoder: RobertaModel,
        task_registry: TaskRegistry,
        dropout_prob: float = 0.1,
        include_mlm_head: bool = False,
    ):
        """
        Args:
            encoder: Shared RobertaModel encoder
            task_registry: Registry defining supervised tasks
            dropout_prob: Dropout for task heads
            include_mlm_head: Whether to include MLM head for mixed training
        """
        super().__init__()
        self.encoder = encoder
        self.task_registry = task_registry
        self.hidden_size = encoder.config.hidden_size
        self.vocab_size = encoder.config.vocab_size

        # Task heads (supervised training scaffolding)
        self.task_heads = nn.ModuleDict()
        for task_spec in task_registry:
            self.task_heads[task_spec.name] = TaskHead(
                task_spec=task_spec,
                hidden_size=self.hidden_size,
                dropout_prob=dropout_prob,
            )

        # Optional MLM head for mixed training
        self.mlm_head = None
        if include_mlm_head:
            self._init_mlm_head()

        self._log_creation()

    def _init_mlm_head(self):
        """Initialize MLM prediction head."""
        # Standard RoBERTa LM head architecture
        self.mlm_head = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.GELU(),
            nn.LayerNorm(self.hidden_size),
            nn.Linear(self.hidden_size, self.vocab_size),
        )
        self.mlm_loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    def _log_creation(self):
        """Log model creation details."""
        logger.info(f"Created MultiTaskModel:")
        logger.info(f"  Encoder hidden_size: {self.hidden_size}")
        logger.info(f"  Task heads: {list(self.task_heads.keys())}")
        logger.info(f"  MLM head: {'yes' if self.mlm_head else 'no'}")
        for name, head in self.task_heads.items():
            logger.info(f"    {name}: {head.task_type.value} -> dim={head.output_dim}")

    def forward_supervised(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[Dict[str, torch.Tensor]] = None,
        label_masks: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, Any]:
        """
        Forward pass for supervised task prediction.

        Args:
            input_ids: [batch, seq_len] token IDs
            attention_mask: [batch, seq_len] attention mask
            labels: Dict[task_name -> [batch] labels]
            label_masks: Dict[task_name -> [batch] validity masks]

        Returns:
            Dict with 'loss', 'logits', 'task_losses'
        """
        # Encode
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0, :]  # [CLS] token

        # Forward through task heads
        all_logits = {}
        task_losses = {}
        total_loss = None

        for task_name, head in self.task_heads.items():
            task_labels = labels.get(task_name) if labels else None
            task_mask = label_masks.get(task_name) if label_masks else None

            head_out = head(pooled, labels=task_labels, mask=task_mask)
            all_logits[task_name] = head_out['logits']

            if 'loss' in head_out and head_out.get('n_valid', 0) > 0:
                task_losses[task_name] = head_out['loss']
                weight = self.task_registry.get(task_name).loss_weight
                weighted = head_out['loss'] * weight
                total_loss = weighted if total_loss is None else total_loss + weighted

        return {
            'loss': total_loss,
            'logits': all_logits,
            'task_losses': task_losses,
            'pooled_output': pooled,
        }

    def forward_mlm(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        mlm_labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """
        Forward pass for masked language modeling.

        Args:
            input_ids: [batch, seq_len] with some tokens masked
            attention_mask: [batch, seq_len]
            mlm_labels: [batch, seq_len] with -100 for non-masked positions

        Returns:
            Dict with 'loss', 'logits'
        """
        if self.mlm_head is None:
            raise RuntimeError("MLM head not initialized. Set include_mlm_head=True")

        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state  # [batch, seq, hidden]
        mlm_logits = self.mlm_head(sequence_output)  # [batch, seq, vocab]

        result = {'logits': mlm_logits}

        if mlm_labels is not None:
            loss = self.mlm_loss_fn(
                mlm_logits.view(-1, self.vocab_size),
                mlm_labels.view(-1),
            )
            result['loss'] = loss

        return result

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[Dict[str, torch.Tensor]] = None,
        label_masks: Optional[Dict[str, torch.Tensor]] = None,
        mlm_labels: Optional[torch.Tensor] = None,
        supervised_weight: float = 1.0,
        mlm_weight: float = 1.0,
    ) -> Dict[str, Any]:
        """
        Combined forward pass for mixed training.

        Can handle supervised-only, MLM-only, or mixed based on provided labels.

        Args:
            input_ids: [batch, seq_len]
            attention_mask: [batch, seq_len]
            labels: Dict[task_name -> [batch] labels] for supervised
            label_masks: Dict[task_name -> [batch] masks] for supervised
            mlm_labels: [batch, seq_len] for MLM (-100 = ignore)
            supervised_weight: Weight for supervised loss
            mlm_weight: Weight for MLM loss

        Returns:
            Dict with 'loss' (combined), 'supervised_loss', 'mlm_loss', etc.
        """
        result = {}
        total_loss = None

        # Supervised forward
        if labels is not None:
            sup_out = self.forward_supervised(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                label_masks=label_masks,
            )
            result['supervised_loss'] = sup_out.get('loss')
            result['task_losses'] = sup_out.get('task_losses', {})
            result['logits'] = sup_out['logits']
            result['pooled_output'] = sup_out['pooled_output']

            if sup_out.get('loss') is not None:
                total_loss = sup_out['loss'] * supervised_weight

        # MLM forward
        if mlm_labels is not None:
            mlm_out = self.forward_mlm(
                input_ids=input_ids,
                attention_mask=attention_mask,
                mlm_labels=mlm_labels,
            )
            result['mlm_loss'] = mlm_out['loss']
            result['mlm_logits'] = mlm_out['logits']

            mlm_contribution = mlm_out['loss'] * mlm_weight
            total_loss = mlm_contribution if total_loss is None else total_loss + mlm_contribution

        result['loss'] = total_loss
        return result

    # =========================================================================
    # Encoder extraction and saving
    # =========================================================================

    def get_encoder(self) -> RobertaModel:
        """Get the encoder (primary output of pretraining)."""
        return self.encoder

    def save_encoder(self, path: str) -> None:
        """
        Save ONLY the encoder for downstream use.

        This is what you use after pretraining - task heads are discarded.
        """
        os.makedirs(path, exist_ok=True)
        self.encoder.save_pretrained(path)
        logger.info(f"Saved encoder to {path}")

    def save_full(self, path: str) -> None:
        """
        Save complete model (encoder + task heads + registry).

        Use this for checkpointing during training.
        """
        os.makedirs(path, exist_ok=True)

        # Encoder
        self.encoder.save_pretrained(path)

        # Task heads
        torch.save(self.task_heads.state_dict(), os.path.join(path, "task_heads.pt"))

        # MLM head if present
        if self.mlm_head is not None:
            torch.save(self.mlm_head.state_dict(), os.path.join(path, "mlm_head.pt"))

        # Task registry
        with open(os.path.join(path, "task_registry.json"), 'w') as f:
            json.dump(self.task_registry.to_dict(), f, indent=2)

        # Model metadata
        meta = {
            'hidden_size': self.hidden_size,
            'vocab_size': self.vocab_size,
            'has_mlm_head': self.mlm_head is not None,
            'task_names': list(self.task_heads.keys()),
        }
        with open(os.path.join(path, "multitask_meta.json"), 'w') as f:
            json.dump(meta, f, indent=2)

        logger.info(f"Saved full MultiTaskModel to {path}")

    # =========================================================================
    # Loading methods
    # =========================================================================

    @classmethod
    def from_pretrained_encoder(
        cls,
        encoder_path: str,
        task_registry: TaskRegistry,
        dropout_prob: float = 0.1,
        include_mlm_head: bool = False,
        use_flash_attention: bool = True,
    ) -> 'MultiTaskModel':
        """
        Create model from a pretrained encoder (e.g., from MLM pretraining).

        Task heads are freshly initialized.
        """
        logger.info(f"Loading encoder from: {encoder_path}")

        config = RobertaConfig.from_pretrained(encoder_path)
        attn_impl = _get_attn_implementation(use_flash_attention)

        if attn_impl:
            encoder = RobertaModel.from_pretrained(
                encoder_path, config=config, attn_implementation=attn_impl
            )
        else:
            encoder = RobertaModel.from_pretrained(encoder_path, config=config)

        return cls(
            encoder=encoder,
            task_registry=task_registry,
            dropout_prob=dropout_prob,
            include_mlm_head=include_mlm_head,
        )

    @classmethod
    def from_encoder_config(
        cls,
        config: Union[RobertaConfig, dict],
        task_registry: TaskRegistry,
        dropout_prob: float = 0.1,
        include_mlm_head: bool = False,
        use_flash_attention: bool = True,
    ) -> 'MultiTaskModel':
        """
        Create model with randomly initialized encoder.
        """
        if isinstance(config, dict):
            config = RobertaConfig(**config)

        attn_impl = _get_attn_implementation(use_flash_attention)
        if attn_impl:
            encoder = RobertaModel(config, attn_implementation=attn_impl)
        else:
            encoder = RobertaModel(config)

        return cls(
            encoder=encoder,
            task_registry=task_registry,
            dropout_prob=dropout_prob,
            include_mlm_head=include_mlm_head,
        )

    @classmethod
    def from_checkpoint(
        cls,
        path: str,
        use_flash_attention: bool = True,
    ) -> 'MultiTaskModel':
        """
        Load from a full checkpoint (encoder + task heads + registry).
        """
        # Load metadata
        with open(os.path.join(path, "multitask_meta.json")) as f:
            meta = json.load(f)

        # Load task registry
        with open(os.path.join(path, "task_registry.json")) as f:
            registry_config = json.load(f)
        task_registry = TaskRegistry.from_config(registry_config)

        # Load encoder
        config = RobertaConfig.from_pretrained(path)
        attn_impl = _get_attn_implementation(use_flash_attention)

        if attn_impl:
            encoder = RobertaModel.from_pretrained(
                path, config=config, attn_implementation=attn_impl
            )
        else:
            encoder = RobertaModel.from_pretrained(path, config=config)

        # Create model
        model = cls(
            encoder=encoder,
            task_registry=task_registry,
            include_mlm_head=meta.get('has_mlm_head', False),
        )

        # Load task heads
        task_heads_path = os.path.join(path, "task_heads.pt")
        model.task_heads.load_state_dict(
            torch.load(task_heads_path, map_location='cpu', weights_only=True)
        )

        # Load MLM head if present
        mlm_head_path = os.path.join(path, "mlm_head.pt")
        if os.path.exists(mlm_head_path) and model.mlm_head is not None:
            model.mlm_head.load_state_dict(
                torch.load(mlm_head_path, map_location='cpu', weights_only=True)
            )

        logger.info(f"Loaded MultiTaskModel from checkpoint: {path}")
        return model


def _get_attn_implementation(use_flash_attention: bool) -> Optional[str]:
    """Check if Flash Attention 2 is available."""
    if not use_flash_attention:
        return None
    try:
        import flash_attn
        if torch.cuda.is_available():
            return "flash_attention_2"
    except ImportError:
        pass
    return None


def get_model_info(model: MultiTaskModel) -> Dict[str, Any]:
    """Get model statistics."""
    n_total = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_encoder = sum(p.numel() for p in model.encoder.parameters())
    n_heads = sum(p.numel() for p in model.task_heads.parameters())
    n_mlm = sum(p.numel() for p in model.mlm_head.parameters()) if model.mlm_head else 0

    return {
        "total_parameters": n_total,
        "trainable_parameters": n_trainable,
        "encoder_parameters": n_encoder,
        "task_heads_parameters": n_heads,
        "mlm_head_parameters": n_mlm,
        "num_tasks": len(model.task_heads),
        "task_names": list(model.task_heads.keys()),
        "hidden_size": model.hidden_size,
        "has_mlm_head": model.mlm_head is not None,
    }
