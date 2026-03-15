"""token_budget.py
Utilities for token-budgeted training and loss-vs-tokens logging.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any

import torch
from transformers import Trainer, TrainerCallback


def _count_tokens_from_inputs(inputs: Dict[str, Any]) -> int:
    """Count non-pad tokens from attention_mask in a Trainer batch."""
    if not inputs:
        return 0
    attn = inputs.get("attention_mask")
    if attn is None:
        return 0
    if isinstance(attn, torch.Tensor):
        return int(attn.sum().item())
    try:
        return int(torch.tensor(attn).sum().item())
    except Exception:
        return 0


@dataclass
class TokenBudgetTracker:
    """Track tokens seen across training steps."""
    token_budget: Optional[int]
    tokens_seen: int = 0

    def update(self, inputs: Dict[str, Any]) -> int:
        tokens = _count_tokens_from_inputs(inputs)
        self.tokens_seen += tokens
        return tokens


class TokenBudgetCallback(TrainerCallback):
    """Stop training when token budget is reached and log metrics to JSONL."""

    def __init__(
        self,
        tracker: TokenBudgetTracker,
        metrics_path: Optional[str],
        run_id: str,
        phase: str,
    ) -> None:
        self.tracker = tracker
        self.metrics_path = Path(metrics_path) if metrics_path else None
        self.run_id = run_id
        self.phase = phase

        if self.metrics_path:
            self.metrics_path.parent.mkdir(parents=True, exist_ok=True)

    def on_step_end(self, args, state, control, **kwargs):
        if self.tracker.token_budget is not None and self.tracker.tokens_seen >= self.tracker.token_budget:
            control.should_training_stop = True
        return control

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs or not self.metrics_path:
            return
        record = {
            "timestamp": time.time(),
            "run_id": self.run_id,
            "phase": self.phase,
            "global_step": state.global_step,
            "tokens_seen": self.tracker.tokens_seen,
            "loss": logs.get("loss"),
            "learning_rate": logs.get("learning_rate"),
            "epoch": logs.get("epoch"),
        }
        with self.metrics_path.open("a") as f:
            f.write(json.dumps(record) + "\n")


class TokenBudgetTrainer(Trainer):
    """Trainer subclass that updates a TokenBudgetTracker each training step."""

    def __init__(self, *args, token_budget_tracker: Optional[TokenBudgetTracker] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.token_budget_tracker = token_budget_tracker

    def training_step(self, model, inputs, num_items_in_batch=None):
        try:
            loss = super().training_step(model, inputs, num_items_in_batch=num_items_in_batch)
        except TypeError:
            # Backward compat with older Transformers that don't pass num_items_in_batch.
            loss = super().training_step(model, inputs)
        if self.token_budget_tracker is not None:
            self.token_budget_tracker.update(inputs)
        return loss
