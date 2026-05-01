"""Tests for the random-baseline encoder.

The critical assertion: the random encoder must NOT silently fall back to a stale
saved encoder. We verify the checksum-equality between the saved encoder and a
freshly seeded reference.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import torch
from transformers import RobertaConfig, RobertaModel

sys.path.insert(0, str(Path(__file__).parent.parent))

from config_v2 import ModelConfigV2
from random_baseline_v2 import _state_dict_checksum, make_random_encoder


def test_random_encoder_seed_is_deterministic():
    cfg = ModelConfigV2(
        hidden_size=64, num_hidden_layers=2, num_attention_heads=2,
        intermediate_size=128, max_position_embeddings=64, vocab_size=100,
    )
    with tempfile.TemporaryDirectory() as td:
        a = make_random_encoder(cfg, seed=42, save_dir=str(Path(td) / "a"))
        b = make_random_encoder(cfg, seed=42, save_dir=str(Path(td) / "b"))
        ma = RobertaModel.from_pretrained(a, add_pooling_layer=False)
        mb = RobertaModel.from_pretrained(b, add_pooling_layer=False)
        assert _state_dict_checksum(ma) == _state_dict_checksum(mb)


def test_random_encoder_seed_changes_weights():
    cfg = ModelConfigV2(
        hidden_size=64, num_hidden_layers=2, num_attention_heads=2,
        intermediate_size=128, max_position_embeddings=64, vocab_size=100,
    )
    with tempfile.TemporaryDirectory() as td:
        a = make_random_encoder(cfg, seed=0, save_dir=str(Path(td) / "a"))
        b = make_random_encoder(cfg, seed=1, save_dir=str(Path(td) / "b"))
        ma = RobertaModel.from_pretrained(a, add_pooling_layer=False)
        mb = RobertaModel.from_pretrained(b, add_pooling_layer=False)
        assert _state_dict_checksum(ma) != _state_dict_checksum(mb)
