"""Tests for gradient instrumentation."""

import math
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.config import InterleavedMoEConfig
from src.instrumentation.gradients import (
    GradientTracker,
    _classify_param,
    summarize_gradients_by_kind,
)
from src.model import InterleavedMoEModel


@pytest.fixture
def config():
    return InterleavedMoEConfig(
        hidden_size=64,
        num_layers=4,
        num_attention_heads=4,
        num_kv_heads=2,
        intermediate_size=128,
        num_experts=4,
        num_experts_per_tok=2,
        expert_intermediate_size=128,
        moe_every_n_layers=2,
        vocab_size=256,
        max_position_embeddings=128,
    )


@pytest.fixture
def trained_model(config):
    """Model with .grad populated by a single backward pass."""
    model = InterleavedMoEModel(config)
    input_ids = torch.randint(0, config.vocab_size, (2, 16))
    labels = torch.randint(0, config.vocab_size, (2, 16))
    out = model(input_ids, labels=labels)
    out["loss"].backward()
    return model


class TestClassifyParam:
    def test_embedding(self):
        idx, kind, _ = _classify_param("embed_tokens.weight")
        assert idx is None and kind == "embedding"

    def test_attention(self):
        idx, kind, _ = _classify_param("layers.3.attn.q_proj.weight")
        assert idx == 3 and kind == "attention"

    def test_attention_direct_proj(self):
        idx, kind, _ = _classify_param("layers.0.q_proj.weight")
        assert idx == 0 and kind == "attention"

    def test_dense_ffn(self):
        # In a dense block, ffn.gate_proj is an nn.Linear (named ffn.gate_proj.weight)
        idx, kind, _ = _classify_param("layers.1.ffn.gate_proj.weight")
        assert idx == 1 and kind == "dense_ffn"

    def test_moe_expert_packed(self):
        # In an MoE block, ffn.gate_proj is a packed Parameter (no .weight suffix)
        idx, kind, _ = _classify_param("layers.0.ffn.gate_proj")
        assert idx == 0 and kind == "moe_expert"

    def test_moe_router(self):
        idx, kind, _ = _classify_param("layers.0.ffn.router.gate.weight")
        assert idx == 0 and kind == "moe_router"

    def test_norm(self):
        idx, kind, _ = _classify_param("layers.2.attn_norm.weight")
        assert idx == 2 and kind == "norm"

    def test_outer_norm(self):
        idx, kind, _ = _classify_param("norm.weight")
        assert idx is None and kind == "norm"


class TestSummarize:
    def test_global_norm_matches_manual(self, trained_model):
        manual = 0.0
        for p in trained_model.parameters():
            if p.grad is not None:
                manual += p.grad.float().pow(2).sum().item()
        manual = math.sqrt(manual)
        summary = summarize_gradients_by_kind(trained_model)
        assert summary.global_norm == pytest.approx(manual, rel=1e-5)

    def test_kinds_present(self, trained_model):
        summary = summarize_gradients_by_kind(trained_model)
        # We expect dense_ffn, moe_expert, moe_router, attention, embedding, norm
        kinds = set(summary.by_layer_kind.keys())
        assert "attention" in kinds
        assert "moe_expert" in kinds
        assert "moe_router" in kinds
        assert "dense_ffn" in kinds
        assert "embedding" in kinds
        assert "norm" in kinds

    def test_per_layer_norms_one_per_block(self, trained_model, config):
        summary = summarize_gradients_by_kind(trained_model)
        for i in range(config.num_layers):
            assert i in summary.by_layer
            assert summary.by_layer[i] >= 0.0

    def test_per_expert_norms_only_for_moe_layers(self, trained_model, config):
        summary = summarize_gradients_by_kind(trained_model)
        moe_layer_indices = {
            i for i in range(config.num_layers)
            if i % config.moe_every_n_layers == 0
        }
        # Every (layer, expert) entry must be in an MoE layer
        for (layer_idx, _expert_idx) in summary.by_expert.keys():
            assert layer_idx in moe_layer_indices

    def test_per_expert_count(self, trained_model, config):
        summary = summarize_gradients_by_kind(trained_model)
        moe_layer_indices = [
            i for i in range(config.num_layers)
            if i % config.moe_every_n_layers == 0
        ]
        expected_count = len(moe_layer_indices) * config.num_experts
        assert len(summary.by_expert) == expected_count

    def test_per_kind_squared_sums_to_global(self, trained_model):
        summary = summarize_gradients_by_kind(trained_model)
        total_sq = sum(v ** 2 for v in summary.by_layer_kind.values())
        assert math.sqrt(total_sq) == pytest.approx(summary.global_norm, rel=1e-5)

    def test_no_grad_returns_empty(self, config):
        model = InterleavedMoEModel(config)
        # No backward pass → no .grad
        summary = summarize_gradients_by_kind(model)
        assert summary.global_norm == 0.0
        assert summary.by_layer == {}
        assert summary.by_layer_kind == {}


class TestGradientTracker:
    def test_records_history(self, config):
        tracker = GradientTracker()
        model = InterleavedMoEModel(config)
        for _ in range(3):
            input_ids = torch.randint(0, config.vocab_size, (1, 8))
            out = model(input_ids, labels=input_ids)
            model.zero_grad()
            out["loss"].backward()
            tracker.step(model)
        assert len(tracker.history) == 3
        assert all(s.global_norm > 0 for s in tracker.history)

    def test_latest_returns_most_recent(self, config):
        tracker = GradientTracker()
        assert tracker.latest() is None
        model = InterleavedMoEModel(config)
        input_ids = torch.randint(0, config.vocab_size, (1, 4))
        out = model(input_ids, labels=input_ids)
        out["loss"].backward()
        tracker.step(model)
        assert tracker.latest() is not None

    def test_reset_clears_history(self, config):
        tracker = GradientTracker()
        model = InterleavedMoEModel(config)
        input_ids = torch.randint(0, config.vocab_size, (1, 4))
        out = model(input_ids, labels=input_ids)
        out["loss"].backward()
        tracker.step(model)
        tracker.reset()
        assert tracker.history == []
        assert tracker.latest() is None
