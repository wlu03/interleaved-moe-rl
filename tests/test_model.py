"""Tests for the interleaved MoE model — Milestone 1: Complete the Model."""

import torch
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.config import InterleavedMoEConfig
from src.model import (
    InterleavedMoEModel,
    RMSNorm,
    DenseFFN,
    MoELayer,
    MoERouter,
    Attention,
    TransformerBlock,
    precompute_rope_freqs,
    apply_rope,
)


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
def tiny_model(config):
    return InterleavedMoEModel(config)


class TestRMSNorm:
    def test_output_shape(self):
        norm = RMSNorm(64)
        x = torch.randn(2, 8, 64)
        out = norm(x)
        assert out.shape == x.shape

    def test_unit_norm_property(self):
        norm = RMSNorm(64)
        x = torch.randn(2, 8, 64)
        out = norm(x)
        rms = out.float().pow(2).mean(-1).sqrt()
        # After normalization, RMS should be approximately 1 (scaled by weight=1)
        assert torch.allclose(rms, torch.ones_like(rms), atol=0.1)

    def test_preserves_dtype(self):
        norm = RMSNorm(64)
        x = torch.randn(2, 8, 64, dtype=torch.float16)
        out = norm(x)
        assert out.dtype == torch.float16


class TestRoPE:
    def test_freqs_shape(self):
        freqs = precompute_rope_freqs(dim=32, max_seq_len=128)
        assert freqs.shape == (128, 16)  # [seq_len, dim//2]
        assert freqs.dtype == torch.complex64

    def test_apply_preserves_shape(self):
        freqs = precompute_rope_freqs(dim=32, max_seq_len=128)
        x = torch.randn(2, 4, 64, 32)  # [B, H, S, D]
        out = apply_rope(x, freqs)
        assert out.shape == x.shape

    def test_rope_is_rotation(self):
        """RoPE should preserve vector norms (it's a rotation)."""
        freqs = precompute_rope_freqs(dim=32, max_seq_len=128)
        x = torch.randn(1, 1, 16, 32)
        out = apply_rope(x, freqs)
        # Norms should be approximately preserved
        x_norms = x.float().norm(dim=-1)
        out_norms = out.float().norm(dim=-1)
        assert torch.allclose(x_norms, out_norms, atol=1e-4)


class TestDenseFFN:
    def test_output_shape(self):
        ffn = DenseFFN(hidden_size=64, intermediate_size=128)
        x = torch.randn(2, 8, 64)
        out = ffn(x)
        assert out.shape == (2, 8, 64)

    def test_gradient_flows(self):
        ffn = DenseFFN(hidden_size=64, intermediate_size=128)
        x = torch.randn(2, 8, 64, requires_grad=True)
        out = ffn(x)
        out.sum().backward()
        assert x.grad is not None
        assert x.grad.shape == x.shape


class TestMoERouter:
    def test_output_shapes(self, config):
        router = MoERouter(config)
        x = torch.randn(16, config.hidden_size)  # [N, D]
        weights, indices, stats = router(x)
        assert weights.shape == (16, config.num_experts_per_tok)
        assert indices.shape == (16, config.num_experts_per_tok)

    def test_weights_sum_to_one(self, config):
        router = MoERouter(config)
        x = torch.randn(16, config.hidden_size)
        weights, _, _ = router(x)
        sums = weights.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)

    def test_indices_in_range(self, config):
        router = MoERouter(config)
        x = torch.randn(32, config.hidden_size)
        _, indices, _ = router(x)
        assert indices.min() >= 0
        assert indices.max() < config.num_experts

    def test_stats_keys(self, config):
        router = MoERouter(config)
        x = torch.randn(16, config.hidden_size)
        _, _, stats = router(x)
        assert "router_logits" in stats
        assert "router_entropy" in stats
        assert "expert_load" in stats
        assert stats["expert_load"].shape == (config.num_experts,)


class TestMoELayer:
    def test_output_shape(self, config):
        moe = MoELayer(config)
        x = torch.randn(2, 8, config.hidden_size)
        out, stats = moe(x)
        assert out.shape == x.shape

    def test_gradient_flows_through_experts(self, config):
        moe = MoELayer(config)
        x = torch.randn(2, 8, config.hidden_size, requires_grad=True)
        out, _ = moe(x)
        out.sum().backward()
        assert x.grad is not None
        # Expert weights should also have gradients
        assert moe.gate_proj.grad is not None
        assert moe.up_proj.grad is not None
        assert moe.down_proj.grad is not None

    def test_routing_stats_present(self, config):
        moe = MoELayer(config)
        x = torch.randn(2, 8, config.hidden_size)
        _, stats = moe(x)
        assert "router_entropy" in stats
        assert "expert_load" in stats

    def test_deterministic(self, config):
        moe = MoELayer(config)
        x = torch.randn(2, 8, config.hidden_size)
        out1, _ = moe(x)
        out2, _ = moe(x)
        assert torch.allclose(out1, out2)


class TestAttention:
    def test_output_shape(self, config):
        attn = Attention(config)
        rope_freqs = precompute_rope_freqs(
            config.hidden_size // config.num_attention_heads,
            config.max_position_embeddings,
            config.rope_theta,
        )
        x = torch.randn(2, 16, config.hidden_size)
        out = attn(x, rope_freqs)
        assert out.shape == x.shape

    def test_causal_masking(self, config):
        """Verify output at position t doesn't depend on future positions."""
        attn = Attention(config)
        rope_freqs = precompute_rope_freqs(
            config.hidden_size // config.num_attention_heads,
            config.max_position_embeddings,
            config.rope_theta,
        )
        x = torch.randn(1, 8, config.hidden_size)
        out_full = attn(x, rope_freqs)

        # Output at position 0 should be same if we truncate to length 1
        x_single = x[:, :1, :]
        out_single = attn(x_single, rope_freqs)
        assert torch.allclose(out_full[:, :1, :], out_single, atol=1e-5)


class TestTransformerBlock:
    def test_dense_block(self, config):
        # Layer 1 is dense (moe_every_n_layers=2, so 0,2 are MoE, 1,3 are dense)
        block = TransformerBlock(config, layer_idx=1)
        assert not block.is_moe
        rope_freqs = precompute_rope_freqs(
            config.hidden_size // config.num_attention_heads,
            config.max_position_embeddings,
        )
        x = torch.randn(2, 8, config.hidden_size)
        out, stats = block(x, rope_freqs)
        assert out.shape == x.shape
        assert stats == {}  # dense blocks return empty stats

    def test_moe_block(self, config):
        block = TransformerBlock(config, layer_idx=0)
        assert block.is_moe
        rope_freqs = precompute_rope_freqs(
            config.hidden_size // config.num_attention_heads,
            config.max_position_embeddings,
        )
        x = torch.randn(2, 8, config.hidden_size)
        out, stats = block(x, rope_freqs)
        assert out.shape == x.shape
        assert "router_entropy" in stats


class TestInterleavedMoEModel:
    def test_forward_shape(self, tiny_model, config):
        input_ids = torch.randint(0, config.vocab_size, (2, 16))
        result = tiny_model(input_ids)
        assert result["logits"].shape == (2, 16, config.vocab_size)

    def test_loss_computation(self, tiny_model, config):
        input_ids = torch.randint(0, config.vocab_size, (2, 16))
        labels = torch.randint(0, config.vocab_size, (2, 16))
        result = tiny_model(input_ids, labels=labels)
        assert "loss" in result
        assert result["loss"].ndim == 0  # scalar
        assert result["loss"].item() > 0

    def test_routing_stats_populated(self, tiny_model, config):
        input_ids = torch.randint(0, config.vocab_size, (2, 16))
        result = tiny_model(input_ids)
        stats = result["routing_stats"]
        # With moe_every_n_layers=2 and 4 layers, MoE is on layers 0 and 2
        assert 0 in stats
        assert 2 in stats
        assert 1 not in stats
        assert 3 not in stats

    def test_backward_pass(self, tiny_model, config):
        input_ids = torch.randint(0, config.vocab_size, (2, 16))
        labels = torch.randint(0, config.vocab_size, (2, 16))
        result = tiny_model(input_ids, labels=labels)
        result["loss"].backward()
        # All parameters should have gradients
        for name, p in tiny_model.named_parameters():
            if p.requires_grad:
                assert p.grad is not None, f"No gradient for {name}"

    def test_parameter_counts(self, tiny_model):
        total = tiny_model.num_parameters()
        active = tiny_model.num_active_parameters()
        # Active < total because not all experts are used per forward pass
        assert active < total
        assert total > 0
        assert active > 0

    def test_tied_embeddings(self, config):
        config.tie_word_embeddings = True
        model = InterleavedMoEModel(config)
        assert model.lm_head is None
        # Logits should still work
        input_ids = torch.randint(0, config.vocab_size, (1, 4))
        result = model(input_ids)
        assert result["logits"].shape == (1, 4, config.vocab_size)

    def test_untied_embeddings(self, config):
        config.tie_word_embeddings = False
        model = InterleavedMoEModel(config)
        assert model.lm_head is not None

    def test_interleaving_pattern(self, config):
        model = InterleavedMoEModel(config)
        for i, layer in enumerate(model.layers):
            if i % config.moe_every_n_layers == 0:
                assert layer.is_moe, f"Layer {i} should be MoE"
                assert isinstance(layer.ffn, MoELayer)
            else:
                assert not layer.is_moe, f"Layer {i} should be dense"
                assert isinstance(layer.ffn, DenseFFN)

    def test_reproducibility(self, config):
        """Same input, same model → same output."""
        model = InterleavedMoEModel(config)
        model.eval()
        input_ids = torch.randint(0, config.vocab_size, (1, 8))
        with torch.no_grad():
            r1 = model(input_ids)
            r2 = model(input_ids)
        assert torch.allclose(r1["logits"], r2["logits"])

    def test_different_sequence_lengths(self, config):
        model = InterleavedMoEModel(config)
        for seq_len in [1, 4, 32, 64]:
            input_ids = torch.randint(0, config.vocab_size, (1, seq_len))
            result = model(input_ids)
            assert result["logits"].shape == (1, seq_len, config.vocab_size)
