"""Tests for representation drift metrics."""

import math
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.config import InterleavedMoEConfig
from src.instrumentation.drift import (
    collect_layer_activations,
    compute_layer_drift,
    linear_cka,
    procrustes_distance,
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


class TestLinearCKA:
    def test_identity(self):
        torch.manual_seed(0)
        X = torch.randn(64, 32)
        assert linear_cka(X, X) == pytest.approx(1.0, abs=1e-6)

    def test_invariant_to_scale_and_shift(self):
        torch.manual_seed(1)
        X = torch.randn(64, 32)
        Y = 3.7 * X + 2.5  # isotropic scale + bias
        assert linear_cka(X, Y) == pytest.approx(1.0, abs=1e-5)

    def test_low_for_independent_random(self):
        torch.manual_seed(2)
        X = torch.randn(256, 32)
        Y = torch.randn(256, 32)  # independent
        cka = linear_cka(X, Y)
        # Independent gaussians → CKA should be small (< 0.2 typical at this size)
        assert cka < 0.2

    def test_handles_different_widths(self):
        torch.manual_seed(3)
        X = torch.randn(64, 32)
        Y = torch.randn(64, 16)
        cka = linear_cka(X, Y)
        # Should return a finite scalar in [0, 1]
        assert 0.0 <= cka <= 1.0

    def test_rejects_mismatched_rows(self):
        with pytest.raises(ValueError):
            linear_cka(torch.randn(8, 4), torch.randn(7, 4))

    def test_rejects_non_2d(self):
        with pytest.raises(ValueError):
            linear_cka(torch.randn(8), torch.randn(8))

    def test_returns_float(self):
        X = torch.randn(16, 8)
        result = linear_cka(X, X)
        assert isinstance(result, float)

    def test_float64_internal_precision(self):
        # Cast input down to float16 — should still give a stable answer
        X = torch.randn(64, 32).half()
        cka = linear_cka(X, X)
        assert cka == pytest.approx(1.0, abs=1e-3)


class TestProcrustesDistance:
    def test_zero_for_identical(self):
        torch.manual_seed(0)
        X = torch.randn(64, 16)
        d = procrustes_distance(X, X)
        assert d == pytest.approx(0.0, abs=1e-6)

    def test_zero_for_rotation(self):
        # Procrustes should be 0 for an orthogonal rotation (it finds the
        # optimal rotation and undoes it).
        torch.manual_seed(1)
        D = 16
        X = torch.randn(64, D)
        A = torch.randn(D, D)
        Q, _ = torch.linalg.qr(A)  # random orthogonal matrix
        Y = X @ Q
        d = procrustes_distance(X, Y)
        assert d == pytest.approx(0.0, abs=1e-5)

    def test_triangle_inequality(self):
        # True metric → triangle inequality holds for any 3 points.
        torch.manual_seed(2)
        X = torch.randn(64, 16)
        Y = X + 0.5 * torch.randn(64, 16)
        Z = X + 1.0 * torch.randn(64, 16)
        d_xy = procrustes_distance(X, Y)
        d_yz = procrustes_distance(Y, Z)
        d_xz = procrustes_distance(X, Z)
        # d(x, z) <= d(x, y) + d(y, z)
        assert d_xz <= d_xy + d_yz + 1e-5

    def test_sym(self):
        torch.manual_seed(3)
        X = torch.randn(32, 8)
        Y = torch.randn(32, 8)
        assert procrustes_distance(X, Y) == pytest.approx(
            procrustes_distance(Y, X), abs=1e-5
        )

    def test_bounded(self):
        torch.manual_seed(4)
        X = torch.randn(64, 16)
        Y = torch.randn(64, 16)
        d = procrustes_distance(X, Y)
        assert 0.0 <= d <= math.sqrt(2.0) + 1e-5

    def test_rejects_mismatched_shape(self):
        with pytest.raises(ValueError):
            procrustes_distance(torch.randn(8, 4), torch.randn(8, 5))


class TestComputeLayerDrift:
    def test_returns_per_layer_metrics(self):
        torch.manual_seed(0)
        a = {0: torch.randn(32, 16), 1: torch.randn(32, 16)}
        b = {0: a[0].clone(), 1: torch.randn(32, 16)}
        result = compute_layer_drift(a, b)
        assert set(result.keys()) == {0, 1}
        assert result[0]["cka"] == pytest.approx(1.0, abs=1e-5)
        assert result[0]["procrustes"] == pytest.approx(0.0, abs=1e-5)

    def test_intersection_of_layer_indices(self):
        a = {0: torch.randn(8, 4), 1: torch.randn(8, 4), 2: torch.randn(8, 4)}
        b = {1: torch.randn(8, 4), 2: torch.randn(8, 4), 3: torch.randn(8, 4)}
        result = compute_layer_drift(a, b)
        assert set(result.keys()) == {1, 2}


class TestCollectLayerActivations:
    def test_one_entry_per_layer(self, config):
        model = InterleavedMoEModel(config)
        input_ids = torch.randint(0, config.vocab_size, (2, 16))
        acts = collect_layer_activations(model, input_ids)
        assert set(acts.keys()) == set(range(config.num_layers))
        for v in acts.values():
            assert v.shape == (2, config.hidden_size)

    def test_completion_mask_only_uses_marked_positions(self, config):
        model = InterleavedMoEModel(config)
        input_ids = torch.randint(0, config.vocab_size, (1, 8))
        # All-ones mask should equal mean over all positions
        mask = torch.ones(1, 8)
        acts_with_mask = collect_layer_activations(model, input_ids, completion_mask=mask)
        acts_no_mask = collect_layer_activations(model, input_ids)
        for i in range(config.num_layers):
            assert torch.allclose(acts_with_mask[i], acts_no_mask[i], atol=1e-5)

    def test_does_not_change_model_mode(self, config):
        model = InterleavedMoEModel(config)
        model.train()
        input_ids = torch.randint(0, config.vocab_size, (1, 4))
        _ = collect_layer_activations(model, input_ids)
        assert model.training

    def test_no_grad_during_collection(self, config):
        model = InterleavedMoEModel(config)
        input_ids = torch.randint(0, config.vocab_size, (1, 4))
        acts = collect_layer_activations(model, input_ids)
        for v in acts.values():
            assert not v.requires_grad
