"""Tests for routing instrumentation."""

import math
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.config import InterleavedMoEConfig
from src.instrumentation.routing import (
    RoutingStats,
    RoutingTracker,
    collapse_signal,
    compute_routing_metrics,
    gini_coefficient,
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


class TestGini:
    def test_perfect_equality(self):
        x = torch.ones(8)
        assert gini_coefficient(x) == pytest.approx(0.0, abs=1e-6)

    def test_maximal_inequality(self):
        x = torch.zeros(8)
        x[0] = 1.0
        # Theoretical max gini for n=8 is (n-1)/n = 0.875
        g = gini_coefficient(x)
        assert g == pytest.approx(0.875, abs=1e-3)

    def test_monotonic_in_inequality(self):
        n = 16
        equal = torch.ones(n)
        skewed = torch.ones(n)
        skewed[0] = 10.0
        very_skewed = torch.ones(n)
        very_skewed[0] = 100.0
        g_equal = gini_coefficient(equal)
        g_skewed = gini_coefficient(skewed)
        g_very = gini_coefficient(very_skewed)
        assert g_equal < g_skewed < g_very

    def test_rejects_non_1d(self):
        with pytest.raises(ValueError):
            gini_coefficient(torch.ones(2, 3))

    def test_handles_empty(self):
        assert gini_coefficient(torch.empty(0)) == 0.0


class TestComputeRoutingMetrics:
    def _raw_stats(self, num_experts=4, num_tokens=16, K=2):
        """Build a synthetic raw_stats dict mimicking MoELayer.forward."""
        torch.manual_seed(0)
        logits = torch.randn(num_tokens, num_experts)
        scores = torch.softmax(logits, dim=-1)
        topk_weights, topk_indices = scores.topk(K, dim=-1)
        topk_weights = topk_weights / topk_weights.sum(-1, keepdim=True)
        return {
            "router_logits": logits,
            "router_entropy": -(scores * (scores + 1e-10).log()).sum(-1).mean(),
            "expert_load": scores.mean(0),
            "topk_indices": topk_indices,
            "topk_weights": topk_weights,
        }

    def test_returns_routing_stats(self):
        stats = compute_routing_metrics(0, self._raw_stats())
        assert isinstance(stats, RoutingStats)
        assert stats.layer_idx == 0
        assert stats.num_experts == 4
        assert stats.num_experts_per_tok == 2

    def test_uniform_routing_has_log_n_entropy(self):
        # Construct stats where every expert is equally likely
        N, E, K = 32, 4, 2
        scores = torch.full((N, E), 1.0 / E)
        topk_weights, topk_indices = scores.topk(K, dim=-1)
        topk_weights = topk_weights / topk_weights.sum(-1, keepdim=True)
        raw = {
            "router_logits": torch.zeros(N, E),
            "router_entropy": torch.tensor(math.log(E)),
            "expert_load": scores.mean(0),
            "topk_indices": topk_indices,
            "topk_weights": topk_weights,
        }
        s = compute_routing_metrics(0, raw)
        assert s.entropy == pytest.approx(math.log(E), abs=1e-4)
        # Equal load → very low Gini
        assert s.expert_load_gini < 0.05

    def test_one_hot_routing_has_zero_entropy(self):
        N, E, K = 16, 4, 2
        scores = torch.zeros(N, E)
        scores[:, 0] = 1.0
        topk_weights, topk_indices = scores.topk(K, dim=-1)
        topk_weights = topk_weights / topk_weights.sum(-1, keepdim=True)
        raw = {
            "router_logits": torch.zeros(N, E),
            "router_entropy": torch.tensor(0.0),
            "expert_load": scores.mean(0),
            "topk_indices": topk_indices,
            "topk_weights": topk_weights,
        }
        s = compute_routing_metrics(0, raw)
        assert s.entropy == pytest.approx(0.0, abs=1e-4)
        assert s.expert_load_max == pytest.approx(1.0)
        # All mass on expert 0 → high Gini, near (n-1)/n = 0.75
        assert s.expert_load_gini > 0.7

    def test_top1_top2_gap_ordering(self):
        N, E, K = 16, 4, 2
        # Sharp routing: top-1 gets 0.9, top-2 gets 0.1
        topk_weights = torch.tensor([[0.9, 0.1]] * N)
        topk_indices = torch.tensor([[0, 1]] * N)
        raw = {
            "router_logits": torch.zeros(N, E),
            "router_entropy": torch.tensor(0.5),
            "expert_load": torch.tensor([0.5, 0.5, 0.0, 0.0]),
            "topk_indices": topk_indices,
            "topk_weights": topk_weights,
        }
        s = compute_routing_metrics(0, raw)
        assert s.top1_top2_gap == pytest.approx(0.8, abs=1e-5)


class TestRoutingTracker:
    def _raw(self, top1_pattern):
        """Build raw stats where the top-1 expert is determined by `top1_pattern`."""
        N = len(top1_pattern)
        E = 4
        K = 2
        topk_indices = torch.zeros(N, K, dtype=torch.long)
        topk_indices[:, 0] = torch.tensor(top1_pattern)
        topk_indices[:, 1] = (topk_indices[:, 0] + 1) % E
        topk_weights = torch.full((N, K), 0.5)
        return {
            "router_logits": torch.zeros(N, E),
            "router_entropy": torch.tensor(1.0),
            "expert_load": torch.full((E,), 1.0 / E),
            "topk_indices": topk_indices,
            "topk_weights": topk_weights,
        }

    def test_first_call_no_churn(self):
        tracker = RoutingTracker()
        out = tracker.update({0: self._raw([0, 0, 1, 1])})
        assert out[0].token_churn is None

    def test_zero_churn_when_routing_unchanged(self):
        tracker = RoutingTracker()
        raw = self._raw([0, 0, 1, 1])
        tracker.update({0: raw})
        out = tracker.update({0: raw})
        assert out[0].token_churn == pytest.approx(0.0)

    def test_full_churn_when_routing_flipped(self):
        tracker = RoutingTracker()
        tracker.update({0: self._raw([0, 0, 1, 1])})
        out = tracker.update({0: self._raw([2, 2, 3, 3])})
        assert out[0].token_churn == pytest.approx(1.0)

    def test_partial_churn(self):
        tracker = RoutingTracker()
        tracker.update({0: self._raw([0, 1, 2, 3])})
        out = tracker.update({0: self._raw([0, 1, 0, 0])})  # 2 of 4 changed
        assert out[0].token_churn == pytest.approx(0.5)

    def test_reset_clears_state(self):
        tracker = RoutingTracker()
        tracker.update({0: self._raw([0, 0, 0, 0])})
        tracker.reset()
        out = tracker.update({0: self._raw([1, 1, 1, 1])})
        assert out[0].token_churn is None


class TestCollapseSignal:
    def test_below_threshold(self):
        s = RoutingStats(
            layer_idx=0, num_experts=4, num_experts_per_tok=2,
            entropy=1.0, expert_load=torch.zeros(4),
            expert_load_max=0.5, expert_load_gini=0.1, top1_top2_gap=0.0,
        )
        assert not collapse_signal(s, threshold=0.7)

    def test_above_threshold(self):
        s = RoutingStats(
            layer_idx=0, num_experts=4, num_experts_per_tok=2,
            entropy=0.1, expert_load=torch.zeros(4),
            expert_load_max=0.85, expert_load_gini=0.7, top1_top2_gap=0.8,
        )
        assert collapse_signal(s, threshold=0.7)


class TestRoutingEndToEnd:
    """Integration: run the real model and feed routing_stats through the tracker."""

    def test_real_model_routing_aggregation(self, config):
        model = InterleavedMoEModel(config)
        model.eval()
        tracker = RoutingTracker()
        input_ids = torch.randint(0, config.vocab_size, (2, 16))
        with torch.no_grad():
            r = model(input_ids)
        out = tracker.update(r["routing_stats"])
        # MoE layers are 0 and 2 with moe_every_n_layers=2
        assert set(out.keys()) == {0, 2}
        for stats in out.values():
            assert 0.0 <= stats.entropy <= math.log(config.num_experts) + 1e-5
            assert 0.0 <= stats.expert_load_max <= 1.0
            assert 0.0 <= stats.expert_load_gini <= 1.0
            assert stats.token_churn is None  # first call

    def test_real_model_token_churn_after_two_calls(self, config):
        model = InterleavedMoEModel(config)
        model.eval()
        tracker = RoutingTracker()
        input_ids = torch.randint(0, config.vocab_size, (2, 16))
        with torch.no_grad():
            r1 = model(input_ids)
            tracker.update(r1["routing_stats"])
            r2 = model(input_ids)
            out = tracker.update(r2["routing_stats"])
        for stats in out.values():
            # Same input → same routing → zero churn
            assert stats.token_churn == pytest.approx(0.0)
