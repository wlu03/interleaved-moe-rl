"""Tests for the Tracker orchestrator."""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.config import InterleavedMoEConfig
from src.instrumentation.drift import collect_layer_activations
from src.instrumentation.tracker import Tracker, TrackerConfig
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


class TestTracker:
    def test_log_step_buffers_record(self):
        t = Tracker(TrackerConfig(use_wandb=False))
        t.log_step(0, loss=1.5, reward=0.3)
        records = t.get_records()
        assert len(records) == 1
        assert records[0]["step"] == 0
        assert records[0]["loss"] == 1.5
        assert records[0]["reward"] == 0.3

    def test_log_routing_emits_per_layer(self, config):
        t = Tracker(TrackerConfig(use_wandb=False))
        model = InterleavedMoEModel(config)
        model.eval()
        input_ids = torch.randint(0, config.vocab_size, (1, 8))
        with torch.no_grad():
            r = model(input_ids)
        t.log_routing(0, r["routing_stats"])
        records = t.get_records()
        assert len(records) == 1
        rec = records[0]
        # MoE layers are 0 and 2
        assert "routing/L0/entropy" in rec
        assert "routing/L2/entropy" in rec
        assert "routing/L0/load_max" in rec
        assert "routing/L0/load_gini" in rec

    def test_log_gradients_emits_per_layer_and_kind(self, config):
        t = Tracker(TrackerConfig(use_wandb=False))
        model = InterleavedMoEModel(config)
        input_ids = torch.randint(0, config.vocab_size, (1, 8))
        out = model(input_ids, labels=input_ids)
        out["loss"].backward()
        t.log_gradients(0, model)
        rec = t.get_records()[0]
        assert "grad/global_norm" in rec
        assert "grad/by_kind/attention" in rec
        assert "grad/by_kind/moe_expert" in rec
        assert "grad/by_layer/L0" in rec

    def test_log_drift_emits_per_kind_means(self, config):
        t = Tracker(TrackerConfig(use_wandb=False))
        model = InterleavedMoEModel(config)
        input_ids = torch.randint(0, config.vocab_size, (2, 8))

        baseline = collect_layer_activations(model, input_ids)
        # Same model + same input → drift should be ~0
        current = collect_layer_activations(model, input_ids)

        layer_kinds = {
            i: ("moe" if i % config.moe_every_n_layers == 0 else "dense")
            for i in range(config.num_layers)
        }
        t.log_drift(0, baseline, current, layer_kinds=layer_kinds)
        rec = t.get_records()[0]

        # Per-layer drift entries
        for i in range(config.num_layers):
            assert f"drift/cka/L{i}" in rec
            assert f"drift/procrustes/L{i}" in rec
            # Identical activations → CKA ≈ 1, Procrustes ≈ 0
            assert rec[f"drift/cka/L{i}"] == pytest.approx(1.0, abs=1e-4)
            assert rec[f"drift/procrustes/L{i}"] == pytest.approx(0.0, abs=1e-4)

        # By-kind aggregates
        assert "drift/cka/by_kind/dense_mean" in rec
        assert "drift/cka/by_kind/moe_mean" in rec
        assert "drift/procrustes/by_kind/dense_mean" in rec
        assert "drift/procrustes/by_kind/moe_mean" in rec

    def test_collapse_warning_fires_once(self, config, capsys):
        t = Tracker(TrackerConfig(use_wandb=False, collapse_threshold=0.7))

        # Synthesize a "collapsed" routing_stats dict where one expert dominates
        E = config.num_experts
        N = 8
        K = config.num_experts_per_tok
        load = torch.zeros(E)
        load[0] = 0.9
        load[1:] = 0.1 / (E - 1)
        topk_indices = torch.zeros(N, K, dtype=torch.long)
        topk_weights = torch.full((N, K), 0.5)
        raw = {
            0: {
                "router_logits": torch.zeros(N, E),
                "router_entropy": torch.tensor(0.5),
                "expert_load": load,
                "topk_indices": topk_indices,
                "topk_weights": topk_weights,
            }
        }
        t.log_routing(10, raw)
        t.log_routing(20, raw)
        captured = capsys.readouterr()
        # Warning should appear exactly once for layer 0
        assert captured.out.count("ROUTING COLLAPSE WARNING") == 1

    def test_wandb_off_doesnt_crash(self):
        t = Tracker(TrackerConfig(use_wandb=False))
        t.start()  # no-op when wandb off
        t.log_step(0, loss=1.0)
        t.finish()

    def test_records_have_step_field(self):
        t = Tracker(TrackerConfig(use_wandb=False))
        t.log_step(0, loss=1.0)
        t.log_step(5, loss=0.8)
        t.log_step(10, loss=0.6)
        steps = [r["step"] for r in t.get_records()]
        assert steps == [0, 5, 10]
