"""Tests for the GRPO training loop (src/rl/train.py)."""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.config import InterleavedMoEConfig
from src.rl.grpo import GRPOConfig
from src.rl.train import (
    TrainConfig,
    build_model,
    layer_kinds,
    load_checkpoint,
    load_config,
    main,
    register_config,
    save_checkpoint,
)


# =====================================================================
#  Fixtures
# =====================================================================
class FakeTokenizer:
    def __init__(self, vocab_size: int = 64):
        self.vocab_size = vocab_size
        self.pad_token_id = 0
        self.eos_token_id = 1

    def encode(self, text: str) -> list[int]:
        ids = [(ord(c) % (self.vocab_size - 2)) + 2 for c in text]
        return ids if ids else [2]

    def decode(self, ids) -> str:
        return "".join(
            chr((int(t) - 2) + 32)
            for t in ids
            if int(t) not in (self.pad_token_id, self.eos_token_id)
        )


def _tiny_train_config(name="unit_tiny"):
    return TrainConfig(
        name=name,
        model=InterleavedMoEConfig(
            hidden_size=32, num_layers=2, num_attention_heads=4,
            num_kv_heads=2, intermediate_size=64, expert_intermediate_size=64,
            num_experts=4, num_experts_per_tok=2, moe_every_n_layers=2,
            vocab_size=64, max_position_embeddings=128,
        ),
        grpo=GRPOConfig(group_size=2, max_completion_len=8),
        lr=1e-3, max_steps=4, batch_prompts=2,
        eval_every=2, drift_every=2, checkpoint_every=2, probe_size=8,
    )


@pytest.fixture(autouse=True)
def _register_tiny():
    register_config("unit_tiny", _tiny_train_config)
    yield


@pytest.fixture
def fake_tokenizer():
    return FakeTokenizer(vocab_size=64)


def fake_sample_batch(b, step):
    prompts = [f"q{step}_{j}" for j in range(b)]
    golds = [str(step + j) for j in range(b)]
    return prompts, golds


# =====================================================================
#  Config registry
# =====================================================================
class TestConfigRegistry:
    def test_known_configs_load(self):
        for name in [
            "moe_interleaved", "dense_baseline",
            "moe_interleaved_smoke", "dense_baseline_smoke",
        ]:
            cfg = load_config(name)
            assert cfg.name == name
            assert isinstance(cfg.grpo, GRPOConfig)

    def test_unknown_config_raises(self):
        with pytest.raises(KeyError):
            load_config("does_not_exist")

    def test_dense_baseline_has_no_moe_layers(self):
        cfg = load_config("dense_baseline")
        model = build_model(cfg.model)
        kinds = layer_kinds(model)
        assert set(kinds.values()) == {"dense"}

    def test_moe_interleaved_has_moe_layers(self):
        cfg = load_config("moe_interleaved")
        model = build_model(cfg.model)
        kinds = layer_kinds(model)
        assert "moe" in kinds.values()
        assert "dense" in kinds.values()


# =====================================================================
#  Checkpointing
# =====================================================================
class TestCheckpoint:
    def test_round_trip(self, tmp_path):
        cfg = _tiny_train_config()
        model = build_model(cfg.model)
        optim = torch.optim.AdamW(model.parameters(), lr=1e-3)
        # mutate a param so we can tell load actually restores it
        with torch.no_grad():
            next(model.parameters()).add_(1.0)
        path = tmp_path / "latest.pt"
        save_checkpoint(model, optim, step=7, path=path)

        model2 = build_model(cfg.model)
        optim2 = torch.optim.AdamW(model2.parameters(), lr=1e-3)
        step = load_checkpoint(model2, optim2, path)
        assert step == 7
        for a, b in zip(model.parameters(), model2.parameters()):
            assert torch.equal(a, b)

    def test_atomic_write_leaves_no_tmp(self, tmp_path):
        cfg = _tiny_train_config()
        model = build_model(cfg.model)
        optim = torch.optim.AdamW(model.parameters(), lr=1e-3)
        path = tmp_path / "latest.pt"
        save_checkpoint(model, optim, step=1, path=path)
        assert path.exists()
        assert not (tmp_path / "latest.pt.tmp").exists()


# =====================================================================
#  Loop
# =====================================================================
class TestMainLoop:
    def test_runs_and_returns_summary(self, tmp_path, fake_tokenizer):
        summary = main(
            "unit_tiny", ckpt_dir=tmp_path, resume=False,
            tokenizer=fake_tokenizer, sample_batch=fake_sample_batch,
            device="cpu",
        )
        assert summary["config"] == "unit_tiny"
        assert summary["final_step"] == 4
        assert "mean_reward" in summary["last_metrics"]
        assert (tmp_path / "unit_tiny" / "latest.pt").exists()

    def test_tracker_records_steps(self, tmp_path, fake_tokenizer):
        from src.instrumentation.tracker import Tracker, TrackerConfig

        tracker = Tracker(TrackerConfig(use_wandb=False))
        main(
            "unit_tiny", ckpt_dir=tmp_path, resume=False,
            tokenizer=fake_tokenizer, sample_batch=fake_sample_batch,
            tracker=tracker, device="cpu",
        )
        records = tracker.get_records()
        steps = {r["step"] for r in records}
        assert steps  # something logged
        # loss logged every step 0..3
        assert any("loss" in r for r in records)

    def test_eval_fn_called(self, tmp_path, fake_tokenizer):
        calls = []

        def eval_fn(model, tok, step):
            calls.append(step)
            return {"eval/acc": 0.5}

        main(
            "unit_tiny", ckpt_dir=tmp_path, resume=False,
            tokenizer=fake_tokenizer, sample_batch=fake_sample_batch,
            eval_fn=eval_fn, device="cpu",
        )
        # eval_every=2, max_steps=4 → steps 0 and 2
        assert 0 in calls and 2 in calls

    def test_resume_from_checkpoint(self, tmp_path, fake_tokenizer):
        # First run to step 4, leaving latest.pt at step=max_steps.
        main(
            "unit_tiny", ckpt_dir=tmp_path, resume=False,
            tokenizer=fake_tokenizer, sample_batch=fake_sample_batch,
            device="cpu",
        )
        latest = tmp_path / "unit_tiny" / "latest.pt"
        assert latest.exists()

        # Resume: start_step becomes max_steps (4) so the loop body runs 0 times,
        # but it must not crash and must re-save.
        register_config(
            "unit_tiny_more",
            lambda: TrainConfig(
                **{**_tiny_train_config("unit_tiny_more").__dict__, "max_steps": 6}
            ),
        )
        # point the resumed run at the same checkpoint dir/name
        summary = main(
            "unit_tiny", ckpt_dir=tmp_path, resume=True,
            tokenizer=fake_tokenizer, sample_batch=fake_sample_batch,
            device="cpu", max_steps=6,
        )
        assert summary["final_step"] == 6

    def test_max_steps_override(self, tmp_path, fake_tokenizer):
        summary = main(
            "unit_tiny", ckpt_dir=tmp_path, resume=False,
            tokenizer=fake_tokenizer, sample_batch=fake_sample_batch,
            device="cpu", max_steps=2,
        )
        assert summary["final_step"] == 2
