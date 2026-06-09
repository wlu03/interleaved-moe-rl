"""Tests for the SFT warm-start (src/rl/sft.py) and its data helpers."""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.config import InterleavedMoEConfig
from src.rl.data import (
    SFTExample,
    collate_sft_batch,
    encode_sft_example,
    format_prompt,
)
from src.rl.sft import SFTConfig, sft_config_for, train_sft
from src.rl.train import build_model, load_checkpoint, load_config, main


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


@pytest.fixture
def fake_tokenizer():
    return FakeTokenizer()


def _sft_examples(n=6):
    return [
        SFTExample(format_prompt(f"add {i}"), f"so \\boxed{{{i + 1}}}", "toy")
        for i in range(n)
    ]


def _tiny_model_cfg():
    return InterleavedMoEConfig(
        hidden_size=32, num_layers=2, num_attention_heads=4, num_kv_heads=2,
        intermediate_size=64, expert_intermediate_size=64, num_experts=4,
        num_experts_per_tok=2, moe_every_n_layers=2, vocab_size=64,
        max_position_embeddings=128,
    )


# Encoding / labels
class TestEncodeSFT:
    def test_prompt_tokens_masked(self, fake_tokenizer):
        ex = _sft_examples(1)[0]
        prompt_len = len(fake_tokenizer.encode(ex.prompt))
        input_ids, labels = encode_sft_example(ex, fake_tokenizer, max_len=128)
        assert len(input_ids) == len(labels)
        assert all(l == -100 for l in labels[:prompt_len])

    def test_completion_supervised_and_ends_in_eos(self, fake_tokenizer):
        ex = _sft_examples(1)[0]
        prompt_len = len(fake_tokenizer.encode(ex.prompt))
        input_ids, labels = encode_sft_example(ex, fake_tokenizer, max_len=128)
        # completion labels are real token ids, not -100
        assert all(l != -100 for l in labels[prompt_len:])
        # last supervised token is EOS, and input/labels agree there
        assert labels[-1] == fake_tokenizer.eos_token_id
        assert input_ids[-1] == fake_tokenizer.eos_token_id

    def test_truncation_respects_max_len(self, fake_tokenizer):
        ex = SFTExample("x" * 50, "y" * 50, "toy")
        input_ids, labels = encode_sft_example(ex, fake_tokenizer, max_len=10)
        assert len(input_ids) == 10 and len(labels) == 10


class TestCollateSFT:
    def test_pads_to_uniform_length(self, fake_tokenizer):
        examples = [
            SFTExample(format_prompt("a"), "short", "toy"),
            SFTExample(format_prompt("a much longer question here"), "longer answer", "toy"),
        ]
        input_ids, labels = collate_sft_batch(examples, fake_tokenizer, max_len=128)
        assert input_ids.shape == labels.shape
        assert input_ids.shape[0] == 2

    def test_padding_positions_are_ignored(self, fake_tokenizer):
        examples = [
            SFTExample(format_prompt("a"), "x", "toy"),
            SFTExample(format_prompt("aaaaaaaa"), "xxxxxxxx", "toy"),
        ]
        input_ids, labels = collate_sft_batch(examples, fake_tokenizer, max_len=128)
        # The short row's tail must be pad in input_ids and -100 in labels.
        row0_len = len(encode_sft_example(examples[0], fake_tokenizer)[0])
        assert (input_ids[0, row0_len:] == fake_tokenizer.pad_token_id).all()
        assert (labels[0, row0_len:] == -100).all()


# Config derivation
class TestSFTConfig:
    def test_derives_arch_from_rl_config(self):
        rl = load_config("moe_interleaved_smoke")
        sft = sft_config_for("moe_interleaved_smoke")
        assert sft.model == rl.model
        assert sft.name == "moe_interleaved_smoke_sft"

    def test_overrides_apply(self):
        sft = sft_config_for("moe_interleaved_smoke", max_steps=3, lr=5e-4)
        assert sft.max_steps == 3 and sft.lr == 5e-4


# Loop
class TestTrainSFT:
    def _cfg(self):
        return SFTConfig(
            name="sft_unit", model=_tiny_model_cfg(),
            lr=1e-3, max_steps=4, batch_size=2, max_len=64,
            log_every=2, checkpoint_every=2,
        )

    def _next_batch(self, fake_tokenizer):
        examples = _sft_examples(6)

        def next_batch(tokenizer, batch_size, step, max_len):
            start = (step * batch_size) % len(examples)
            chunk = [examples[(start + j) % len(examples)] for j in range(batch_size)]
            return collate_sft_batch(chunk, tokenizer, max_len=max_len)

        return next_batch

    def test_runs_and_writes_checkpoint(self, tmp_path, fake_tokenizer):
        summary = train_sft(
            self._cfg(), ckpt_dir=tmp_path,
            tokenizer=fake_tokenizer, next_batch=self._next_batch(fake_tokenizer),
            device="cpu",
        )
        assert summary["final_step"] == 4
        assert "last_loss" in summary
        assert (tmp_path / "sft_unit" / "latest.pt").exists()

    def test_updates_parameters(self, tmp_path, fake_tokenizer):
        cfg = self._cfg()
        model = build_model(cfg.model)
        before = [p.detach().clone() for p in model.parameters()]
        train_sft(
            cfg, ckpt_dir=tmp_path, tokenizer=fake_tokenizer,
            next_batch=self._next_batch(fake_tokenizer), model=model, device="cpu",
        )
        changed = any(not torch.equal(b, a) for b, a in zip(before, model.parameters()))
        assert changed

    def test_checkpoint_loadable_as_rl_init(self, tmp_path, fake_tokenizer):
        cfg = self._cfg()
        train_sft(
            cfg, ckpt_dir=tmp_path, tokenizer=fake_tokenizer,
            next_batch=self._next_batch(fake_tokenizer), device="cpu",
        )
        sft_ckpt = tmp_path / "sft_unit" / "latest.pt"
        # A fresh model of the same arch can load just the weights.
        fresh = build_model(cfg.model)
        step = load_checkpoint(fresh, None, sft_ckpt)
        assert step == cfg.max_steps


# init_from wiring into the RL loop
class TestSFTThenRL:
    def test_main_init_from_loads_sft_weights(self, tmp_path, fake_tokenizer):
        from src.rl.train import register_config, TrainConfig
        from src.rl.grpo import GRPOConfig

        model_cfg = _tiny_model_cfg()
        register_config(
            "sft_then_rl_unit",
            lambda: TrainConfig(
                name="sft_then_rl_unit", model=model_cfg,
                grpo=GRPOConfig(group_size=2, max_completion_len=8),
                lr=1e-3, max_steps=2, batch_prompts=2,
                eval_every=10, drift_every=10, checkpoint_every=10, probe_size=8,
            ),
        )

        # Produce an SFT checkpoint with a distinctive weight.
        sft_cfg = SFTConfig(name="sft_src", model=model_cfg, max_steps=1, batch_size=2)
        sft_model = build_model(model_cfg)
        with torch.no_grad():
            next(sft_model.parameters()).fill_(0.123)
        from src.rl.data import collate_sft_batch

        def nb(tok, bs, step, ml):
            ex = _sft_examples(bs)
            return collate_sft_batch(ex, tok, max_len=ml)

        # max_steps=0 so weights are saved unchanged.
        train_sft(SFTConfig(name="sft_src", model=model_cfg, max_steps=0, batch_size=2),
                  ckpt_dir=tmp_path, tokenizer=fake_tokenizer, next_batch=nb,
                  model=sft_model, device="cpu")
        sft_ckpt = tmp_path / "sft_src" / "latest.pt"

        def sample_batch(b, step):
            return [f"q{j}" for j in range(b)], [str(j) for j in range(b)]

        summary = main(
            "sft_then_rl_unit", ckpt_dir=tmp_path, resume=False,
            tokenizer=fake_tokenizer, sample_batch=sample_batch,
            device="cpu", init_from=sft_ckpt,
        )
        assert summary["final_step"] == 2
