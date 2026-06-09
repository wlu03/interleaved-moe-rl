"""Tests for the dataset + eval providers (src/rl/data.py).

Only the `datasets`-free parts are exercised: prompt formatting, gold
extraction, batch assembly, probe building, scoring, and the eval_fn (run
against a stub model). The HuggingFace loaders are integration-tested on
Modal where `datasets` is installed.
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.config import InterleavedMoEConfig
from src.model import InterleavedMoEModel
from src.rl.data import (
    Example,
    build_probe_input_ids,
    format_prompt,
    gsm8k_gold,
    make_eval_fn,
    make_sample_batch,
    math_gold,
    score_completion,
)


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


def _examples(n=10):
    return [
        Example(format_prompt(f"what is {i}+1?"), str(i + 1), "gsm8k")
        for i in range(n)
    ]


# =====================================================================
#  Prompt format
# =====================================================================
class TestFormatPrompt:
    def test_contains_system_user_assistant(self):
        p = format_prompt("2+2?")
        assert "System:" in p and "User: 2+2?" in p and p.rstrip().endswith("Assistant:")

    def test_includes_boxed_instruction(self):
        assert "\\boxed{}" in format_prompt("x")


# =====================================================================
#  Gold extraction
# =====================================================================
class TestGoldExtraction:
    def test_gsm8k_gold_from_terminator(self):
        ans = "Some reasoning here.\n#### 42"
        assert gsm8k_gold(ans) == "42"

    def test_gsm8k_gold_fallback_to_raw(self):
        # No #### and no "answer is" → returns stripped raw text
        assert gsm8k_gold("  72  ") == "72"

    def test_math_gold_from_boxed(self):
        sol = "We compute ... so the answer is \\boxed{\\frac{1}{2}}."
        assert math_gold(sol) == "\\frac{1}{2}"

    def test_math_gold_last_boxed_wins(self):
        sol = "First \\boxed{1}, but actually \\boxed{2}."
        assert math_gold(sol) == "2"


# =====================================================================
#  sample_batch provider
# =====================================================================
class TestSampleBatch:
    def test_returns_parallel_prompts_and_golds(self):
        sb = make_sample_batch(_examples(10), seed=0)
        prompts, golds = sb(4, 0)
        assert len(prompts) == 4 and len(golds) == 4

    def test_deterministic_for_same_seed_and_step(self):
        a = make_sample_batch(_examples(10), seed=7)
        b = make_sample_batch(_examples(10), seed=7)
        assert a(3, 5) == b(3, 5)

    def test_different_seed_changes_order(self):
        a = make_sample_batch(_examples(20), seed=1)
        b = make_sample_batch(_examples(20), seed=2)
        # Highly unlikely to match across 20 elements with different seeds
        assert a(5, 0) != b(5, 0)

    def test_wraps_around_pool(self):
        sb = make_sample_batch(_examples(4), seed=0)
        prompts, golds = sb(6, 0)  # batch bigger than pool → wraps
        assert len(prompts) == 6

    def test_empty_pool_raises(self):
        with pytest.raises(ValueError):
            make_sample_batch([], seed=0)


# =====================================================================
#  Probe builder
# =====================================================================
class TestProbe:
    def test_shape_and_padding(self, fake_tokenizer):
        probe = build_probe_input_ids(_examples(10), fake_tokenizer, n=5)
        assert probe.shape[0] == 5
        assert probe.dtype == torch.long

    def test_caps_at_pool_size(self, fake_tokenizer):
        probe = build_probe_input_ids(_examples(3), fake_tokenizer, n=10)
        assert probe.shape[0] == 3


# =====================================================================
#  Scoring
# =====================================================================
class TestScoreCompletion:
    def test_boxed_correct(self):
        assert score_completion("the answer is \\boxed{42}", "42")

    def test_boxed_wrong(self):
        assert not score_completion("\\boxed{41}", "42")

    def test_gsm8k_terminator_fallback(self):
        assert score_completion("reasoning ... #### 7", "7")

    def test_raw_text_fallback(self):
        assert score_completion("12", "12")


# =====================================================================
#  eval_fn against a stub model
# =====================================================================
class TestEvalFn:
    def test_accuracy_reported_per_set(self, fake_tokenizer):
        torch.manual_seed(0)
        cfg = InterleavedMoEConfig(
            hidden_size=32, num_layers=2, num_attention_heads=4, num_kv_heads=2,
            intermediate_size=64, expert_intermediate_size=64, num_experts=4,
            num_experts_per_tok=2, moe_every_n_layers=2, vocab_size=64,
            max_position_embeddings=128,
        )
        model = InterleavedMoEModel(cfg)
        eval_fn = make_eval_fn(
            {"tiny": _examples(3)}, max_examples=3, max_new_tokens=4,
        )
        out = eval_fn(model, fake_tokenizer, step=0)
        assert "eval/tiny/acc" in out
        assert 0.0 <= out["eval/tiny/acc"] <= 1.0

    def test_restores_training_mode(self, fake_tokenizer):
        torch.manual_seed(0)
        cfg = InterleavedMoEConfig(
            hidden_size=32, num_layers=2, num_attention_heads=4, num_kv_heads=2,
            intermediate_size=64, expert_intermediate_size=64, num_experts=4,
            num_experts_per_tok=2, moe_every_n_layers=2, vocab_size=64,
            max_position_embeddings=128,
        )
        model = InterleavedMoEModel(cfg)
        model.train()
        eval_fn = make_eval_fn({"tiny": _examples(2)}, max_new_tokens=3)
        eval_fn(model, fake_tokenizer, step=0)
        assert model.training  # restored
