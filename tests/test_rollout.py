"""Tests for the rollout sampler."""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.config import InterleavedMoEConfig
from src.model import InterleavedMoEModel
from src.rl.rollout import (
    RolloutBatch,
    group_advantages,
    rollout,
    sample_completions,
    sample_next_token,
    sample_one_prompt,
)


# =====================================================================
#  Test fixtures: tiny config + a fake tokenizer
# =====================================================================
@pytest.fixture
def tiny_config():
    return InterleavedMoEConfig(
        hidden_size=32,
        num_layers=2,
        num_attention_heads=4,
        num_kv_heads=2,
        intermediate_size=64,
        num_experts=4,
        num_experts_per_tok=2,
        expert_intermediate_size=64,
        moe_every_n_layers=2,
        vocab_size=64,
        max_position_embeddings=128,
    )


@pytest.fixture
def tiny_model(tiny_config):
    torch.manual_seed(0)
    return InterleavedMoEModel(tiny_config)


class FakeTokenizer:
    """Minimal tokenizer matching the Protocol used by `rollout`.

    Each character maps to its ASCII code (mod vocab_size). Reserves IDs 0
    and 1 for pad/EOS so they don't accidentally appear in encode().
    """

    def __init__(self, vocab_size: int = 64):
        self.vocab_size = vocab_size
        self.pad_token_id = 0
        self.eos_token_id = 1

    def encode(self, text: str) -> list[int]:
        # Map chars to ids in [2, vocab_size). 0/1 are reserved.
        ids = [(ord(c) % (self.vocab_size - 2)) + 2 for c in text]
        return ids if ids else [2]  # never empty

    def decode(self, ids) -> str:
        out = []
        for tid in ids:
            t = int(tid)
            if t in (self.pad_token_id, self.eos_token_id):
                continue
            out.append(chr((t - 2) + 32))  # arbitrary printable mapping
        return "".join(out)


@pytest.fixture
def fake_tokenizer(tiny_config):
    return FakeTokenizer(vocab_size=tiny_config.vocab_size)


# =====================================================================
#  sample_next_token
# =====================================================================
class TestSampleNextToken:
    def test_greedy_picks_argmax(self):
        logits = torch.tensor([[1.0, 2.0, 3.0, 0.5]])
        out = sample_next_token(logits, temperature=0.0)
        assert out.shape == (1,)
        assert int(out.item()) == 2

    def test_temperature_returns_valid_id(self):
        torch.manual_seed(0)
        logits = torch.randn(4, 16)
        out = sample_next_token(logits, temperature=1.0)
        assert out.shape == (4,)
        assert (out >= 0).all() and (out < 16).all()

    def test_top_p_zero_stays_top1(self):
        # top_p with extremely low budget falls back to top-1 (we always include it)
        logits = torch.tensor([[10.0, 0.0, 0.0]] * 4)
        out = sample_next_token(logits, temperature=1.0, top_p=0.01)
        assert (out == 0).all()

    def test_top_p_one_unrestricted(self):
        torch.manual_seed(0)
        logits = torch.randn(4, 32)
        out = sample_next_token(logits, temperature=1.0, top_p=1.0)
        assert (out >= 0).all() and (out < 32).all()


# =====================================================================
#  sample_one_prompt
# =====================================================================
class TestSampleOnePrompt:
    def test_shapes(self, tiny_model, fake_tokenizer):
        ids = fake_tokenizer.encode("hello")
        seq, attn, mask = sample_one_prompt(
            tiny_model, ids,
            eos_token_id=fake_tokenizer.eos_token_id,
            pad_token_id=fake_tokenizer.pad_token_id,
            group_size=4, max_new_tokens=8, temperature=0.0,
        )
        assert seq.shape[0] == 4
        assert attn.shape == seq.shape
        assert mask.shape == seq.shape
        assert seq.shape[1] == len(ids) + 8  # greedy never stops via EOS in this random model

    def test_completion_mask_zero_on_prompt(self, tiny_model, fake_tokenizer):
        ids = fake_tokenizer.encode("abc")
        prompt_len = len(ids)
        _, _, mask = sample_one_prompt(
            tiny_model, ids,
            eos_token_id=fake_tokenizer.eos_token_id,
            pad_token_id=fake_tokenizer.pad_token_id,
            group_size=2, max_new_tokens=4, temperature=0.0,
        )
        assert (mask[:, :prompt_len] == 0).all()
        assert (mask[:, prompt_len:] == 1).all()  # nothing finished early w/ greedy

    def test_attention_covers_prompt_and_active_completion(self, tiny_model, fake_tokenizer):
        ids = fake_tokenizer.encode("abc")
        seq, attn, mask = sample_one_prompt(
            tiny_model, ids,
            eos_token_id=fake_tokenizer.eos_token_id,
            pad_token_id=fake_tokenizer.pad_token_id,
            group_size=2, max_new_tokens=4, temperature=0.0,
        )
        prompt_len = len(ids)
        # Prompt portion always attended
        assert (attn[:, :prompt_len] == 1).all()
        # Completion portion attended iff completion_mask is set
        assert (attn[:, prompt_len:] == mask[:, prompt_len:]).all()

    def test_eos_inclusion_then_no_more_generation(self, tiny_config, fake_tokenizer):
        """When the model produces EOS, that EOS is included in the
        completion mask, but no further generated tokens for that row are."""
        # Use a stub model whose logits always favor EOS, so greedy decoding
        # emits EOS at the very first generated step. (Patching the real
        # model's tied embeddings doesn't work: zeroing the embedding table
        # makes the input embeddings — and thus all logits — zero, so argmax
        # would pick token 0, not EOS.)
        eos_id = fake_tokenizer.eos_token_id
        vocab_size = tiny_config.vocab_size

        class EosModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # A real parameter so `next(model.parameters()).device` works.
                self._p = torch.nn.Parameter(torch.zeros(1))

            def forward(self, seq):
                logits = torch.zeros(
                    seq.shape[0], seq.shape[1], vocab_size, device=seq.device
                )
                logits[..., eos_id] = 10.0
                return {"logits": logits, "routing_stats": {}}

        model = EosModel()

        ids = fake_tokenizer.encode("xy")
        prompt_len = len(ids)
        seq, attn, mask = sample_one_prompt(
            model, ids,
            eos_token_id=fake_tokenizer.eos_token_id,
            pad_token_id=fake_tokenizer.pad_token_id,
            group_size=2, max_new_tokens=5, temperature=0.0,
        )
        # First generated token across every row should be EOS (mask=1)
        assert (seq[:, prompt_len] == fake_tokenizer.eos_token_id).all()
        assert (mask[:, prompt_len] == 1).all()
        # Subsequent positions: no further generation, mask=0, padded
        if seq.shape[1] > prompt_len + 1:
            assert (seq[:, prompt_len + 1:] == fake_tokenizer.pad_token_id).all()
            assert (mask[:, prompt_len + 1:] == 0).all()

    def test_empty_prompt_raises(self, tiny_model, fake_tokenizer):
        with pytest.raises(ValueError):
            sample_one_prompt(
                tiny_model, [],
                eos_token_id=fake_tokenizer.eos_token_id,
                pad_token_id=fake_tokenizer.pad_token_id,
                group_size=1, max_new_tokens=2,
            )

    def test_invalid_group_size_raises(self, tiny_model, fake_tokenizer):
        with pytest.raises(ValueError):
            sample_one_prompt(
                tiny_model, [2, 3],
                eos_token_id=fake_tokenizer.eos_token_id,
                pad_token_id=fake_tokenizer.pad_token_id,
                group_size=0, max_new_tokens=2,
            )

    def test_zero_max_new_tokens_returns_only_prompt(self, tiny_model, fake_tokenizer):
        ids = fake_tokenizer.encode("hi")
        seq, attn, mask = sample_one_prompt(
            tiny_model, ids,
            eos_token_id=fake_tokenizer.eos_token_id,
            pad_token_id=fake_tokenizer.pad_token_id,
            group_size=3, max_new_tokens=0, temperature=0.0,
        )
        assert seq.shape == (3, len(ids))
        assert mask.sum() == 0
        assert (attn == 1).all()

    def test_does_not_change_training_mode(self, tiny_model, fake_tokenizer):
        tiny_model.train()
        sample_one_prompt(
            tiny_model, [2, 3, 4],
            eos_token_id=fake_tokenizer.eos_token_id,
            pad_token_id=fake_tokenizer.pad_token_id,
            group_size=1, max_new_tokens=2,
        )
        assert tiny_model.training


# =====================================================================
#  sample_completions
# =====================================================================
class TestSampleCompletions:
    def test_uniform_padding_across_prompts(self, tiny_model, fake_tokenizer):
        ids_a = fake_tokenizer.encode("a")
        ids_b = fake_tokenizer.encode("longer")
        seqs, attns, masks, p_idx, p_lens = sample_completions(
            tiny_model, [ids_a, ids_b],
            eos_token_id=fake_tokenizer.eos_token_id,
            pad_token_id=fake_tokenizer.pad_token_id,
            group_size=2, max_new_tokens=3, temperature=0.0,
        )
        # Expected total rows = 2 prompts × 2 group = 4
        assert seqs.shape[0] == 4
        # All rows share T_max
        assert seqs.shape == attns.shape == masks.shape

    def test_prompt_indices_grouped(self, tiny_model, fake_tokenizer):
        ids_a = fake_tokenizer.encode("a")
        ids_b = fake_tokenizer.encode("b")
        ids_c = fake_tokenizer.encode("c")
        _, _, _, p_idx, _ = sample_completions(
            tiny_model, [ids_a, ids_b, ids_c],
            eos_token_id=fake_tokenizer.eos_token_id,
            pad_token_id=fake_tokenizer.pad_token_id,
            group_size=4, max_new_tokens=2,
        )
        # Each prompt should appear group_size times, consecutively
        assert p_idx.tolist() == [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]

    def test_prompt_lens_correct(self, tiny_model, fake_tokenizer):
        ids_a = fake_tokenizer.encode("ab")
        ids_b = fake_tokenizer.encode("xyz")
        _, _, _, _, p_lens = sample_completions(
            tiny_model, [ids_a, ids_b],
            eos_token_id=fake_tokenizer.eos_token_id,
            pad_token_id=fake_tokenizer.pad_token_id,
            group_size=2, max_new_tokens=2,
        )
        assert p_lens.tolist() == [2, 2, 3, 3]

    def test_empty_prompt_list(self, tiny_model, fake_tokenizer):
        seqs, attns, masks, p_idx, p_lens = sample_completions(
            tiny_model, [],
            eos_token_id=fake_tokenizer.eos_token_id,
            pad_token_id=fake_tokenizer.pad_token_id,
            group_size=2, max_new_tokens=2,
        )
        assert seqs.numel() == 0
        assert p_idx.numel() == 0


# =====================================================================
#  group_advantages
# =====================================================================
class TestGroupAdvantages:
    def test_centers_per_group(self):
        # Group 0: rewards [0, 1, 0, 1] → mean 0.5 → advantages [-0.5, 0.5, -0.5, 0.5]
        # Group 1: rewards [1, 1, 1, 1] → mean 1.0 → advantages [0, 0, 0, 0]
        rewards = torch.tensor([0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        adv = group_advantages(rewards, group_size=4, scale_rewards=False)
        assert adv[:4].tolist() == pytest.approx([-0.5, 0.5, -0.5, 0.5])
        assert adv[4:].tolist() == pytest.approx([0.0, 0.0, 0.0, 0.0])

    def test_scaled_normalizes_std(self):
        rewards = torch.tensor([0.0, 1.0, 0.0, 1.0])
        adv = group_advantages(rewards, group_size=4, scale_rewards=True)
        assert adv.mean().item() == pytest.approx(0.0, abs=1e-6)
        assert adv.std(unbiased=False).item() == pytest.approx(1.0, abs=1e-3)

    def test_degenerate_group_zero_advantage(self):
        # All-equal rewards → after centering → all zeros
        rewards = torch.tensor([1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0])
        adv_unscaled = group_advantages(rewards, group_size=4, scale_rewards=False)
        adv_scaled = group_advantages(rewards, group_size=4, scale_rewards=True)
        assert torch.allclose(adv_unscaled, torch.zeros_like(adv_unscaled))
        # Scaled: 0 / clamped_std = 0, no NaNs
        assert not torch.isnan(adv_scaled).any()
        assert torch.allclose(adv_scaled, torch.zeros_like(adv_scaled), atol=1e-5)

    def test_size_mismatch_raises(self):
        with pytest.raises(ValueError):
            group_advantages(torch.tensor([1.0, 2.0, 3.0]), group_size=2)

    def test_non_1d_raises(self):
        with pytest.raises(ValueError):
            group_advantages(torch.zeros(4, 4), group_size=2)

    def test_empty_input(self):
        out = group_advantages(torch.zeros(0), group_size=4)
        assert out.numel() == 0


# =====================================================================
#  rollout (top-level API)
# =====================================================================
class TestRollout:
    def test_basic_shapes_and_types(self, tiny_model, fake_tokenizer):
        torch.manual_seed(0)
        prompts = ["abc", "defg"]
        golds = ["1", "2"]

        def fake_reward(completion, gold, step=0):
            return 1.0 if gold == "1" else 0.0

        out = rollout(
            tiny_model, fake_tokenizer, prompts, golds,
            group_size=4, max_new_tokens=4, temperature=0.0,
            reward_fn=fake_reward,
        )
        assert isinstance(out, RolloutBatch)
        N = 2 * 4
        assert out.input_ids.shape[0] == N
        assert out.attention_mask.shape == out.input_ids.shape
        assert out.completion_mask.shape == out.input_ids.shape
        assert out.rewards.shape == (N,)
        assert out.advantages.shape == (N,)
        assert out.prompt_indices.shape == (N,)
        assert out.prompt_lens.shape == (N,)
        assert len(out.completion_texts) == N

    def test_rewards_dispatched_correctly(self, tiny_model, fake_tokenizer):
        prompts = ["a", "b"]
        golds = ["X", "Y"]
        calls = []

        def reward_fn(completion, gold, step=0):
            calls.append((completion, gold, step))
            return 0.5

        out = rollout(
            tiny_model, fake_tokenizer, prompts, golds,
            group_size=2, max_new_tokens=2, temperature=0.0,
            reward_fn=reward_fn, step=42,
        )
        # 2 prompts × 2 group = 4 reward calls
        assert len(calls) == 4
        # First two calls should reference "X", next two "Y"
        assert calls[0][1] == "X" and calls[1][1] == "X"
        assert calls[2][1] == "Y" and calls[3][1] == "Y"
        # step propagated
        assert all(c[2] == 42 for c in calls)
        assert torch.allclose(out.rewards, torch.full((4,), 0.5))

    def test_empty_prompts(self, tiny_model, fake_tokenizer):
        out = rollout(tiny_model, fake_tokenizer, [], [], group_size=2, max_new_tokens=2)
        assert out.num_rollouts == 0
        assert out.completion_texts == []

    def test_length_mismatch_raises(self, tiny_model, fake_tokenizer):
        with pytest.raises(ValueError):
            rollout(tiny_model, fake_tokenizer, ["a", "b"], ["1"], group_size=2, max_new_tokens=2)

    def test_advantages_centered_per_group(self, tiny_model, fake_tokenizer):
        # Use deterministic fake reward depending on rollout index
        seen = {"i": 0}

        def reward_fn(completion, gold, step=0):
            i = seen["i"]
            seen["i"] += 1
            # Group 0 (first 4 calls) rewards 0,1,0,1; group 1 all 0.5
            if i < 4:
                return float(i % 2)
            return 0.5

        torch.manual_seed(0)
        out = rollout(
            tiny_model, fake_tokenizer, ["a", "b"], ["x", "y"],
            group_size=4, max_new_tokens=2, temperature=0.0,
            reward_fn=reward_fn,
        )
        # Group 0: rewards [0,1,0,1], advantages [-0.5,0.5,-0.5,0.5]
        # Group 1: all 0.5, advantages [0,0,0,0]
        assert out.advantages[:4].tolist() == pytest.approx([-0.5, 0.5, -0.5, 0.5])
        assert out.advantages[4:].tolist() == pytest.approx([0.0, 0.0, 0.0, 0.0])

    def test_determinism_under_greedy(self, tiny_model, fake_tokenizer):
        prompts = ["abc", "xyz"]
        golds = ["g1", "g2"]
        out1 = rollout(
            tiny_model, fake_tokenizer, prompts, golds,
            group_size=2, max_new_tokens=4, temperature=0.0,
            reward_fn=lambda *a, **k: 0.0,
        )
        out2 = rollout(
            tiny_model, fake_tokenizer, prompts, golds,
            group_size=2, max_new_tokens=4, temperature=0.0,
            reward_fn=lambda *a, **k: 0.0,
        )
        assert torch.equal(out1.input_ids, out2.input_ids)
        assert torch.equal(out1.completion_mask, out2.completion_mask)

    def test_to_device_no_op_on_cpu(self, tiny_model, fake_tokenizer):
        out = rollout(
            tiny_model, fake_tokenizer, ["a"], ["x"],
            group_size=2, max_new_tokens=2,
            reward_fn=lambda *a, **k: 0.0,
        )
        out2 = out.to("cpu")
        assert torch.equal(out.input_ids, out2.input_ids)

    def test_completion_texts_decoded_for_each_rollout(self, tiny_model, fake_tokenizer):
        out = rollout(
            tiny_model, fake_tokenizer, ["a", "b"], ["x", "y"],
            group_size=3, max_new_tokens=2, temperature=0.0,
            reward_fn=lambda *a, **k: 0.0,
        )
        assert len(out.completion_texts) == 2 * 3
        # All decodes are strings (possibly empty if model never generated valid tokens)
        for t in out.completion_texts:
            assert isinstance(t, str)
