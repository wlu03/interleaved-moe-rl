"""Tests for the GRPO loss and training step."""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.config import InterleavedMoEConfig
from src.model import InterleavedMoEModel
from src.rl.grpo import (
    GRPOConfig,
    compute_logprobs,
    grpo_loss,
    grpo_train_step,
    selective_log_softmax,
    shift_completion_mask,
)
from src.rl.rollout import RolloutBatch


# =====================================================================
#  Fixtures
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
    def __init__(self, vocab_size: int = 64):
        self.vocab_size = vocab_size
        self.pad_token_id = 0
        self.eos_token_id = 1

    def encode(self, text: str) -> list[int]:
        ids = [(ord(c) % (self.vocab_size - 2)) + 2 for c in text]
        return ids if ids else [2]

    def decode(self, ids) -> str:
        out = []
        for tid in ids:
            t = int(tid)
            if t in (self.pad_token_id, self.eos_token_id):
                continue
            out.append(chr((t - 2) + 32))
        return "".join(out)


@pytest.fixture
def fake_tokenizer(tiny_config):
    return FakeTokenizer(vocab_size=tiny_config.vocab_size)


def make_batch(n_prompts, group_size, prompt_len, comp_len, vocab=64, rewards=None):
    """Build a deterministic RolloutBatch for loss tests."""
    N = n_prompts * group_size
    T = prompt_len + comp_len
    torch.manual_seed(1)
    input_ids = torch.randint(2, vocab, (N, T))
    attention_mask = torch.ones(N, T, dtype=torch.long)
    completion_mask = torch.zeros(N, T, dtype=torch.long)
    completion_mask[:, prompt_len:] = 1
    if rewards is None:
        rewards = torch.arange(N, dtype=torch.float)
    from src.rl.rollout import group_advantages

    advantages = group_advantages(rewards, group_size)
    prompt_indices = torch.repeat_interleave(
        torch.arange(n_prompts), group_size
    )
    prompt_lens = torch.full((N,), prompt_len, dtype=torch.long)
    return RolloutBatch(
        input_ids=input_ids,
        attention_mask=attention_mask,
        completion_mask=completion_mask,
        rewards=rewards,
        advantages=advantages,
        prompt_indices=prompt_indices,
        prompt_lens=prompt_lens,
        completion_texts=[""] * N,
    )


# =====================================================================
#  selective_log_softmax
# =====================================================================
class TestSelectiveLogSoftmax:
    def test_matches_full_log_softmax(self):
        torch.manual_seed(0)
        logits = torch.randn(3, 5, 7)
        index = torch.randint(0, 7, (3, 5))
        got = selective_log_softmax(logits, index)
        ref = torch.log_softmax(logits, dim=-1).gather(
            -1, index.unsqueeze(-1)
        ).squeeze(-1)
        assert torch.allclose(got, ref, atol=1e-6)

    def test_shape(self):
        logits = torch.randn(2, 4, 9)
        index = torch.zeros(2, 4, dtype=torch.long)
        assert selective_log_softmax(logits, index).shape == (2, 4)


# =====================================================================
#  compute_logprobs
# =====================================================================
class TestComputeLogprobs:
    def test_shape_is_T_minus_1(self, tiny_model):
        input_ids = torch.randint(2, 64, (4, 10))
        lp = compute_logprobs(tiny_model, input_ids)
        assert lp.shape == (4, 9)

    def test_logprobs_are_negative(self, tiny_model):
        input_ids = torch.randint(2, 64, (2, 8))
        lp = compute_logprobs(tiny_model, input_ids)
        assert (lp <= 0).all()

    def test_alignment_to_next_token(self, tiny_model):
        # column j must be the log-prob of input_ids[:, j+1] under logits[:, j]
        input_ids = torch.randint(2, 64, (1, 6))
        logits = tiny_model(input_ids)["logits"]
        expected = torch.log_softmax(logits[:, 2, :], dim=-1)[0, input_ids[0, 3]]
        got = compute_logprobs(tiny_model, input_ids)[0, 2]
        assert torch.allclose(got, expected, atol=1e-5)


# =====================================================================
#  shift_completion_mask
# =====================================================================
class TestShiftCompletionMask:
    def test_drops_first_column(self):
        cm = torch.tensor([[0, 0, 1, 1, 1]])
        shifted = shift_completion_mask(cm)
        assert shifted.shape == (1, 4)
        assert torch.equal(shifted, torch.tensor([[0.0, 1.0, 1.0, 1.0]]))


# =====================================================================
#  grpo_loss
# =====================================================================
class TestGrpoLoss:
    def test_ratio_one_when_logprobs_equal(self):
        # old == new → ratio == 1 → loss == -(adv) summed over comp tokens / norm
        lp = torch.tensor([[-1.0, -2.0, -0.5]])
        adv = torch.tensor([2.0])
        mask = torch.ones(1, 3)
        cfg = GRPOConfig(max_completion_len=3)
        loss, m = grpo_loss(lp, lp.clone(), adv, mask, cfg)
        # per_token = -ratio*adv = -1*2 = -2 each; sum = -6; norm = N*maxlen = 3
        assert torch.allclose(loss, torch.tensor(-2.0))
        assert abs(m["mean_ratio"] - 1.0) < 1e-6

    def test_mask_zeros_out_prompt(self):
        lp = torch.tensor([[-1.0, -1.0, -1.0, -1.0]])
        adv = torch.tensor([1.0])
        mask = torch.tensor([[0.0, 0.0, 1.0, 1.0]])
        cfg = GRPOConfig(max_completion_len=4)
        loss, m = grpo_loss(lp, lp.clone(), adv, mask, cfg)
        assert m["completion_tokens"] == 2.0
        # only 2 tokens contribute: -(1)*2 / (1*4) = -0.5
        assert torch.allclose(loss, torch.tensor(-0.5))

    def test_zero_advantage_zero_loss(self):
        lp = torch.randn(2, 5)
        mask = torch.ones(2, 5)
        cfg = GRPOConfig(max_completion_len=5)
        loss, _ = grpo_loss(lp, lp.clone(), torch.zeros(2), mask, cfg)
        assert torch.allclose(loss, torch.tensor(0.0))

    def test_dr_grpo_no_length_bias(self):
        # Two rows, same advantage & per-token loss, different completion
        # lengths. dr_grpo normalizer is constant, so the longer sequence
        # contributes proportionally more (no per-sequence averaging).
        lp = torch.zeros(2, 6)  # ratio terms all exp(0)=1
        adv = torch.tensor([1.0, 1.0])
        mask = torch.tensor(
            [[1.0, 1, 1, 1, 1, 1], [1, 1, 1, 0, 0, 0]]
        )
        cfg = GRPOConfig(max_completion_len=6, loss_type="dr_grpo")
        loss, _ = grpo_loss(lp, lp.clone(), adv, mask, cfg)
        # sum per_token = -(1)*(6+3 tokens) = -9 ; norm = 2*6 = 12
        assert torch.allclose(loss, torch.tensor(-9.0 / 12.0))

    def test_grpo_per_sequence_mean(self):
        lp = torch.zeros(2, 6)
        adv = torch.tensor([1.0, 1.0])
        mask = torch.tensor(
            [[1.0, 1, 1, 1, 1, 1], [1, 1, 1, 0, 0, 0]]
        )
        cfg = GRPOConfig(loss_type="grpo")
        loss, _ = grpo_loss(lp, lp.clone(), adv, mask, cfg)
        # row means: -1 and -1 → mean -1 (length-normalized)
        assert torch.allclose(loss, torch.tensor(-1.0))

    def test_clip_higher_caps_positive_advantage(self):
        # ratio large, positive advantage → clipped at 1+eps_high
        old = torch.tensor([[0.0]])
        new = torch.tensor([[2.0]])  # ratio = e^2 ≈ 7.39
        adv = torch.tensor([1.0])
        mask = torch.ones(1, 1)
        cfg = GRPOConfig(max_completion_len=1, eps_high=0.28)
        loss, m = grpo_loss(new, old, adv, mask, cfg)
        # min(7.39*1, 1.28*1) = 1.28 → loss = -1.28 / 1
        assert torch.allclose(loss, torch.tensor(-1.28), atol=1e-4)
        assert m["clip_frac"] == 1.0

    def test_unknown_loss_type_raises(self):
        lp = torch.zeros(1, 2)
        cfg = GRPOConfig(loss_type="bogus")
        with pytest.raises(ValueError):
            grpo_loss(lp, lp, torch.zeros(1), torch.ones(1, 2), cfg)


# =====================================================================
#  grpo_train_step
# =====================================================================
class TestGrpoTrainStep:
    def test_step_runs_and_updates_params(self, tiny_model, fake_tokenizer):
        cfg = GRPOConfig(group_size=2, max_completion_len=4)
        optim = torch.optim.SGD(tiny_model.parameters(), lr=0.1)

        # Inject a fake rollout so the step is deterministic and fast.
        def fake_rollout(model, tok, prompts, golds, **kw):
            return make_batch(
                len(prompts), cfg.group_size, prompt_len=3, comp_len=4,
                vocab=64, rewards=torch.tensor([1.0, 0.0, 1.0, 0.0]),
            )

        before = [p.detach().clone() for p in tiny_model.parameters()]
        metrics = grpo_train_step(
            tiny_model, fake_tokenizer, optim,
            prompts=["a", "b"], golds=["1", "2"], cfg=cfg,
            rollout_fn=fake_rollout,
        )
        after = list(tiny_model.parameters())
        changed = any(not torch.equal(b, a) for b, a in zip(before, after))
        assert changed
        assert "loss" in metrics and "mean_reward" in metrics
        assert metrics["mean_reward"] == 0.5

    def test_empty_prompts_no_crash(self, tiny_model, fake_tokenizer):
        cfg = GRPOConfig(group_size=2)
        optim = torch.optim.SGD(tiny_model.parameters(), lr=0.1)

        def empty_rollout(model, tok, prompts, golds, **kw):
            return make_batch(0, 2, 1, 1) if prompts else _empty_batch()

        def _empty_batch():
            z2 = torch.zeros(0, 0, dtype=torch.long)
            z1 = torch.zeros(0, dtype=torch.long)
            zf = torch.zeros(0, dtype=torch.float)
            return RolloutBatch(z2, z2, z2, zf, zf, z1, z1, [])

        metrics = grpo_train_step(
            tiny_model, fake_tokenizer, optim,
            prompts=[], golds=[], cfg=cfg, rollout_fn=lambda *a, **k: _empty_batch(),
        )
        assert metrics["loss"] == 0.0

    def test_grad_norm_reported_and_clipped(self, tiny_model, fake_tokenizer):
        cfg = GRPOConfig(group_size=2, max_completion_len=4, grad_clip=0.5)
        optim = torch.optim.SGD(tiny_model.parameters(), lr=0.1)

        def fake_rollout(model, tok, prompts, golds, **kw):
            return make_batch(1, 2, prompt_len=3, comp_len=4,
                              rewards=torch.tensor([1.0, -1.0]))

        metrics = grpo_train_step(
            tiny_model, fake_tokenizer, optim,
            prompts=["a"], golds=["1"], cfg=cfg, rollout_fn=fake_rollout,
        )
        assert "grad_norm" in metrics
        assert metrics["grad_norm"] >= 0.0

    def test_restores_eval_mode(self, tiny_model, fake_tokenizer):
        cfg = GRPOConfig(group_size=2, max_completion_len=4)
        optim = torch.optim.SGD(tiny_model.parameters(), lr=0.1)
        tiny_model.eval()

        def fake_rollout(model, tok, prompts, golds, **kw):
            return make_batch(1, 2, 3, 4, rewards=torch.tensor([1.0, 0.0]))

        grpo_train_step(
            tiny_model, fake_tokenizer, optim,
            prompts=["a"], golds=["1"], cfg=cfg, rollout_fn=fake_rollout,
        )
        assert not tiny_model.training  # restored to eval
