"""Rollout sampler for GRPO.

Produces a RolloutBatch ready for the loss: G completions per prompt with the
right masks, group-normalized advantages, and decoded text for logging.

We sample natively in PyTorch (no vLLM at this scale) and batch per prompt:
replicate one prompt G times and generate all G completions together. Because
the rollouts within a prompt share the same prompt length, we sidestep the
left-padding RoPE headaches you'd hit batching across prompts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Protocol

import torch
import torch.nn.functional as F


class TokenizerLike(Protocol):
    """Just enough of a tokenizer for rollout; lets tests use a fake."""

    eos_token_id: int
    pad_token_id: int

    def encode(self, text: str) -> list[int]: ...
    def decode(self, ids) -> str: ...


@dataclass
class RolloutBatch:
    """A batch of N = B * G rollouts ready to feed into GRPO.

    All 2-D tensors are [N, T_max], right-padded:
        input_ids        prompt + generated tokens
        attention_mask   1 on the prompt and active completion tokens
        completion_mask  1 only on generated tokens (EOS included)
    The 1-D tensors are [N]: rewards, advantages (group-mean centered),
    prompt_indices (which prompt, 0..B-1), and prompt_lens.
    """

    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    completion_mask: torch.Tensor
    rewards: torch.Tensor
    advantages: torch.Tensor
    prompt_indices: torch.Tensor
    prompt_lens: torch.Tensor
    completion_texts: list[str] = field(default_factory=list)

    @property
    def num_rollouts(self) -> int:
        return int(self.input_ids.shape[0])

    @property
    def seq_len(self) -> int:
        return int(self.input_ids.shape[1]) if self.input_ids.ndim == 2 else 0

    def to(self, device) -> "RolloutBatch":
        return RolloutBatch(
            input_ids=self.input_ids.to(device),
            attention_mask=self.attention_mask.to(device),
            completion_mask=self.completion_mask.to(device),
            rewards=self.rewards.to(device),
            advantages=self.advantages.to(device),
            prompt_indices=self.prompt_indices.to(device),
            prompt_lens=self.prompt_lens.to(device),
            completion_texts=self.completion_texts,
        )


def sample_next_token(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_p: float = 1.0,
) -> torch.Tensor:
    """Sample one token per row from [B, V] logits, returning [B] ids.

    temperature == 0 is greedy argmax. top_p < 1 applies nucleus filtering.
    """
    if temperature <= 0:
        return logits.argmax(dim=-1)

    logits = logits.float() / temperature

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        sorted_probs = torch.softmax(sorted_logits, dim=-1)
        cumulative = sorted_probs.cumsum(dim=-1)
        keep = cumulative <= top_p
        keep[..., 0] = True  # always keep the top token
        sorted_logits = sorted_logits.masked_fill(~keep, float("-inf"))
        logits_filtered = torch.full_like(logits, float("-inf"))
        logits_filtered.scatter_(-1, sorted_indices, sorted_logits)
        logits = logits_filtered

    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)


@torch.no_grad()
def sample_one_prompt(
    model: torch.nn.Module,
    prompt_ids: list[int],
    *,
    eos_token_id: int,
    pad_token_id: int,
    group_size: int = 8,
    max_new_tokens: int = 512,
    temperature: float = 1.0,
    top_p: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate `group_size` completions from a single prompt.

    Returns (seq, attention_mask, completion_mask), each [G, prompt_len + n].
    Rows that hit EOS early are right-padded so all G share a length.
    """
    if group_size <= 0:
        raise ValueError(f"group_size must be >= 1, got {group_size}")
    if max_new_tokens < 0:
        raise ValueError(f"max_new_tokens must be >= 0, got {max_new_tokens}")
    if not prompt_ids:
        raise ValueError("prompt_ids must be non-empty")

    device = next(model.parameters()).device
    prompt_len = len(prompt_ids)

    seq = (
        torch.tensor(prompt_ids, dtype=torch.long, device=device)
        .unsqueeze(0)
        .repeat(group_size, 1)
    )
    completion_mask = torch.zeros(group_size, prompt_len, dtype=torch.long, device=device)
    finished = torch.zeros(group_size, dtype=torch.bool, device=device)

    was_training = model.training
    model.eval()
    try:
        for _ in range(max_new_tokens):
            if bool(finished.all().item()):
                break
            logits = model(seq)["logits"][:, -1, :]
            next_tokens = sample_next_token(logits, temperature=temperature, top_p=top_p)

            # Finished rows emit pad and stop contributing.
            next_tokens = torch.where(
                finished,
                torch.full_like(next_tokens, pad_token_id),
                next_tokens,
            )

            # Append and mark mask=1 *before* updating `finished`, so the EOS
            # token itself counts as a completion token but nothing after it.
            seq = torch.cat([seq, next_tokens.unsqueeze(-1)], dim=-1)
            new_mask_col = (~finished).long().unsqueeze(-1)
            completion_mask = torch.cat([completion_mask, new_mask_col], dim=-1)

            finished = finished | (next_tokens == eos_token_id)
    finally:
        if was_training:
            model.train()

    attention_mask = torch.zeros_like(seq)
    attention_mask[:, :prompt_len] = 1
    if seq.shape[1] > prompt_len:
        attention_mask[:, prompt_len:] = completion_mask[:, prompt_len:]

    return seq, attention_mask, completion_mask


@torch.no_grad()
def sample_completions(
    model: torch.nn.Module,
    prompt_token_ids: list[list[int]],
    *,
    eos_token_id: int,
    pad_token_id: int,
    group_size: int = 8,
    max_new_tokens: int = 512,
    temperature: float = 1.0,
    top_p: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample G completions per prompt and pad everything to a common length.

    Returns (input_ids, attention_mask, completion_mask, prompt_indices,
    prompt_lens); the first three are [B*G, T_max], the rest [B*G].
    """
    if not prompt_token_ids:
        empty2d = torch.zeros(0, 0, dtype=torch.long)
        empty1d = torch.zeros(0, dtype=torch.long)
        return empty2d, empty2d, empty2d, empty1d, empty1d

    seqs: list[torch.Tensor] = []
    attns: list[torch.Tensor] = []
    masks: list[torch.Tensor] = []
    prompt_indices: list[int] = []
    prompt_lens: list[int] = []

    for prompt_idx, prompt_ids in enumerate(prompt_token_ids):
        seq, attn, mask = sample_one_prompt(
            model,
            prompt_ids,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            group_size=group_size,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        seqs.append(seq)
        attns.append(attn)
        masks.append(mask)
        prompt_indices.extend([prompt_idx] * group_size)
        prompt_lens.extend([len(prompt_ids)] * group_size)

    T_max = max(s.shape[1] for s in seqs)
    padded_seqs, padded_attns, padded_masks = [], [], []
    for seq, attn, mask in zip(seqs, attns, masks):
        pad_len = T_max - seq.shape[1]
        if pad_len > 0:
            seq = F.pad(seq, (0, pad_len), value=pad_token_id)
            attn = F.pad(attn, (0, pad_len), value=0)
            mask = F.pad(mask, (0, pad_len), value=0)
        padded_seqs.append(seq)
        padded_attns.append(attn)
        padded_masks.append(mask)

    return (
        torch.cat(padded_seqs, dim=0),
        torch.cat(padded_attns, dim=0),
        torch.cat(padded_masks, dim=0),
        torch.tensor(prompt_indices, dtype=torch.long),
        torch.tensor(prompt_lens, dtype=torch.long),
    )


def group_advantages(
    rewards: torch.Tensor,
    group_size: int,
    scale_rewards: bool = False,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Per-group advantages from per-rollout rewards.

    Rewards must be laid out as consecutive blocks of `group_size` (matching
    sample_completions). By default we just subtract the group mean (Dr.GRPO);
    set scale_rewards=True to also divide by the group std (vanilla GRPO),
    which introduces a difficulty bias toward groups that nearly agree.
    """
    if rewards.ndim != 1:
        raise ValueError(f"rewards must be 1-D, got shape {tuple(rewards.shape)}")
    if rewards.numel() == 0:
        return rewards.clone()
    if rewards.numel() % group_size != 0:
        raise ValueError(
            f"rewards length ({rewards.numel()}) not divisible by group_size ({group_size})"
        )

    B = rewards.numel() // group_size
    grouped = rewards.float().view(B, group_size)
    centered = grouped - grouped.mean(dim=1, keepdim=True)
    if scale_rewards:
        # eps keeps an all-equal group (std == 0) at advantage 0 rather than NaN.
        std = grouped.std(dim=1, keepdim=True, unbiased=False).clamp_min(eps)
        centered = centered / std
    return centered.reshape(-1)


def rollout(
    model: torch.nn.Module,
    tokenizer: TokenizerLike,
    prompts: list[str],
    golds: list[str],
    *,
    group_size: int = 8,
    max_new_tokens: int = 512,
    temperature: float = 1.0,
    top_p: float = 1.0,
    step: int = 0,
    reward_fn: Callable[..., float] | None = None,
    scale_rewards: bool = False,
) -> RolloutBatch:
    """Sample, decode, grade, and group-normalize rollouts in one call.

    `tokenizer` needs encode/decode plus eos_token_id/pad_token_id. `golds`
    runs parallel to `prompts`. `reward_fn(completion, gold, step=...)`
    defaults to src.rl.rewards.reward.
    """
    if len(prompts) != len(golds):
        raise ValueError(
            f"rollout: len(prompts)={len(prompts)} != len(golds)={len(golds)}"
        )

    if reward_fn is None:
        from .rewards import reward as default_reward
        reward_fn = default_reward

    if len(prompts) == 0:
        empty2d = torch.zeros(0, 0, dtype=torch.long)
        empty1d = torch.zeros(0, dtype=torch.long)
        empty1f = torch.zeros(0, dtype=torch.float)
        return RolloutBatch(
            input_ids=empty2d,
            attention_mask=empty2d,
            completion_mask=empty2d,
            rewards=empty1f,
            advantages=empty1f,
            prompt_indices=empty1d,
            prompt_lens=empty1d,
            completion_texts=[],
        )

    prompt_token_ids = [tokenizer.encode(p) for p in prompts]
    if any(len(p) == 0 for p in prompt_token_ids):
        raise ValueError("rollout: every prompt must encode to >= 1 token")

    input_ids, attention_mask, completion_mask, prompt_indices, prompt_lens = (
        sample_completions(
            model,
            prompt_token_ids,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            group_size=group_size,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )
    )

    completion_texts: list[str] = []
    rewards_list: list[float] = []
    N = input_ids.shape[0]
    for i in range(N):
        p_idx = int(prompt_indices[i].item())
        p_len = int(prompt_lens[i].item())
        comp_ids = input_ids[i, p_len:]
        comp_mask = completion_mask[i, p_len:]
        valid = comp_ids[comp_mask.bool()].tolist()
        # Drop the trailing EOS before decoding (it stays in input_ids /
        # completion_mask for the loss).
        if valid and valid[-1] == tokenizer.eos_token_id:
            valid = valid[:-1]
        text = tokenizer.decode(valid)
        completion_texts.append(text)
        rewards_list.append(float(reward_fn(text, golds[p_idx], step=step)))

    rewards = torch.tensor(rewards_list, dtype=torch.float)
    advantages = group_advantages(rewards, group_size, scale_rewards=scale_rewards)

    return RolloutBatch(
        input_ids=input_ids,
        attention_mask=attention_mask,
        completion_mask=completion_mask,
        rewards=rewards,
        advantages=advantages,
        prompt_indices=prompt_indices,
        prompt_lens=prompt_lens,
        completion_texts=completion_texts,
    )
