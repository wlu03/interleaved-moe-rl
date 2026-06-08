"""Rollout sampler for GRPO.

Produces a RolloutBatch ready for the GRPO loss: G completions per prompt,
each with the right masks, group-normalized advantages, and decoded text
for diagnostic logging.

Design choices (locked in NEXT_STEPS.md / IMPLEMENTATION_LOG.md):
    - Native HF-style autoregressive sampling. No vLLM at this stage.
    - Per-prompt batching: replicate one prompt G times and sample G
      completions in one batched forward loop. Avoids left-padding RoPE
      issues since all rollouts within one prompt share the prompt length.
    - Dr.GRPO advantage by default (subtract group mean, no std division).
    - Greedy when temperature == 0; top-p nucleus otherwise.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Protocol

import torch
import torch.nn.functional as F


# ---- Tokenizer protocol so tests don't need a real HF tokenizer --------
class TokenizerLike(Protocol):
    eos_token_id: int
    pad_token_id: int
    def encode(self, text: str) -> list[int]: ...
    def decode(self, ids) -> str: ...


# ---- Outputs -----------------------------------------------------------
@dataclass
class RolloutBatch:
    """A batch of G * B rollouts ready to feed into GRPO.

    Tensor shapes — N = B * G (number of rollouts in the batch):
        input_ids        [N, T_max]   prompt tokens + generated tokens, right-padded
        attention_mask   [N, T_max]   1 on prompt + active completion tokens
        completion_mask  [N, T_max]   1 only on generated tokens (incl. EOS)
        rewards          [N]
        advantages       [N]          group-mean-centered (Dr.GRPO style)
        prompt_indices   [N]          which prompt each rollout came from (0..B-1)
        prompt_lens      [N]          length of the original prompt for that row
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


# ---- Sampling primitives ----------------------------------------------
def sample_next_token(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_p: float = 1.0,
) -> torch.Tensor:
    """
    Sample one token per row from logits.

    Args:
        logits: [B, V]
        temperature: 0 → greedy argmax. >0 → temperature-scaled softmax.
        top_p: top-p (nucleus) cutoff. 1.0 disables filtering.

    Returns:
        [B] sampled token ids
    """
    if temperature <= 0:
        return logits.argmax(dim=-1)

    logits = logits.float() / temperature

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        sorted_probs = torch.softmax(sorted_logits, dim=-1)
        cumulative = sorted_probs.cumsum(dim=-1)
        # Keep tokens where cumulative <= top_p, but always include the top-1
        keep = cumulative <= top_p
        keep[..., 0] = True
        sorted_logits = sorted_logits.masked_fill(~keep, float("-inf"))
        # Restore original token ordering
        logits_filtered = torch.full_like(logits, float("-inf"))
        logits_filtered.scatter_(-1, sorted_indices, sorted_logits)
        logits = logits_filtered

    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)


# ---- Per-prompt sampling loop -----------------------------------------
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
    """
    Generate `group_size` completions from a single prompt.

    Returns:
        seq:             [G, prompt_len + n_generated]  full sequences
        attention_mask:  [G, ...]  1 on prompt + active completion tokens
        completion_mask: [G, ...]  1 only on generated tokens (incl. EOS)

    Note: short sequences (those that hit EOS early) are right-padded with
    `pad_token_id` so all G rows share the same length.
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
    )  # [G, L]
    completion_mask = torch.zeros(group_size, prompt_len, dtype=torch.long, device=device)
    finished = torch.zeros(group_size, dtype=torch.bool, device=device)

    was_training = model.training
    model.eval()
    try:
        for _ in range(max_new_tokens):
            if bool(finished.all().item()):
                break
            out = model(seq)
            logits = out["logits"][:, -1, :]  # [G, V]
            next_tokens = sample_next_token(logits, temperature=temperature, top_p=top_p)

            # Where this row already finished, replace with pad (don't generate further)
            next_tokens = torch.where(
                finished,
                torch.full_like(next_tokens, pad_token_id),
                next_tokens,
            )

            # Append before updating `finished` so the EOS itself is included with mask=1
            seq = torch.cat([seq, next_tokens.unsqueeze(-1)], dim=-1)
            new_mask_col = (~finished).long().unsqueeze(-1)
            completion_mask = torch.cat([completion_mask, new_mask_col], dim=-1)

            # An EOS token marks the end of *this* row's completion
            finished = finished | (next_tokens == eos_token_id)
    finally:
        if was_training:
            model.train()

    # attention_mask = prompt portion (always 1) ∪ valid completion tokens
    attention_mask = torch.zeros_like(seq)
    attention_mask[:, :prompt_len] = 1
    if seq.shape[1] > prompt_len:
        attention_mask[:, prompt_len:] = completion_mask[:, prompt_len:]

    return seq, attention_mask, completion_mask


# ---- Multi-prompt aggregation -----------------------------------------
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
    """
    Sample G completions for each prompt and aggregate into uniform tensors.

    Returns:
        input_ids        [B*G, T_max]
        attention_mask   [B*G, T_max]
        completion_mask  [B*G, T_max]
        prompt_indices   [B*G]
        prompt_lens      [B*G]
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


# ---- Advantage normalization (Dr.GRPO default) ------------------------
def group_advantages(
    rewards: torch.Tensor,
    group_size: int,
    scale_rewards: bool = False,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Compute per-group advantages from per-rollout rewards.

    Assumes rewards are arranged so consecutive blocks of `group_size`
    rollouts come from the same prompt — this matches `sample_completions`'s
    output ordering.

    Args:
        rewards: [B*G] scalar rewards
        group_size: G — group size used during rollout
        scale_rewards: if True, divide by per-group std (vanilla GRPO).
                       If False (default, Dr.GRPO), only subtract mean.
                       Avoids difficulty bias from std normalization.
        eps: numerical guard for std

    Returns:
        [B*G] advantages
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
        std = grouped.std(dim=1, keepdim=True, unbiased=False).clamp_min(eps)
        # When all rewards in a group are equal, std is ~0 → advantage stays 0
        # (eps in denom ⇒ 0/eps = 0). The clamp is for numerical safety only.
        centered = centered / std
    return centered.reshape(-1)


# ---- Top-level rollout API --------------------------------------------
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
    """
    Sample, decode, grade, and group-normalize rollouts in one shot.

    Args:
        model: an InterleavedMoEModel (or compatible).
        tokenizer: must expose .encode(str)→list[int], .decode(ids)→str,
                   .eos_token_id, .pad_token_id
        prompts: B prompts to sample G completions from each.
        golds: B ground-truth answer strings (parallel to `prompts`).
        group_size: rollouts per prompt.
        max_new_tokens: hard cap on generated tokens.
        temperature, top_p: sampling controls.
        step: passed to reward_fn (governs format-bonus phase).
        reward_fn: signature reward(completion, gold, step=...) → float.
                   Defaults to src.rl.rewards.reward.
        scale_rewards: if True, std-normalize advantages within each group
                       (vanilla GRPO). Default False = Dr.GRPO.
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
        raise ValueError("rollout: every prompt must encode to ≥1 token")

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

    # Decode each row's completion portion and grade.
    completion_texts: list[str] = []
    rewards_list: list[float] = []
    N = input_ids.shape[0]
    for i in range(N):
        p_idx = int(prompt_indices[i].item())
        p_len = int(prompt_lens[i].item())
        # Completion tokens live in [p_len, end], filtered by completion_mask
        comp_slice_ids = input_ids[i, p_len:]
        comp_slice_mask = completion_mask[i, p_len:]
        valid = comp_slice_ids[comp_slice_mask.bool()].tolist()
        # Strip the EOS token from the decoded text (still kept in input_ids
        # / completion_mask for loss computation)
        if valid and valid[-1] == tokenizer.eos_token_id:
            valid_for_decode = valid[:-1]
        else:
            valid_for_decode = valid
        text = tokenizer.decode(valid_for_decode)
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
