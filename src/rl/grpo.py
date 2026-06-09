"""GRPO loss and training step.

The loss never sees strings: it works off a RolloutBatch (token ids, masks,
precomputed group advantages) and returns a scalar to backprop.

Defaults follow Dr.GRPO with no KL term (beta=0, so no reference model). The
loss is normalized by a constant (N * max_completion_len) rather than per
sequence, which avoids the length bias that per-sequence averaging causes.
Clipping uses DAPO's clip-higher (eps_low < eps_high). With num_inner_epochs=1
the ratio is exactly 1 at the gradient step, so the clip never bites -- but
it's implemented properly so going multi-epoch is just a config change.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import torch
import torch.nn.functional as F

from .rollout import RolloutBatch, rollout as default_rollout


@dataclass
class GRPOConfig:
    group_size: int = 8
    eps_low: float = 0.2
    eps_high: float = 0.28          # DAPO clip-higher
    beta: float = 0.0               # KL coef; 0 => no reference model
    loss_type: str = "dr_grpo"      # "dr_grpo" | "grpo"
    max_completion_len: int = 512   # normalizer for dr_grpo
    temperature: float = 1.0
    top_p: float = 1.0
    scale_rewards: bool = False
    num_inner_epochs: int = 1       # mu; 1 => pure on-policy, ratio == 1
    grad_clip: float = 1.0          # 0 disables


def selective_log_softmax(logits: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """log p(index) under softmax(logits), gathered per position.

    logits is [..., V], index is [...]; returns [...].
    """
    logps = F.log_softmax(logits, dim=-1)
    return logps.gather(dim=-1, index=index.unsqueeze(-1)).squeeze(-1)


def compute_logprobs(model: torch.nn.Module, input_ids: torch.Tensor) -> torch.Tensor:
    """Per-token log-probs aligned to the target tokens.

    The model predicts token t from the logits at position t-1, so we drop the
    last logit and the first input id. Result is [N, T-1] where column j holds
    log p(input_ids[:, j+1]).
    """
    logits = model(input_ids)["logits"][:, :-1, :]
    targets = input_ids[:, 1:]
    return selective_log_softmax(logits, targets)


def shift_completion_mask(completion_mask: torch.Tensor) -> torch.Tensor:
    """Align a [N, T] completion mask to the [N, T-1] target positions."""
    return completion_mask[:, 1:].to(torch.float32)


def grpo_loss(
    logprobs: torch.Tensor,
    old_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    completion_mask: torch.Tensor,
    cfg: GRPOConfig,
) -> tuple[torch.Tensor, dict]:
    """Clipped GRPO surrogate over completion tokens.

    logprobs/old_logprobs are [N, T-1]; advantages [N] broadcast over tokens;
    completion_mask [N, T-1]. Returns (loss, metrics).
    """
    mask = completion_mask.to(logprobs.dtype)
    adv = advantages.to(logprobs.dtype).unsqueeze(-1)

    ratio = torch.exp(logprobs - old_logprobs)
    unclipped = ratio * adv
    clipped = torch.clamp(ratio, 1.0 - cfg.eps_low, 1.0 + cfg.eps_high) * adv
    per_token = -torch.min(unclipped, clipped) * mask

    if cfg.loss_type == "dr_grpo":
        # Constant normalizer -> no length bias.
        normalizer = logprobs.shape[0] * cfg.max_completion_len
        loss = per_token.sum() / normalizer
    elif cfg.loss_type == "grpo":
        # Per-sequence mean over its own tokens, then mean across rows.
        tok_per_row = mask.sum(dim=-1).clamp_min(1.0)
        loss = (per_token.sum(dim=-1) / tok_per_row).mean()
    else:
        raise ValueError(f"unknown loss_type: {cfg.loss_type!r}")

    with torch.no_grad():
        n_tokens = mask.sum().clamp_min(1.0)
        clip_frac = (
            ((unclipped > clipped) & (mask > 0)).to(logprobs.dtype).sum() / n_tokens
        )
        metrics = {
            "loss": float(loss.detach()),
            "clip_frac": float(clip_frac),
            "mean_ratio": float((ratio * mask).sum() / n_tokens),
            "completion_tokens": float(mask.sum()),
        }
    return loss, metrics


def grpo_train_step(
    model: torch.nn.Module,
    tokenizer,
    optimizer: torch.optim.Optimizer,
    prompts: list[str],
    golds: list[str],
    cfg: GRPOConfig,
    *,
    step: int = 0,
    reward_fn: Optional[Callable[..., float]] = None,
    rollout_fn: Optional[Callable[..., RolloutBatch]] = None,
    tracker=None,
) -> dict:
    """One GRPO update: rollout -> loss -> backward -> step.

    `rollout_fn` and `reward_fn` are injectable so tests can hand in a fixed
    batch. Returns a metrics dict including mean_reward and loss.
    """
    if rollout_fn is None:
        rollout_fn = default_rollout

    batch: RolloutBatch = rollout_fn(
        model,
        tokenizer,
        prompts,
        golds,
        group_size=cfg.group_size,
        max_new_tokens=cfg.max_completion_len,
        temperature=cfg.temperature,
        top_p=cfg.top_p,
        step=step,
        reward_fn=reward_fn,
        scale_rewards=cfg.scale_rewards,
    )

    if batch.num_rollouts == 0:
        return {"loss": 0.0, "mean_reward": 0.0, "completion_tokens": 0.0}

    device = next(model.parameters()).device
    batch = batch.to(device)

    completion_mask = shift_completion_mask(batch.completion_mask)

    # Sampling-policy log-probs. With num_inner_epochs=1 these equal the first
    # inner step's logprobs, so ratio == 1 and the clip is inert.
    with torch.no_grad():
        old_logprobs = compute_logprobs(model, batch.input_ids)

    was_training = model.training
    model.train()
    last_metrics: dict = {}
    try:
        for _ in range(cfg.num_inner_epochs):
            logprobs = compute_logprobs(model, batch.input_ids)
            loss, metrics = grpo_loss(
                logprobs, old_logprobs, batch.advantages, completion_mask, cfg
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if cfg.grad_clip and cfg.grad_clip > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), cfg.grad_clip
                )
                metrics["grad_norm"] = float(grad_norm)
            optimizer.step()
            last_metrics = metrics
    finally:
        if not was_training:
            model.eval()

    last_metrics["mean_reward"] = float(batch.rewards.mean())
    last_metrics["reward_std"] = float(batch.rewards.std(unbiased=False))
    return last_metrics
