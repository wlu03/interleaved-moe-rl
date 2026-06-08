"""GRPO loss and training step.

The loss never touches strings — it consumes a `RolloutBatch` (token ids,
masks, and precomputed group advantages) and produces a scalar to backprop.

Design choices (locked in NEXT_STEPS.md):
    - Dr.GRPO normalization: divide the summed per-token loss by a constant
      (num_rollouts * max_completion_len), NOT per-sequence length. This
      removes the length bias that per-sequence averaging introduces.
    - beta = 0: no KL term, no reference model. DAPO / Open-Reasoner-Zero /
      Dr.GRPO all drop it. Saves a forward pass and ~15% memory.
    - Clipped surrogate with DAPO clip-higher (eps_low < eps_high). With
      num_inner_epochs = 1 the ratio is identically 1 at the gradient step
      (old_logprobs == new_logprobs), so clipping is inert — but it is
      implemented faithfully so multi-epoch upgrades are a config change.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import torch
import torch.nn.functional as F

from .rollout import RolloutBatch, rollout as default_rollout


# ---- Config ------------------------------------------------------------
@dataclass
class GRPOConfig:
    """GRPO hyperparameters (Dr.GRPO defaults, beta=0)."""

    group_size: int = 8           # rollouts per prompt
    eps_low: float = 0.2          # PPO clip lower bound
    eps_high: float = 0.28        # DAPO clip-higher upper bound
    beta: float = 0.0             # KL coefficient; 0 = no reference model
    loss_type: str = "dr_grpo"    # "dr_grpo" | "grpo" (per-sequence mean)
    max_completion_len: int = 512  # normalizer constant for dr_grpo
    temperature: float = 1.0
    top_p: float = 1.0
    scale_rewards: bool = False   # std-normalize advantages within group
    num_inner_epochs: int = 1     # mu; 1 = pure on-policy, ratio == 1
    grad_clip: float = 1.0        # global grad-norm clip (0 disables)


# ---- Log-prob helpers --------------------------------------------------
def selective_log_softmax(logits: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """Per-token log-probability of the chosen `index`, without materializing
    the full [*, V] log-softmax for every row at once.

    Args:
        logits: [..., V]
        index:  [...]  token id to select at each position

    Returns:
        [...] log p(index) under softmax(logits)
    """
    logps = F.log_softmax(logits, dim=-1)
    return logps.gather(dim=-1, index=index.unsqueeze(-1)).squeeze(-1)


def compute_logprobs(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
) -> torch.Tensor:
    """Per-token log-probs aligned to the *target* tokens.

    For an autoregressive model, the log-prob of the token at position t is
    read from the logits at position t-1. We therefore return a tensor of
    shape [N, T-1] where column j holds log p(input_ids[:, j+1]).

    Args:
        model: returns dict with "logits" [N, T, V]
        input_ids: [N, T]

    Returns:
        [N, T-1] per-token log-probs
    """
    logits = model(input_ids)["logits"]          # [N, T, V]
    logits = logits[:, :-1, :]                    # predict tokens 1..T-1
    targets = input_ids[:, 1:]                    # [N, T-1]
    return selective_log_softmax(logits, targets)


def shift_completion_mask(completion_mask: torch.Tensor) -> torch.Tensor:
    """Align a completion mask [N, T] to the target positions [N, T-1].

    Column j of the shifted mask corresponds to predicting input_ids[:, j+1],
    so it takes its value from completion_mask[:, j+1].
    """
    return completion_mask[:, 1:].to(torch.float32)


# ---- Loss --------------------------------------------------------------
def grpo_loss(
    logprobs: torch.Tensor,
    old_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    completion_mask: torch.Tensor,
    cfg: GRPOConfig,
) -> tuple[torch.Tensor, dict]:
    """Clipped GRPO surrogate loss over completion tokens.

    Args:
        logprobs:     [N, T-1] current-policy per-token log-probs (grad)
        old_logprobs: [N, T-1] sampling-policy log-probs (no grad). With
                      num_inner_epochs=1 these equal `logprobs` numerically.
        advantages:   [N] per-rollout group advantage (broadcast over tokens)
        completion_mask: [N, T-1] 1 on generated-token target positions
        cfg: GRPOConfig

    Returns:
        (loss scalar, metrics dict)
    """
    mask = completion_mask.to(logprobs.dtype)
    adv = advantages.to(logprobs.dtype).unsqueeze(-1)  # [N, 1]

    ratio = torch.exp(logprobs - old_logprobs)         # [N, T-1]
    unclipped = ratio * adv
    clipped = torch.clamp(ratio, 1.0 - cfg.eps_low, 1.0 + cfg.eps_high) * adv
    per_token = -torch.min(unclipped, clipped)         # [N, T-1]

    per_token = per_token * mask

    if cfg.loss_type == "dr_grpo":
        # Constant normalizer → no length bias. N rollouts * max len.
        normalizer = logprobs.shape[0] * cfg.max_completion_len
        loss = per_token.sum() / normalizer
    elif cfg.loss_type == "grpo":
        # Per-sequence mean over its own completion tokens, then mean over rows.
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


# ---- Train step --------------------------------------------------------
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
    """One GRPO update: rollout → loss → backward → step.

    Args:
        model: policy (InterleavedMoEModel or compatible).
        tokenizer: exposes encode/decode/eos_token_id/pad_token_id.
        optimizer: already constructed over model.parameters().
        prompts, golds: parallel lists, B each.
        cfg: GRPOConfig.
        step: global step (governs the reward format-bonus phase).
        reward_fn: override the default math reward.
        rollout_fn: override the sampler (tests inject a fake batch).
        tracker: optional object with .log_routing(step, routing_stats).

    Returns:
        metrics dict including mean_reward and loss.
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

    completion_mask = shift_completion_mask(batch.completion_mask)  # [N, T-1]

    # Sampling-policy log-probs (no grad). num_inner_epochs=1 → equal to the
    # first inner step's logprobs, so the ratio is 1 and clipping is inert.
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
