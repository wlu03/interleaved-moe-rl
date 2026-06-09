"""Supervised fine-tuning warm-start.

The `sft_then_rl` arm of the ablation needs a model that already produces
well-formed `\\boxed{}` solutions before GRPO starts. This module does that:
plain next-token cross-entropy on GSM8K + MATH reference solutions, with the
prompt tokens masked out of the loss.

The output checkpoint is loaded as the GRPO init -- so SFT writes the same
{step, model, optimizer} format train.save_checkpoint does, but the RL stage
only restores the model weights (fresh optimizer for the new objective).
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable, Optional, Sequence

import torch

from ..config import InterleavedMoEConfig
from ..model import InterleavedMoEModel
from .train import (
    _load_real_tokenizer,
    build_model,
    load_config as load_rl_config,
    save_checkpoint,
)


@dataclass
class SFTConfig:
    name: str = "sft_default"
    model: InterleavedMoEConfig = field(default_factory=InterleavedMoEConfig)

    lr: float = 1e-4
    weight_decay: float = 0.0
    warmup_steps: int = 0

    max_steps: int = 500
    batch_size: int = 8
    max_len: int = 1024
    grad_clip: float = 1.0
    seed: int = 0

    log_every: int = 10
    checkpoint_every: int = 200


def sft_config_for(rl_config_name: str, **overrides) -> SFTConfig:
    """Derive an SFT config from an RL config so the two stages share an arch.

    Matching the model architecture is what makes the SFT checkpoint loadable
    as the RL init. `overrides` tweak the SFT hyperparameters.
    """
    rl_cfg = load_rl_config(rl_config_name)
    base = dict(
        name=f"{rl_config_name}_sft",
        model=rl_cfg.model,
        max_len=rl_cfg.grpo.max_completion_len * 2,
        seed=rl_cfg.seed,
    )
    base.update(overrides)
    return SFTConfig(**base)


def _synthetic_sft_provider(vocab_size: int):
    """Toy (input_ids, labels) batches so the loop runs without `datasets`."""
    from .data import SFTExample, collate_sft_batch, format_prompt

    examples = [
        SFTExample(format_prompt(f"add {i}"), f"it is \\boxed{{{i + 2}}}", "toy")
        for i in range(64)
    ]

    def next_batch(tokenizer, batch_size: int, step: int, max_len: int):
        start = (step * batch_size) % len(examples)
        chunk = [examples[(start + j) % len(examples)] for j in range(batch_size)]
        return collate_sft_batch(chunk, tokenizer, max_len=max_len)

    return next_batch


def _lr_at(step: int, cfg: SFTConfig) -> float:
    """Linear warmup then constant. Returns a multiplier-applied lr."""
    if cfg.warmup_steps > 0 and step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / cfg.warmup_steps
    return cfg.lr


def train_sft(
    cfg: SFTConfig,
    ckpt_dir: str | Path = "checkpoints",
    *,
    tokenizer=None,
    next_batch: Optional[Callable[..., tuple[torch.Tensor, torch.Tensor]]] = None,
    model: Optional[InterleavedMoEModel] = None,
    device: Optional[str] = None,
    max_steps: Optional[int] = None,
) -> dict:
    """Run SFT and write a checkpoint at ckpt_dir/<cfg.name>/latest.pt.

    `next_batch(tokenizer, batch_size, step, max_len) -> (input_ids, labels)`
    is injectable; defaults to the synthetic provider so this runs locally.
    """
    if max_steps is not None:
        cfg = replace(cfg, max_steps=max_steps)

    torch.manual_seed(cfg.seed)
    dev = torch.device(device) if device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    if model is None:
        model = build_model(cfg.model)
    model = model.to(dev)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )

    if tokenizer is None:
        tokenizer = _load_real_tokenizer(cfg.model.vocab_size)
    if next_batch is None:
        next_batch = _synthetic_sft_provider(cfg.model.vocab_size)

    out_dir = Path(ckpt_dir) / cfg.name
    latest = out_dir / "latest.pt"

    model.train()
    last_loss = 0.0
    for step in range(cfg.max_steps):
        input_ids, labels = next_batch(tokenizer, cfg.batch_size, step, cfg.max_len)
        input_ids = input_ids.to(dev)
        labels = labels.to(dev)

        out = model(input_ids, labels=labels)
        loss = out["loss"]

        lr = _lr_at(step, cfg)
        for group in optimizer.param_groups:
            group["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if cfg.grad_clip and cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()

        last_loss = float(loss.detach())
        if step % cfg.log_every == 0:
            print(f"[sft] step={step} loss={last_loss:.4f} lr={lr:.2e}")
        if step % cfg.checkpoint_every == 0 and step > 0:
            save_checkpoint(model, optimizer, step, latest)

    save_checkpoint(model, optimizer, cfg.max_steps, latest)
    return {
        "config": cfg.name,
        "final_step": cfg.max_steps,
        "last_loss": last_loss,
        "ckpt": str(latest),
    }
