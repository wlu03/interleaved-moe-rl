"""GRPO RL components for interleaved MoE experiments."""

from .rewards import (
    extract_gsm8k_answer,
    extract_boxed_answer,
    answers_equivalent,
    reward,
    batch_reward,
    RewardBreakdown,
)
from .rollout import (
    RolloutBatch,
    sample_next_token,
    sample_one_prompt,
    sample_completions,
    group_advantages,
    rollout,
)
from .grpo import (
    GRPOConfig,
    selective_log_softmax,
    compute_logprobs,
    shift_completion_mask,
    grpo_loss,
    grpo_train_step,
)
from .train import (
    TrainConfig,
    load_config,
    register_config,
    build_model,
    layer_kinds,
    save_checkpoint,
    load_checkpoint,
    main,
    smoke_train_loop,
)

__all__ = [
    "extract_gsm8k_answer",
    "extract_boxed_answer",
    "answers_equivalent",
    "reward",
    "batch_reward",
    "RewardBreakdown",
    "RolloutBatch",
    "sample_next_token",
    "sample_one_prompt",
    "sample_completions",
    "group_advantages",
    "rollout",
    "GRPOConfig",
    "selective_log_softmax",
    "compute_logprobs",
    "shift_completion_mask",
    "grpo_loss",
    "grpo_train_step",
    "TrainConfig",
    "load_config",
    "register_config",
    "build_model",
    "layer_kinds",
    "save_checkpoint",
    "load_checkpoint",
    "main",
    "smoke_train_loop",
]
