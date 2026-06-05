"""MoE routing statistics.

Each MoE layer's forward pass returns a stats dict with:
    router_logits  [N, E]
    router_entropy scalar (mean per-token entropy)
    expert_load    [E]      mean softmax probability per expert
    topk_indices   [N, K]
    topk_weights   [N, K]

This module aggregates those into research-grade metrics:
    - per-expert load (Gini coefficient for imbalance detection)
    - top-1/top-2 gap (routing sharpness)
    - token churn between consecutive forward passes (RL drift signal)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass
class RoutingStats:
    """Aggregated metrics for a single MoE layer at a single step."""

    layer_idx: int
    num_experts: int
    num_experts_per_tok: int

    entropy: float                  # mean per-token routing entropy (nats)
    expert_load: torch.Tensor       # [E] fraction of routing prob mass per expert
    expert_load_max: float          # max load (collapse signal)
    expert_load_gini: float         # Gini coefficient of expert load (imbalance)
    top1_top2_gap: float            # mean (p_top1 - p_top2), routing sharpness
    token_churn: float | None = None  # fraction of tokens whose top-1 expert
                                      # changed since the last call (None on
                                      # first call)

    extra: dict = field(default_factory=dict)


def gini_coefficient(x: torch.Tensor, eps: float = 1e-12) -> float:
    """
    Gini coefficient over a non-negative 1-D tensor.

    Returns 0 for perfect equality, approaches 1 for maximal inequality.

    Reference: standard formula G = (2 * Σᵢ i*xᵢ) / (n * Σ xᵢ) - (n+1)/n
    after sorting x in ascending order.
    """
    if x.dim() != 1:
        raise ValueError(f"gini expects 1-D tensor, got shape {tuple(x.shape)}")
    x = x.float().abs()
    n = x.numel()
    if n == 0:
        return 0.0
    sorted_x, _ = torch.sort(x)
    indices = torch.arange(1, n + 1, dtype=sorted_x.dtype, device=sorted_x.device)
    total = sorted_x.sum().clamp_min(eps)
    g = (2.0 * (indices * sorted_x).sum() / (n * total)) - (n + 1.0) / n
    return float(g.clamp(min=0.0).item())


def compute_routing_metrics(
    layer_idx: int,
    raw_stats: dict,
    previous_topk_indices: torch.Tensor | None = None,
) -> RoutingStats:
    """
    Aggregate raw per-layer routing stats from MoELayer.forward into RoutingStats.

    Args:
        layer_idx: which transformer layer this is
        raw_stats: dict returned by MoELayer.forward (must contain
                   'router_entropy', 'expert_load', 'topk_indices',
                   'topk_weights', 'router_logits')
        previous_topk_indices: [N, K] from the prior forward, or None on first call.
                               Used to compute token churn.
    """
    expert_load = raw_stats["expert_load"].float()  # [E]
    topk_indices = raw_stats["topk_indices"]  # [N, K]
    topk_weights = raw_stats["topk_weights"]  # [N, K]

    num_experts = expert_load.numel()
    K = topk_indices.shape[-1]

    # Top-1 / top-2 gap. K>=2 by construction in this codebase but guard anyway.
    if K >= 2:
        top1_top2_gap = float((topk_weights[:, 0] - topk_weights[:, 1]).mean().item())
    else:
        top1_top2_gap = float(topk_weights[:, 0].mean().item())

    # Token churn: fraction of tokens whose top-1 expert changed since last call.
    churn: float | None = None
    if previous_topk_indices is not None:
        prev_top1 = previous_topk_indices[:, 0]
        cur_top1 = topk_indices[:, 0]
        # If batch shapes differ (different number of tokens), churn is undefined.
        if prev_top1.shape == cur_top1.shape:
            changed = (prev_top1 != cur_top1).float().mean()
            churn = float(changed.item())

    return RoutingStats(
        layer_idx=layer_idx,
        num_experts=num_experts,
        num_experts_per_tok=K,
        entropy=float(raw_stats["router_entropy"].item()),
        expert_load=expert_load,
        expert_load_max=float(expert_load.max().item()),
        expert_load_gini=gini_coefficient(expert_load),
        top1_top2_gap=top1_top2_gap,
        token_churn=churn,
    )


class RoutingTracker:
    """
    Stateful tracker that maintains previous-step topk_indices per MoE layer
    so token churn can be computed without the caller threading state.

    Use one tracker instance per training run.
    """

    def __init__(self) -> None:
        self._previous: dict[int, torch.Tensor] = {}

    def update(self, routing_stats_by_layer: dict[int, dict]) -> dict[int, RoutingStats]:
        """
        Args:
            routing_stats_by_layer: {layer_idx: raw_stats_dict} as returned by
                                    InterleavedMoEModel.forward().routing_stats
        Returns:
            {layer_idx: RoutingStats} with churn populated when previous data exists
        """
        out: dict[int, RoutingStats] = {}
        for idx, raw in routing_stats_by_layer.items():
            prev = self._previous.get(idx)
            metrics = compute_routing_metrics(idx, raw, previous_topk_indices=prev)
            out[idx] = metrics
            # Detach + clone to avoid keeping the autograd graph alive
            self._previous[idx] = raw["topk_indices"].detach().clone()
        return out

    def reset(self) -> None:
        self._previous.clear()


def collapse_signal(stats: RoutingStats, threshold: float = 0.7) -> bool:
    """Returns True if routing has likely collapsed to a single dominant expert."""
    return stats.expert_load_max > threshold
