"""Per-layer and per-expert gradient norm tracking.

The core question of this project is whether dense FFN layers and MoE
layers absorb RL gradients differently. To answer that, we need
per-layer-kind aggregations:
    - dense FFN layers
    - MoE expert weights (the bulk of MoE params)
    - MoE router weights (small but critical)
    - Attention layers (Q/K/V/O)
    - Embeddings + output norm
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import torch
import torch.nn as nn


@dataclass
class GradientSummary:
    """Per-step gradient norm summary."""

    by_layer: dict[int, float] = field(default_factory=dict)         # {layer_idx: norm}
    by_layer_kind: dict[str, float] = field(default_factory=dict)    # {"dense_ffn", "moe_expert", "moe_router", ...}
    by_expert: dict[tuple[int, int], float] = field(default_factory=dict)  # {(layer_idx, expert_idx): norm}
    global_norm: float = 0.0


def _classify_param(name: str) -> tuple[int | None, str, int | None]:
    """
    Map a parameter name to (layer_idx, kind, expert_idx).

    Returns:
        layer_idx: int or None for params outside the transformer stack
        kind: one of {"dense_ffn", "moe_expert", "moe_router",
                      "attention", "norm", "embedding", "lm_head", "other"}
        expert_idx: int for moe_expert kind, else None
    """
    parts = name.split(".")

    # Embedding / output
    if parts[0] == "embed_tokens":
        return None, "embedding", None
    if parts[0] == "lm_head":
        return None, "lm_head", None
    if parts[0] == "norm":
        return None, "norm", None

    # Transformer block params: layers.<i>.<...>
    if parts[0] == "layers" and len(parts) >= 2:
        try:
            layer_idx = int(parts[1])
        except ValueError:
            return None, "other", None

        rest = ".".join(parts[2:])

        # Norms first — they contain "attn" or "ffn" substrings and would
        # otherwise be misclassified
        if "norm" in rest:
            return layer_idx, "norm", None
        # Attention sublayer
        if rest.startswith("attn.") or rest.startswith("attn") and "norm" not in rest:
            return layer_idx, "attention", None
        if any(rest.startswith(p + ".") for p in ("q_proj", "k_proj", "v_proj", "o_proj")):
            return layer_idx, "attention", None
        # MoE router (ffn.router.*)
        if rest.startswith("ffn.router"):
            return layer_idx, "moe_router", None
        # MoE expert weights — packed 3-D Parameters with no .weight suffix
        if rest in ("ffn.gate_proj", "ffn.up_proj", "ffn.down_proj"):
            return layer_idx, "moe_expert", None
        # Dense FFN — nn.Linear modules with .weight suffix
        if rest.startswith("ffn."):
            return layer_idx, "dense_ffn", None

    return None, "other", None


def per_expert_grad_norms(
    layer_idx: int,
    gate_proj: torch.Tensor,
    up_proj: torch.Tensor,
    down_proj: torch.Tensor,
) -> dict[tuple[int, int], float]:
    """
    Given the three packed 3-D expert tensors and their gradients,
    compute the per-expert gradient norm.

    Each tensor has shape [E, ...] where the leading dim is the expert index.
    """
    out: dict[tuple[int, int], float] = {}
    if gate_proj.grad is None or up_proj.grad is None or down_proj.grad is None:
        return out
    E = gate_proj.shape[0]
    for e in range(E):
        n_sq = (
            gate_proj.grad[e].float().pow(2).sum()
            + up_proj.grad[e].float().pow(2).sum()
            + down_proj.grad[e].float().pow(2).sum()
        )
        out[(layer_idx, e)] = float(n_sq.sqrt().item())
    return out


def summarize_gradients_by_kind(model: nn.Module) -> GradientSummary:
    """
    Aggregate current .grad attributes on `model` into a GradientSummary.

    Call this AFTER loss.backward() but BEFORE optim.step() (and BEFORE clip,
    if you want true raw norms).
    """
    by_layer_sq: dict[int, float] = defaultdict(float)
    by_kind_sq: dict[str, float] = defaultdict(float)
    global_sq = 0.0

    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        g_sq = param.grad.float().pow(2).sum().item()
        global_sq += g_sq

        layer_idx, kind, _ = _classify_param(name)
        by_kind_sq[kind] += g_sq
        if layer_idx is not None:
            by_layer_sq[layer_idx] += g_sq

    summary = GradientSummary(
        by_layer={i: v ** 0.5 for i, v in by_layer_sq.items()},
        by_layer_kind={k: v ** 0.5 for k, v in by_kind_sq.items()},
        global_norm=global_sq ** 0.5,
    )

    # Per-expert norms — only for MoE layers
    if hasattr(model, "layers"):
        for i, block in enumerate(model.layers):
            ffn = getattr(block, "ffn", None)
            is_moe = bool(getattr(block, "is_moe", False))
            if not is_moe or ffn is None:
                continue
            if not (
                hasattr(ffn, "gate_proj")
                and hasattr(ffn, "up_proj")
                and hasattr(ffn, "down_proj")
            ):
                continue
            summary.by_expert.update(
                per_expert_grad_norms(i, ffn.gate_proj, ffn.up_proj, ffn.down_proj)
            )

    return summary


class GradientTracker:
    """Persistent tracker that accumulates gradient summaries across steps."""

    def __init__(self) -> None:
        self.history: list[GradientSummary] = []

    def step(self, model: nn.Module) -> GradientSummary:
        summary = summarize_gradients_by_kind(model)
        self.history.append(summary)
        return summary

    def latest(self) -> GradientSummary | None:
        return self.history[-1] if self.history else None

    def reset(self) -> None:
        self.history.clear()
