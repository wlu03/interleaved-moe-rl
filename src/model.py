"""
Interleaved MoE Transformer — a minimal, correct implementation for research.

Architecture mirrors MAI-Thinking-1 / Llama 4 style:
- Alternating dense FFN and MoE layers
- GQA with RoPE
- RMSNorm pre/post residual (MAI style)
- Efficient MoE dispatch via scatter/gather (no Python loops over experts)
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import InterleavedMoEConfig


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x_float = x.float()
        norm = x_float.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x_float * norm * self.weight.float()).to(dtype)


def precompute_rope_freqs(dim: int, max_seq_len: int, theta: float = 10000.0) -> torch.Tensor:
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(max_seq_len).float()
    freqs = torch.outer(t, freqs)  # [seq_len, dim//2]
    return torch.polar(torch.ones_like(freqs), freqs)  # complex64


def apply_rope(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    """Apply rotary embeddings. x: [B, H, S, D] where D is head_dim."""
    B, H, S, D = x.shape
    x_complex = torch.view_as_complex(x.float().reshape(B, H, S, D // 2, 2))
    freqs = freqs[:S].unsqueeze(0).unsqueeze(0)  # [1, 1, S, D//2]
    x_rotated = torch.view_as_real(x_complex * freqs).reshape(B, H, S, D)
    return x_rotated.to(x.dtype)


class Attention(nn.Module):
    def __init__(self, config: InterleavedMoEConfig):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.num_kv_groups = self.num_heads // self.num_kv_heads

        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=False)

    def forward(self, x: torch.Tensor, rope_freqs: torch.Tensor) -> torch.Tensor:
        B, S, _ = x.shape

        q = self.q_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2)

        q = apply_rope(q, rope_freqs)
        k = apply_rope(k, rope_freqs)

        # Expand KV for GQA
        if self.num_kv_groups > 1:
            k = k.repeat_interleave(self.num_kv_groups, dim=1)
            v = v.repeat_interleave(self.num_kv_groups, dim=1)

        # Efficient causal attention via PyTorch SDPA (uses FlashAttention-2 when available)
        attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        attn_out = attn_out.transpose(1, 2).contiguous().view(B, S, -1)
        return self.o_proj(attn_out)


class DenseFFN(nn.Module):
    """SwiGLU feed-forward network."""

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class MoERouter(nn.Module):
    def __init__(self, config: InterleavedMoEConfig):
        super().__init__()
        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        self.num_experts = config.num_experts
        self.num_experts_per_tok = config.num_experts_per_tok

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, dict]:
        """
        Args:
            x: [N, D] flattened token representations
        Returns:
            topk_weights: [N, K] normalized routing weights
            topk_indices: [N, K] expert indices
            stats: routing statistics dict
        """
        logits = self.gate(x.float())
        scores = F.softmax(logits, dim=-1)

        topk_weights, topk_indices = torch.topk(scores, self.num_experts_per_tok, dim=-1)
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

        stats = {
            "router_logits": logits.detach(),
            "router_entropy": -(scores * (scores + 1e-10).log()).sum(-1).mean().detach(),
            "expert_load": scores.detach().mean(0),       # [num_experts]
            "topk_indices": topk_indices.detach(),        # [N, K] for token-churn analysis
            "topk_weights": topk_weights.detach(),        # [N, K] for routing-sharpness analysis
        }
        return topk_weights, topk_indices, stats


class MoELayer(nn.Module):
    """
    Efficient MoE using grouped dispatch via scatter/gather.
    Avoids Python for-loops over experts by using batched linear ops.
    """

    def __init__(self, config: InterleavedMoEConfig):
        super().__init__()
        self.router = MoERouter(config)
        self.num_experts = config.num_experts
        self.num_experts_per_tok = config.num_experts_per_tok

        # All expert weights packed into single tensors for efficient dispatch
        D = config.hidden_size
        I = config.expert_intermediate_size
        self.gate_proj = nn.Parameter(torch.empty(config.num_experts, D, I))
        self.up_proj = nn.Parameter(torch.empty(config.num_experts, D, I))
        self.down_proj = nn.Parameter(torch.empty(config.num_experts, I, D))
        self._init_expert_weights()

    def _init_expert_weights(self):
        for p in [self.gate_proj, self.up_proj, self.down_proj]:
            nn.init.kaiming_uniform_(p, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict]:
        """
        Args:
            x: [B, S, D]
        Returns:
            output: [B, S, D]
            stats: routing statistics
        """
        B, S, D = x.shape
        x_flat = x.view(-1, D)  # [N, D] where N = B*S
        N = x_flat.shape[0]

        topk_weights, topk_indices, stats = self.router(x_flat)
        # topk_weights: [N, K], topk_indices: [N, K]
        K = self.num_experts_per_tok

        # Group tokens by expert and run each expert's shared [D, I] weights on
        # only the tokens routed to it. The earlier approach gathered a
        # per-(token, slot) copy of the weights (gate_proj[expert_indices] is
        # [N*K, D, I]); at real N that tensor is hundreds of GB. Looping over the
        # handful of experts keeps the footprint at O(N*I) activations instead.
        flat_expert = topk_indices.reshape(-1)                              # [N*K]
        flat_weight = topk_weights.reshape(-1)                              # [N*K]
        flat_token = torch.arange(N, device=x.device).repeat_interleave(K)  # [N*K]

        out = torch.zeros(N, D, dtype=x_flat.dtype, device=x.device)
        for e in range(self.num_experts):
            sel = torch.nonzero(flat_expert == e, as_tuple=True)[0]
            if sel.numel() == 0:
                continue
            tokens = flat_token[sel]
            xe = x_flat[tokens]                          # [m, D]
            gate_out = xe @ self.gate_proj[e]            # [m, I]
            up_out = xe @ self.up_proj[e]                # [m, I]
            expert_out = (F.silu(gate_out) * up_out) @ self.down_proj[e]  # [m, D]
            expert_out = expert_out * flat_weight[sel].unsqueeze(-1)
            out.index_add_(0, tokens, expert_out.to(out.dtype))

        return out.view(B, S, D), stats


class TransformerBlock(nn.Module):
    def __init__(self, config: InterleavedMoEConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        # moe_every_n_layers <= 0 means "all dense" (the dense_baseline config).
        self.is_moe = (
            config.moe_every_n_layers > 0
            and layer_idx % config.moe_every_n_layers == 0
        )

        self.attn_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.attn = Attention(config)
        self.post_attn_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)

        self.ffn_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_ffn_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)

        if self.is_moe:
            self.ffn = MoELayer(config)
        else:
            self.ffn = DenseFFN(config.hidden_size, config.intermediate_size)

        self.dropout = nn.Dropout(config.dropout) if config.dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor, rope_freqs: torch.Tensor) -> tuple[torch.Tensor, dict]:
        # Attention with pre-norm and post-norm before residual (MAI style)
        residual = x
        x_normed = self.attn_norm(x)
        attn_out = self.attn(x_normed, rope_freqs)
        attn_out = self.post_attn_norm(attn_out)
        x = residual + self.dropout(attn_out)

        # FFN with pre-norm and post-norm before residual
        residual = x
        x_normed = self.ffn_norm(x)

        stats = {}
        if self.is_moe:
            ffn_out, stats = self.ffn(x_normed)
        else:
            ffn_out = self.ffn(x_normed)

        ffn_out = self.post_ffn_norm(ffn_out)
        x = residual + self.dropout(ffn_out)

        return x, stats


class InterleavedMoEModel(nn.Module):
    """
    Full interleaved MoE language model.

    Returns logits and a dict of per-layer routing statistics for analysis.
    """

    def __init__(self, config: InterleavedMoEConfig):
        super().__init__()
        self.config = config

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([
            TransformerBlock(config, layer_idx=i) for i in range(config.num_layers)
        ])
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)

        if config.tie_word_embeddings:
            self.lm_head = None  # reuse embed_tokens.weight
        else:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Precompute RoPE frequencies (registered as buffer, not a parameter)
        head_dim = config.hidden_size // config.num_attention_heads
        rope_freqs = precompute_rope_freqs(head_dim, config.max_position_embeddings, config.rope_theta)
        self.register_buffer("rope_freqs", rope_freqs, persistent=False)

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        compute_logits: bool = True,
    ) -> dict:
        """
        Args:
            input_ids: [B, S] token ids
            labels: [B, S] optional targets for cross-entropy loss
            compute_logits: if False, skip the lm_head projection and return
                logits=None. The vocab projection is a [B, S, vocab_size]
                tensor (hundreds of GB at the real 4096-row drift probe), so
                callers that only need hidden states (drift) or routing stats
                pass False to avoid materializing it. Ignored when labels are
                given, since the loss needs logits.

        Returns:
            dict with keys: logits, loss (if labels), routing_stats
        """
        B, S = input_ids.shape
        x = self.embed_tokens(input_ids)

        routing_stats = {}
        for i, layer in enumerate(self.layers):
            x, stats = layer(x, self.rope_freqs)
            if stats:
                routing_stats[i] = stats

        x = self.norm(x)

        if not compute_logits and labels is None:
            return {"logits": None, "routing_stats": routing_stats}

        # Compute logits
        if self.lm_head is not None:
            logits = self.lm_head(x)
        else:
            logits = F.linear(x, self.embed_tokens.weight)

        result = {"logits": logits, "routing_stats": routing_stats}

        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )
            result["loss"] = loss

        return result

    def num_parameters(self, only_trainable: bool = True) -> int:
        return sum(p.numel() for p in self.parameters() if not only_trainable or p.requires_grad)

    def num_active_parameters(self) -> int:
        """Active params per forward pass (dense + top-k experts, not all experts)."""
        total = 0
        for name, p in self.named_parameters():
            if "gate_proj" in name or "up_proj" in name or "down_proj" in name:
                if p.dim() == 3:  # packed expert weights [E, ...]
                    # Only top-k are active
                    per_expert = p[0].numel()
                    total += per_expert * self.config.num_experts_per_tok
                else:
                    total += p.numel()
            else:
                total += p.numel()
        return total
