"""Representation drift metrics: linear CKA + orthogonal Procrustes.

Both metrics compare two activation matrices X, Y of shape [N, D] (with N
examples and D features per layer) and return a scalar.

Linear CKA  ∈ [0, 1]   1 = identical (up to isotropic scale + bias)
Procrustes  ∈ [0, √2]   0 = identical, larger = more drift

Best practice (per Kornblith 2019, Williams 2021, Ding 2021):
    - Always compute in float64. Float32 catastrophic cancellation on RL'd
      checkpoints is a known failure mode.
    - CKA is dominated by top eigenvectors. Always report Procrustes alongside
      to catch tail-direction changes (Ding 2021).
    - Probe set size: n >= 2 * D for reasonable bootstrap CIs.
"""

from __future__ import annotations

import torch


def linear_cka(X: torch.Tensor, Y: torch.Tensor, eps: float = 1e-30) -> float:
    """
    Centered linear CKA, feature-space form.

        CKA(X, Y) = ||Y^T X||_F^2 / (||X^T X||_F * ||Y^T Y||_F)

    Args:
        X: [N, D1] activations from network A
        Y: [N, D2] activations from network B
        eps: numerical guard for the denominator

    Returns:
        scalar in [0, 1] (clamped). Returns NaN if both denominators are zero
        (i.e., one or both layers have collapsed to a constant).
    """
    if X.dim() != 2 or Y.dim() != 2:
        raise ValueError(
            f"linear_cka expects 2-D inputs, got X={tuple(X.shape)}, Y={tuple(Y.shape)}"
        )
    if X.shape[0] != Y.shape[0]:
        raise ValueError(
            f"linear_cka: row counts must match, got X={X.shape[0]}, Y={Y.shape[0]}"
        )
    if X.shape[0] < 2:
        raise ValueError("linear_cka requires at least 2 rows for centering")

    X = X.to(torch.float64)
    Y = Y.to(torch.float64)
    X = X - X.mean(0, keepdim=True)
    Y = Y - Y.mean(0, keepdim=True)

    num = (Y.T @ X).pow(2).sum()
    xtx_norm = (X.T @ X).pow(2).sum().sqrt()
    yty_norm = (Y.T @ Y).pow(2).sum().sqrt()
    den = xtx_norm * yty_norm

    if den.item() < eps:
        return float("nan")
    return float((num / den).clamp(0.0, 1.0).item())


def procrustes_distance(X: torch.Tensor, Y: torch.Tensor, eps: float = 1e-30) -> float:
    """
    Williams 2021 orthogonal Procrustes shape distance.

    Both matrices are centered and rescaled to unit Frobenius norm, then we
    find the optimal rotation. The distance is

        d^2 = ||X||^2 + ||Y||^2 - 2 * tr(Σ)
            = 2 * (1 - tr(Σ))

    where Σ is the singular values of X^T Y. This is a true metric: it
    satisfies the triangle inequality, unlike CKA.

    Returns:
        d in [0, sqrt(2)]. 0 = identical up to rotation; sqrt(2) = orthogonal.
    """
    if X.dim() != 2 or Y.dim() != 2:
        raise ValueError(
            f"procrustes_distance expects 2-D inputs, got X={tuple(X.shape)}, Y={tuple(Y.shape)}"
        )
    if X.shape != Y.shape:
        raise ValueError(
            f"procrustes_distance requires X.shape == Y.shape, got "
            f"X={tuple(X.shape)}, Y={tuple(Y.shape)}"
        )

    X = X.to(torch.float64)
    Y = Y.to(torch.float64)
    X = X - X.mean(0, keepdim=True)
    Y = Y - Y.mean(0, keepdim=True)

    x_norm = X.norm()
    y_norm = Y.norm()
    if x_norm.item() < eps or y_norm.item() < eps:
        return float("nan")

    X = X / x_norm
    Y = Y / y_norm
    s = torch.linalg.svdvals(X.T @ Y)
    d_sq = (2.0 * (1.0 - s.sum())).clamp_min(0.0)
    return float(d_sq.sqrt().item())


def compute_layer_drift(
    activations_a: dict[int, torch.Tensor],
    activations_b: dict[int, torch.Tensor],
) -> dict[int, dict[str, float]]:
    """
    Compute CKA + Procrustes drift for every shared layer index.

    Args:
        activations_a: {layer_idx: [N, D]}
        activations_b: {layer_idx: [N, D]}

    Returns:
        {layer_idx: {"cka": float, "procrustes": float}}
    """
    out: dict[int, dict[str, float]] = {}
    for idx in sorted(set(activations_a.keys()) & set(activations_b.keys())):
        X = activations_a[idx]
        Y = activations_b[idx]
        out[idx] = {
            "cka": linear_cka(X, Y),
            "procrustes": procrustes_distance(X, Y),
        }
    return out


def collect_layer_activations(
    model,
    input_ids: torch.Tensor,
    completion_mask: torch.Tensor | None = None,
    chunk_size: int = 16,
) -> dict[int, torch.Tensor]:
    """
    Run model.forward and return per-layer activations mean-pooled over the
    positions selected by `completion_mask` (or all positions if no mask).

    Per the drift research, pooling only over assistant-turn / completion
    positions gives a higher-signal probe than pooling over all positions
    (RL changes are localized there).

    The forward is run in row chunks of `chunk_size` and with
    compute_logits=False, so the full-vocab [B, S, vocab] logits tensor is
    never materialized -- at the real 4096-row probe that tensor alone is
    hundreds of GB and would OOM the GPU before the activations are even read.

    Args:
        model: an InterleavedMoEModel (or any model with .layers)
        input_ids: [B, S] token ids
        completion_mask: [B, S] boolean / float mask, 1 on positions to pool,
                         0 elsewhere. If None, pool over all positions.
        chunk_size: number of rows per forward pass.

    Returns:
        {layer_idx: [B, D]} mean-pooled activations per layer.
    """
    chunks: dict[int, list[torch.Tensor]] = {}
    handles = []

    def make_hook(idx: int, mask_chunk):
        def hook(module, inputs, output):
            x = output[0] if isinstance(output, tuple) else output
            # x: [b, S, D]
            if mask_chunk is not None:
                m = mask_chunk.to(x.dtype).unsqueeze(-1)  # [b, S, 1]
                pooled = (x * m).sum(dim=1) / m.sum(dim=1).clamp_min(1.0)
            else:
                pooled = x.mean(dim=1)
            chunks.setdefault(idx, []).append(pooled.detach().float())
        return hook

    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            for start in range(0, input_ids.shape[0], chunk_size):
                ids_chunk = input_ids[start:start + chunk_size]
                mask_chunk = (
                    completion_mask[start:start + chunk_size]
                    if completion_mask is not None else None
                )
                handles = [
                    layer.register_forward_hook(make_hook(i, mask_chunk))
                    for i, layer in enumerate(model.layers)
                ]
                try:
                    _ = model(ids_chunk, compute_logits=False)
                finally:
                    for h in handles:
                        h.remove()
                    handles = []
    finally:
        if was_training:
            model.train()

    return {idx: torch.cat(parts, dim=0) for idx, parts in chunks.items()}
