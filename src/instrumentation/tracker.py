"""Orchestrator for instrumentation: combines routing, drift, and gradient
tracking with cadences (drift every 100 steps, routing every 10, etc.).

W&B integration is optional — if wandb isn't installed or no run is active,
metrics are emitted to stdout / a list buffer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from .gradients import GradientTracker
from .routing import RoutingTracker, collapse_signal


@dataclass
class TrackerConfig:
    routing_every: int = 10
    drift_every: int = 100
    grad_every: int = 1
    eval_every: int = 50
    use_wandb: bool = False
    wandb_project: str | None = None
    wandb_run_name: str | None = None
    collapse_threshold: float = 0.7


class Tracker:
    """
    Single entry point for all logging during training. Owns cadences for
    each instrumentation kind and emits flattened metrics to W&B (or buffers
    them locally).

    Usage:
        tracker = Tracker(cfg)
        tracker.start()                                       # init wandb if requested
        for step in range(num_steps):
            ...
            if step % cfg.routing_every == 0:
                tracker.log_routing(step, routing_stats)
            if step % cfg.grad_every == 0:
                tracker.log_gradients(step, model)
            if step % cfg.drift_every == 0:
                tracker.log_drift(step, init_acts, cur_acts, layer_kinds)
            tracker.log_step(step, loss=..., reward=...)
        tracker.finish()
    """

    def __init__(self, cfg: TrackerConfig | None = None) -> None:
        self.cfg = cfg or TrackerConfig()
        self.routing = RoutingTracker()
        self.gradients = GradientTracker()
        self.local_buffer: list[dict[str, Any]] = []
        self._wandb_run = None
        self._collapse_warned: set[int] = set()

    # ---- W&B lifecycle ------------------------------------------------
    def start(self) -> None:
        if not self.cfg.use_wandb:
            return
        try:
            import wandb  # type: ignore
        except ImportError:
            print("[Tracker] wandb not installed; falling back to stdout")
            self.cfg.use_wandb = False
            return
        self._wandb_run = wandb.init(
            project=self.cfg.wandb_project,
            name=self.cfg.wandb_run_name,
        )

    def finish(self) -> None:
        if self._wandb_run is not None:
            try:
                self._wandb_run.finish()
            except Exception:
                pass
            self._wandb_run = None

    # ---- Core emit ---------------------------------------------------
    def _emit(self, step: int, metrics: dict[str, Any]) -> None:
        """Emit a flat dict of metrics to wandb (if active) and the local buffer."""
        record = {"step": step, **metrics}
        self.local_buffer.append(record)
        if self._wandb_run is not None:
            self._wandb_run.log(metrics, step=step)

    # ---- Convenience loggers ----------------------------------------
    def log_step(self, step: int, **scalars: float) -> None:
        self._emit(step, dict(scalars))

    def log_routing(self, step: int, routing_stats_by_layer: dict[int, dict]) -> None:
        agg = self.routing.update(routing_stats_by_layer)
        flat: dict[str, Any] = {}
        for layer_idx, stats in agg.items():
            prefix = f"routing/L{layer_idx}"
            flat[f"{prefix}/entropy"] = stats.entropy
            flat[f"{prefix}/load_max"] = stats.expert_load_max
            flat[f"{prefix}/load_gini"] = stats.expert_load_gini
            flat[f"{prefix}/top1_top2_gap"] = stats.top1_top2_gap
            if stats.token_churn is not None:
                flat[f"{prefix}/token_churn"] = stats.token_churn

            # Collapse warning (one-shot per layer)
            if (
                collapse_signal(stats, self.cfg.collapse_threshold)
                and layer_idx not in self._collapse_warned
            ):
                print(
                    f"[Tracker] step={step} layer={layer_idx} "
                    f"ROUTING COLLAPSE WARNING: load_max={stats.expert_load_max:.3f} "
                    f"> threshold {self.cfg.collapse_threshold}"
                )
                self._collapse_warned.add(layer_idx)
        if flat:
            self._emit(step, flat)

    def log_gradients(self, step: int, model: torch.nn.Module) -> None:
        summary = self.gradients.step(model)
        flat: dict[str, Any] = {"grad/global_norm": summary.global_norm}
        for kind, norm in summary.by_layer_kind.items():
            flat[f"grad/by_kind/{kind}"] = norm
        for layer_idx, norm in summary.by_layer.items():
            flat[f"grad/by_layer/L{layer_idx}"] = norm
        for (layer_idx, expert_idx), norm in summary.by_expert.items():
            flat[f"grad/by_expert/L{layer_idx}_E{expert_idx}"] = norm
        self._emit(step, flat)

    def log_drift(
        self,
        step: int,
        baseline_acts: dict[int, torch.Tensor],
        current_acts: dict[int, torch.Tensor],
        layer_kinds: dict[int, str] | None = None,
    ) -> None:
        """
        Compute and log per-layer drift between baseline and current activations.

        layer_kinds maps layer_idx → "dense" or "moe" so we can produce the
        per-kind aggregates that are the core analysis output.
        """
        from .drift import compute_layer_drift

        per_layer = compute_layer_drift(baseline_acts, current_acts)
        flat: dict[str, Any] = {}
        cka_by_kind: dict[str, list[float]] = {}
        proc_by_kind: dict[str, list[float]] = {}

        for layer_idx, metrics in per_layer.items():
            cka = metrics["cka"]
            proc = metrics["procrustes"]
            flat[f"drift/cka/L{layer_idx}"] = cka
            flat[f"drift/procrustes/L{layer_idx}"] = proc

            if layer_kinds is not None:
                kind = layer_kinds.get(layer_idx, "unknown")
                if cka == cka:  # not NaN
                    cka_by_kind.setdefault(kind, []).append(cka)
                if proc == proc:
                    proc_by_kind.setdefault(kind, []).append(proc)

        for kind, vals in cka_by_kind.items():
            flat[f"drift/cka/by_kind/{kind}_mean"] = sum(vals) / len(vals)
        for kind, vals in proc_by_kind.items():
            flat[f"drift/procrustes/by_kind/{kind}_mean"] = sum(vals) / len(vals)

        self._emit(step, flat)

    def get_records(self) -> list[dict[str, Any]]:
        """Return everything emitted so far (useful for tests / offline analysis)."""
        return list(self.local_buffer)
