"""Instrumentation for interleaved MoE RL experiments.

Modules:
    routing    — per-MoE-layer statistics (entropy, load, churn, top-k gap)
    drift      — CKA + Procrustes between checkpoints (representation drift)
    gradients  — per-layer / per-expert gradient norms
    tracker    — orchestrator that wires everything together with cadences
"""

from .routing import RoutingStats, compute_routing_metrics
from .drift import linear_cka, procrustes_distance, compute_layer_drift
from .gradients import GradientTracker, summarize_gradients_by_kind
from .tracker import Tracker

__all__ = [
    "RoutingStats",
    "compute_routing_metrics",
    "linear_cka",
    "procrustes_distance",
    "compute_layer_drift",
    "GradientTracker",
    "summarize_gradients_by_kind",
    "Tracker",
]
