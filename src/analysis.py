"""Offline analysis of training runs.

Each run writes a records.json under its checkpoint dir (the Tracker buffer:
a list of flat {step, metric: value} dicts). This module loads those, pulls
out the series we care about, and renders the six charts from the experiment
plan:

    1. reward curve, configs overlaid
    2. per-layer CKA vs. init over time (dense/MoE colored)
    3. per-layer Procrustes distance, same axes
    4. routing entropy per MoE layer over training
    5. per-expert load distribution at the final step
    6. per-layer grad norm over training

The parsing/series extraction is pure and unit-tested. Plotting imports
matplotlib lazily (Modal-only locally), mirroring how the rest of the repo
keeps heavy deps off the local Py3.9 venv.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class RunRecords:
    """One run's metrics, indexed for series queries."""

    name: str
    records: list[dict] = field(default_factory=list)

    @classmethod
    def from_file(cls, path, name: Optional[str] = None) -> "RunRecords":
        path = Path(path)
        with open(path) as f:
            records = json.load(f)
        # Default the run name to the parent dir (e.g. .../moe_interleaved/records.json).
        return cls(name=name or path.parent.name, records=records)

    def series(self, key: str) -> tuple[list[int], list[float]]:
        """(steps, values) for a metric key, in step order, skipping records
        that don't carry it. NaN values are dropped (drift can be NaN on
        degenerate probe batches)."""
        pairs = []
        for r in self.records:
            if key in r and "step" in r and _is_finite(r[key]):
                pairs.append((r["step"], r[key]))
        pairs.sort(key=lambda p: p[0])
        return [s for s, _ in pairs], [v for _, v in pairs]

    def keys_matching(self, pattern: str) -> list[str]:
        """All metric keys (across records) matching a regex, sorted."""
        rx = re.compile(pattern)
        found = set()
        for r in self.records:
            found.update(k for k in r if rx.search(k))
        return sorted(found)

    def latest(self, key: str) -> Optional[float]:
        """The value of a metric at the highest step that has it."""
        steps, vals = self.series(key)
        return vals[-1] if vals else None


def _is_finite(x) -> bool:
    return isinstance(x, (int, float)) and x == x and abs(x) != float("inf")


# Layer-key helpers ----------------------------------------------------------
_LAYER_RE = re.compile(r"/L(\d+)(?:/|$)")


def layer_index(key: str) -> Optional[int]:
    """Pull the layer index out of a key like 'drift/cka/L2' -> 2."""
    m = _LAYER_RE.search(key)
    return int(m.group(1)) if m else None


def per_layer_series(
    run: RunRecords,
    prefix: str,
) -> dict[int, tuple[list[int], list[float]]]:
    """Map layer index -> (steps, values) for all keys '<prefix>/L<idx>'.

    e.g. per_layer_series(run, "drift/cka") -> {0: (...), 1: (...), ...}
    """
    out: dict[int, tuple[list[int], list[float]]] = {}
    for key in run.keys_matching(re.escape(prefix) + r"/L\d+$"):
        idx = layer_index(key)
        if idx is not None:
            out[idx] = run.series(key)
    return out


def expert_load_at(
    run: RunRecords,
    layer_idx: int,
    step: Optional[int] = None,
) -> dict[int, float]:
    """Per-expert load for one MoE layer at a given step (default: latest).

    The Tracker logs aggregate load_max/gini per layer but not per-expert
    fractions, so this reads the per-expert grad-norm keys as a stand-in proxy
    only if explicit load keys are absent. Prefer explicit 'routing/L*/load/E*'
    keys when present.
    """
    explicit = {}
    for key in run.keys_matching(rf"routing/L{layer_idx}/load/E\d+$"):
        m = re.search(r"/E(\d+)$", key)
        if m:
            val = run.latest(key) if step is None else _value_at(run, key, step)
            if val is not None:
                explicit[int(m.group(1))] = val
    return explicit


def _value_at(run: RunRecords, key: str, step: int) -> Optional[float]:
    steps, vals = run.series(key)
    for s, v in zip(steps, vals):
        if s == step:
            return v
    return None


def drift_gap_summary(run: RunRecords) -> dict[str, Optional[float]]:
    """Final-step mean CKA for MoE vs. dense layers and their gap.

    Reads the by-kind aggregates the Tracker emits. The gap (dense - moe) is
    the headline number for H1: positive means MoE layers drifted more (lower
    CKA) than dense layers in the same model.
    """
    moe = run.latest("drift/cka/by_kind/moe_mean")
    dense = run.latest("drift/cka/by_kind/dense_mean")
    gap = (dense - moe) if (moe is not None and dense is not None) else None
    return {"moe_cka": moe, "dense_cka": dense, "dense_minus_moe": gap}


# Plotting -------------------------------------------------------------------
def plot_all(
    runs: list[RunRecords],
    out_dir,
    layer_kinds: Optional[dict[int, str]] = None,
) -> list[str]:
    """Render the six charts into out_dir; return the written file paths.

    `runs` are typically the 2x2 ablation cells. `layer_kinds` colors the
    per-layer plots (moe vs. dense); if None they're all one color.
    """
    import matplotlib

    matplotlib.use("Agg")  # headless
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []

    def _save(fig, name):
        p = out_dir / name
        fig.savefig(p, bbox_inches="tight", dpi=120)
        plt.close(fig)
        written.append(str(p))

    def _color(idx):
        if layer_kinds is None:
            return None
        return "tab:red" if layer_kinds.get(idx) == "moe" else "tab:blue"

    # 1. reward curve, configs overlaid
    fig, ax = plt.subplots()
    for run in runs:
        steps, vals = run.series("mean_reward")
        if steps:
            ax.plot(steps, vals, label=run.name)
    ax.set(xlabel="step", ylabel="mean reward", title="Reward")
    ax.legend()
    _save(fig, "01_reward.png")

    # 2 & 3. per-layer CKA and Procrustes over time (one fig per run)
    for metric, fname, ylabel in [
        ("drift/cka", "02_cka", "CKA(init, step)"),
        ("drift/procrustes", "03_procrustes", "Procrustes distance"),
    ]:
        for run in runs:
            layers = per_layer_series(run, metric)
            if not layers:
                continue
            fig, ax = plt.subplots()
            for idx in sorted(layers):
                steps, vals = layers[idx]
                if steps:
                    kind = layer_kinds.get(idx) if layer_kinds else None
                    label = f"L{idx}" + (f" ({kind})" if kind else "")
                    ax.plot(steps, vals, label=label, color=_color(idx))
            ax.set(xlabel="step", ylabel=ylabel, title=f"{ylabel} — {run.name}")
            ax.legend(fontsize="small")
            _save(fig, f"{fname}_{run.name}.png")

    # 4. routing entropy per MoE layer over training (one fig per run)
    for run in runs:
        layers = per_layer_series(run, "routing/entropy") or _entropy_layers(run)
        if not layers:
            continue
        fig, ax = plt.subplots()
        for idx in sorted(layers):
            steps, vals = layers[idx]
            if steps:
                ax.plot(steps, vals, label=f"L{idx}", color=_color(idx))
        ax.set(xlabel="step", ylabel="routing entropy (nats)",
               title=f"Routing entropy — {run.name}")
        ax.legend(fontsize="small")
        _save(fig, f"04_routing_entropy_{run.name}.png")

    # 5. per-expert load distribution at the final step (one fig per run)
    for run in runs:
        moe_layers = sorted({
            layer_index(k)
            for k in run.keys_matching(r"routing/L\d+/entropy$")
            if layer_index(k) is not None
        })
        if not moe_layers:
            continue
        fig, ax = plt.subplots()
        plotted = False
        for idx in moe_layers:
            loads = expert_load_at(run, idx)
            if loads:
                xs = sorted(loads)
                ax.bar([x + 0.1 * idx for x in xs], [loads[x] for x in xs],
                       width=0.1, label=f"L{idx}")
                plotted = True
        if plotted:
            ax.set(xlabel="expert", ylabel="load fraction",
                   title=f"Expert load @ final — {run.name}")
            ax.legend(fontsize="small")
            _save(fig, f"05_expert_load_{run.name}.png")
        else:
            plt.close(fig)

    # 6. per-layer grad norm over training (one fig per run)
    for run in runs:
        layers = per_layer_series(run, "grad/by_layer")
        if not layers:
            continue
        fig, ax = plt.subplots()
        for idx in sorted(layers):
            steps, vals = layers[idx]
            if steps:
                kind = layer_kinds.get(idx) if layer_kinds else None
                label = f"L{idx}" + (f" ({kind})" if kind else "")
                ax.plot(steps, vals, label=label, color=_color(idx))
        ax.set(xlabel="step", ylabel="grad norm", title=f"Grad norm — {run.name}")
        ax.legend(fontsize="small")
        _save(fig, f"06_grad_norm_{run.name}.png")

    return written


def _entropy_layers(run: RunRecords) -> dict[int, tuple[list[int], list[float]]]:
    """Per-layer routing entropy series, keyed by layer index.

    The entropy key is 'routing/L<idx>/entropy', which per_layer_series's
    '<prefix>/L<idx>$' pattern doesn't match, so we collect it directly.
    """
    out: dict[int, tuple[list[int], list[float]]] = {}
    for key in run.keys_matching(r"routing/L\d+/entropy$"):
        idx = layer_index(key)
        if idx is not None:
            out[idx] = run.series(key)
    return out


def load_runs(paths_or_dir) -> list[RunRecords]:
    """Load runs from either a list of records.json paths or a parent directory
    containing <run>/records.json files."""
    p = Path(paths_or_dir) if not isinstance(paths_or_dir, list) else None
    if p is not None and p.is_dir():
        files = sorted(p.glob("*/records.json"))
        return [RunRecords.from_file(f) for f in files]
    paths = paths_or_dir if isinstance(paths_or_dir, list) else [paths_or_dir]
    return [RunRecords.from_file(f) for f in paths]
