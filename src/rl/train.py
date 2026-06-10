"""GRPO training loop.

Ties together rollout, the GRPO step, and the instrumentation Tracker, running
the periodic cadences (gradients every step, eval/drift/checkpoint on their own
intervals).

Configs are a Python registry rather than YAML so the loop imports cleanly on
the local Py3.9 venv (no pyyaml/datasets there). For the same reason the
tokenizer, data provider, and eval are injectable: tests drive the loop with a
fake tokenizer and synthetic data, while Modal passes the real GSM8K+MATH
providers.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable, Optional

import torch

from ..config import InterleavedMoEConfig
from ..model import InterleavedMoEModel
from ..instrumentation.tracker import Tracker, TrackerConfig
from ..instrumentation.drift import collect_layer_activations
from .grpo import GRPOConfig, grpo_train_step


@dataclass
class TrainConfig:
    name: str = "default"
    model: InterleavedMoEConfig = field(default_factory=InterleavedMoEConfig)
    grpo: GRPOConfig = field(default_factory=GRPOConfig)

    lr: float = 1e-6
    weight_decay: float = 0.0

    max_steps: int = 1000
    batch_prompts: int = 8       # B prompts per step; each yields G rollouts
    seed: int = 0

    eval_every: int = 50
    drift_every: int = 100
    checkpoint_every: int = 200
    routing_every: int = 10
    grad_every: int = 1

    probe_size: int = 4096       # drift probe set size

    use_wandb: bool = False
    wandb_project: Optional[str] = None


def _moe_config(**overrides) -> InterleavedMoEConfig:
    return replace(InterleavedMoEConfig(), **overrides)


def _dense_config(**overrides) -> InterleavedMoEConfig:
    # moe_every_n_layers <= 0 makes every layer dense.
    base = dict(moe_every_n_layers=0)
    base.update(overrides)
    return replace(InterleavedMoEConfig(), **base)


# Named configs. The *_smoke variants are tiny enough to run on CPU in seconds;
# the others are the paper-scale experiment.
_REGISTRY: dict[str, Callable[[], TrainConfig]] = {
    "moe_interleaved": lambda: TrainConfig(
        name="moe_interleaved",
        model=_moe_config(),
        grpo=GRPOConfig(),
        max_steps=1000,
    ),
    "dense_baseline": lambda: TrainConfig(
        name="dense_baseline",
        model=_dense_config(),
        grpo=GRPOConfig(),
        max_steps=1000,
    ),
    "moe_interleaved_smoke": lambda: TrainConfig(
        name="moe_interleaved_smoke",
        model=_moe_config(
            hidden_size=64, num_layers=4, num_attention_heads=4,
            num_kv_heads=2, intermediate_size=128, expert_intermediate_size=128,
            num_experts=4, num_experts_per_tok=2, moe_every_n_layers=2,
            vocab_size=2048, max_position_embeddings=512,
        ),
        grpo=GRPOConfig(group_size=4, max_completion_len=64),
        lr=1e-4, max_steps=50, batch_prompts=4,
        eval_every=25, drift_every=25, checkpoint_every=25, probe_size=64,
    ),
    "dense_baseline_smoke": lambda: TrainConfig(
        name="dense_baseline_smoke",
        model=_dense_config(
            hidden_size=64, num_layers=4, num_attention_heads=4,
            num_kv_heads=2, intermediate_size=128,
            vocab_size=2048, max_position_embeddings=512,
        ),
        grpo=GRPOConfig(group_size=4, max_completion_len=64),
        lr=1e-4, max_steps=50, batch_prompts=4,
        eval_every=25, drift_every=25, checkpoint_every=25, probe_size=64,
    ),
}


def load_config(name: str) -> TrainConfig:
    if name not in _REGISTRY:
        raise KeyError(f"unknown config {name!r}; known: {sorted(_REGISTRY)}")
    return _REGISTRY[name]()


def register_config(name: str, factory: Callable[[], TrainConfig]) -> None:
    _REGISTRY[name] = factory


def build_model(model_cfg: InterleavedMoEConfig) -> InterleavedMoEModel:
    return InterleavedMoEModel(model_cfg)


def layer_kinds(model: InterleavedMoEModel) -> dict[int, str]:
    """Map layer index -> "moe" | "dense" for per-kind drift aggregation."""
    return {
        i: ("moe" if getattr(layer, "is_moe", False) else "dense")
        for i, layer in enumerate(model.layers)
    }


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    path: Path,
    extra: Optional[dict] = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Write to a temp file then rename so we never leave a half-written latest.pt.
    tmp = path.with_suffix(path.suffix + ".tmp")
    payload = {
        "step": step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    if extra:
        payload.update(extra)
    torch.save(payload, tmp)
    tmp.replace(path)


def load_checkpoint(
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    path: Path,
) -> int:
    ckpt = torch.load(Path(path), map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    return int(ckpt.get("step", 0))


def load_checkpoint_field(path: Path, key: str):
    """Read a single auxiliary field (e.g. the persisted drift baseline) from a
    checkpoint without touching model/optimizer state. Returns None if absent."""
    ckpt = torch.load(Path(path), map_location="cpu", weights_only=False)
    return ckpt.get(key)


def _synthetic_data_provider(vocab_size: int):
    """Deterministic toy data so the loop runs without `datasets` installed."""
    prompts = [f"problem {i}: 2 + {i} =" for i in range(64)]
    golds = [str(2 + i) for i in range(64)]

    def sample_batch(b: int, step: int):
        start = (step * b) % len(prompts)
        idx = [(start + j) % len(prompts) for j in range(b)]
        return [prompts[i] for i in idx], [golds[i] for i in idx]

    return sample_batch, prompts, golds


def collect_init_snapshot(
    model: InterleavedMoEModel,
    probe_input_ids: torch.Tensor,
    probe_mask: torch.Tensor | None = None,
) -> dict[int, torch.Tensor]:
    """Per-layer activation snapshot used as the drift baseline."""
    return collect_layer_activations(model, probe_input_ids, probe_mask)


def _probe_routing_stats(
    model: InterleavedMoEModel,
    probe_input_ids: torch.Tensor,
    chunk_size: int = 16,
) -> dict[int, dict]:
    """Per-MoE-layer raw routing stats over the probe batch, in the same shape
    MoELayer emits so RoutingTracker can consume it.

    Runs in row chunks with compute_logits=False so the full-vocab logits
    tensor is never built. entropy and expert_load are token-count-weighted
    means across chunks; the small [N, K] topk index/weight tensors are
    concatenated (K=2 so this is cheap) so the tracker's top1/top2 gap and
    token churn still work.
    """
    was_training = model.training
    model.eval()
    ent: dict[int, torch.Tensor] = {}
    load: dict[int, torch.Tensor] = {}
    idxs: dict[int, list] = {}
    wts: dict[int, list] = {}
    counts: dict[int, int] = {}
    try:
        with torch.no_grad():
            for start in range(0, probe_input_ids.shape[0], chunk_size):
                ids = probe_input_ids[start:start + chunk_size]
                stats = model(ids, compute_logits=False)["routing_stats"]
                n = ids.shape[0]
                for li, raw in stats.items():
                    ent[li] = raw["router_entropy"] * n + ent.get(li, 0.0)
                    load[li] = raw["expert_load"] * n + (load[li] if li in load else 0.0)
                    idxs.setdefault(li, []).append(raw["topk_indices"])
                    wts.setdefault(li, []).append(raw["topk_weights"])
                    counts[li] = counts.get(li, 0) + n
    finally:
        if was_training:
            model.train()
    return {
        li: {
            "router_entropy": ent[li] / counts[li],
            "expert_load": load[li] / counts[li],
            "topk_indices": torch.cat(idxs[li], dim=0),
            "topk_weights": torch.cat(wts[li], dim=0),
        }
        for li in counts
    }


def main(
    config_name: str = "moe_interleaved",
    ckpt_dir: str | Path = "checkpoints",
    resume: bool = True,
    *,
    tokenizer=None,
    reward_fn: Optional[Callable[..., float]] = None,
    sample_batch: Optional[Callable[[int, int], tuple[list[str], list[str]]]] = None,
    probe_input_ids: Optional[torch.Tensor] = None,
    eval_fn: Optional[Callable[..., dict]] = None,
    tracker: Optional[Tracker] = None,
    device: Optional[str] = None,
    max_steps: Optional[int] = None,
    init_from: Optional[str | Path] = None,
) -> dict:
    """Run GRPO training and return a small summary dict.

    Everything past `resume` is injectable for testing; on Modal only
    config_name / ckpt_dir / resume (and the real providers) are passed.

    `init_from` points at an SFT checkpoint whose model weights seed the policy
    (fresh optimizer) -- this is the `sft_then_rl` warm-start. It's ignored when
    resuming from an existing RL checkpoint, which already reflects it.
    """
    cfg = load_config(config_name)
    if max_steps is not None:
        cfg = replace(cfg, max_steps=max_steps)

    torch.manual_seed(cfg.seed)

    dev = torch.device(device) if device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = build_model(cfg.model).to(dev)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )

    if tokenizer is None:
        tokenizer = _load_real_tokenizer(cfg.model.vocab_size)

    if sample_batch is None:
        sample_batch, _prompts, _golds = _synthetic_data_provider(cfg.model.vocab_size)

    # Build a probe batch for drift if one wasn't supplied.
    if probe_input_ids is None:
        probe_prompts, _ = sample_batch(min(cfg.probe_size, 32), 0)
        probe_ids = [tokenizer.encode(p) for p in probe_prompts]
        max_len = max(len(p) for p in probe_ids)
        pad = tokenizer.pad_token_id
        probe_input_ids = torch.tensor(
            [p + [pad] * (max_len - len(p)) for p in probe_ids], dtype=torch.long
        )
    probe_input_ids = probe_input_ids.to(dev)
    # Pool drift over real (non-pad) positions only; padding hidden states are
    # length-dependent noise that dilutes the signal.
    probe_mask = (probe_input_ids != tokenizer.pad_token_id).long()

    if tracker is None:
        tracker = Tracker(
            TrackerConfig(
                routing_every=cfg.routing_every,
                drift_every=cfg.drift_every,
                grad_every=cfg.grad_every,
                eval_every=cfg.eval_every,
                use_wandb=cfg.use_wandb,
                wandb_project=cfg.wandb_project,
                wandb_run_name=cfg.name,
            )
        )
    tracker.start()

    ckpt_dir = Path(ckpt_dir) / cfg.name
    latest = ckpt_dir / "latest.pt"
    start_step = 0
    resuming = resume and latest.exists()

    # Warm-start from SFT weights before snapshotting, so drift is measured
    # relative to the SFT init -- but only on a fresh run, since an existing RL
    # checkpoint already carries the warm-started weights.
    if init_from is not None and not resuming:
        load_checkpoint(model, None, Path(init_from))
        print(f"[train] initialized policy from SFT checkpoint {init_from}")

    kinds = layer_kinds(model)

    if resuming:
        start_step = load_checkpoint(model, optimizer, latest)
        print(f"[train] resumed from {latest} at step {start_step}")
        # Restore the drift baseline saved on the original fresh run. Recomputing
        # it here would snapshot the *resumed* weights, making all post-resume
        # drift meaningless. Fall back to current weights only for legacy
        # checkpoints that predate baseline persistence.
        init_acts = load_checkpoint_field(latest, "init_acts")
        if init_acts is not None:
            init_acts = {int(k): v.to(dev) for k, v in init_acts.items()}
        else:
            init_acts = collect_init_snapshot(model, probe_input_ids, probe_mask)
    else:
        init_acts = collect_init_snapshot(model, probe_input_ids, probe_mask)

    last_step = start_step
    last_metrics: dict = {}
    try:
        for step in range(start_step, cfg.max_steps):
            prompts, golds = sample_batch(cfg.batch_prompts, step)
            metrics = grpo_train_step(
                model, tokenizer, optimizer,
                prompts=prompts, golds=golds, cfg=cfg.grpo,
                step=step, reward_fn=reward_fn,
            )
            last_metrics = metrics

            tracker.log_step(
                step,
                loss=metrics.get("loss", 0.0),
                mean_reward=metrics.get("mean_reward", 0.0),
                reward_std=metrics.get("reward_std", 0.0),
                clip_frac=metrics.get("clip_frac", 0.0),
                nonzero_adv_frac=metrics.get("nonzero_adv_frac", 0.0),
            )

            if step % cfg.grad_every == 0:
                tracker.log_gradients(step, model)

            if step % cfg.routing_every == 0:
                routing_stats = _probe_routing_stats(model, probe_input_ids)
                if routing_stats:
                    tracker.log_routing(step, routing_stats)

            if step % cfg.eval_every == 0 and step > start_step and eval_fn is not None:
                tracker.log_step(step, **eval_fn(model, tokenizer, step))

            if step % cfg.drift_every == 0 and step > start_step:
                cur_acts = collect_layer_activations(model, probe_input_ids, probe_mask)
                tracker.log_drift(step, init_acts, cur_acts, layer_kinds=kinds)

            if step % cfg.checkpoint_every == 0 and step > start_step:
                save_checkpoint(model, optimizer, step, latest,
                                extra={"init_acts": init_acts})

            last_step = step + 1
    finally:
        # Stamp the real last-completed step (not cfg.max_steps) so a crash mid-run
        # leaves a checkpoint that resumes from where it stopped, not one that
        # falsely reports completion. Persist the drift baseline alongside it.
        save_checkpoint(model, optimizer, last_step, latest,
                        extra={"init_acts": init_acts})
        tracker.dump_records(ckpt_dir / "records.json")
        tracker.finish()

    return {
        "config": cfg.name,
        "final_step": last_step,
        "last_metrics": last_metrics,
        "ckpt": str(latest),
    }


def smoke_train_loop(config_name: str = "moe_interleaved_smoke", steps: int = 50) -> dict:
    """Run the full pipeline for a few steps on synthetic data.

    Used by modal/modal_app.py::smoke_train to verify the wiring before paying
    for a real GPU run.
    """
    return main(config_name=config_name, ckpt_dir="checkpoints", resume=False,
                max_steps=steps)


def _load_real_tokenizer(vocab_size: int):
    """Qwen2.5-Math tokenizer on Modal, with a byte-level fallback ONLY when
    `transformers` isn't installed (the local Py3.9 venv).

    The except is narrowed to ImportError on purpose: a download/network/HF-Hub
    failure must propagate and crash the run (Modal retries handle it) rather
    than silently swap in the byte tokenizer and train the real model on
    garbage token ids. We also assert the real tokenizer's vocab fits the model
    config, so a mismatched tokenizer fails loudly instead of indexing OOB.
    """
    try:
        from transformers import AutoTokenizer
    except ImportError:
        return _ByteTokenizer(vocab_size)

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Math-1.5B")
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    if len(tok) > vocab_size:
        raise ValueError(
            f"tokenizer vocab {len(tok)} exceeds model vocab_size {vocab_size}; "
            "update config.vocab_size to match the tokenizer"
        )
    return tok


class _ByteTokenizer:
    """Minimal byte-level tokenizer for local smoke runs without transformers."""

    def __init__(self, vocab_size: int):
        self.vocab_size = vocab_size
        self.pad_token_id = 0
        self.eos_token_id = 1

    def encode(self, text: str) -> list[int]:
        ids = [(b % (self.vocab_size - 2)) + 2 for b in text.encode("utf-8")]
        return ids or [2]

    def decode(self, ids) -> str:
        out = bytearray()
        for t in ids:
            t = int(t)
            if t in (self.pad_token_id, self.eos_token_id):
                continue
            out.append((t - 2) % 256)
        return out.decode("utf-8", errors="replace")
