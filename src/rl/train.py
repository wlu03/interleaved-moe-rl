"""GRPO training loop — the final M3 slice.

Wires together the rollout sampler (`rollout`), the GRPO step
(`grpo_train_step`), and the instrumentation `Tracker`, with the periodic
cadences from NEXT_STEPS.md §3:

    - routing stats   every cfg.routing_every steps (Tracker owns the gate)
    - gradient norms  every step
    - eval            every cfg.eval_every steps
    - drift (CKA/Proc) every cfg.drift_every steps vs. a stashed init snapshot
    - checkpoint      every cfg.checkpoint_every steps, to ckpt_dir/latest.pt

Configs are a Python registry (not YAML) so the loop is importable and
testable on the local Python 3.9 venv where pyyaml / datasets aren't
installed. Dataset and eval providers are injectable for the same reason —
the loop logic is exercised by tests with tiny synthetic data, while Modal
passes the real GSM8K + MATH mix.
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


# ---- Loop config -------------------------------------------------------
@dataclass
class TrainConfig:
    """Everything the loop needs: model arch, GRPO hyperparams, cadences."""

    name: str = "default"
    model: InterleavedMoEConfig = field(default_factory=InterleavedMoEConfig)
    grpo: GRPOConfig = field(default_factory=GRPOConfig)

    # Optimizer
    lr: float = 1e-6
    weight_decay: float = 0.0

    # Loop
    max_steps: int = 1000
    batch_prompts: int = 8            # B prompts per step (each yields G rollouts)
    seed: int = 0

    # Cadences
    eval_every: int = 50
    drift_every: int = 100
    checkpoint_every: int = 200
    routing_every: int = 10
    grad_every: int = 1

    # Probe set for drift
    probe_size: int = 4096

    # W&B
    use_wandb: bool = False
    wandb_project: Optional[str] = None


# ---- Config registry ---------------------------------------------------
def _moe_config(**overrides) -> InterleavedMoEConfig:
    return replace(InterleavedMoEConfig(), **overrides)


def _dense_config(**overrides) -> InterleavedMoEConfig:
    # "Dense baseline": moe_every_n_layers <= 0 makes every layer dense.
    base = dict(moe_every_n_layers=0)
    base.update(overrides)
    return replace(InterleavedMoEConfig(), **base)


# Named configs. Smoke variants are tiny so the pipeline runs on CPU in
# seconds; full variants match the paper-scale experiment in NEXT_STEPS.md §5.
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
        raise KeyError(
            f"unknown config {name!r}; known: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[name]()


def register_config(name: str, factory: Callable[[], TrainConfig]) -> None:
    """Register a custom config factory (used by tests / ablations)."""
    _REGISTRY[name] = factory


# ---- Builders ----------------------------------------------------------
def build_model(model_cfg: InterleavedMoEConfig) -> InterleavedMoEModel:
    return InterleavedMoEModel(model_cfg)


def layer_kinds(model: InterleavedMoEModel) -> dict[int, str]:
    """Map layer index → "moe" | "dense" for per-kind drift aggregation."""
    return {
        i: ("moe" if getattr(layer, "is_moe", False) else "dense")
        for i, layer in enumerate(model.layers)
    }


# ---- Checkpointing -----------------------------------------------------
def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    path: Path,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "step": step,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        },
        tmp,
    )
    tmp.replace(path)  # atomic on POSIX — never leave a half-written latest.pt


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


# ---- Default data / eval providers (overridable) -----------------------
def _synthetic_data_provider(vocab_size: int):
    """Tiny deterministic data so the loop is runnable without `datasets`.

    Returns (sample_batch, probe_ids) where sample_batch(B, step) → (prompts,
    golds) of decoded-ish strings, and probe_ids is a [n, S] int tensor for
    drift snapshots.
    """
    prompts = [f"problem {i}: 2 + {i} =" for i in range(64)]
    golds = [str(2 + i) for i in range(64)]

    def sample_batch(b: int, step: int):
        # Deterministic rotating window, no RNG (keeps runs reproducible).
        start = (step * b) % len(prompts)
        idx = [(start + j) % len(prompts) for j in range(b)]
        return [prompts[i] for i in idx], [golds[i] for i in idx]

    return sample_batch, prompts, golds


def collect_init_snapshot(
    model: InterleavedMoEModel,
    probe_input_ids: torch.Tensor,
) -> dict[int, torch.Tensor]:
    """Per-layer activation snapshot used as the drift baseline."""
    return collect_layer_activations(model, probe_input_ids)


# ---- Main loop ---------------------------------------------------------
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
) -> dict:
    """Run GRPO training.

    Most arguments are injectable so the loop can be unit-tested with a fake
    tokenizer and synthetic data. On Modal, only config_name / ckpt_dir /
    resume are passed and the real GSM8K+MATH providers are wired here.

    Returns a small summary dict (final step, last metrics).
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

    # Data provider: real on Modal, synthetic fallback locally.
    if sample_batch is None:
        sample_batch, _prompts, _golds = _synthetic_data_provider(cfg.model.vocab_size)

    # Probe set for drift: encode a held-out batch of prompts.
    if probe_input_ids is None:
        probe_prompts, _ = sample_batch(min(cfg.probe_size, 32), 0)
        probe_ids = [tokenizer.encode(p) for p in probe_prompts]
        max_len = max(len(p) for p in probe_ids)
        pad = tokenizer.pad_token_id
        probe_input_ids = torch.tensor(
            [p + [pad] * (max_len - len(p)) for p in probe_ids],
            dtype=torch.long,
        )
    probe_input_ids = probe_input_ids.to(dev)

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

    kinds = layer_kinds(model)
    init_acts = collect_init_snapshot(model, probe_input_ids)

    ckpt_dir = Path(ckpt_dir) / cfg.name
    latest = ckpt_dir / "latest.pt"
    start_step = 0
    if resume and latest.exists():
        start_step = load_checkpoint(model, optimizer, latest)
        print(f"[train] resumed from {latest} at step {start_step}")

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
            )

            if step % cfg.grad_every == 0:
                tracker.log_gradients(step, model)

            if step % cfg.eval_every == 0 and eval_fn is not None:
                eval_metrics = eval_fn(model, tokenizer, step)
                tracker.log_step(step, **eval_metrics)

            if step % cfg.drift_every == 0 and step > start_step:
                cur_acts = collect_layer_activations(model, probe_input_ids)
                tracker.log_drift(step, init_acts, cur_acts, layer_kinds=kinds)

            if step % cfg.checkpoint_every == 0 and step > start_step:
                save_checkpoint(model, optimizer, step, latest)
    finally:
        # Always leave a final checkpoint behind.
        save_checkpoint(model, optimizer, cfg.max_steps, latest)
        tracker.finish()

    return {
        "config": cfg.name,
        "final_step": cfg.max_steps,
        "last_metrics": last_metrics,
        "ckpt": str(latest),
    }


def smoke_train_loop(config_name: str = "moe_interleaved_smoke", steps: int = 50) -> dict:
    """Entry point used by modal/modal_app.py::smoke_train.

    Runs the full pipeline (rollout → loss → step → instrumentation →
    checkpoint) for a handful of steps on the synthetic provider, verifying
    the wiring end-to-end before paying for a real GPU run.
    """
    return main(config_name=config_name, ckpt_dir="checkpoints", resume=False,
                max_steps=steps)


# ---- Real tokenizer (Modal-only; lazy import) --------------------------
def _load_real_tokenizer(vocab_size: int):
    """Load the Qwen2.5-Math tokenizer on Modal. Falls back to a byte-level
    stub locally (where `transformers` may be absent) so imports never fail.
    """
    try:
        from transformers import AutoTokenizer  # type: ignore

        tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Math-1.5B")
        if tok.pad_token_id is None:
            tok.pad_token_id = tok.eos_token_id
        return tok
    except Exception:
        return _ByteTokenizer(vocab_size)


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
