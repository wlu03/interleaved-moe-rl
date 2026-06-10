"""
Modal app for running interleaved MoE RL experiments on cloud GPUs.

Verified facts (June 2026, from Modal docs):
- Per-second billing (failed 90s crash on H100 ≈ $0.10, not a full hour)
- 24h max execution per attempt; chain longer runs via Volume + retries
- H100: ~$3.95/hr  L40S: ~$1.95/hr  A100-80GB: ~$2.50/hr
- Use spawn().get(), NOT .remote() — Function Calls expire at 24h
- modal run --detach to survive laptop closing
- uv_pip_install is faster than pip_install
- add_local_python_source mounts at container start, no rebuild on edits
- Volume.commit() is automatic in background + on shutdown
- Multi-node is private beta; single-node multi-GPU works via "H100:2"

Usage:
    modal token new                                          # one-time auth
    modal run modal/modal_app.py::run_tests                  # pytest sanity
    modal run modal/modal_app.py::smoke_train                # 50-step pipeline check
    modal run --detach modal/modal_app.py::cli               # full run
"""

import modal

# ----- Image: pinned, fast-cached, no flash-attn (we use F.scaled_dot_product_attention) -----
# Order matters: stable layers first, code last. Layer caching makes code edits fast.
image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install(
        # Pin tightly. Torch 2.5.1 is current stable that we tested locally.
        "torch==2.5.1",
        "transformers==4.57.0",
        "datasets>=3.0",
        "math-verify[antlr4_13_2]",   # SymPy-aware reward verification for MATH
        "wandb",
        "numpy",
        "scipy",
        "matplotlib",                 # offline chart rendering in analyze()
        "pytest",
        "tqdm",
    )
    # Code last so edits don't bust the dependency cache.
    .add_local_python_source("src")
    .add_local_dir("./tests", remote_path="/root/tests")
    .add_local_dir("./configs", remote_path="/root/configs", ignore=["*.pyc"])
)

# ----- Volumes: persistent across function invocations -----
checkpoints = modal.Volume.from_name("interleaved-moe-checkpoints", create_if_missing=True)
hf_cache = modal.Volume.from_name("interleaved-moe-hf-cache", create_if_missing=True)

app = modal.App("interleaved-moe-rl", image=image)


# Pre-flight: pytest on cheap GPU. ALWAYS run this before a long job.
@app.function(
    gpu="L40S",            # cheapest GPU, ~$1.95/hr — same CUDA stack as H100
    timeout=15 * 60,       # 15 min ceiling
    volumes={"/cache/hf": hf_cache},
)
def run_tests():
    """Run pytest on Modal GPU hardware. Catches CUDA/import/device issues cheaply."""
    import os
    import subprocess
    import sys

    os.environ["HF_HOME"] = "/cache/hf"

    print("=" * 60)
    print("Environment check")
    print("=" * 60)
    import torch
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f"Device: {torch.cuda.get_device_name(0)}")
        print(f"Mem: {props.total_memory / 1e9:.1f} GB")
        print(f"Compute: {props.major}.{props.minor}")

    print("\n" + "=" * 60)
    print("Running pytest")
    print("=" * 60)
    result = subprocess.run(
        ["python", "-m", "pytest", "/root/tests/", "-v", "--tb=short"],
        cwd="/root",
    )
    sys.exit(result.returncode)


# Smoke training: 50 steps, cheapest GPU. Verifies the pipeline runs.
@app.function(
    gpu="L40S",                # smoke test on cheap GPU first
    timeout=60 * 30,           # 30 minutes
    volumes={
        "/checkpoints": checkpoints,
        "/cache/hf": hf_cache,
    },
)
def smoke_train(steps: int = 50, config_name: str = "moe_interleaved_smoke"):
    """50-step end-to-end smoke run. Confirms data loading, rollout, reward, grad."""
    import os
    os.environ["HF_HOME"] = "/cache/hf"
    os.environ["CHECKPOINT_DIR"] = f"/checkpoints/{config_name}"

    from src.rl.train import main

    print(f"[smoke_train] config={config_name}, steps={steps}")
    providers = _build_providers(config_name, eval_cap=8)
    summary = main(
        config_name=config_name,
        ckpt_dir="/checkpoints",
        resume=False,
        max_steps=steps,
        **providers,
    )
    checkpoints.commit()
    print(f"[smoke_train] done: {summary}")
    return {"status": "ok", "steps": steps, "config": config_name, **summary}


# Provider builder: real GSM8K+MATH data + MATH-500/AIME eval.
def _build_providers(config_name: str, eval_cap=None):
    """Build the real {tokenizer, sample_batch, eval_fn, probe_input_ids} that
    `train.main` injects. Imports `datasets`/`transformers` (Modal-only)."""
    from src.rl.train import load_config, _load_real_tokenizer
    from src.rl.data import (
        load_train_mix, load_eval_set, make_sample_batch,
        make_eval_fn, build_probe_input_ids,
    )

    cfg = load_config(config_name)
    tokenizer = _load_real_tokenizer(cfg.model.vocab_size)

    print("[providers] loading GSM8K + MATH L1-L3 train mix ...")
    train_examples = load_train_mix()
    print(f"[providers] {len(train_examples)} train examples")

    sample_batch = make_sample_batch(train_examples, seed=cfg.seed)
    probe_input_ids = build_probe_input_ids(
        train_examples, tokenizer, n=cfg.probe_size, seed=cfg.seed
    )

    eval_sets = {
        "math500": load_eval_set("math500"),
        "aime2024": load_eval_set("aime2024"),
    }
    eval_fn = make_eval_fn(
        eval_sets,
        max_examples=eval_cap,
        max_new_tokens=cfg.grpo.max_completion_len,
        temperature=0.0,
    )
    return {
        "tokenizer": tokenizer,
        "sample_batch": sample_batch,
        "eval_fn": eval_fn,
        "probe_input_ids": probe_input_ids,
    }


# SFT warm-start: produces the init checkpoint for the sft_then_rl arm.
@app.function(
    gpu="H100",
    timeout=24 * 60 * 60,
    volumes={
        "/checkpoints": checkpoints,
        "/cache/hf": hf_cache,
    },
    secrets=[
        modal.Secret.from_name("wandb-secret", required_keys=[]),
        modal.Secret.from_name("huggingface-secret", required_keys=[]),
    ],
    retries=modal.Retries(initial_delay=0, max_retries=10),
)
def sft(config_name: str = "moe_interleaved", steps: int = 500):
    """Supervised fine-tune the arch behind `config_name` on GSM8K+MATH
    solutions. Writes /checkpoints/<config_name>_sft/latest.pt."""
    import os
    os.environ["HF_HOME"] = "/cache/hf"

    from src.rl.sft import sft_config_for, train_sft
    from src.rl.train import _load_real_tokenizer
    from src.rl.data import load_sft_mix, collate_sft_batch

    cfg = sft_config_for(config_name, max_steps=steps)
    tokenizer = _load_real_tokenizer(cfg.model.vocab_size)

    print("[sft] loading GSM8K + MATH L1-L3 solutions ...")
    examples = load_sft_mix()
    print(f"[sft] {len(examples)} SFT examples")

    import torch
    g = torch.Generator().manual_seed(cfg.seed)
    order = torch.randperm(len(examples), generator=g).tolist()

    def next_batch(tok, batch_size, step, max_len):
        start = (step * batch_size) % len(examples)
        chunk = [examples[order[(start + j) % len(examples)]] for j in range(batch_size)]
        return collate_sft_batch(chunk, tok, max_len=max_len)

    summary = train_sft(cfg, ckpt_dir="/checkpoints", tokenizer=tokenizer,
                        next_batch=next_batch)
    checkpoints.commit()
    print(f"[sft] done: {summary}")
    return {"status": "ok", "config": cfg.name, **summary}


# Full training: H100, up to 24h, resumable from Volume.
@app.function(
    gpu="H100",                                # ~$3.95/hr per-second billed
    timeout=24 * 60 * 60,                      # max allowed per attempt
    volumes={
        "/checkpoints": checkpoints,
        "/cache/hf": hf_cache,
    },
    secrets=[
        # Optional: create at modal.com/secrets if you want W&B / HF Hub access.
        # If absent, training still works; logging falls back to stdout.
        modal.Secret.from_name("wandb-secret", required_keys=[]),
        modal.Secret.from_name("huggingface-secret", required_keys=[]),
    ],
    retries=modal.Retries(initial_delay=0, max_retries=10),
)
def train(config_name: str = "moe_interleaved", resume: bool = True, sft_init: bool = False):
    """Full training run. Resumes from /checkpoints/<config>/latest.pt if present.

    With sft_init=True, the policy warm-starts from
    /checkpoints/<config>_sft/latest.pt (the sft_then_rl arm). Run sft() first.
    """
    import os
    from pathlib import Path

    os.environ["HF_HOME"] = "/cache/hf"
    # main() appends cfg.name to ckpt_dir, so pass the base, not /<config_name>.
    base_ckpt_dir = Path("/checkpoints")
    latest = base_ckpt_dir / config_name / "latest.pt"

    init_from = None
    if sft_init:
        init_from = base_ckpt_dir / f"{config_name}_sft" / "latest.pt"
        if not init_from.exists():
            raise FileNotFoundError(
                f"sft_init=True but {init_from} is missing; run the sft() step first"
            )

    print(f"[train] config={config_name}")
    print(f"[train] resume={resume and latest.exists()} (latest.pt {'present' if latest.exists() else 'absent'})")
    print(f"[train] init_from={init_from}")
    print(f"[train] ckpt_dir={base_ckpt_dir / config_name}")

    from src.rl.train import main

    # Cap eval to a fixed subset so the no-KV-cache greedy decode of the full
    # 530-prompt eval set (x20 cycles) doesn't erode the 24h budget.
    providers = _build_providers(config_name, eval_cap=200)
    summary = main(
        config_name=config_name,
        ckpt_dir=str(base_ckpt_dir),
        resume=resume,
        init_from=str(init_from) if init_from else None,
        **providers,
    )
    checkpoints.commit()
    print(f"[train] done: {summary}")
    return {"status": "ok", "config": config_name, **summary}


# Analysis: render the experiment charts from persisted records.json.
@app.function(
    timeout=60 * 15,
    volumes={"/checkpoints": checkpoints},
)
def analyze(config_names: str = "moe_interleaved,dense_baseline"):
    """Render the six charts into /checkpoints/figures from the records.json
    each run wrote. `config_names` is a comma-separated list of run dirs."""
    from pathlib import Path

    from src.analysis import RunRecords, drift_gap_summary, plot_all
    from src.rl.train import build_model, layer_kinds, load_config

    names = [n.strip() for n in config_names.split(",") if n.strip()]
    runs = []
    for name in names:
        rec = Path("/checkpoints") / name / "records.json"
        if rec.exists():
            runs.append(RunRecords.from_file(rec, name=name))
        else:
            print(f"[analyze] skipping {name}: {rec} missing")

    if not runs:
        return {"status": "no-runs"}

    # Color the per-layer plots using whichever config has MoE layers.
    kinds = None
    for name in names:
        cfg = load_config(name)
        k = layer_kinds(build_model(cfg.model))
        if "moe" in k.values():
            kinds = k
            break

    out_dir = Path("/checkpoints") / "figures"
    written = plot_all(runs, out_dir, layer_kinds=kinds)
    checkpoints.commit()

    summaries = {r.name: drift_gap_summary(r) for r in runs}
    print(f"[analyze] wrote {len(written)} figures to {out_dir}")
    for name, s in summaries.items():
        print(f"[analyze] {name}: {s}")
    return {"status": "ok", "figures": written, "drift_gap": summaries}


# Local entrypoint: dispatches to the right Modal function.
@app.local_entrypoint()
def cli(target: str = "tests", config_name: str = "moe_interleaved", steps: int = 50):
    """
    Usage:
        modal run modal/modal_app.py::cli --target tests
        modal run modal/modal_app.py::cli --target smoke --steps 50
        modal run --detach modal/modal_app.py::cli --target full --config-name moe_interleaved
        modal run --detach modal/modal_app.py::cli --target sft --config-name moe_interleaved --steps 500
        modal run --detach modal/modal_app.py::cli --target sft_then_rl --config-name moe_interleaved
        modal run modal/modal_app.py::cli --target analyze --config-name moe_interleaved,dense_baseline
    """
    if target == "tests":
        run_tests.spawn().get()
    elif target == "smoke":
        out = smoke_train.spawn(steps=steps, config_name=config_name).get()
        print(f"\n[cli] smoke output: {out}")
    elif target == "full":
        # spawn().get() instead of .remote() — .remote() Function Calls expire after 24h
        out = train.spawn(config_name=config_name).get()
        print(f"\n[cli] train output: {out}")
    elif target == "sft":
        out = sft.spawn(config_name=config_name, steps=steps).get()
        print(f"\n[cli] sft output: {out}")
    elif target == "sft_then_rl":
        # Two stages back to back: SFT warm-start, then GRPO from those weights.
        sft_out = sft.spawn(config_name=config_name, steps=steps).get()
        print(f"\n[cli] sft output: {sft_out}")
        rl_out = train.spawn(config_name=config_name, sft_init=True).get()
        print(f"\n[cli] train output: {rl_out}")
    elif target == "analyze":
        # config_name is a comma-separated list of run dirs here.
        out = analyze.spawn(config_names=config_name).get()
        print(f"\n[cli] analyze output: {out}")
    else:
        print(f"Unknown target: {target}. "
              "Choices: tests | smoke | full | sft | sft_then_rl | analyze")
