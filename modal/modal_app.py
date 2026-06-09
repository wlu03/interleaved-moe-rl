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


# =====================================================================
#  Pre-flight: pytest on cheap GPU. ALWAYS run this before a long job.
# =====================================================================
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


# =====================================================================
#  Smoke training: 50 steps, cheapest GPU. Verifies the pipeline runs.
# =====================================================================
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
    summary = main(
        config_name=config_name,
        ckpt_dir="/checkpoints",
        resume=False,
        max_steps=steps,
    )
    checkpoints.commit()
    print(f"[smoke_train] done: {summary}")
    return {"status": "ok", "steps": steps, "config": config_name, **summary}


# =====================================================================
#  Full training: H100, up to 24h, resumable from Volume.
# =====================================================================
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
def train(config_name: str = "moe_interleaved", resume: bool = True):
    """Full training run. Resumes from /checkpoints/<config>/latest.pt if present."""
    import os
    from pathlib import Path

    os.environ["HF_HOME"] = "/cache/hf"
    # main() appends cfg.name to ckpt_dir, so pass the base, not /<config_name>.
    base_ckpt_dir = Path("/checkpoints")
    latest = base_ckpt_dir / config_name / "latest.pt"
    print(f"[train] config={config_name}")
    print(f"[train] resume={resume and latest.exists()} (latest.pt {'present' if latest.exists() else 'absent'})")
    print(f"[train] ckpt_dir={base_ckpt_dir / config_name}")

    from src.rl.train import main

    summary = main(
        config_name=config_name,
        ckpt_dir=str(base_ckpt_dir),
        resume=resume,
    )
    checkpoints.commit()
    print(f"[train] done: {summary}")
    return {"status": "ok", "config": config_name, **summary}


# =====================================================================
#  Local entrypoint: dispatches to the right Modal function.
# =====================================================================
@app.local_entrypoint()
def cli(target: str = "tests", config_name: str = "moe_interleaved", steps: int = 50):
    """
    Usage:
        modal run modal/modal_app.py::cli --target tests
        modal run modal/modal_app.py::cli --target smoke --steps 50
        modal run --detach modal/modal_app.py::cli --target full --config-name moe_interleaved
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
    else:
        print(f"Unknown target: {target}. Choices: tests | smoke | full")
