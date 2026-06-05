# Modal Setup

Cloud GPU compute for the interleaved MoE RL experiments. Verified against Modal's June-2026 docs and pricing.

---

## Prerequisites

```bash
.venv/bin/pip install modal
.venv/bin/modal token new   # browser-based auth, ~30 seconds
```

Optional secrets (if you want W&B logging or private HF datasets):
- Create at https://modal.com/secrets
- Name them `wandb-secret` (with `WANDB_API_KEY=...`) and `huggingface-secret` (with `HF_TOKEN=...`)
- If absent, training still works; logging falls back to stdout

---

## What's in the scaffold

| Function | GPU | Purpose | Cost (~) |
|---|---|---|---|
| `run_tests()` | L40S | pytest on real GPU hardware | ~$0.50 (15 min ceiling) |
| `smoke_train(steps=50)` | L40S | 50-step end-to-end pipeline check | ~$1 (30 min) |
| `train(config_name=...)` | H100 | The actual training run, up to 24h | ~$48/12h, ~$95/24h |

Verified GPU pricing (Modal pricing page, per-second billing):

| GPU | $/hr | Use case |
|---|---|---|
| H100 | **$3.95** | Training (default) |
| A100-80GB | $2.50 | Cheaper alt; ~30% slower than H100 on attention |
| L40S | **$1.95** | Tests, smoke runs, eval |
| A10 | $1.10 | Way too small for training a 30-50M MoE |

Per-second billing means a failed 90-second crash costs ~$0.10 on H100, not a full hour. Fail fast and iterate.

---

## Workflow

```bash
# 1. ALWAYS run tests on Modal hardware before paying for H100
modal run modal/modal_app.py::cli --target tests
# Expected: 30/30 tests pass on L40S in ~5 minutes (~$0.20)

# 2. Smoke run: confirms full pipeline works end-to-end
modal run modal/modal_app.py::cli --target smoke --steps 50
# Expected: ~10 minutes on L40S, ~$0.40

# 3. Full run: detach so the job survives your laptop closing
modal run --detach modal/modal_app.py::cli --target full \
  --config-name moe_interleaved
# Expected: ~12h on H100, ~$48
```

The `--detach` flag is critical for long jobs. Modal docs explicitly recommend it for the long-training pattern.

---

## Volumes (persistent storage)

Two named volumes auto-create on first run:

- `interleaved-moe-checkpoints` — model checkpoints, training logs (mounted at `/checkpoints`)
- `interleaved-moe-hf-cache` — HuggingFace dataset cache (mounted at `/cache/hf`)

Inspect / manage:

```bash
modal volume ls
modal volume get interleaved-moe-checkpoints /  # download
modal volume rm interleaved-moe-checkpoints     # nuke (careful)
```

Modal does background commits during long runs and a final commit on container shutdown — you don't need to call `vol.commit()` explicitly unless you need cross-container visibility mid-run.

---

## Cost ceiling for the smallest meaningful experiment

**See `COST_ESTIMATE.md` in the repo root for the full verified breakdown.**

Headline numbers (after $30/month Starter free credit):

| Scenario | Cost |
|---|---|
| Mid estimate **with vLLM** added before final runs | **~$95** |
| Mid estimate **without vLLM** (current scaffold) | ~$145 |
| Worst case | $185-230 |
| With academic credits ($10K available) | **$0** |

Single-line takeaway: **adding vLLM rollouts before Phase 5 cuts paired-run cost from ~$115 to ~$35.** Highest-leverage cost optimization in the project.

---

## Gotchas (verified facts)

1. **24h is the hard per-attempt timeout.** Use the resumable pattern (Volume + `retries=10` + checkpoint loading).
2. **Use `spawn().get()`, not `.remote()`.** Function Calls created by `.remote()` expire after 24h. The local entrypoint already does this correctly.
3. **`add_local_python_source` mounts at container start, not build time.** Code edits don't trigger image rebuilds — you can iterate fast.
4. **`single_use_containers=True` is unnecessary** at this scale (no flash-attn build, no GPU-hungry image build). Modal's default container reuse is fine.
5. **Multi-node is private beta.** Stay on single-node multi-GPU (`gpu="H100:2"`) for now. We don't need >1 GPU for a 30-50M MoE.
6. **No flash-attn install.** The model uses `F.scaled_dot_product_attention` which dispatches to FlashAttention-2 automatically on Hopper. Skipping the flash-attn build saves ~10 minutes of cold-start image build time.
7. **uv_pip_install** is faster than pip_install (Modal's recommendation). We use it.
8. **Image layer ordering matters.** Stable layers first (deps), code last (`add_local_python_source`). Code edits don't bust the deps cache.
