# Interleaved MoE + RL Dynamics

**Investigating dense vs. MoE layer specialization during reinforcement learning in interleaved architectures.**

## Overview

This repository contains experiments, code, and analysis for studying a key open question in modern LLM architectures:

> In **interleaved** Mixture-of-Experts models (alternating dense FFN and MoE layers), do dense layers preferentially absorb RL gradients for reasoning, while MoE layers better preserve pre-training knowledge? Does interleaving provide implicit stability against router drift?

This question is motivated by recent models like **MAI-Thinking-1**, **Llama 4 (Scout/Maverick)**, and findings from full-MoE systems (Qwen, DeepSeek) showing router drift and "Super Expert" invariance under GRPO-style RL.

## Research Goals

- Measure gradient flow through dense vs. MoE layers during RL
- Quantify representation drift (CKA/SVCCA) pre- vs. post-RL by layer type
- Test whether interleaving mitigates router drift and collapse
- Perform causal ablations (freeze/zero dense vs. MoE layers) to localize reasoning vs. knowledge capabilities
- Understand why interleaved designs (like MAI) appear more stable during long RL runs

## Key Hypotheses

1. **Gradient Absorption**: Dense layers receive stronger RL gradient signals (stable reasoning substrate).
2. **Stability Anchor**: Dense layers reduce input distribution shift to MoE routers → less router drift.
3. **Capability Split**: Reasoning emerges primarily in dense layers; core knowledge remains distributed in MoE experts.
4. **Super Expert Invariance**: Holds more robustly in interleaved setups.

## Key Features

- Clean implementation of **interleaved** Transformer blocks (dense ↔ MoE)
- Efficient MoE dispatch with routing statistics
- Comprehensive tracking:
  - Per-layer gradient norms
  - Linear CKA for representation drift
  - Router entropy, Gini coefficient, weight drift
- Ablation manager (freeze dense/MoE, zero at inference)
- Ready for GRPO / PPO-style RL (TRL compatible)

## Citation
```bash
@misc{interleaved-moe-rl,
  title = {Interleaved MoE + RL Dynamics},
  author = {Wesley Lu},
  year = {2026},
  note = {https://github.com/wlu314/interleaved-moe-rl}
}
```
