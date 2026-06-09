"""Dataset and eval providers for the GRPO loop.

train.main takes `sample_batch` and `eval_fn` as injectable callables; this
module builds the real ones backed by GSM8K + MATH.

`datasets` only exists on Modal (not the local Py3.9 venv), so the heavy
loaders import it lazily inside the functions. Everything that doesn't touch
it -- prompt formatting, gold extraction, batching, scoring -- is pure and
unit-tested locally.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import torch

from .rewards import answers_equivalent, extract_boxed_answer, extract_gsm8k_answer


SYSTEM_PROMPT = "Solve the problem step by step. Put your final answer in \\boxed{}."


def format_prompt(problem: str) -> str:
    return f"System: {SYSTEM_PROMPT}\nUser: {problem}\nAssistant:"


@dataclass(frozen=True)
class Example:
    prompt: str   # fully formatted, ready to tokenize
    gold: str     # ground-truth answer
    source: str   # "gsm8k" | "math" | eval set name


def gsm8k_gold(answer_field: str) -> str:
    """GSM8K answers end with `#### N`; return the numeric token."""
    extracted = extract_gsm8k_answer(answer_field)
    return extracted if extracted is not None else answer_field.strip()


def math_gold(solution_field: str) -> str:
    """MATH solutions put the answer in the last `\\boxed{}`."""
    boxed = extract_boxed_answer(solution_field)
    return boxed if boxed is not None else solution_field.strip()


def load_gsm8k_train() -> list[Example]:
    from datasets import load_dataset

    ds = load_dataset("openai/gsm8k", "main", split="train")
    return [
        Example(format_prompt(row["question"]), gsm8k_gold(row["answer"]), "gsm8k")
        for row in ds
    ]


_MATH_CONFIGS = ["algebra", "prealgebra", "number_theory"]
_MATH_LEVELS = {"Level 1", "Level 2", "Level 3"}


def load_math_l13_train() -> list[Example]:
    from datasets import concatenate_datasets, load_dataset

    parts = [
        load_dataset("EleutherAI/hendrycks_math", c, split="train").filter(
            lambda x: x["level"] in _MATH_LEVELS
        )
        for c in _MATH_CONFIGS
    ]
    ds = concatenate_datasets(parts)
    return [
        Example(format_prompt(row["problem"]), math_gold(row["solution"]), "math")
        for row in ds
    ]


def load_train_mix() -> list[Example]:
    """GSM8K + MATH L1-L3, the training mix."""
    return load_gsm8k_train() + load_math_l13_train()


def load_eval_set(name: str) -> list[Example]:
    """Load a held-out eval set: "math500" or "aime2024"."""
    from datasets import load_dataset

    if name == "math500":
        ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
        return [
            Example(format_prompt(r["problem"]), math_gold(r["solution"]), "math500")
            for r in ds
        ]
    if name == "aime2024":
        ds = load_dataset("Maxwell-Jia/AIME_2024", split="train")
        return [
            Example(format_prompt(r["Problem"]), str(r["Answer"]).strip(), "aime2024")
            for r in ds
        ]
    raise ValueError(f"unknown eval set: {name!r}")


def make_sample_batch(
    examples: Sequence[Example],
    seed: int = 0,
) -> Callable[[int, int], tuple[list[str], list[str]]]:
    """Build a `sample_batch(b, step)` over a fixed example pool.

    The pool is permuted once (seeded) and then walked in order, so a resumed
    run sees the same prompts at the same step. That determinism is what lets
    the paired dense/MoE runs share an identical data order.
    """
    n = len(examples)
    if n == 0:
        raise ValueError("make_sample_batch: empty example pool")

    g = torch.Generator().manual_seed(seed)
    order = torch.randperm(n, generator=g).tolist()

    def sample_batch(b: int, step: int) -> tuple[list[str], list[str]]:
        start = (step * b) % n
        idx = [order[(start + j) % n] for j in range(b)]
        return (
            [examples[i].prompt for i in idx],
            [examples[i].gold for i in idx],
        )

    return sample_batch


def build_probe_input_ids(
    examples: Sequence[Example],
    tokenizer,
    n: int,
    seed: int = 0,
) -> torch.Tensor:
    """Encode n held-out prompts into a right-padded [n, S] probe tensor for
    the drift snapshots."""
    n = min(n, len(examples))
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(examples), generator=g)[:n].tolist()
    ids = [tokenizer.encode(examples[i].prompt) for i in idx]
    max_len = max(len(x) for x in ids)
    pad = tokenizer.pad_token_id
    return torch.tensor(
        [x + [pad] * (max_len - len(x)) for x in ids], dtype=torch.long
    )


def score_completion(completion: str, gold: str) -> bool:
    """True iff the completion's extracted answer matches gold.

    Prefer `\\boxed{}` (what we train toward), then the GSM8K `#### N`
    extractor, then the raw text.
    """
    pred = extract_boxed_answer(completion)
    if pred is None:
        pred = extract_gsm8k_answer(completion)
    if pred is None:
        pred = completion.strip()
    return answers_equivalent(pred, gold)


def make_eval_fn(
    eval_sets: dict[str, Sequence[Example]],
    *,
    max_examples: Optional[int] = None,
    max_new_tokens: int = 512,
    temperature: float = 0.0,
) -> Callable[..., dict]:
    """Build `eval_fn(model, tokenizer, step) -> {metric: value}`.

    Greedy-decodes each eval prompt once and reports per-set accuracy. Runs in
    eval/no_grad and restores training mode afterward.
    """
    from .rollout import sample_one_prompt

    def eval_fn(model, tokenizer, step: int) -> dict:
        was_training = model.training
        model.eval()
        out: dict[str, float] = {}
        try:
            for set_name, examples in eval_sets.items():
                pool = list(examples)
                if max_examples is not None:
                    pool = pool[:max_examples]
                if not pool:
                    continue
                correct = 0
                for ex in pool:
                    prompt_ids = tokenizer.encode(ex.prompt)
                    seq, _attn, mask = sample_one_prompt(
                        model, prompt_ids,
                        eos_token_id=tokenizer.eos_token_id,
                        pad_token_id=tokenizer.pad_token_id,
                        group_size=1,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                    )
                    comp_ids = seq[0, len(prompt_ids):]
                    comp_mask = mask[0, len(prompt_ids):].bool()
                    valid = comp_ids[comp_mask].tolist()
                    if valid and valid[-1] == tokenizer.eos_token_id:
                        valid = valid[:-1]
                    if score_completion(tokenizer.decode(valid), ex.gold):
                        correct += 1
                out[f"eval/{set_name}/acc"] = correct / len(pool)
        finally:
            if was_training:
                model.train()
        return out

    return eval_fn
