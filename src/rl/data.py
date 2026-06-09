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


@dataclass(frozen=True)
class SFTExample:
    prompt: str       # same formatted prompt as RL
    completion: str   # full reference solution, ending in \boxed{answer}
    source: str


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


def _gsm8k_completion(answer_field: str) -> str:
    """Turn a GSM8K answer into a reference completion ending in \\boxed{}.

    The dataset answer is the chain of thought followed by `#### N`. We keep the
    reasoning, drop the `####` line, and append a boxed final answer so the
    target matches the format the RL reward looks for.
    """
    final = gsm8k_gold(answer_field)
    body = answer_field.split("####")[0].strip()
    eos_marker = " "
    return f"{body}{eos_marker}The final answer is \\boxed{{{final}}}."


def load_gsm8k_sft() -> list[SFTExample]:
    from datasets import load_dataset

    ds = load_dataset("openai/gsm8k", "main", split="train")
    return [
        SFTExample(
            format_prompt(row["question"]),
            _gsm8k_completion(row["answer"]),
            "gsm8k",
        )
        for row in ds
    ]


def load_math_l13_sft() -> list[SFTExample]:
    from datasets import concatenate_datasets, load_dataset

    parts = [
        load_dataset("EleutherAI/hendrycks_math", c, split="train").filter(
            lambda x: x["level"] in _MATH_LEVELS
        )
        for c in _MATH_CONFIGS
    ]
    ds = concatenate_datasets(parts)
    # MATH solutions already contain a \boxed{} answer, so the raw solution is a
    # well-formatted completion as-is.
    return [
        SFTExample(format_prompt(row["problem"]), row["solution"].strip(), "math")
        for row in ds
    ]


def load_sft_mix() -> list[SFTExample]:
    """GSM8K + MATH L1-L3 reference solutions for the SFT warm-start."""
    return load_gsm8k_sft() + load_math_l13_sft()


def encode_sft_example(
    ex: SFTExample,
    tokenizer,
    max_len: int = 1024,
) -> tuple[list[int], list[int]]:
    """Tokenize one SFT example into (input_ids, labels).

    Labels mask the prompt with -100 so the loss only supervises completion
    tokens, and append EOS so the model learns to stop.

    When the pair is longer than max_len we trim the *prompt* from the left
    rather than the completion from the right -- truncating the completion
    would leave a row with no supervised tokens (all -100), which makes the
    cross-entropy NaN. Only if the completion alone exceeds max_len do we trim
    it too.
    """
    prompt_ids = tokenizer.encode(ex.prompt)
    completion_ids = tokenizer.encode(ex.completion) + [tokenizer.eos_token_id]

    if len(completion_ids) >= max_len:
        # Pathological: keep the tail of the completion (the boxed answer), drop
        # the prompt entirely.
        completion_ids = completion_ids[-max_len:]
        prompt_ids = []
    else:
        budget = max_len - len(completion_ids)
        prompt_ids = prompt_ids[-budget:]

    input_ids = prompt_ids + completion_ids
    labels = [-100] * len(prompt_ids) + completion_ids
    return input_ids, labels


def collate_sft_batch(
    examples: Sequence[SFTExample],
    tokenizer,
    max_len: int = 1024,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode and right-pad a batch of SFT examples into (input_ids, labels).

    Padded positions get pad_token_id in input_ids and -100 in labels so they
    don't contribute to the loss.
    """
    encoded = [encode_sft_example(ex, tokenizer, max_len) for ex in examples]
    batch_len = max(len(ids) for ids, _ in encoded)
    pad = tokenizer.pad_token_id

    input_rows, label_rows = [], []
    for ids, labels in encoded:
        gap = batch_len - len(ids)
        input_rows.append(ids + [pad] * gap)
        label_rows.append(labels + [-100] * gap)
    return (
        torch.tensor(input_rows, dtype=torch.long),
        torch.tensor(label_rows, dtype=torch.long),
    )


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
